from math import sqrt
from itertools import product
from collections import namedtuple
import torch
from einops import rearrange, repeat

from .basis import get_basis
from .utils import exists, default, uniq, batched_index_select, masked_mean, to_order, fourier_encode, cast_tuple, safe_cat, fast_split, rand_uniform, broadcat, row_zscore, rank_dropout
from .reversible import SequentialSequence

FiberEl = namedtuple("FiberEl", ["degrees", "dim"])


class Fiber(torch.nn.Module):
    def __init__(self, structure):
        super().__init__()
        if isinstance(structure, dict):
            structure = [FiberEl(degree, dim) for degree, dim in structure.items()]
        self.structure = structure

    @property
    def dims(self):
        return uniq(map(lambda t: t[1], self.structure))

    @property
    def degrees(self):
        return map(lambda t: t[0], self.structure)

    @staticmethod
    def create(num_degrees, dim):
        dim_tuple = dim if isinstance(dim, tuple) else ((dim,) * num_degrees)
        return Fiber([FiberEl(degree, dim) for degree, dim in zip(range(num_degrees), dim_tuple)])

    def __getitem__(self, degree):
        return dict(self.structure)[degree]

    def __iter__(self):
        return iter(self.structure)

    def __mul__(self, fiber):
        return product(self.structure, fiber.structure)

    def __and__(self, fiber):
        out = []
        for degree, dim in self:
            if degree in fiber.degrees:
                dim_out = fiber[degree]
                out.append((degree, dim, dim_out))
        return out


def get_tensor_device_and_dtype(features):
    first_tensor = next(iter(features.items()))[1]
    return first_tensor.device, first_tensor.dtype


class ResidualSE3(torch.nn.Module):
    def forward(self, x, res):
        out = {}
        for degree, tensor in x.items():
            degree = str(degree)
            out[degree] = tensor
            if degree in res:
                out[degree] = out[degree] + res[degree]
        return out


class LinearSE3(torch.nn.Module):
    def __init__(self, fiber_in, fiber_out):
        super().__init__()
        self.weights = torch.nn.ParameterDict()
        for degree, dim_in, dim_out in fiber_in & fiber_out:
            key = str(degree)
            self.weights[key] = torch.nn.Parameter(torch.randn(dim_in, dim_out) / sqrt(dim_in))

    def forward(self, x):
        out = {}
        for degree, weight in self.weights.items():
            out[degree] = torch.einsum("b n d m, d e -> b n e m", x[degree], weight)
        return out


class NormSE3(torch.nn.Module):
    """
    Norm-based SE(3)-equivariant nonlinearity.

                 ┌──> feature_norm ──> LayerNorm? ──> ReLU() ──┐
    feature_in ──┤                                              * ──> feature_out
                 └──> feature_phase ────────────────────────────┘
    """

    NORM_CLAMP = 2**-24  # Minimum positive subnormal for FP16

    def __init__(self, fiber, nonlin=torch.nn.GELU(), use_layernorm=True):
        super().__init__()
        self.fiber = fiber
        self.nonlinearity = nonlin
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.layer_norms = torch.nn.ModuleDict({str(degree): torch.nn.LayerNorm(channels) for degree, channels in fiber})

    def forward(self, features):
        output = {}
        for degree, feat in features.items():
            norm = feat.norm(dim=-1, keepdim=True).clamp(min=self.NORM_CLAMP)
            norm_squeezed = norm.squeeze(-1)

            if self.use_layernorm:
                norm_processed = self.layer_norms[degree](norm_squeezed)
            else:
                norm_processed = norm_squeezed
            new_norm = self.nonlinearity(norm_processed).unsqueeze(-1)
            output[degree] = new_norm * feat / norm
        return output


class ConvSE3(torch.nn.Module):

    def __init__(
        self,
        fiber_in,
        fiber_out,
        self_interaction: bool = True,
        pool: bool = True,
        edge_dim: int = 0,
        fourier_encode_dist: bool = True,
        num_fourier_features: int = 4,
        splits: int = 4,
        *,
        center_deg1_by_ca: bool = False,
    ):
        super().__init__()
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.self_interaction = self_interaction
        self.pool = pool
        self.fourier_encode_dist = fourier_encode_dist
        self.num_fourier_features = num_fourier_features
        self.splits = splits
        self.center_deg1_by_ca = center_deg1_by_ca

        edge_dim += 0 if not fourier_encode_dist else (num_fourier_features * 2)

        self.kernel_unary = torch.nn.ModuleDict()
        for (di, mi), (do, mo) in self.fiber_in * self.fiber_out:
            self.kernel_unary[f"({di},{do})"] = PairwiseConv(di, mi, do, mo, edge_dim=edge_dim, splits=splits)

        if self_interaction:
            assert self.pool, "must pool edges if followed with self interaction"
            self.self_interact = LinearSE3(fiber_in, fiber_out)
            self.self_interact_sum = ResidualSE3()

    def forward(self, inp, edge_info, rel_dist=None, basis=None, atom14_masks=None):
        neighbor_indices, neighbor_masks, edges = edge_info
        outputs = {}

        rel_dist = rearrange(rel_dist, "b m n -> b m n ()")
        if self.fourier_encode_dist:
            rel_dist = fourier_encode(rel_dist, num_encodings=self.num_fourier_features)

        basis_keys = basis.keys()
        split_basis_values = list(zip(*[fast_split(t, self.splits, dim=1) for t in basis.values()]))
        split_basis = [dict(zip(basis_keys, v)) for v in split_basis_values]

        for degree_out in self.fiber_out.degrees:
            degree_out_key = str(degree_out)
            output = 0

            for degree_in, m_in in self.fiber_in:
                etype = f"({degree_in},{degree_out})"
                x = inp[str(degree_in)]

                if int(degree_in) == 1 and self.center_deg1_by_ca:
                    ca_center = x[:, :, 1, :]  # [B, N, 3]
                    x = batched_index_select(x, neighbor_indices, dim=1)  # [B, N, K, 14, 3]

                    if atom14_masks is None:
                        raise ValueError("atom14_masks must be provided when center_deg1_by_ca=True")

                    present = batched_index_select(atom14_masks, neighbor_indices, dim=1)  # [B, N, K, 14]
                    present = present.to(x.dtype)

                    anchor = ca_center.unsqueeze(2).unsqueeze(3)  # [B, N, 1, 1, 3]
                    x = (x - anchor) * present[..., None]  # [B, N, K, 14, 3]
                else:
                    x = batched_index_select(x, neighbor_indices, dim=1)

                x = x.reshape(*x.shape[:3], to_order(degree_in) * m_in, 1)

                edge_feat = torch.cat((rel_dist, edges), dim=-1) if exists(edges) else rel_dist
                split_x = fast_split(x, self.splits, dim=1)
                split_edge_feat = fast_split(edge_feat, self.splits, dim=1)

                output_chunk = None
                for x_chunk, e_chunk, bdict in zip(split_x, split_edge_feat, split_basis):
                    kernel = self.kernel_unary[etype](e_chunk, basis=bdict)
                    chunk = torch.einsum("... o i, ... i c -> ... o c", kernel, x_chunk)
                    output_chunk = safe_cat(output_chunk, chunk, dim=1)

                output = output + output_chunk

            if self.pool:
                output = masked_mean(output, neighbor_masks, dim=2) if exists(neighbor_masks) else output.mean(dim=2)

            leading_shape = x.shape[:2] if self.pool else x.shape[:3]
            output = output.view(*leading_shape, -1, to_order(degree_out))
            outputs[degree_out_key] = output

        if self.self_interaction:
            inp_for_self = inp

            if self.center_deg1_by_ca and ("1" in inp):
                x1 = inp["1"]  # [B, N, 14, 3]

                if atom14_masks is None:
                    raise ValueError("atom14_masks must be provided when center_deg1_by_ca=True")

                present = atom14_masks.to(x1.dtype)[..., None]  # [B, N, 14, 1]

                ca_center = x1[:, :, 1:2, :]  # [B, N, 1, 3]
                x1_rel = (x1 - ca_center) * present
                inp_for_self = dict(inp)
                inp_for_self["1"] = x1_rel

            self_interact_out = self.self_interact(inp_for_self)
            outputs = self.self_interact_sum(outputs, self_interact_out)

        return outputs


class RadialFunc(torch.nn.Module):
    def __init__(
        self,
        num_freq,
        in_dim,
        out_dim,
        edge_dim=None,
        mid_dim=128,
    ):
        super().__init__()
        self.num_freq = num_freq
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.edge_dim = default(edge_dim, 0)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.edge_dim + 1, mid_dim),
            torch.nn.LayerNorm(mid_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mid_dim, mid_dim),
            torch.nn.LayerNorm(mid_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mid_dim, num_freq * in_dim * out_dim),
        )

    def forward(self, x):
        y = self.net(x)
        return rearrange(y, "... (o i f) -> ... o () i () f", i=self.in_dim, o=self.out_dim)


class PairwiseConv(torch.nn.Module):
    def __init__(
        self,
        degree_in,
        nc_in,
        degree_out,
        nc_out,
        edge_dim=0,
        splits=4,
    ):
        super().__init__()
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out
        self.num_freq = to_order(min(degree_in, degree_out))
        self.d_out = to_order(degree_out)
        self.edge_dim = edge_dim
        self.rp = RadialFunc(self.num_freq, nc_in, nc_out, edge_dim)
        self.splits = splits

    def forward(self, feat, basis):
        R = self.rp(feat)
        B = basis[f"{self.degree_in},{self.degree_out}"]
        out_shape = (*R.shape[:3], self.d_out * self.nc_out, -1)
        out = 0
        for i in range(R.shape[-1]):
            out += R[..., i] * B[..., i]
        out = rearrange(out, "b n h s ... -> (b n h s) ...")
        return out.view(*out_shape)


class FeedForwardSE3(torch.nn.Module):
    def __init__(self, fiber, mult=4):
        super().__init__()
        self.fiber = fiber
        fiber_hidden = Fiber(list(map(lambda t: (t[0], t[1] * mult), fiber)))
        self.project_in = LinearSE3(fiber, fiber_hidden)
        self.nonlin = NormSE3(fiber_hidden, nonlin=torch.nn.GELU(), use_layernorm=False)
        self.project_out = LinearSE3(fiber_hidden, fiber)

    def forward(self, features):
        outputs = self.project_in(features)
        outputs = self.nonlin(outputs)
        outputs = self.project_out(outputs)
        return outputs


class FeedForwardBlockSE3(torch.nn.Module):
    def __init__(self, fiber):
        super().__init__()
        self.fiber = fiber
        self.prenorm = NormSE3(fiber, nonlin=torch.nn.Identity(), use_layernorm=True)
        self.feedforward = FeedForwardSE3(fiber)
        self.residual = ResidualSE3()

    def forward(self, features):
        res = features
        out = self.prenorm(features)
        out = self.feedforward(out)
        return self.residual(out, res)


class AttentionSE3(torch.nn.Module):
    def __init__(
        self,
        fiber,
        dim_head=64,
        heads=8,
        attend_self=False,
        edge_dim=None,
        fourier_encode_dist=False,
        rel_dist_num_fourier_features=4,
        splits=4,
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        hidden_fiber = Fiber(list(map(lambda t: (t[0], hidden_dim), fiber)))

        project_out = not (heads == 1 and len(fiber.dims) == 1 and dim_head == fiber.dims[0])
        self.scale = dim_head**-0.5
        self.heads = heads
        self.to_q = LinearSE3(fiber, hidden_fiber)
        self.to_v = ConvSE3(
            fiber,
            hidden_fiber,
            edge_dim=edge_dim,
            pool=False,
            self_interaction=False,
            fourier_encode_dist=fourier_encode_dist,
            num_fourier_features=rel_dist_num_fourier_features,
            splits=splits,
        )
        self.to_k = ConvSE3(
            fiber,
            hidden_fiber,
            edge_dim=edge_dim,
            pool=False,
            self_interaction=False,
            fourier_encode_dist=fourier_encode_dist,
            num_fourier_features=rel_dist_num_fourier_features,
            splits=splits,
        )

        self.to_out = LinearSE3(hidden_fiber, fiber) if project_out else torch.nn.Identity()
        self.attend_self = attend_self
        if attend_self:
            self.to_self_k = LinearSE3(fiber, hidden_fiber)
            self.to_self_v = LinearSE3(fiber, hidden_fiber)

        self.use_edge_attn_bias = exists(edge_dim) and (edge_dim is not None) and (edge_dim > 0)
        self.use_edge_attn_bias = False

        if self.use_edge_attn_bias:
            self.edge_to_attn_bias = torch.nn.Sequential(
                torch.nn.LayerNorm(edge_dim),
                torch.nn.Linear(edge_dim, edge_dim),
                torch.nn.GELU(),
                torch.nn.Linear(edge_dim, heads),  # -> per-head bias
            )
            # start from 0 so behavior initially == old model
            self.edge_attn_scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, features, edge_info, rel_dist, basis, pos_emb=None):
        h, attend_self = self.heads, self.attend_self
        neighbor_indices, neighbor_mask, edges = edge_info
        if exists(neighbor_mask):
            neighbor_mask = rearrange(neighbor_mask, "b i j -> b () i j")
        queries = self.to_q(features)
        values = self.to_v(features, edge_info, rel_dist, basis)
        keys = self.to_k(features, edge_info, rel_dist, basis)
        if attend_self:
            self_keys, self_values = self.to_self_k(features), self.to_self_v(features)
        outputs = {}
        for degree in features.keys():
            q, k, v = map(lambda t: t[degree], (queries, keys, values))
            q = rearrange(q, "b i (h d) m -> b h i d m", h=h)
            k, v = map(lambda t: rearrange(t, "b i j (h d) m -> b h i j d m", h=h), (k, v))
            if attend_self:
                self_k, self_v = map(lambda t: t[degree], (self_keys, self_values))
                self_k, self_v = map(
                    lambda t: rearrange(t, "b n (h d) m -> b h n () d m", h=h),
                    (self_k, self_v),
                )
                k = torch.cat((self_k, k), dim=3)
                v = torch.cat((self_v, v), dim=3)
            sim = torch.einsum("b h i d m, b h i j d m -> b h i j", q, k) * self.scale
            if self.use_edge_attn_bias and exists(edges):
                # edges: [B, N, K, edge_dim]
                eb = self.edge_to_attn_bias(edges)  # [B, N, K, H]
                eb = rearrange(eb, "b i j h -> b h i j")  # [B, H, N, K]

                if attend_self:
                    # when attend_self, we prepend 1 slot to k/v at dim=3 (j dim)
                    # so we also pad bias with 0 for self slot
                    eb = torch.nn.functional.pad(eb, (1, 0), value=0.0)

                sim = sim + torch.tanh(self.edge_attn_scale) * eb

            if exists(neighbor_mask):
                num_left_pad = sim.shape[-1] - neighbor_mask.shape[-1]
                pad_mask = torch.nn.functional.pad(neighbor_mask, (num_left_pad, 0), value=True)
                sim = sim.masked_fill(~pad_mask, -torch.finfo(sim.dtype).max)
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b h i j, b h i j d m -> b h i d m", attn, v)
            outputs[degree] = rearrange(out, "b h n d m -> b n (h d) m")
        return self.to_out(outputs), attn


class AttentionBlockSE3(torch.nn.Module):
    def __init__(
        self,
        fiber,
        dim_head=24,
        heads=8,
        attend_self=False,
        edge_dim=None,
        fourier_encode_dist=False,
        rel_dist_num_fourier_features=4,
        splits=4,
    ):
        super().__init__()
        self.attn = AttentionSE3(
            fiber,
            heads=heads,
            dim_head=dim_head,
            attend_self=attend_self,
            edge_dim=edge_dim,
            rel_dist_num_fourier_features=rel_dist_num_fourier_features,
            fourier_encode_dist=fourier_encode_dist,
            splits=splits,
        )
        self.prenorm = NormSE3(fiber, nonlin=torch.nn.GELU(), use_layernorm=True)
        self.residual = ResidualSE3()
        self.last_attn = None

    def forward(self, features, edge_info, rel_dist, basis, global_feats=None, pos_emb=None):
        res = features
        outputs = self.prenorm(features)
        outputs, attn = self.attn(outputs, edge_info, rel_dist, basis, pos_emb)
        self.last_attn = attn
        return self.residual(outputs, res)


class SE3Transformer(torch.nn.Module):
    def __init__(
        self,
        *,
        heads=4,
        dim_head=64,
        depth=1,
        input_degrees=2,
        num_degrees=2,
        attend_self=True,
        differentiable_coors=False,
        fourier_encode_dist=True,
        rel_dist_num_fourier_features=4,
        adj_dim=0,
        dim_in=(320, 14),
        splits=4,
        hidden_fiber_dict=None,
        out_fiber_dict=None,
        rankH=256,
        geo_neighbor=1 / 2,
        epi_neighbor=1 / 3,
        esm_dropout=0.3,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.input_degrees = input_degrees

        self.num_degrees = num_degrees if exists(num_degrees) else (max(hidden_fiber_dict.keys()) + 1)

        self.differentiable_coors = differentiable_coors

        self.edge_dim = adj_dim

        self.esm_dropout = esm_dropout

        fiber_in = Fiber.create(input_degrees, dim_in)

        fiber_hidden = Fiber(hidden_fiber_dict)

        fiber_out = Fiber(out_fiber_dict)

        conv_in_kwargs = dict(
            edge_dim=self.edge_dim,
            fourier_encode_dist=fourier_encode_dist,
            num_fourier_features=rel_dist_num_fourier_features,
            splits=splits,
            center_deg1_by_ca=True,
        )

        conv_out_kwargs = dict(
            edge_dim=self.edge_dim,
            fourier_encode_dist=fourier_encode_dist,
            num_fourier_features=rel_dist_num_fourier_features,
            splits=splits,
        )

        self.conv_in = ConvSE3(fiber_in, fiber_hidden, **conv_in_kwargs)

        self.attend_self = attend_self
        layers = torch.nn.ModuleList([])
        for _ in range(depth):
            layers.append(
                torch.nn.ModuleList(
                    [
                        AttentionBlockSE3(
                            fiber_hidden,
                            heads=heads,
                            dim_head=dim_head,
                            attend_self=attend_self,
                            edge_dim=self.edge_dim,
                            fourier_encode_dist=fourier_encode_dist,
                            rel_dist_num_fourier_features=rel_dist_num_fourier_features,
                            splits=splits,
                        ),
                        FeedForwardBlockSE3(fiber_hidden),
                    ]
                )
            )
        self.net = SequentialSequence(layers)
        self.conv_out = ConvSE3(fiber_hidden, fiber_out, **conv_out_kwargs)

        self.linear_out = torch.nn.Sequential(
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 20),
        )

        self.edge_proj = torch.nn.Sequential(
            # torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 200),
            torch.nn.GELU(),
            torch.nn.Linear(200, self.edge_dim),
        )

        self.rankH = rankH
        self.aa_embed = torch.nn.Parameter(torch.randn(20, self.rankH) * 0.2)
        self.high_in = torch.nn.Sequential(
            torch.nn.LayerNorm(128 + out_fiber_dict[1] + 2),  # 128+32+2
            torch.nn.Linear(128 + out_fiber_dict[1] + 2, 128),
            torch.nn.GELU(),
        )
        self.high_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, self.rankH),
        )

        self.u_mix = torch.nn.Linear(self.rankH, self.rankH, bias=False)

        self.geo_neighbor = geo_neighbor
        self.epi_neighbor = epi_neighbor

        self.esm_single_head = torch.nn.Sequential(
            torch.nn.LayerNorm(1280),
            torch.nn.Linear(1280, 640),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(640, 320),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(320, 20),
        )
        self.beta = torch.nn.Parameter(torch.tensor(0.1))

        self.node_transform = torch.nn.Sequential(
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 320),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(320, 320),
            torch.nn.LayerNorm(320),
        )

    def forward(self, data):

        esm2_embedding = data["embedding"]
        if esm2_embedding.dim() == 2:
            esm2_embedding = esm2_embedding.unsqueeze(dim=0)  # [1,L,1280]

        node_embedding = data["node_embedding"]
        if node_embedding.dim() == 2:
            node_embedding = node_embedding.unsqueeze(dim=0)  # [1,L,256]

        atom14_coords = data["atom14_coords"]
        if atom14_coords.dim() == 3:
            atom14_coords = data["atom14_coords"].unsqueeze(dim=0)  # [1,L,14,3]

        ca_coords = atom14_coords[:, :, 1, :]  # [B, L, 3]

        atom14_masks = data["atom14_masks"]
        if atom14_masks.dim() == 2:
            atom14_masks = atom14_masks.unsqueeze(dim=0)  # [1,L,14]

        edge_embedding = data["edge_embedding"]
        if edge_embedding.dim() == 3:
            edge_embedding = edge_embedding.unsqueeze(dim=0)  # [1,L,L,128]

        # zeroshot_single_mut = data["single_mutation_effects"]
        # if zeroshot_single_mut.dim() == 2:
        #    zeroshot_single_mut = zeroshot_single_mut.unsqueeze(dim=0)  # [1,L,20]
        wt_idx = data["wt_idx"]

        edges = self.edge_proj(edge_embedding)

        seq0 = self.node_transform(node_embedding)  # [B,L,320]

        feats = {"0": seq0[..., None], "1": atom14_coords}

        b, n, d, *_, device = *feats["0"].shape, feats["0"].device
        assert d == self.dim_in[0], f"feature dimension {d} must be equal to dimension given at init {self.dim_in[0]}"
        assert set(map(int, feats.keys())) == set(range(self.input_degrees)), f"input must have {self.input_degrees} degree"
        num_degrees = self.num_degrees

        protein_length = ca_coords.shape[1]

        k_geo = int(round(protein_length**self.geo_neighbor))

        k_epi = int(round(protein_length**self.epi_neighbor))
        total_neighbors = k_geo + k_epi
        total_neighbors = min(total_neighbors, n - 1)

        exclude_self_mask = rearrange(~torch.eye(n, dtype=torch.bool, device=device), "i j -> () i j")

        indices = repeat(torch.arange(n, device=device), "j -> b i j", b=b, i=n)
        rel_pos = rearrange(ca_coords, "b n d -> b n () d") - rearrange(ca_coords, "b n d -> b () n d")

        indices = indices.masked_select(exclude_self_mask).reshape(b, n, n - 1)

        rel_pos_masked_self = rel_pos.masked_select(exclude_self_mask[..., None]).reshape(b, n, n - 1, 3)

        rel_dist = rel_pos_masked_self.norm(dim=-1)  # shape: (b, l, l-1)

        _, geo_local_indices = rel_dist.topk(k_geo, dim=-1, largest=False)

        epi_score = edge_embedding.norm(dim=-1)  # [1, N, N]

        epi_score_masked = epi_score.masked_select(exclude_self_mask).reshape(b, n, n - 1)

        epi_score_for_selection = epi_score_masked.clone()
        min_value = -torch.finfo(epi_score_for_selection.dtype).max  # 或者直接 float('-inf')
        epi_score_for_selection.scatter_(dim=2, index=geo_local_indices, value=min_value)
        _, epi_local_indices = epi_score_for_selection.topk(k_epi, dim=-1, largest=True)

        nearest_indices = torch.cat([geo_local_indices, epi_local_indices], dim=-1)

        neighbor_mask = torch.ones((b, n, total_neighbors), device=device, dtype=torch.bool)

        neighbor_rel_dist = batched_index_select(rel_dist, nearest_indices, dim=2)  # shape: (b, l, total_neighbors)
        neighbor_rel_pos = batched_index_select(rel_pos_masked_self, nearest_indices, dim=2)  # shape: (b, l, total_neighbors, 3)
        neighbor_indices = batched_index_select(indices, nearest_indices, dim=2)  # shape: (b, l, total_neighbors)

        if exists(edges):
            edges = batched_index_select(edges, neighbor_indices, dim=2)

        basis = get_basis(neighbor_rel_pos, num_degrees - 1, differentiable=self.differentiable_coors)
        edge_info = (neighbor_indices, neighbor_mask, edges)
        x = feats

        x = self.conv_in(
            x,
            edge_info,
            rel_dist=neighbor_rel_dist,
            basis=basis,
            atom14_masks=atom14_masks,
        )

        x = self.net(x, edge_info=edge_info, rel_dist=neighbor_rel_dist, basis=basis)

        x = self.conv_out(x, edge_info, rel_dist=neighbor_rel_dist, basis=basis)

        z = x["0"].squeeze(dim=-1).squeeze(0)  # [L,128]

        single_mut = self.linear_out(z).squeeze(dim=0)
        esm_correction = self.esm_single_head(esm2_embedding).squeeze(dim=0)

        keep_prob = 1.0 - self.esm_dropout

        if self.training:
            if torch.rand((), device=single_mut.device) < keep_prob:
                single_mut = single_mut + (self.beta / keep_prob) * esm_correction
        else:
            single_mut = single_mut + self.beta * esm_correction

        wt_val = single_mut.gather(1, wt_idx[:, None])  # [L, 1]
        single_mut = single_mut - wt_val  # [L,20]

        v1 = x["1"].squeeze(0)  # [L,32,3]
        v_norm = v1.norm(dim=-1)  # [L,32]

        dist = neighbor_rel_dist.squeeze(0)  # [L,K]
        dist_stats = torch.stack([dist.mean(dim=-1), dist.std(dim=-1)], dim=-1)  # [L,2]

        high_in = torch.cat([z, v_norm, dist_stats], dim=-1)  # [L,162]
        high_in = self.high_in(high_in)  # [L,128]

        h = self.high_mlp(high_in)  # [L,rH]

        U = h[:, None, :] * self.aa_embed[None, :, :]  # [L,20,rH]
        U = self.u_mix(U)  # [L,20,rH]

        U_wt = U.gather(1, wt_idx[:, None, None].expand(-1, 1, self.rankH))  # [L,1,rH]
        high_delta = U - U_wt

        return single_mut, high_delta
