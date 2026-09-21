import torch

import pandas as pd
import numpy as np

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


class EpistasisMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_combined):
        return self.net(h_combined).squeeze(-1)


def build_mutation_table(mutation_lists):
    """
    Args:
        mutation_lists: list[str], e.g. ["A1G", "A5M,Y10D", ...] (0-based positions)
    Returns:
        DataFrame with columns: sample_id, pos, wt, mt
    """
    data = []
    for i, s in enumerate(mutation_lists):
        for mut in s.split(","):
            if not mut:
                continue  # Handle empty strings if any
            wt = mut[0]
            mt = mut[-1]
            try:
                pos = int(mut[1:-1])
            except ValueError:
                continue  # Skip malformed strings
            data.append({"sample_id": i, "pos": pos, "wt": AA_TO_IDX[wt], "mt": AA_TO_IDX[mt]})
    return pd.DataFrame(data)


def df_to_tensor(df, max_mut=10, device="cpu"):
    """
    Convert mutation DataFrame to a padded tensor.
    """
    if df.empty:
        # Handle empty case safely
        return torch.zeros((0, max_mut, 3), dtype=torch.long, device=device), torch.zeros((0, max_mut), dtype=torch.bool, device=device)

    df = df.copy()
    sample_ids, inverse = np.unique(df["sample_id"].values, return_inverse=True)
    df["mapped_id"] = inverse
    df["mut_idx"] = df.groupby("mapped_id").cumcount()

    B = len(sample_ids)
    tensor = torch.full((B, max_mut, 3), -1, dtype=torch.long, device=device)

    idx_sample = torch.tensor(df["mapped_id"].values, dtype=torch.long, device=device)
    idx_mut = torch.tensor(df["mut_idx"].values, dtype=torch.long, device=device)
    pos = torch.tensor(df["pos"].values, dtype=torch.long, device=device)
    wt = torch.tensor(df["wt"].values, dtype=torch.long, device=device)
    mt = torch.tensor(df["mt"].values, dtype=torch.long, device=device)

    # Safe indexing
    mask_valid = idx_mut < max_mut
    tensor[idx_sample[mask_valid], idx_mut[mask_valid], 0] = pos[mask_valid]
    tensor[idx_sample[mask_valid], idx_mut[mask_valid], 1] = wt[mask_valid]
    tensor[idx_sample[mask_valid], idx_mut[mask_valid], 2] = mt[mask_valid]

    mask = tensor[:, :, 0] != -1
    return tensor, mask


def calculate_batch_prediction_mlp(
    single_mut_matrix,
    mut_name_list,
    U,
    mlp_model,
    max_mut=10,
    device="cpu",
    return_epi=False,
    sqrt_scale=False,
):
    df = build_mutation_table(mut_name_list)
    mutations_tensor, mask = df_to_tensor(df, max_mut=max_mut, device=device)

    pos = mutations_tensor[..., 0]
    mt = mutations_tensor[..., 2]

    pos_safe = pos.clone()
    mt_safe = mt.clone()
    pos_safe[~mask] = 0
    mt_safe[~mask] = 0

    u_vecs = U[pos_safe, mt_safe]
    u_vecs = u_vecs * mask.unsqueeze(-1).to(u_vecs.dtype)

    n_mut = mask.sum(dim=1)
    n_mut_float = n_mut.clamp(min=1).to(u_vecs.dtype)

    h_combined = u_vecs.sum(dim=1)
    if sqrt_scale:
        h_combined = h_combined / torch.sqrt(n_mut_float[:, None])

    g_sum = mlp_model(h_combined)

    single_vals = single_mut_matrix[pos_safe, mt_safe]
    single_sum = (single_vals * mask.to(single_vals.dtype)).sum(dim=1)

    preds = single_sum + g_sum

    if not return_epi:
        return preds.to(torch.float32)

    B = mask.shape[0]
    sample_idx = mask.nonzero(as_tuple=False)[:, 0]
    u_valid = u_vecs[mask]

    g_each_valid = mlp_model(u_valid)
    sum_g_each = torch.zeros(B, dtype=g_each_valid.dtype, device=g_each_valid.device)
    sum_g_each.index_add_(0, sample_idx, g_each_valid)

    pred_epi = g_sum - sum_g_each

    return preds.to(torch.float32), pred_epi.to(torch.float32)
