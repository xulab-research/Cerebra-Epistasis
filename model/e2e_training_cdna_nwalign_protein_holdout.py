"""End-to-end Cerebra-Seq and Cerebra-Epistasis training with NW-align-based protein-level cross-validation.

Run folds serially on cuda:0 by default; override the device with --device.
Read ESM2, ESMC, ESM3 and optional mmCIF-derived structure features from .pt files.
Resume a trusted training checkpoint with --resume and --fold_id.
"""

import re
import argparse
import random
import itertools
import subprocess
import gc
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

base_dir = Path(__file__).resolve().parent
proj_root = base_dir.parent

from cerebra_epistasis.model import SE3Transformer
from utils.metrics import spearman_corr
from utils.utils_func import (
    read_wt_idx_from_fasta,
    to_gpu,
    max_mut_from_list,
    set_seed_everywhere,
)
from utils.calculate_nbodys_mutation_effect import calculate_batch_prediction_mlp, EpistasisMLP
from utils.Cerebra_Seq_utils import *
from utils.Cerebra_Seq_loss import comp_structure_loss, STRUCTURE_LOSS_ITEM_NAMES, STRUCTURE_TEST_REPORT_NAMES
from utils.structure_utils.mmcif_labels import FEATURE_FILENAME, load_structure_features
from utils.structure_utils import residue_constants as rc
from utils.structure_utils.atom_geometry import hu_model_pred_to_atom14_pos, make_atom14_masks

CEREBRA_REVISION = "aab7318429599d6efc711049e1ef614374102d69"


def mutation_depth(mut_name: str) -> int:
    return len([x for x in str(mut_name).split(",") if x.strip()])


def compute_twobody_epistasis_labels(
    train_df: pd.DataFrame,
    train_mean: float,
    train_denom: float,
) -> Tuple[np.ndarray, np.ndarray]:
    df = train_df.copy().reset_index(drop=True)
    df["mutation_name"] = df["mutation_name"].astype(str).str.strip()
    df["mut_depth"] = df["mutation_name"].map(mutation_depth)

    label_map = dict(zip(df["mutation_name"], df["label"].astype(float)))
    double_indices = []
    epi_labels_z = []

    for idx, row in df.iterrows():
        if int(row["mut_depth"]) != 2:
            continue

        muts = [m.strip() for m in row["mutation_name"].split(",") if m.strip()]
        if len(muts) != 2 or muts[0] not in label_map or muts[1] not in label_map:
            continue

        epi_raw = float(row["label"]) - float(label_map[muts[0]]) - float(label_map[muts[1]])
        epi_labels_z.append((epi_raw + float(train_mean)) / float(train_denom))
        double_indices.append(idx)

    return np.asarray(double_indices, dtype=np.int64), np.asarray(epi_labels_z, dtype=np.float32)


def load_features(assay_dir: Path) -> Dict[str, torch.Tensor]:
    sequence = read_fasta_sequence(assay_dir / "wt.fasta").upper()
    feature = torch.load(assay_dir / FEATURE_FILENAME, map_location="cpu", weights_only=True)
    if feature["sequence"] != sequence:
        raise ValueError(f"Feature sequence differs from wt.fasta: {assay_dir}")
    for key, dim in (("esmc", 1152), ("esm3", 1536)):
        if key not in feature or tuple(feature[key].shape) != (len(sequence), dim):
            raise ValueError(f"Missing or invalid {key}; regenerate {FEATURE_FILENAME}: {assay_dir}")
    embedding = torch.load(
        assay_dir / "embedding_ESM2_650M_for_Cerebra_Epistasis.pt",
        map_location="cpu",
        weights_only=True,
    )
    if tuple(embedding.shape) != (len(sequence), 1280):
        raise ValueError(f"Invalid ESM2 embedding shape: {assay_dir}")
    return {
        "embedding": embedding,
        "wt_idx": torch.as_tensor(read_wt_idx_from_fasta(assay_dir / "wt.fasta"), dtype=torch.long),
        **build_cerebra_batch(sequence, feature["esmc"], feature["esm3"], torch.device("cpu")),
        "aatype": torch.tensor([rc.restype_order[aa] for aa in sequence], dtype=torch.long),
    }


def configure_cerebra_training(cerebra_model, training_mode):
    train_cerebra = training_mode == "e2e"
    cerebra_model.requires_grad_(train_cerebra)
    cerebra_model.train(train_cerebra)


def build_training_optimizer(cerebra_model, model, mlp, args):
    groups = []
    if args.training_mode == "e2e":
        groups.append({"params": cerebra_model.parameters(), "lr": 5e-6})
    groups.extend([{"params": model.parameters(), "lr": args.lr}, {"params": mlp.parameters(), "lr": args.lr}])
    return torch.optim.Adam(groups, eps=1e-6)


def build_cerebra_model(args, device):
    cerebra_model = load_cerebra_model(device, checkpoint="model1", revision=args.cerebra_revision, cache_dir=args.hf_cache_dir, training=args.training_mode == "e2e")
    configure_cerebra_training(cerebra_model, args.training_mode)
    return cerebra_model


def save_training_checkpoint(path, cerebra_model, model, mlp, optimizer, scheduler, epoch, fold, args, train_mean, train_std, best_val_spearman, loss_records, eval_records, train_items, test_items, device):
    state = {
        "format_version": 2,
        "cerebra_model": cerebra_model.state_dict(),
        "model": model.state_dict(),
        "mlp": mlp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_epoch": epoch + 1,
        "fold": fold,
        "args": vars(args),
        "train_mean": train_mean,
        "train_std": train_std,
        "best_val_spearman": best_val_spearman,
        "loss_records": loss_records,
        "eval_records": eval_records,
        "train_order": [x["assay_id"] for x in train_items],
        "test_assays": [x["assay_id"] for x in test_items],
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_training_checkpoint(state, cerebra_model, model, mlp, optimizer, scheduler, fold, train_mean, train_std, train_items, test_items, device):
    if state.get("format_version") != 2:
        raise ValueError("Resume requires a checkpoint saved by this HF training script")
    if state["fold"] != fold:
        raise ValueError("Resume checkpoint belongs to a different fold")
    if not np.allclose([state["train_mean"], state["train_std"]], [train_mean, train_std], rtol=0, atol=1e-12):
        raise ValueError("Training-label normalization differs from the checkpoint")
    by_id = {x["assay_id"]: x for x in train_items}
    if set(by_id) != set(state["train_order"]) or set(state["test_assays"]) != {x["assay_id"] for x in test_items}:
        raise ValueError("Resume checkpoint has different train/test assays")
    for module, key in [(cerebra_model, "cerebra_model"), (model, "model"), (mlp, "mlp")]:
        module.load_state_dict(state[key], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    train_items[:] = [by_id[key] for key in state["train_order"]]
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if device.type == "cuda" and state["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state(state["rng"]["cuda"], device)
    return state["epoch"] + 1, state["best_val_spearman"], state["loss_records"], state["eval_records"]


def build_model(args, geo_neighbor: float, epi_neighbor: float):
    model = SE3Transformer(
        depth=1,
        hidden_fiber_dict={0: 320, 1: 32},
        out_fiber_dict={0: 128, 1: 32},
        adj_dim=args.adj_dim,
        rankH=args.rankH,
        geo_neighbor=geo_neighbor,
        epi_neighbor=epi_neighbor,
    )
    mlp = EpistasisMLP(
        input_dim=args.rankH,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.mlp_dropout,
    )
    return model, mlp


def set_model_neighbors(model, geo_neighbor, epi_neighbor):
    for module in model.modules():
        if hasattr(module, "geo_neighbor"):
            module.geo_neighbor = float(geo_neighbor)
        if hasattr(module, "epi_neighbor"):
            module.epi_neighbor = float(epi_neighbor)


def pairwise_identity(nwalign_bin, fasta1, fasta2):
    def run(a, b):
        output = subprocess.check_output(
            [str(nwalign_bin), str(a), str(b)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        score = float(re.search(r"Sequence identity:\s*([0-9.]+)", output).group(1))
        return score / 100.0 if score > 1.0 else score

    return max(run(fasta1, fasta2), run(fasta2, fasta1))


def build_or_load_folds(assays, args, train_log):
    cluster_dir = train_log / "protein_similarity_folds"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    identity_csv = cluster_dir / "identity.csv"
    distance_csv = cluster_dir / "distance.csv"
    cluster_csv = cluster_dir / f"nwalign_{args.k_folds}fold_cluster.csv"

    if cluster_csv.exists() and not args.recompute_clusters:
        print(f"[cluster] load {cluster_csv}")
        return pd.read_csv(cluster_csv)[["assay_id", "protein_fold"]]

    names = [x["assay_id"] for x in assays]
    fasta_paths = [x["assay_dir"] / "wt.fasta" for x in assays]

    identity = pd.DataFrame(
        np.eye(len(names), dtype=float),
        index=names,
        columns=names,
    )

    pairs = list(itertools.combinations(range(len(names)), 2))
    print(f"[cluster] build nwalign identity matrix | n={len(names)} | pairs={len(pairs)}")

    for step, (i, j) in enumerate(pairs, start=1):
        score = pairwise_identity(args.nwalign_bin, fasta_paths[i], fasta_paths[j])
        identity.iat[i, j] = score
        identity.iat[j, i] = score

        if step == 1 or step % args.cluster_print_every == 0 or step == len(pairs):
            print(f"[cluster] nwalign pairs {step}/{len(pairs)}")

    identity.to_csv(identity_csv)

    distance = 1.0 - identity
    np.fill_diagonal(distance.values, 0.0)
    distance.to_csv(distance_csv)

    linkage = hierarchy.linkage(
        squareform(distance.values, checks=False),
        method=args.linkage_method,
    )

    cluster_index = hierarchy.fcluster(
        linkage,
        args.k_folds,
        criterion="maxclust",
    )

    if len(np.unique(cluster_index)) != args.k_folds:
        cluster_index = (
            hierarchy.cut_tree(
                linkage,
                n_clusters=args.k_folds,
            ).reshape(-1)
            + 1
        )

    fold_map = {c: i for i, c in enumerate(sorted(np.unique(cluster_index)))}
    protein_fold = [fold_map[c] for c in cluster_index]

    fold_df = pd.DataFrame(
        {
            "assay_id": names,
            "cluster_index": cluster_index.astype(int),
            "protein_fold": protein_fold,
        }
    )

    fold_df.to_csv(cluster_csv, index=False)

    print(f"[cluster] saved identity -> {identity_csv}")
    print(f"[cluster] saved distance -> {distance_csv}")
    print(f"[cluster] saved folds -> {cluster_csv}")

    return fold_df[["assay_id", "protein_fold"]]


def make_split(cv_df, train_mean, train_std):
    df = cv_df[["mutation_name", "label"]].copy().reset_index(drop=True)
    df["mutation_name"] = df["mutation_name"].astype(str).str.strip()
    df["label"] = df["label"].astype(float)

    y_raw = df["label"].to_numpy(dtype=float)
    mut_list = df["mutation_name"].tolist()

    epi_indices, epi_y = compute_twobody_epistasis_labels(
        df,
        train_mean,
        train_std,
    )

    return {
        "mut_list": mut_list,
        "y_z": ((y_raw - train_mean) / train_std).astype(np.float32),
        "y_raw": y_raw,
        "train_mean": train_mean,
        "train_std": train_std,
        "epi_indices": epi_indices,
        "epi_y": epi_y,
        "max_mut": int(max_mut_from_list(mut_list)),
        "n": len(df),
        "n_epi": len(epi_indices),
    }


def prepare_cdna_structure_batch(structure_label, device):
    # Derived supervision and device conversion are handled by comp_structure_loss.
    return structure_label


def cast_tree_fp32(obj):
    if torch.is_tensor(obj):
        if obj.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            return obj.float()
        return obj
    if isinstance(obj, dict):
        return {key: cast_tree_fp32(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [cast_tree_fp32(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(cast_tree_fp32(value) for value in obj)
    return obj


def evaluate_cdna_structure_loss(
    cerebra_outputs,
    anchor_list,
    structure_label,
    device,
):
    structure_batch = prepare_cdna_structure_batch(structure_label, device)
    with torch.amp.autocast(device_type=device.type, enabled=False):
        structure_loss_raw, structure_loss_items = comp_structure_loss(
            cast_tree_fp32(cerebra_outputs),
            structure_batch,
            anchor_list,
        )

    values = np.asarray(structure_loss_items, dtype=np.float64).reshape(-1)
    if len(values) != len(STRUCTURE_LOSS_ITEM_NAMES):
        raise ValueError("Unexpected comp_structure_loss output length: " f"expected={len(STRUCTURE_LOSS_ITEM_NAMES)}, got={len(values)}")

    metrics = dict(zip(STRUCTURE_LOSS_ITEM_NAMES, values.tolist()))
    returned_raw = float(structure_loss_raw.detach().cpu().item())
    if not np.isclose(
        returned_raw,
        metrics["structure_loss_raw"],
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True,
    ):
        raise ValueError("Structure-loss return value does not match its reported item: " f"return={returned_raw}, item={metrics['structure_loss_raw']}")

    return {name: metrics[name] for name in STRUCTURE_TEST_REPORT_NAMES}


def attach_cdna_structure_labels(train_items):
    loaded = 0
    for item in train_items:
        structure_pt = item.get("structure_pt")
        if structure_pt is None:
            item["structure_label"] = None
            continue
        item["structure_label"] = load_structure_features(
            Path(structure_pt),
            read_fasta_sequence(item["assay_dir"] / "wt.fasta"),
        )
        loaded += int(item["structure_label"] is not None)
    return loaded


def _with_batch_dim(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() > 1 and x.shape[0] == 1 else x.unsqueeze(0)


def run_cycle(batch, cerebra_model, mode="train"):
    if mode == "train":
        num_iters = (torch.rand(1) * 4).long() + 1
        num_iters = torch.clip(num_iters, min=1, max=4).item()
    else:
        num_iters = 4
    prevs = [None, None, None]
    is_grad_enabled = torch.is_grad_enabled()

    batch = dict(batch)
    for key in ("X1D_esm_c", "X1D_esm3", "target_feat", "residue_index", "aatype"):
        batch[key] = _with_batch_dim(batch[key])

    length = batch["X1D_esm_c"].shape[1]
    n_clusters = 12 if length < 128 else 24
    anchor_list = np.array([(x + 1) * length / (n_clusters + 1) for x in range(n_clusters)]) + np.random.randint(round(length / n_clusters), size=(n_clusters))
    anchor_list = np.clip(anchor_list.astype(int), a_min=1, a_max=length - 2)

    device = next(cerebra_model.parameters()).device

    for cycle_no in range(num_iters):
        feats = prepare_cerebra_inputs(batch, cerebra_model, clone=True)

        is_final_iter = cycle_no == num_iters - 1
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                m_prev, z_prev, x_prev, outputs = cerebra_model(
                    feats,
                    prevs,
                    anchor_list,
                    _recycle=not is_final_iter,
                    return_aux=is_final_iter,
                    return_dist=True,
                    return_pae=True,
                    reduce_plddt=False,
                    keep_structure_all=True,
                )

                if is_final_iter:
                    return outputs, batch, anchor_list

                prevs = [m_prev.detach(), z_prev.detach(), x_prev.detach()]

    return outputs, batch, anchor_list


def pdb_feats_write(outputs, batch, anchor_list, use_comb=False, main_anchor_id=8):
    device = batch["target_feat"].device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch = make_atom14_masks(batch)

    if use_comb:
        pred_q, pred_t = AnchorFrameConsensus(outputs, main_anchor_id)
    else:
        pred_q = outputs["quaternion"][-1][:, main_anchor_id]
        pred_t = outputs["translation"][-1][:, main_anchor_id]

    _, angles = outputs["angles"]
    pred_all_atoms_pos_14 = hu_model_pred_to_atom14_pos(
        pred_q,
        pred_t,
        angles.to(device),
        batch["aatype"],
    )

    return {
        "final_atom_positions": pred_all_atoms_pos_14[0],
        "final_atom_mask": batch["atom14_atom_exists"][0],
        "node_embedding": outputs["x1D"][0],
        "edge_embedding": outputs["x2D"][0],
    }


def apply_cerebra_features(
    data,
    cerebra_model,
    mode="train",
    return_structure_context=False,
):
    outputs, batch, anchor_list = run_cycle(data, cerebra_model, mode)
    pdb_feats = pdb_feats_write(outputs, batch, anchor_list)

    data["atom14_coords"] = pdb_feats["final_atom_positions"].float()
    data["atom14_masks"] = pdb_feats["final_atom_mask"].float()
    data["node_embedding"] = pdb_feats["node_embedding"].float()
    data["edge_embedding"] = pdb_feats["edge_embedding"].permute(1, 2, 0).float()
    if return_structure_context:
        return data, outputs, anchor_list
    return data


def evaluate_fold_once(cerebra_model, model, mlp, test_items, device):
    per_assay_metrics = []

    for item in test_items:
        split = item["split"]
        set_model_neighbors(
            model,
            item["geo_neighbor"],
            item["epi_neighbor"],
        )

        data = to_gpu(load_features(item["assay_dir"]), device)
        pred_raw, structure_metrics = predict_one_protein(
            cerebra_model=cerebra_model,
            model=model,
            mlp=mlp,
            data=data,
            split=split,
            device=device,
            structure_label=item.get("structure_label"),
        )
        del data

        test_spearman = float(spearmanr(split["y_raw"], pred_raw, nan_policy="omit").correlation) if len(split["y_raw"]) >= 2 else np.nan
        row = {
            "assay_id": item["assay_id"],
            "test_spearman": test_spearman,
            "has_structure_label": int(structure_metrics is not None),
        }
        if structure_metrics is not None:
            row.update(structure_metrics)
        per_assay_metrics.append(row)

    metric_df = pd.DataFrame(per_assay_metrics)
    structure_df = metric_df.loc[metric_df["has_structure_label"] == 1]
    result = {
        "n_proteins": int(len(test_items)),
        "mean_test_spearman": float(metric_df["test_spearman"].mean()),
        "n_structure_proteins": int(len(structure_df)),
        "per_assay_metrics": per_assay_metrics,
    }
    for name in STRUCTURE_TEST_REPORT_NAMES:
        result[f"mean_{name}"] = float(structure_df[name].mean()) if len(structure_df) > 0 else np.nan
    return result


def train_one_protein(
    cerebra_model,
    model,
    mlp,
    optimizer,
    data,
    split,
    structure_label,
    args,
    device,
):
    configure_cerebra_training(cerebra_model, args.training_mode)
    model.train()
    mlp.train()
    optimizer.zero_grad(set_to_none=True)

    train_y = torch.as_tensor(split["y_z"], dtype=torch.float32, device=device)

    train_cerebra = args.training_mode == "e2e"
    has_cdna_structure = train_cerebra and structure_label is not None
    if not train_cerebra:
        with torch.no_grad():
            data = apply_cerebra_features(data, cerebra_model, mode="eval")
    elif has_cdna_structure:
        data, cerebra_outputs, anchor_list = apply_cerebra_features(
            data,
            cerebra_model,
            mode="train",
            return_structure_context=True,
        )
    else:
        data = apply_cerebra_features(data, cerebra_model, mode="train")
    single_pred, high_delta = model(data)

    preds, pred_epi_all = calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=split["mut_list"],
        U=high_delta,
        mlp_model=mlp,
        max_mut=split["max_mut"],
        device=str(device),
        return_epi=True,
    )

    fitness_loss = torch.nn.functional.smooth_l1_loss(
        preds.float(),
        train_y.float(),
        beta=args.huber_delta,
    )

    if len(split["epi_indices"]) > 0:
        epi_indices = torch.as_tensor(split["epi_indices"], dtype=torch.long, device=device)
        epi_y = torch.as_tensor(split["epi_y"], dtype=torch.float32, device=device)

        epi_loss = torch.nn.functional.smooth_l1_loss(
            pred_epi_all[epi_indices].float(),
            epi_y.float(),
            beta=args.huber_delta,
        )
    else:
        epi_loss = fitness_loss.new_tensor(0.0)

    cdna_struct_loss_raw = fitness_loss.new_tensor(0.0)
    cdna_struct_loss_items = None
    if has_cdna_structure:
        cdna_structure_batch = prepare_cdna_structure_batch(
            structure_label,
            device,
        )
        with torch.amp.autocast(device_type=device.type, enabled=False):
            cdna_struct_loss_raw, cdna_struct_loss_items = comp_structure_loss(
                cast_tree_fp32(cerebra_outputs),
                cdna_structure_batch,
                anchor_list,
            )

    cdna_struct_loss = args.lambda_cdna_struct * cdna_struct_loss_raw
    loss = fitness_loss + args.lambda_epi * epi_loss + cdna_struct_loss
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        max_norm=args.clip_grad,
    )

    optimizer.step()

    return {
        "n_train": split["n"],
        "n_epi": split["n_epi"],
        "train_total_loss": float(loss.detach().item()),
        "train_fitness_loss": float(fitness_loss.detach().item()),
        "train_epi_loss": float(epi_loss.detach().item()),
        "has_cdna_structure": int(has_cdna_structure),
        "cdna_struct_loss_raw": (float(cdna_struct_loss_raw.detach().item()) if has_cdna_structure else np.nan),
        "cdna_struct_loss_weighted": (float(cdna_struct_loss.detach().item()) if has_cdna_structure else np.nan),
        "cdna_struct_loss_items": ([float(x) for x in np.asarray(cdna_struct_loss_items).reshape(-1)] if has_cdna_structure else None),
        "train_spearman": float(spearman_corr(preds.detach(), train_y.detach()).detach().item()),
        "train_grad_norm": float(grad_norm),
    }


@torch.no_grad()
def predict_one_protein(
    cerebra_model,
    model,
    mlp,
    data,
    split,
    device,
    structure_label=None,
):
    cerebra_model.eval()
    model.eval()
    mlp.eval()

    if structure_label is not None:
        data, cerebra_outputs, anchor_list = apply_cerebra_features(
            data,
            cerebra_model,
            mode="eval",
            return_structure_context=True,
        )
        structure_metrics = evaluate_cdna_structure_loss(
            cerebra_outputs,
            anchor_list,
            structure_label,
            device,
        )
    else:
        data = apply_cerebra_features(data, cerebra_model, mode="eval")
        structure_metrics = None

    single_pred, high_delta = model(data)

    pred_z = calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=split["mut_list"],
        U=high_delta,
        mlp_model=mlp,
        max_mut=split["max_mut"],
        device=str(device),
        return_epi=False,
    )

    pred_z = pred_z.detach().cpu().numpy().reshape(-1)
    pred_raw = pred_z * split["train_std"] + split["train_mean"]
    return pred_raw, structure_metrics


def load_assays(args):
    assays = []
    use_structure = (args.training_mode == "e2e" and args.lambda_cdna_struct > 0) or args.eval_test_structure_loss

    assay_dirs = sorted(p for p in Path(args.data_root).iterdir() if p.is_dir() and p.name.startswith(args.assay_prefix))

    for assay_dir in assay_dirs:
        assay_id = assay_dir.name
        structure_pt = assay_dir / FEATURE_FILENAME
        cv_df = pd.read_csv(assay_dir / "data.csv")

        L = len(read_wt_idx_from_fasta(assay_dir / "wt.fasta"))
        auto_geo, auto_epi = (1.0 / 3.0, 0.0) if L > 200 else (0.5, 1.0 / 3.0)

        assays.append(
            {
                "assay_id": assay_id,
                "assay_dir": assay_dir,
                "cv_df": cv_df,
                "geo_neighbor": auto_geo if args.geo_neighbor < 0 else args.geo_neighbor,
                "epi_neighbor": auto_epi if args.epi_neighbor < 0 else args.epi_neighbor,
                "structure_pt": structure_pt if use_structure and structure_pt.is_file() else None,
            }
        )

    n_structure_labels = sum(item["structure_pt"] is not None for item in assays)
    print(f"[data] n_assays={len(assays)} | " f"structure_feature_files={n_structure_labels} | " f"without_structure_feature_file={len(assays) - n_structure_labels}")
    return assays


def train_one_fold(fold, train_items, test_items, args, device, train_log, train_mean, train_std):
    print(f"\n========== fold {fold}/{args.k_folds - 1} ==========")

    fold_dir = train_log / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fold] n_proteins={len(train_items) + len(test_items)} | " f"train={sum(x['split']['n'] for x in train_items)} | " f"test={sum(x['split']['n'] for x in test_items)} | " f"epi={sum(x['split']['n_epi'] for x in train_items)}")

    model, mlp = build_model(
        args,
        geo_neighbor=train_items[0]["geo_neighbor"],
        epi_neighbor=train_items[0]["epi_neighbor"],
    )
    cerebra_model = build_cerebra_model(args, device)
    model = model.to(device)
    mlp = mlp.to(device)

    optimizer = build_training_optimizer(cerebra_model, model, mlp, args)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=args.warmup_epochs,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.max_epochs - args.warmup_epochs),
                eta_min=args.min_lr,
            ),
        ],
        milestones=[args.warmup_epochs],
    )

    loss_records = []
    eval_records = []
    best_val_spearman = None

    start_epoch = 0
    if args.resume:
        # Checkpoints contain optimizer and NumPy/Python RNG state; load only
        # trusted checkpoints produced by this training script.
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["args"].get("training_mode", "e2e") != args.training_mode:
            raise ValueError("--training_mode must match the resume checkpoint")
        if state["args"]["cerebra_revision"] != args.cerebra_revision:
            raise ValueError("--cerebra_revision must match the resume checkpoint")
        start_epoch, best_val_spearman, loss_records, eval_records = restore_training_checkpoint(
            state,
            cerebra_model,
            model,
            mlp,
            optimizer,
            scheduler,
            fold,
            train_mean,
            train_std,
            train_items,
            test_items,
            device,
        )
        del state
        pd.DataFrame(loss_records).to_csv(fold_dir / "loss.csv", index=False)
        if eval_records:
            pd.DataFrame(eval_records).to_csv(fold_dir / "periodic_eval.csv", index=False)
        elif (fold_dir / "periodic_eval.csv").exists():
            (fold_dir / "periodic_eval.csv").write_text("")
        detail_path = fold_dir / "periodic_test_structure_per_assay.csv"
        if detail_path.exists():
            details = pd.read_csv(detail_path)
            details.loc[details["global_epoch"] <= start_epoch].to_csv(detail_path, index=False)
        print(f"[resume] fold={fold} | next_epoch={start_epoch + 1} | {args.resume}")

    for epoch in range(start_epoch, args.max_epochs):
        lr_now = float(optimizer.param_groups[0]["lr"])
        random.shuffle(train_items)

        batch_records = []

        for item in train_items:
            set_model_neighbors(
                model,
                item["geo_neighbor"],
                item["epi_neighbor"],
            )

            data = to_gpu(load_features(item["assay_dir"]), device)

            batch_records.append(
                train_one_protein(
                    cerebra_model=cerebra_model,
                    model=model,
                    mlp=mlp,
                    optimizer=optimizer,
                    data=data,
                    split=item["split"],
                    structure_label=item["structure_label"],
                    args=args,
                    device=device,
                )
            )

            del data

        batch_df = pd.DataFrame(batch_records)

        epoch_record = {
            "epoch": epoch,
            "lr": lr_now,
            "n_protein_batches": len(batch_df),
            "total_train_variants": int(batch_df["n_train"].sum()),
            "total_epi_samples": int(batch_df["n_epi"].sum()),
            "mean_train_total_loss": float(batch_df["train_total_loss"].mean()),
            "mean_train_fitness_loss": float(batch_df["train_fitness_loss"].mean()),
            "mean_train_epi_loss": float(batch_df["train_epi_loss"].mean()),
            "n_cdna_structure_batches": int(batch_df["has_cdna_structure"].sum()),
            "mean_cdna_struct_loss_raw": float(batch_df["cdna_struct_loss_raw"].mean()),
            "mean_cdna_struct_loss_weighted": float(batch_df["cdna_struct_loss_weighted"].mean()),
            "mean_train_spearman": float(batch_df["train_spearman"].mean()),
            "mean_train_grad_norm": float(batch_df["train_grad_norm"].mean()),
            "lambda_epi": float(args.lambda_epi),
            "lambda_cdna_struct": float(args.lambda_cdna_struct),
            "huber_delta": float(args.huber_delta),
            "train_mean": float(train_mean),
            "train_std": float(train_std),
        }

        loss_records.append(epoch_record)
        pd.DataFrame(loss_records).to_csv(fold_dir / "loss.csv", index=False)

        print(f"[epoch] fold={fold} | {epoch + 1}/{args.max_epochs} | " f"lr={lr_now:.3e} | " f"loss={epoch_record['mean_train_total_loss']:.6f} | " f"fitness={epoch_record['mean_train_fitness_loss']:.6f} | " f"epi={epoch_record['mean_train_epi_loss']:.6f} | " f"cdna_struct_raw={epoch_record['mean_cdna_struct_loss_raw']:.6f} | " f"cdna_struct_weighted={epoch_record['mean_cdna_struct_loss_weighted']:.6f} | " f"spearman={epoch_record['mean_train_spearman']:.6f}")

        scheduler.step()

        if (epoch + 1) % max(1, int(args.ckpt_every)) == 0:
            periodic_eval = evaluate_fold_once(
                cerebra_model=cerebra_model,
                model=model,
                mlp=mlp,
                test_items=test_items,
                device=device,
            )
            eval_record = {
                "epoch": epoch,
                "global_epoch": epoch + 1,
                "fold": fold,
                "lr": lr_now,
                "n_proteins": periodic_eval["n_proteins"],
                "mean_test_spearman": periodic_eval["mean_test_spearman"],
                "n_structure_proteins": periodic_eval["n_structure_proteins"],
            }
            eval_record.update({f"mean_{name}": periodic_eval[f"mean_{name}"] for name in STRUCTURE_TEST_REPORT_NAMES})
            eval_records.append(eval_record)
            periodic_eval_path = fold_dir / "periodic_eval.csv"
            pd.DataFrame([eval_record]).to_csv(
                periodic_eval_path,
                mode="a",
                header=not periodic_eval_path.exists() or periodic_eval_path.stat().st_size == 0,
                index=False,
            )

            periodic_structure_df = pd.DataFrame(periodic_eval["per_assay_metrics"])
            periodic_structure_df = periodic_structure_df.loc[periodic_structure_df["has_structure_label"] == 1].copy()
            if len(periodic_structure_df) > 0:
                periodic_structure_df.insert(0, "fold", fold)
                periodic_structure_df.insert(0, "global_epoch", epoch + 1)
                periodic_structure_path = fold_dir / "periodic_test_structure_per_assay.csv"
                periodic_structure_df.to_csv(
                    periodic_structure_path,
                    mode="a",
                    header=not periodic_structure_path.exists(),
                    index=False,
                )

            val_spearman = eval_record["mean_test_spearman"]
            is_best = best_val_spearman is None or (np.isfinite(val_spearman) and (not np.isfinite(best_val_spearman) or val_spearman > best_val_spearman))
            ckpt_path = fold_dir / "best_checkpoint.pt"
            if is_best:
                best_val_spearman = val_spearman
                save_training_checkpoint(
                    ckpt_path,
                    cerebra_model,
                    model,
                    mlp,
                    optimizer,
                    scheduler,
                    epoch,
                    fold,
                    args,
                    train_mean,
                    train_std,
                    best_val_spearman,
                    loss_records,
                    eval_records,
                    train_items,
                    test_items,
                    device,
                )
            print(f"[periodic-eval] fold={fold} | epoch={epoch + 1} | " f"spearman={val_spearman:.6f} | " f"n_struct={eval_record['n_structure_proteins']} | " f"lddt={eval_record['mean_lddt']:.6f} | " f"real_fape={eval_record['mean_real_fape']:.6f} | " f"ckpt={ckpt_path.name}")

        # Save every completed epoch, independently of the evaluation interval.
        save_training_checkpoint(
            fold_dir / "last_checkpoint.pt",
            cerebra_model,
            model,
            mlp,
            optimizer,
            scheduler,
            epoch,
            fold,
            args,
            train_mean,
            train_std,
            best_val_spearman,
            loss_records,
            eval_records,
            train_items,
            test_items,
            device,
        )

    return cerebra_model, model, mlp


def test_one_fold(fold, cerebra_model, model, mlp, train_items, test_items, pred_dfs, device, train_log, pred_output):
    fold_dir = train_log / f"fold_{fold}"
    protein_metrics = []

    for item in test_items:
        split = item["split"]
        assay_id = item["assay_id"]

        set_model_neighbors(
            model,
            item["geo_neighbor"],
            item["epi_neighbor"],
        )

        data = to_gpu(load_features(item["assay_dir"]), device)

        pred_raw, structure_metrics = predict_one_protein(
            cerebra_model=cerebra_model,
            model=model,
            mlp=mlp,
            data=data,
            split=split,
            device=device,
            structure_label=item.get("structure_label"),
        )

        del data

        test_spearman = float(spearmanr(split["y_raw"], pred_raw, nan_policy="omit").correlation) if len(split["y_raw"]) >= 2 else np.nan

        row = {
            "assay_id": assay_id,
            "fold": fold,
            "n_test": split["n"],
            "test_spearman": test_spearman,
            "has_structure_label": int(structure_metrics is not None),
        }
        if structure_metrics is not None:
            row.update(structure_metrics)
        protein_metrics.append(row)

        pred_dfs[assay_id]["pred"] = pred_raw

    protein_df = pd.DataFrame(protein_metrics)

    final_metrics = pd.DataFrame(
        [
            {
                "assay_id": "PROTEIN_MEAN",
                "fold": fold,
                "n_proteins": int(len(protein_df)),
                "n_train": int(sum(x["split"]["n"] for x in train_items)),
                "n_test": int(protein_df["n_test"].sum()),
                "n_epi": int(sum(x["split"]["n_epi"] for x in train_items)),
                "test_spearman": float(protein_df["test_spearman"].mean()),
                "n_structure_proteins": int(protein_df["has_structure_label"].sum()),
            }
        ]
    )

    structure_columns = [
        "assay_id",
        "fold",
        "n_test",
        *STRUCTURE_TEST_REPORT_NAMES,
    ]
    structure_df = protein_df.loc[protein_df["has_structure_label"] == 1].reindex(columns=structure_columns)
    structure_mean_row = {
        "fold": fold,
        "n_structure_proteins": int(len(structure_df)),
    }
    structure_mean_row.update({f"mean_{name}": (float(structure_df[name].mean()) if len(structure_df) > 0 else np.nan) for name in STRUCTURE_TEST_REPORT_NAMES})

    structure_df.to_csv(
        fold_dir / "final_test_structure_per_assay.csv",
        index=False,
    )
    pd.DataFrame([structure_mean_row]).to_csv(
        fold_dir / "final_test_structure_mean.csv",
        index=False,
    )
    final_metrics.to_csv(fold_dir / "final_test_metrics.csv", index=False)

    print(f"[final-test] fold={fold} | PROTEIN_MEAN | " f"n_proteins={len(protein_df)} | " f"spearman={final_metrics.loc[0, 'test_spearman']:.6f} | " f"n_struct={structure_mean_row['n_structure_proteins']} | " f"lddt={structure_mean_row['mean_lddt']:.6f} | " f"real_fape={structure_mean_row['mean_real_fape']:.6f}")

    for item in test_items:
        assay_id = item["assay_id"]
        df = pred_dfs[assay_id]
        df[["mutation_name", "label", "fold_id", "pred"]].to_csv(
            pred_output / f"{assay_id}.csv",
            index=False,
        )


def resolve_output_paths(args):
    experiment_name = f"{args.training_mode}_cdna_nwalign_5fold"
    if args.train_log_root is None:
        args.train_log_root = str(base_dir / "training_log" / experiment_name)
    if args.pred_output_root is None:
        args.pred_output_root = str(base_dir / "cerebra_outputs" / experiment_name)


def run_train(args):
    resolve_output_paths(args)
    if args.resume and args.fold_id is None:
        raise ValueError("--resume requires --fold_id for its single fold")
    if args.fold_id is not None and not 0 <= args.fold_id < args.k_folds:
        raise ValueError("--fold_id must be in [0, --k_folds)")

    pred_output = Path(args.pred_output_root)
    train_log = Path(args.train_log_root)

    pred_output.mkdir(parents=True, exist_ok=True)
    train_log.mkdir(parents=True, exist_ok=True)

    assays = load_assays(args)

    fold_df = build_or_load_folds(assays, args, train_log)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device}, but CUDA is unavailable")

    fold_map = dict(zip(fold_df["assay_id"].astype(str), fold_df["protein_fold"].astype(int)))

    pred_dfs = {}

    for assay in assays:
        assay["protein_fold"] = fold_map[assay["assay_id"]]

        cv_df = assay["cv_df"]
        pred_dfs[assay["assay_id"]] = pd.DataFrame(
            {
                "mutation_name": cv_df["mutation_name"].astype(str),
                "label": cv_df["label"].astype(float),
                "fold_id": int(assay["protein_fold"]),
                "pred": np.nan,
            }
        )

    print("[fold-count]")
    print(pd.Series([x["protein_fold"] for x in assays]).value_counts().sort_index().to_string())

    folds = range(args.k_folds) if args.fold_id is None else [args.fold_id]
    for fold in folds:
        # Match the independent workers' original seed at the start of each fold.
        set_seed_everywhere(args.seed)
        print(f"[serial] fold={fold} | device={device}")
        train_raw = [x for x in assays if x["protein_fold"] != fold]
        test_raw = [x for x in assays if x["protein_fold"] == fold]

        train_labels = np.concatenate([x["cv_df"]["label"].to_numpy(dtype=float) for x in train_raw])

        train_mean = float(train_labels.mean())
        train_std = float(train_labels.std(ddof=0))
        train_std = train_std if train_std >= 1e-8 else 1.0

        train_items = [{**assay, "split": make_split(assay["cv_df"], train_mean, train_std)} for assay in train_raw]

        test_items = [{**assay, "split": make_split(assay["cv_df"], train_mean, train_std)} for assay in test_raw]

        if args.training_mode == "e2e" and args.lambda_cdna_struct > 0:
            n_loaded_train_structure_labels = attach_cdna_structure_labels(train_items)
        else:
            for item in train_items:
                item["structure_label"] = None
            n_loaded_train_structure_labels = 0

        if args.eval_test_structure_loss:
            n_loaded_test_structure_labels = attach_cdna_structure_labels(test_items)
        else:
            for item in test_items:
                item["structure_label"] = None
            n_loaded_test_structure_labels = 0

        print(f"[fold-structure] fold={fold} | " f"train_assays={len(train_items)} | " f"train_structure_labels={n_loaded_train_structure_labels} | " f"test_assays={len(test_items)} | " f"test_structure_labels={n_loaded_test_structure_labels}")

        cerebra_model, model, mlp = train_one_fold(
            fold=fold,
            train_items=train_items,
            test_items=test_items,
            args=args,
            device=device,
            train_log=train_log,
            train_mean=train_mean,
            train_std=train_std,
        )

        test_one_fold(
            fold=fold,
            cerebra_model=cerebra_model,
            model=model,
            mlp=mlp,
            train_items=train_items,
            test_items=test_items,
            pred_dfs=pred_dfs,
            device=device,
            train_log=train_log,
            pred_output=pred_output,
        )

        del cerebra_model, model, mlp
        gc.collect()

        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    print(f"\n[Done] cluster files -> {train_log / 'protein_similarity_folds'}")
    print(f"[Done] prediction csvs -> {pred_output}")
    print(f"[Done] training logs -> {train_log}")


def build_parser():
    parser = argparse.ArgumentParser("Joint cDNA nwalign protein-holdout training")

    parser.add_argument("--device", type=str, default="cuda:0", help="Single device for serial fold training: cpu or cuda:N (default: cuda:0)")
    parser.add_argument(
        "--training_mode",
        type=str,
        default="e2e",
        choices=["e2e", "downstream"],
        help="Training mode: end-to-end joint training or downstream-only training with Cerebra-Seq frozen.",
    )
    parser.add_argument("--fold_id", type=int, default=None, help="Run only this fold; by default all folds run serially")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--cerebra_revision", type=str, default=CEREBRA_REVISION)
    parser.add_argument("--resume", type=str, default=None, help="Trusted HF training checkpoint; requires --fold_id. Restores all models, optimizer, scheduler and next epoch.")

    parser.add_argument("--data_root", default=str(proj_root / "data" / "1M+_to_1M+"))
    parser.add_argument("--pred_output_root", type=str, default=None, help="Default: cerebra_outputs/<training_mode>_cdna_nwalign_5fold")
    parser.add_argument("--train_log_root", type=str, default=None, help="Default: training_log/<training_mode>_cdna_nwalign_5fold")

    parser.add_argument("--assay_prefix", type=str, default="cdna")
    parser.add_argument("--k_folds", type=int, default=5)

    parser.add_argument("--nwalign_bin", type=str, default=str(proj_root / "other_softwares" / "NW-align" / "NWalign"))
    parser.add_argument("--recompute_clusters", action="store_true")
    parser.add_argument("--linkage_method", type=str, default="complete")
    parser.add_argument("--cluster_print_every", type=int, default=500)

    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--lambda_epi", type=float, default=0.0)
    parser.add_argument(
        "--lambda_cdna_struct",
        type=float,
        default=0.001,
        help="Structure-loss weight for training cDNA assays with aligned mmCIF labels.",
    )
    parser.add_argument(
        "--eval_test_structure_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Evaluate and separately record every comp_structure_loss term " "for test assays with aligned structure labels."),
    )
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--clip_grad", type=float, default=2.0)

    parser.add_argument("--adj_dim", type=int, default=32)
    parser.add_argument("--rankH", type=int, default=64)
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_dropout", type=float, default=0.4)

    parser.add_argument("--geo_neighbor", type=float, default=-1.0)
    parser.add_argument("--epi_neighbor", type=float, default=-1.0)

    return parser


def main():
    args = build_parser().parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
