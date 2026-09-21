import torch
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
)

data_root = Path(__file__).resolve().parents[1] / "data"

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}


def log_param_grad_norms(model):
    print("\n[Grad norms per parameter - in model order]")
    for name, p in model.named_parameters():
        if p.grad is None:
            grad_str = "grad = None"
        else:
            grad_str = f"grad_norm = {p.grad.detach().norm(2).item():.4e}"
        print(f"{name:60s} {grad_str}")


def to_gpu(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [to_gpu(i, device=device) for i in obj]
    elif isinstance(obj, tuple):
        return (to_gpu(i, device=device) for i in obj)
    elif isinstance(obj, dict):
        return {i: to_gpu(j, device=device) for i, j in obj.items()}
    else:
        return obj


def get_position_matrix(mut_list: List[str]) -> np.ndarray:
    positions = sorted({int(part[1:-1]) for m in mut_list for part in m.split(",")})
    pos2i = {p: i for i, p in enumerate(positions)}

    Y = np.zeros((len(mut_list), len(positions)), dtype=np.int8)
    for i, m in enumerate(mut_list):
        for part in m.split(","):
            Y[i, pos2i[int(part[1:-1])]] = 1

    return Y


def zscore_labels(train_labels: pd.Series, val_labels: pd.Series, eps: float = 1e-8):
    train_labels = train_labels.astype(float)
    val_labels = val_labels.astype(float)

    mean = train_labels.mean()
    std = train_labels.std(ddof=0)
    denom = std + eps

    train_z = (train_labels - mean) / denom
    val_z = (val_labels - mean) / denom

    return train_z.tolist(), val_z.tolist()


def row_zscore(x, eps=1e-8):
    # x: [L,20]
    mu = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mu) / (std + eps)


def read_wt_idx_from_fasta(fasta_path: str) -> torch.LongTensor:
    wt_seq = Path(fasta_path).read_text().splitlines()[1].strip().upper()
    return torch.tensor([AA_TO_IDX[aa] for aa in wt_seq], dtype=torch.long)


def max_mut_from_list(mut_list: List[str]) -> int:
    return max(len([x for x in m.split(",") if x]) for m in mut_list)


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_fold_split(
    cv_df: pd.DataFrame,
    split_col: str,
    test_fold: int,
    *,
    seed: int,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str], np.ndarray, float, float]:
    """
    Returns: Train, Val, Test data and Train Mean/Std
    """
    fold_values = cv_df[split_col].astype(int)
    valid = fold_values >= 0

    train_all_df = cv_df[valid & (fold_values != test_fold)].copy()
    test_df = cv_df[valid & (fold_values == test_fold)].copy()

    train_all_muts = train_all_df["mutation_name"].astype(str).tolist()

    try:
        Y = get_position_matrix(train_all_muts)
    except Exception:
        Y = np.random.randint(0, 2, size=(len(train_all_df), 10))

    x_dummy = np.zeros((len(train_all_df), 1))

    mskf = MultilabelStratifiedKFold(n_splits=8, shuffle=True, random_state=seed + test_fold)
    val_fold_id = (seed + test_fold) % 8

    tr_idx, va_idx = None, None
    for fold_id, (tr, va) in enumerate(mskf.split(x_dummy, Y)):
        if fold_id == val_fold_id:
            tr_idx, va_idx = tr, va
            break

    train_df = train_all_df.iloc[tr_idx]
    val_df = train_all_df.iloc[va_idx]

    train_mut_list = train_df["mutation_name"].astype(str).tolist()
    val_mut_list = val_df["mutation_name"].astype(str).tolist()
    test_mut_list = test_df["mutation_name"].astype(str).tolist()

    train_labels = train_df["label"].to_numpy(dtype=float)
    val_labels = val_df["label"].to_numpy(dtype=float)
    test_labels = test_df["label"].to_numpy(dtype=float)

    train_mean = float(np.mean(train_labels)) if len(train_labels) > 0 else 0.0
    train_std = float(np.std(train_labels, ddof=0)) if len(train_labels) > 0 else 1.0
    if not np.isfinite(train_std) or train_std < 1e-12:
        train_std = 1.0

    train_labels_z = (train_labels - train_mean) / train_std
    val_labels_z = (val_labels - train_mean) / train_std
    test_labels_z = (test_labels - train_mean) / train_std

    return (
        train_mut_list,
        train_labels_z,
        val_mut_list,
        val_labels_z,
        test_mut_list,
        test_labels_z,
        train_mean,
        train_std,
    )


def load_features(assay_dir):
    embedding = torch.load(assay_dir / "embedding_ESM2_650M_for_Cerebra_Epistasis.pt", map_location="cpu", weights_only=True)
    cerebra_features = torch.load(assay_dir / "embedding_Cerebra_Seq_for_Cerebra_Epistasis.pt", map_location="cpu", weights_only=True)
    wt_idx = read_wt_idx_from_fasta(assay_dir / "wt.fasta")

    return {
        "embedding": embedding,
        "wt_idx": torch.as_tensor(wt_idx, dtype=torch.long),
        "node_embedding": cerebra_features["node_embedding"].float(),
        "edge_embedding": cerebra_features["edge_embedding"].float().permute(1, 2, 0).contiguous(),
        "atom14_coords": cerebra_features["atom14_coords"].float(),
        "atom14_masks": cerebra_features["atom14_masks"].float(),
    }


def compute_twobody_epistasis_labels(train_df, train_mean, train_std):
    train_df = train_df.copy().reset_index(drop=True)
    label_map = dict(zip(train_df["mutation_name"], train_df["label"]))

    epi_indices, epi_labels = [], []

    for idx, row in train_df.iterrows():
        muts = [m.strip() for m in row["mutation_name"].split(",") if m.strip()]

        if len(muts) != 2:
            continue
        if any(m not in label_map for m in muts):
            continue

        epi_z = (float(row["label"]) - sum(float(label_map[m]) for m in muts) + train_mean) / train_std

        epi_indices.append(idx)
        epi_labels.append(epi_z)

    return np.asarray(epi_indices, dtype=np.int64), np.asarray(epi_labels, dtype=np.float32)
