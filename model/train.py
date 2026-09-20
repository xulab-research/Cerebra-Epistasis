import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cerebra_epistasis.model import SE3Transformer
from utils.metrics import spearman_corr
from utils.utils_func import (
    compute_twobody_epistasis_labels,
    load_features,
    max_mut_from_list,
    read_wt_idx_from_fasta,
    set_seed_everywhere,
    to_gpu,
)
from utils.calculate_nbodys_mutation_effect import calculate_batch_prediction_mlp, EpistasisMLP

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def load_data(data_root, device):
    df = pd.read_csv(data_root / "data.csv").dropna(subset=["mutation_name", "label", "fold_id"]).copy()
    df["mutation_name"] = df["mutation_name"].astype(str).str.strip()
    df["label"] = df["label"].astype(float)
    df["fold_id"] = df["fold_id"].astype(int)
    train_df = df[df["fold_id"] == 0].reset_index(drop=True)
    test_df = df[df["fold_id"] == 1].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("data.csv must contain training rows (fold_id=0) and test rows (fold_id=1).")

    mean = float(train_df["label"].mean())
    std = float(train_df["label"].std(ddof=0)) + 1e-8
    train_y = torch.as_tensor(((train_df["label"].to_numpy() - mean) / std).astype(np.float32), device=device)
    epi_indices, epi_labels = compute_twobody_epistasis_labels(train_df=train_df, train_mean=mean, train_std=std)
    length = len(read_wt_idx_from_fasta(data_root / "wt.fasta"))
    geo_neighbor, epi_neighbor = (1.0 / 3.0, 0.0) if length > 200 else (0.5, 1.0 / 3.0)
    return {
        "data": to_gpu(load_features(data_root), device),
        "train_df": train_df,
        "test_df": test_df,
        "train_y": train_y,
        "epi_indices": torch.as_tensor(epi_indices, dtype=torch.long, device=device),
        "epi_y": torch.as_tensor(epi_labels, dtype=torch.float32, device=device),
        "max_mut": max_mut_from_list(df["mutation_name"].tolist()),
        "geo_neighbor": geo_neighbor,
        "epi_neighbor": epi_neighbor,
        "mean": mean,
        "std": std,
    }


def train(args, unit, device):
    set_seed_everywhere(args.seed)
    model = SE3Transformer(
        depth=1,
        hidden_fiber_dict={0: 320, 1: 32},
        out_fiber_dict={0: 128, 1: 32},
        adj_dim=args.adj_dim,
        rankH=args.rankH,
        geo_neighbor=unit["geo_neighbor"],
        epi_neighbor=unit["epi_neighbor"],
    ).to(device)
    mlp = EpistasisMLP(input_dim=args.rankH, hidden_dim=args.mlp_hidden_dim, dropout=args.mlp_dropout).to(device)
    parameters = list(model.parameters()) + list(mlp.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    mut_list = unit["train_df"]["mutation_name"].tolist()
    logs = []

    for epoch in range(args.epochs):
        model.train()
        mlp.train()
        optimizer.zero_grad(set_to_none=True)
        single_pred, high_delta = model(unit["data"])
        preds, pred_epi = calculate_batch_prediction_mlp(
            single_mut_matrix=single_pred,
            mut_name_list=mut_list,
            U=high_delta,
            mlp_model=mlp,
            max_mut=unit["max_mut"],
            device=str(device),
            return_epi=True,
            sqrt_scale=True,
        )
        fitness_loss = torch.nn.functional.smooth_l1_loss(preds.float(), unit["train_y"], beta=args.huber_delta)
        epi_loss = fitness_loss.new_zeros(())
        if args.lambda_epi > 0 and unit["epi_indices"].numel() > 0:
            epi_loss = torch.nn.functional.smooth_l1_loss(pred_epi[unit["epi_indices"]].float(), unit["epi_y"], beta=args.huber_delta)
        loss = fitness_loss + args.lambda_epi * epi_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad)
        optimizer.step()
        scheduler.step()

        train_spearman = float(spearman_corr(preds.detach(), unit["train_y"]).item())
        logs.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_total_loss": loss.item(),
                "train_fitness_loss": fitness_loss.item(),
                "train_epi_loss": epi_loss.item(),
                "train_spearman": train_spearman,
            }
        )
        print(f"[epoch {epoch:03d}] loss={loss.item():.6f} train_spearman={train_spearman:.6f}")

    log_dir = Path(args.train_log)
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv(log_dir / "train_log.csv", index=False)
    torch.save(
        {
            "epoch": args.epochs - 1,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "mlp_state": {k: v.detach().cpu() for k, v in mlp.state_dict().items()},
            "args": vars(args),
            "train_mean": unit["mean"],
            "train_denom": unit["std"],
            "n_train_epi": int(unit["epi_indices"].numel()),
        },
        log_dir / "last.pt",
    )
    return model, mlp


@torch.no_grad()
def predict_test(args, unit, model, mlp, device):
    model.eval()
    mlp.eval()
    single_pred, high_delta = model(unit["data"])
    test_df = unit["test_df"][["mutation_name", "label", "fold_id"]].copy()
    pred_z = calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=test_df["mutation_name"].tolist(),
        U=high_delta,
        mlp_model=mlp,
        max_mut=unit["max_mut"],
        device=str(device),
        return_epi=False,
        sqrt_scale=True,
    )
    test_df["pred"] = pred_z.detach().cpu().numpy().reshape(-1) * unit["std"] + unit["mean"]
    pred_dir = Path(args.pred_output)
    pred_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(pred_dir / "predictions.csv", index=False)
    print(f"[saved] {pred_dir / 'predictions.csv'} ({len(test_df)} test variants)")


def main():
    parser = argparse.ArgumentParser(description="Train on one assay and predict its held-out test set.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--train_log", type=Path, default=BASE_DIR / "training_log")
    parser.add_argument("--pred_output", type=Path, default=BASE_DIR / "output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--adj_dim", type=int, default=32)
    parser.add_argument("--rankH", type=int, default=64)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--clip_grad", type=float, default=2.0)
    parser.add_argument("--lambda_epi", type=float, default=2.0)
    args = parser.parse_args()
    device = torch.device(args.device)
    unit = load_data(args.data_root, device)
    model, mlp = train(args, unit, device)
    predict_test(args, unit, model, mlp, device)


if __name__ == "__main__":
    main()
