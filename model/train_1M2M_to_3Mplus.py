import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from cerebra_epistasis.model import SE3Transformer
from utils.metrics import spearman_corr
from utils.utils_func import (
    read_wt_idx_from_fasta,
    to_gpu,
    max_mut_from_list,
    set_seed_everywhere,
    zscore_labels,
    compute_twobody_epistasis_labels,
    load_features,
)
from utils.calculate_nbodys_mutation_effect import (
    calculate_batch_prediction_mlp,
    EpistasisMLP,
)

base_dir = Path(__file__).resolve().parent
proj_root = base_dir.parent


def select_assays(data_root: Path, only_assay: str) -> list[Path]:
    assay_dirs = [p for p in sorted(data_root.iterdir()) if p.is_dir()]

    if only_assay:
        assay_dirs = [p for p in assay_dirs if p.name == only_assay]

    return assay_dirs


def prepare_train_unit(assay_dir: Path, eps: float = 1e-8) -> dict:
    df = pd.read_csv(assay_dir / "data.csv")
    df = df.dropna(subset=["mutation_name", "label", "fold_id"]).copy()
    df["mutation_name"] = df["mutation_name"].astype(str).str.strip()
    df["fold_id"] = df["fold_id"].astype(int)
    df["label"] = df["label"].astype(float)

    train_df = df[df["fold_id"] == 0].copy().reset_index(drop=True)
    test_df = df[df["fold_id"] == 1].copy().reset_index(drop=True)

    train_y, _ = zscore_labels(train_df["label"], test_df["label"], eps=eps)
    train_mean = float(train_df["label"].mean())
    train_denom = float(train_df["label"].std(ddof=0)) + eps

    epi_indices, epi_labels = compute_twobody_epistasis_labels(
        train_df=train_df,
        train_mean=train_mean,
        train_std=train_denom,
    )

    wt_idx = read_wt_idx_from_fasta(assay_dir / "wt.fasta")
    geo_neighbor, epi_neighbor = (1.0 / 3.0, 0.0) if len(wt_idx) > 200 else (0.5, 1.0 / 3.0)

    print(f"[unit] {assay_dir.name} | " f"n_train={len(train_df)} | " f"n_test={len(test_df)} | " f"n_train_twobody_epi={len(epi_indices)}")

    return {
        "assay_id": assay_dir.name,
        "data_cpu": load_features(assay_dir),
        "train_mut_list": train_df["mutation_name"].tolist(),
        "train_y": np.asarray(train_y, dtype=float),
        "train_mean": train_mean,
        "train_denom": train_denom,
        "epi_indices": epi_indices,
        "epi_labels": epi_labels,
        "n_train_epi": len(epi_indices),
        "max_mut": max_mut_from_list(df["mutation_name"].tolist()),
        "geo_neighbor": geo_neighbor,
        "epi_neighbor": epi_neighbor,
        "all_df": df.reset_index(drop=True),
        "all_mut_list": df["mutation_name"].tolist(),
    }


def train_one_epoch(
    model,
    mlp_model,
    optimizer,
    data,
    train_y,
    train_mut_list,
    max_mut,
    device_str,
    huber_delta,
    clip_grad,
    epi_indices,
    epi_y,
    lambda_epi,
) -> tuple[float, float, float, float]:
    model.train()
    mlp_model.train()
    optimizer.zero_grad(set_to_none=True)

    single_pred, high_delta = model(data)
    preds, pred_epi_all = calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=train_mut_list,
        U=high_delta,
        mlp_model=mlp_model,
        max_mut=max_mut,
        device=device_str,
        return_epi=True,
        sqrt_scale=True,
    )

    fitness_loss = torch.nn.functional.smooth_l1_loss(
        preds.float(),
        train_y.float(),
        beta=huber_delta,
    )

    if lambda_epi > 0 and epi_indices.numel() > 0:
        epi_loss = torch.nn.functional.smooth_l1_loss(
            pred_epi_all[epi_indices].float(),
            epi_y.float(),
            beta=huber_delta,
        )
        loss = fitness_loss + lambda_epi * epi_loss
    else:
        epi_loss = torch.zeros((), dtype=fitness_loss.dtype, device=fitness_loss.device)
        loss = fitness_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(mlp_model.parameters()),
        max_norm=clip_grad,
    )
    optimizer.step()

    train_spearman = float(spearman_corr(preds.detach(), train_y).item())

    return (
        float(loss.item()),
        float(fitness_loss.item()),
        float(epi_loss.item()),
        train_spearman,
    )


def train_one_assay(args, unit, assay_out: Path, device, device_str):
    assay_id = unit["assay_id"]
    assay_out.mkdir(parents=True, exist_ok=True)
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

    mlp_model = EpistasisMLP(
        input_dim=args.rankH,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.mlp_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(mlp_model.parameters()),
        lr=args.lr,
        eps=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    data = to_gpu(unit["data_cpu"], device)
    train_y = torch.as_tensor(unit["train_y"], dtype=torch.float32, device=device)
    epi_indices = torch.as_tensor(unit["epi_indices"], dtype=torch.long, device=device)
    epi_y = torch.as_tensor(unit["epi_labels"], dtype=torch.float32, device=device)

    print(f"\n[train] assay={assay_id} | " f"train={len(unit['train_mut_list'])} | " f"n_train_epi={unit['n_train_epi']} | " f"epochs={args.epochs} | " f"lambda_epi={args.lambda_epi}")

    records = []
    for epoch in range(args.epochs):
        total_loss, fitness_loss, epi_loss, train_spearman = train_one_epoch(
            model=model,
            mlp_model=mlp_model,
            optimizer=optimizer,
            data=data,
            train_y=train_y,
            train_mut_list=unit["train_mut_list"],
            max_mut=unit["max_mut"],
            device_str=device_str,
            huber_delta=args.huber_delta,
            clip_grad=args.clip_grad,
            epi_indices=epi_indices,
            epi_y=epi_y,
            lambda_epi=args.lambda_epi,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        records.append(
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_total_loss": total_loss,
                "train_fitness_loss": fitness_loss,
                "train_epi_loss": epi_loss,
                "train_spearman": train_spearman,
            }
        )

        print(f"[train] {assay_id} " f"epoch={epoch:03d} " f"lr={lr_now:.6g} " f"total_loss={total_loss:.6f} " f"fitness_loss={fitness_loss:.6f} " f"epi_loss={epi_loss:.6f} " f"train_spearman={train_spearman:.6f}")

    pd.DataFrame(records).to_csv(assay_out / "train_log.csv", index=False)

    torch.save(
        {
            "epoch": args.epochs - 1,
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "mlp_state": {k: v.cpu() for k, v in mlp_model.state_dict().items()},
            "args": vars(args),
            "train_mean": unit["train_mean"],
            "train_denom": unit["train_denom"],
            "n_train_epi": unit["n_train_epi"],
        },
        assay_out / "last.pt",
    )

    return model, mlp_model, data


@torch.no_grad()
def predict_and_save(unit, pred_csv_path: Path, model, mlp_model, data, device_str):
    model.eval()
    mlp_model.eval()

    single_pred, high_delta = model(data)
    pred_z = calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=unit["all_mut_list"],
        U=high_delta,
        mlp_model=mlp_model,
        max_mut=unit["max_mut"],
        device=device_str,
        return_epi=False,
        sqrt_scale=True,
    )

    pred_raw = pred_z.cpu().numpy().astype(float).reshape(-1) * unit["train_denom"] + unit["train_mean"]

    pred_df = unit["all_df"][["mutation_name", "label", "fold_id"]].copy()
    pred_df["split"] = np.where(pred_df["fold_id"] == 0, "train", "test")
    pred_df["pred"] = pred_raw
    pred_df.to_csv(pred_csv_path, index=False)

    print(f"[save] predictions -> {pred_csv_path}")


def run_train(args) -> None:
    device = torch.device(args.device)
    device_str = str(device)

    data_root = Path(args.data_root)
    out_root = Path(args.train_log)
    pred_root = Path(args.pred_output)
    out_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    for assay_dir in select_assays(data_root, args.only_assay):
        unit = prepare_train_unit(assay_dir)
        assay_id = unit["assay_id"]
        assay_out = out_root / assay_id
        pred_csv_path = pred_root / f"{assay_id}.csv"

        model, mlp_model, data = train_one_assay(
            args=args,
            unit=unit,
            assay_out=assay_out,
            device=device,
            device_str=device_str,
        )

        predict_and_save(
            unit=unit,
            pred_csv_path=pred_csv_path,
            model=model,
            mlp_model=mlp_model,
            data=data,
            device_str=device_str,
        )

        del model, mlp_model, data, unit

        print(f"[done] assay={assay_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("direct training pipeline with fold_id split and twobody epistasis loss")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_root", default=str(proj_root / "data" / "1M2M_to_3M+"), help="Assay root directory.")
    parser.add_argument("--only_assay", default="")
    parser.add_argument("--train_log", default=str(base_dir / "training_log" / "1M2M_to_3M+"))
    parser.add_argument("--pred_output", default=str(base_dir / "output" / "1M2M_to_3M+"))
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
    parser.add_argument("--lambda_epi", type=float, default=2.0, help="Weight for twobody epistasis loss.")

    return parser


def main() -> None:
    run_train(build_parser().parse_args())


if __name__ == "__main__":
    main()
