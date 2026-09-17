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


def prepare_assay_base(assay_dir: Path) -> dict:
    df = pd.read_csv(assay_dir / "data.csv")
    df = df.dropna(subset=["mutation_name", "label", "fold_id"]).copy()
    df["mutation_name"] = df["mutation_name"].astype(str).str.strip()
    df["label"] = df["label"].astype(float)
    df["fold_id"] = df["fold_id"].astype(int)
    df["mut_depth"] = df["mutation_name"].str.count(",") + 1

    one_m_df = df[df["fold_id"] == 0].copy().reset_index(drop=True)
    two_m_df = df[df["mut_depth"] == 2].copy().reset_index(drop=True)

    wt_idx = read_wt_idx_from_fasta(assay_dir / "wt.fasta")
    geo_neighbor, epi_neighbor = (1.0 / 3.0, 0.0) if len(wt_idx) > 200 else (0.5, 1.0 / 3.0)

    return {
        "assay_id": assay_dir.name,
        "data_cpu": load_features(assay_dir),
        "one_m_df": one_m_df,
        "two_m_df": two_m_df,
        "max_mut": max_mut_from_list(df["mutation_name"].tolist()),
        "geo_neighbor": geo_neighbor,
        "epi_neighbor": epi_neighbor,
    }


def prepare_fold_unit(base: dict, fold: int, eps: float = 1e-8) -> dict:
    one_m_df = base["one_m_df"]
    two_m_df = base["two_m_df"]

    train_2m_df = two_m_df[two_m_df["fold_id"] != fold].copy().reset_index(drop=True)
    test_2m_df = two_m_df[two_m_df["fold_id"] == fold].copy().reset_index(drop=True)
    train_df = pd.concat([one_m_df, train_2m_df], ignore_index=True)
    pred_df = pd.concat([one_m_df, test_2m_df], ignore_index=True)

    train_y, test_y = zscore_labels(train_df["label"], test_2m_df["label"], eps=eps)
    train_mean = float(train_df["label"].mean())
    train_denom = float(train_df["label"].std(ddof=0)) + eps

    epi_indices, epi_labels = compute_twobody_epistasis_labels(
        train_df=train_df,
        train_mean=train_mean,
        train_std=train_denom,
    )

    print(f"[unit] {base['assay_id']} fold={fold} | " f"n_1m={len(one_m_df)} | " f"n_train_2m={len(train_2m_df)} | " f"n_test_2m={len(test_2m_df)} | " f"n_train={len(train_df)} | " f"n_train_twobody_epi={len(epi_indices)}")

    return {
        "assay_id": base["assay_id"],
        "fold": fold,
        "data_cpu": base["data_cpu"],
        "pred_df": pred_df,
        "train_mut_list": train_df["mutation_name"].tolist(),
        "pred_mut_list": pred_df["mutation_name"].tolist(),
        "train_y": np.asarray(train_y, dtype=float),
        "test_y": np.asarray(test_y, dtype=float),
        "train_mean": train_mean,
        "train_denom": train_denom,
        "epi_indices": epi_indices,
        "epi_labels": epi_labels,
        "n_train_epi": len(epi_indices),
        "max_mut": base["max_mut"],
        "geo_neighbor": base["geo_neighbor"],
        "epi_neighbor": base["epi_neighbor"],
        "n_train": len(train_df),
        "n_test": len(test_2m_df),
    }


def train_one_epoch(
    model,
    mlp_model,
    optimizer,
    data,
    train_y,
    train_mut_list,
    max_mut,
    device,
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
        device=str(device),
        return_epi=True,
        sqrt_scale=False,
    )

    fitness_loss = torch.nn.functional.smooth_l1_loss(preds, train_y, beta=huber_delta)

    if lambda_epi > 0 and epi_indices.numel() > 0:
        epi_loss = torch.nn.functional.smooth_l1_loss(pred_epi_all[epi_indices], epi_y, beta=huber_delta)
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

    train_spearman = spearman_corr(preds.detach(), train_y).item()
    return loss.item(), fitness_loss.item(), epi_loss.item(), train_spearman


def train_one_fold(args, unit, fold_out: Path, device):
    assay_id = unit["assay_id"]
    fold = unit["fold"]
    fold_out.mkdir(parents=True, exist_ok=True)

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

    print(f"\n[train] assay={assay_id} fold={fold} | " f"train={unit['n_train']} | " f"test_2m={unit['n_test']} | " f"n_train_epi={unit['n_train_epi']} | " f"epochs={args.epochs} | " f"lambda_epi={args.lambda_epi}")

    records = []
    for epoch in range(args.epochs):
        lr_now = optimizer.param_groups[0]["lr"]

        total_loss, fitness_loss, epi_loss, train_spearman = train_one_epoch(
            model=model,
            mlp_model=mlp_model,
            optimizer=optimizer,
            data=data,
            train_y=train_y,
            train_mut_list=unit["train_mut_list"],
            max_mut=unit["max_mut"],
            device=device,
            huber_delta=args.huber_delta,
            clip_grad=args.clip_grad,
            epi_indices=epi_indices,
            epi_y=epi_y,
            lambda_epi=args.lambda_epi,
        )
        scheduler.step()

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

        print(f"[train] {assay_id} fold={fold} " f"epoch={epoch:03d} " f"lr={lr_now:.6g} " f"total_loss={total_loss:.6f} " f"fitness_loss={fitness_loss:.6f} " f"epi_loss={epi_loss:.6f} " f"train_spearman={train_spearman:.6f}")

    pd.DataFrame(records).to_csv(fold_out / "train_log.csv", index=False)
    return model, mlp_model, data


def compute_fold_epistasis(pred_df: pd.DataFrame) -> pd.DataFrame:
    label_map = dict(zip(pred_df["mutation_name"], pred_df["label"]))
    pred_map = dict(zip(pred_df["mutation_name"], pred_df["pred"]))
    rows = []

    for _, row in pred_df[pred_df["mut_depth"] == 2].iterrows():
        mut1, mut2 = row["mutation_name"].split(",")
        if mut1 not in label_map or mut2 not in label_map:
            continue

        rows.append(
            {
                "fold_id": row["fold_id"],
                "mutation_name": row["mutation_name"],
                "label": float(row["label"]) - float(label_map[mut1]) - float(label_map[mut2]),
                "pred": float(row["pred"]) - float(pred_map[mut1]) - float(pred_map[mut2]),
            }
        )

    return pd.DataFrame(rows)


@torch.no_grad()
def predict_one_fold(args, unit, fold_out: Path, model, mlp_model, data, device):
    assay_id = unit["assay_id"]
    fold = unit["fold"]

    model.eval()
    mlp_model.eval()

    single_pred, high_delta = model(data)
    pred_z = calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=unit["pred_mut_list"],
        U=high_delta,
        mlp_model=mlp_model,
        max_mut=unit["max_mut"],
        device=str(device),
        return_epi=False,
        sqrt_scale=False,
    )

    pred_df = unit["pred_df"][["mutation_name", "mutated_sequence", "label", "fold_id", "mut_depth"]].copy()

    test_mask = pred_df["mut_depth"].to_numpy() == 2
    test_pred_z = pred_z[test_mask]
    test_y = torch.as_tensor(unit["test_y"], dtype=torch.float32, device=device)

    test_mse = torch.nn.functional.mse_loss(test_pred_z, test_y).item()
    test_spearman = spearman_corr(test_pred_z, test_y).item()

    pred_raw = pred_z.cpu().numpy().reshape(-1) * unit["train_denom"] + unit["train_mean"]
    pred_df["pred"] = pred_raw

    two_m_out = pred_df[pred_df["mut_depth"] == 2][["mutation_name", "mutated_sequence", "pred", "label", "fold_id"]].copy()
    epi_df = compute_fold_epistasis(pred_df)

    pd.DataFrame(
        [
            {
                "fold": fold,
                "test_mse_z": test_mse,
                "test_spearman_z": test_spearman,
                "lambda_epi": args.lambda_epi,
                "n_train_epi": unit["n_train_epi"],
                "n_train": unit["n_train"],
                "n_test_2m": unit["n_test"],
            }
        ]
    ).to_csv(fold_out / "final_test_metrics.csv", index=False)

    print(f"[final-test] assay={assay_id} fold={fold} " f"mse_z={test_mse:.6f} spearman_z={test_spearman:.6f}")

    return two_m_out, epi_df


def run_one_assay(args, assay_dir: Path, device):
    assay_id = assay_dir.name
    assay_train_out = Path(args.train_log) / assay_id
    assay_pred_out = Path(args.pred_output) / assay_id
    assay_train_out.mkdir(parents=True, exist_ok=True)
    assay_pred_out.mkdir(parents=True, exist_ok=True)

    base = prepare_assay_base(assay_dir)
    pred_2m_list = []
    epi_list = []
    set_seed_everywhere(args.seed)

    for fold in range(1, args.n_folds + 1):
        unit = prepare_fold_unit(base, fold)
        fold_out = assay_train_out / f"fold_{fold}"

        model, mlp_model, data = train_one_fold(
            args=args,
            unit=unit,
            fold_out=fold_out,
            device=device,
        )

        two_m_out, epi_df = predict_one_fold(
            args=args,
            unit=unit,
            fold_out=fold_out,
            model=model,
            mlp_model=mlp_model,
            data=data,
            device=device,
        )

        pred_2m_list.append(two_m_out)
        epi_list.append(epi_df)
        del model, mlp_model, data, unit

    two_m_predictions = pd.concat(pred_2m_list, ignore_index=True)
    one_m_out = base["one_m_df"][["mutation_name", "mutated_sequence", "label", "fold_id"]].copy()
    one_m_out.insert(2, "pred", np.nan)
    one_m_out = one_m_out[["mutation_name", "mutated_sequence", "pred", "label", "fold_id"]]

    fitness_df = pd.concat([one_m_out, two_m_predictions], ignore_index=True)
    assay_epi_df = pd.concat(epi_list, ignore_index=True)

    fitness_csv = assay_pred_out / "fitness.csv"
    epi_csv = assay_pred_out / "epistasis.csv"
    fitness_df.to_csv(fitness_csv, index=False)
    assay_epi_df.to_csv(epi_csv, index=False)

    print(f"[assay done] {assay_id} | " f"n_1m={len(one_m_out)} | " f"n_pred_2m={len(two_m_predictions)} | " f"n_epi={len(assay_epi_df)}")
    print(f"[save] {fitness_csv}")
    print(f"[save] {epi_csv}")


def run_train(args) -> None:
    device = torch.device(args.device)
    data_root = Path(args.data_root)

    Path(args.train_log).mkdir(parents=True, exist_ok=True)
    Path(args.pred_output).mkdir(parents=True, exist_ok=True)

    for assay_dir in select_assays(data_root, args.only_assay):
        run_one_assay(args=args, assay_dir=assay_dir, device=device)

    print(f"\n[done] all assay-level predictions saved to: {args.pred_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("5-fold held-out 2M training with fold-specific epistasis prediction")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_root", default=str(proj_root / "data" / "lowN_1M2M_to_heldout2M"), help="Assay root directory.")
    parser.add_argument("--only_assay", default="")
    parser.add_argument("--train_log", default=str(base_dir / "training_log" / "lowN_1M2M_to_heldout2M"))
    parser.add_argument("--pred_output", default=str(base_dir / "output" / "lowN_1M2M_to_heldout2M"))

    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--adj_dim", type=int, default=32)
    parser.add_argument("--rankH", type=int, default=320)
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
