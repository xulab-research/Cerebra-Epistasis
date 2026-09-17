import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from cerebra_epistasis.model import SE3Transformer
from utils.metrics import spearman_corr
from utils.utils_func import (
    read_wt_idx_from_fasta,
    to_gpu,
    max_mut_from_list,
    set_seed_everywhere,
    get_fold_split,
    load_features,
    compute_twobody_epistasis_labels,
)
from utils.calculate_nbodys_mutation_effect import (
    calculate_batch_prediction_mlp,
    EpistasisMLP,
)

base_dir = Path(__file__).resolve().parent
proj_root = base_dir.parent


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_mse = float("inf")
        self.best_epoch = -1
        self.counter = 0

    def step(self, val_mse, epoch):
        if val_mse < self.best_mse - self.min_delta:
            self.best_mse = val_mse
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def select_assays(data_root: Path, prefix: str, only_assay: str):
    assay_dirs = [p for p in sorted(data_root.iterdir()) if p.is_dir() and not p.name.startswith("cdna")]

    if prefix:
        assay_dirs = [p for p in assay_dirs if p.name.startswith(prefix)]
    if only_assay:
        assay_dirs = [p for p in assay_dirs if p.name == only_assay]

    return assay_dirs


def prepare_train_unit(assay_dir: Path, cv_df: pd.DataFrame, data, fold: int, args):
    (
        train_mut_list,
        train_y_z,
        val_mut_list,
        val_y_z,
        test_mut_list,
        _,
        train_mean,
        train_std,
    ) = get_fold_split(cv_df, "fold_id", fold, seed=args.seed)

    # Keep the original float32 conversion because it is part of the numerical path.
    train_y_z = np.asarray(train_y_z, dtype=np.float32)
    val_y_z = np.asarray(val_y_z, dtype=np.float32)
    train_mean = float(train_mean)
    train_std = float(train_std)

    if args.lambda_epi > 0:
        train_df = pd.DataFrame(
            {
                "mutation_name": np.asarray(train_mut_list, dtype=str),
                "label": train_y_z.astype(float) * train_std + train_mean,
            }
        )
        epi_indices, epi_y = compute_twobody_epistasis_labels(
            train_df=train_df,
            train_mean=train_mean,
            train_std=train_std,
        )
    else:
        epi_indices = np.empty(0, dtype=np.int64)
        epi_y = np.empty(0, dtype=np.float32)

    wt_idx = read_wt_idx_from_fasta(assay_dir / "wt.fasta")
    geo_neighbor, epi_neighbor = (1.0 / 3.0, 0.0) if len(wt_idx) > 200 else (0.5, 1.0 / 3.0)

    max_mut = max(
        max_mut_from_list(train_mut_list),
        max_mut_from_list(val_mut_list),
        max_mut_from_list(test_mut_list),
    )

    print(f"[unit] {assay_dir.name} | fold={fold} | " f"train={len(train_mut_list)} | val={len(val_mut_list)} | test={len(test_mut_list)} | " f"n_train_epi={len(epi_indices)} | lambda_epi={args.lambda_epi} | max_mut={max_mut}")

    return {
        "assay_id": assay_dir.name,
        "fold": fold,
        "data": data,
        "geo_neighbor": geo_neighbor,
        "epi_neighbor": epi_neighbor,
        "train_mut_list": train_mut_list,
        "val_mut_list": val_mut_list,
        "test_mut_list": test_mut_list,
        "train_y_z": train_y_z,
        "val_y_z": val_y_z,
        "train_mean": train_mean,
        "train_std": train_std,
        "epi_indices": epi_indices,
        "epi_y": epi_y,
        "n_train_epi": len(epi_indices),
        "max_mut": max_mut,
    }


def predict_mutants(
    model,
    mlp_model,
    unit,
    mut_list,
    device_str,
    return_epi=False,
):
    single_pred, high_delta = model(unit["data"])
    return calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=mut_list,
        U=high_delta,
        mlp_model=mlp_model,
        max_mut=unit["max_mut"],
        device=device_str,
        return_epi=return_epi,
        sqrt_scale=False,
    )


def train_one_epoch(model, mlp_model, optimizer, unit, args, device_str):
    model.train()
    mlp_model.train()
    optimizer.zero_grad(set_to_none=True)

    train_y = torch.as_tensor(
        unit["train_y_z"],
        dtype=torch.float32,
        device=unit["data"]["wt_idx"].device,
    )

    need_epi = args.lambda_epi > 0 and unit["n_train_epi"] > 0
    out = predict_mutants(
        model,
        mlp_model,
        unit,
        unit["train_mut_list"],
        device_str,
        return_epi=need_epi,
    )

    if need_epi:
        preds, pred_epi_all = out
    else:
        preds = out

    fitness_loss = torch.nn.functional.smooth_l1_loss(
        preds.float(),
        train_y.float(),
        beta=args.huber_delta,
    )

    if need_epi:
        epi_indices = torch.as_tensor(
            unit["epi_indices"],
            dtype=torch.long,
            device=train_y.device,
        )
        epi_y = torch.as_tensor(
            unit["epi_y"],
            dtype=torch.float32,
            device=train_y.device,
        )
        epi_loss = torch.nn.functional.smooth_l1_loss(
            pred_epi_all[epi_indices].float(),
            epi_y.float(),
            beta=args.huber_delta,
        )
        loss = fitness_loss + args.lambda_epi * epi_loss
    else:
        epi_loss = fitness_loss.new_zeros(())
        loss = fitness_loss

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(mlp_model.parameters()),
        max_norm=args.clip_grad,
    )
    optimizer.step()

    return {
        "train_total_loss": loss.item(),
        "train_fitness_loss": fitness_loss.item(),
        "train_epi_loss": epi_loss.item(),
        "train_spearman": spearman_corr(preds.detach(), train_y).item(),
        "train_grad_norm": float(grad_norm),
    }


@torch.no_grad()
def eval_model(model, mlp_model, unit, mut_list, labels_z, device_str):
    model.eval()
    mlp_model.eval()

    y = torch.as_tensor(
        labels_z,
        dtype=torch.float32,
        device=unit["data"]["wt_idx"].device,
    )
    preds = predict_mutants(model, mlp_model, unit, mut_list, device_str)

    mse = torch.nn.functional.mse_loss(preds.float(), y.float())
    corr = spearman_corr(preds, y)

    return mse.item(), corr.item()


def train_one_assay(args, unit, assay_out: Path, device, device_str):
    assay_id = unit["assay_id"]
    fold = unit["fold"]
    fold_dir = assay_out / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

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
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=max(1, args.warmup_epochs),
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        threshold=args.min_delta,
        threshold_mode="abs",
        min_lr=args.min_lr,
    )
    early_stop = EarlyStopping(args.stop_patience, args.min_delta)

    print(f"\n[train] assay={assay_id} | fold={fold}/{args.k_folds - 1} | " f"n_train_epi={unit['n_train_epi']} | lambda_epi={args.lambda_epi}")

    best_metric = float("inf") if args.select_best_by == "val_mse" else -float("inf")
    best_epoch = -1
    best_val_mse = np.nan
    best_val_spearman = np.nan
    records = []

    for epoch in range(args.max_epochs):
        train_record = train_one_epoch(model, mlp_model, optimizer, unit, args, device_str)
        val_mse, val_spearman = eval_model(
            model,
            mlp_model,
            unit,
            unit["val_mut_list"],
            unit["val_y_z"],
            device_str,
        )

        current_metric = val_mse if args.select_best_by == "val_mse" else val_spearman
        is_better = current_metric < best_metric if args.select_best_by == "val_mse" else current_metric > best_metric

        if is_better:
            best_metric = current_metric
            best_epoch = epoch
            best_val_mse = val_mse
            best_val_spearman = val_spearman

            torch.save(
                {
                    "model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "mlp": {k: v.cpu() for k, v in mlp_model.state_dict().items()},
                    "best_epoch": best_epoch,
                    "select_best_by": args.select_best_by,
                    "best_ckpt_metric": best_metric,
                    "best_val_mse": best_val_mse,
                    "best_val_spearman": best_val_spearman,
                },
                fold_dir / "best_val_model.pt",
            )

        records.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                **train_record,
                "val_mse": val_mse,
                "val_spearman": val_spearman,
            }
        )

        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_mse)

        if early_stop.step(val_mse, epoch):
            print(f"[early-stop] epoch={epoch} | best_epoch={early_stop.best_epoch} | " f"best_val_mse={early_stop.best_mse:.6f}")
            break

    pd.DataFrame(records).to_csv(fold_dir / "loss.csv", index=False)

    best_ckpt = torch.load(fold_dir / "best_val_model.pt", map_location="cpu")
    model.load_state_dict(best_ckpt["model"])
    mlp_model.load_state_dict(best_ckpt["mlp"])

    print(f"[done] Best checkpoint by {best_ckpt['select_best_by']} | " f"epoch={best_ckpt['best_epoch']} | metric={best_ckpt['best_ckpt_metric']:.6f} | " f"val_mse={best_ckpt['best_val_mse']:.6f} | " f"val_spearman={best_ckpt['best_val_spearman']:.6f}")

    return model, mlp_model


@torch.no_grad()
def predict_and_save(unit, pred_df, pred_csv_path: Path, model, mlp_model, device_str):
    fold = unit["fold"]

    model.eval()
    mlp_model.eval()

    pred_z = predict_mutants(
        model,
        mlp_model,
        unit,
        unit["test_mut_list"],
        device_str,
    )
    pred_z = pred_z.cpu().numpy().astype(float).reshape(-1)
    pred_raw = pred_z * unit["train_std"] + unit["train_mean"]

    pred_map = dict(zip(map(str, unit["test_mut_list"]), pred_raw))
    fold_mask = pred_df["fold_id"] == fold
    target_muts = pred_df.loc[fold_mask, "mutation_name"].astype(str)
    pred_df.loc[fold_mask, "pred"] = target_muts.map(pred_map).to_numpy()
    pred_df.to_csv(pred_csv_path, index=False)

    print(f"[save] assay={unit['assay_id']} fold={fold} -> {pred_csv_path}")


def run_train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device_str = str(device)

    pred_output = Path(args.pred_output_root)
    train_log = Path(args.train_log_root)
    pred_output.mkdir(parents=True, exist_ok=True)
    train_log.mkdir(parents=True, exist_ok=True)

    set_seed_everywhere(args.seed)

    for assay_dir in select_assays(Path(args.data_root), args.prefix, args.only_assay):
        print(f"\nASSAY: {assay_dir.name}")

        cv_df = pd.read_csv(assay_dir / "data.csv")
        data = to_gpu(load_features(assay_dir), device)
        pred_df = pd.DataFrame(
            {
                "mutation_name": cv_df["mutation_name"].astype(str),
                "label": cv_df["label"].astype(float),
                "fold_id": cv_df["fold_id"].astype(int),
                "pred": np.nan,
            }
        )

        assay_out = train_log / assay_dir.name
        pred_csv_path = pred_output / f"{assay_dir.name}.csv"

        for fold in range(args.k_folds):
            unit = prepare_train_unit(
                assay_dir=assay_dir,
                cv_df=cv_df,
                data=data,
                fold=fold,
                args=args,
            )
            model, mlp_model = train_one_assay(
                args=args,
                unit=unit,
                assay_out=assay_out,
                device=device,
                device_str=device_str,
            )
            predict_and_save(
                unit=unit,
                pred_df=pred_df,
                pred_csv_path=pred_csv_path,
                model=model,
                mlp_model=mlp_model,
                device_str=device_str,
            )

            # Release the previous fold before constructing the next GPU model.
            del model, mlp_model, unit

        # Release assay-level data before loading the next assay.
        del data, cv_df, pred_df

        print(f"[done] assay {assay_dir.name} finished.")

    print("\nAll assays finished.")


def build_parser():
    parser = argparse.ArgumentParser("Train one SE3+MLP model per assay fold")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_root", default=str(proj_root / "data" / "1M+_to_1M+"))
    parser.add_argument("--pred_output_root", default=str(base_dir / "output" / "1Mplus_to_1Mplus"))
    parser.add_argument("--train_log_root", default=str(base_dir / "training_log" / "1Mplus_to_1Mplus"))

    parser.add_argument("--prefix", default="")
    parser.add_argument("--only_assay", default="")
    parser.add_argument("--k_folds", type=int, default=5)

    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_patience", type=int, default=20)
    parser.add_argument("--stop_patience", type=int, default=80)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    parser.add_argument("--lambda_epi", type=float, default=0.0)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--clip_grad", type=float, default=2.0)

    parser.add_argument("--adj_dim", type=int, default=32)
    parser.add_argument("--rankH", type=int, default=320)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)

    parser.add_argument(
        "--select_best_by",
        default="val_mse",
        choices=["val_mse", "val_spearman"],
    )

    return parser


def main():
    run_train(build_parser().parse_args())


if __name__ == "__main__":
    main()
