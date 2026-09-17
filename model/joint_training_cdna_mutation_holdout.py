import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cerebra_epistasis.model import SE3Transformer
from utils.metrics import spearman_corr
from utils.utils_func import read_wt_idx_from_fasta, to_gpu, max_mut_from_list, set_seed_everywhere, load_features
from utils.calculate_nbodys_mutation_effect import calculate_batch_prediction_mlp, EpistasisMLP

base_dir = Path(__file__).resolve().parent
proj_root = base_dir.parent


def build_model(args, geo_neighbor, epi_neighbor):
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
            module.geo_neighbor = geo_neighbor
        if hasattr(module, "epi_neighbor"):
            module.epi_neighbor = epi_neighbor


def make_split(cv_df, fold):
    train_df = cv_df.loc[cv_df["fold_id"] != fold, ["mutation_name", "label"]].reset_index(drop=True)
    test_df = cv_df.loc[cv_df["fold_id"] == fold, ["mutation_name", "label"]].reset_index(drop=True)

    train_y = train_df["label"].to_numpy(dtype=float)
    test_y = test_df["label"].to_numpy(dtype=float)
    train_mean = train_y.mean()
    train_std = train_y.std(ddof=0)

    train_mut_list = train_df["mutation_name"].tolist()
    test_mut_list = test_df["mutation_name"].tolist()

    return {
        "train_mut_list": train_mut_list,
        "test_mut_list": test_mut_list,
        "train_y_z": ((train_y - train_mean) / train_std).astype(np.float32),
        "test_y_raw": test_y,
        "train_mean": train_mean,
        "train_std": train_std,
        "max_mut": max_mut_from_list(train_mut_list + test_mut_list),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


def train_one_protein(model, mlp, optimizer, data, split, args, device):
    model.train()
    mlp.train()
    optimizer.zero_grad(set_to_none=True)

    train_y = torch.as_tensor(split["train_y_z"], device=device)
    single_pred, high_delta = model(data)

    preds = calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=split["train_mut_list"],
        U=high_delta,
        mlp_model=mlp,
        max_mut=split["max_mut"],
        device=str(device),
        return_epi=False,
        sqrt_scale=False,
    )

    loss = torch.nn.functional.smooth_l1_loss(preds, train_y, beta=args.huber_delta)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(mlp.parameters()), max_norm=2.0)
    optimizer.step()

    return loss.item(), spearman_corr(preds.detach(), train_y).item()


@torch.no_grad()
def predict_one_protein(model, mlp, data, split, device):
    model.eval()
    mlp.eval()

    single_pred, high_delta = model(data)
    pred_z = calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=split["test_mut_list"],
        U=high_delta,
        mlp_model=mlp,
        max_mut=split["max_mut"],
        device=str(device),
        return_epi=False,
        sqrt_scale=False,
    )

    pred_z = pred_z.cpu().numpy().reshape(-1)
    return pred_z * split["train_std"] + split["train_mean"]


def load_assays(args):
    assays = []
    pred_dfs = {}

    for assay_dir in sorted(p for p in Path(args.data_root).iterdir() if p.is_dir() and p.name.startswith("cdna")):
        assay_id = assay_dir.name
        cv_df = pd.read_csv(assay_dir / "data.csv")
        length = len(read_wt_idx_from_fasta(assay_dir / "wt.fasta"))
        auto_geo, auto_epi = (1.0 / 3.0, 0.0) if length > 200 else (0.5, 1.0 / 3.0)

        assays.append(
            {
                "assay_id": assay_id,
                "assay_dir": assay_dir,
                "cv_df": cv_df,
                "geo_neighbor": auto_geo if args.geo_neighbor < 0 else args.geo_neighbor,
                "epi_neighbor": auto_epi if args.epi_neighbor < 0 else args.epi_neighbor,
            }
        )

        pred_dfs[assay_id] = pd.DataFrame(
            {
                "mutation_name": cv_df["mutation_name"],
                "label": cv_df["label"],
                "fold_id": cv_df["fold_id"],
                "pred": np.nan,
            }
        )

    print(f"[data] n_cdna_assays={len(assays)}")
    return assays, pred_dfs


def train_one_fold(fold, split_items, args, device, train_log):
    print(f"\n========== fold {fold}/{args.k_folds - 1} ==========")

    fold_dir = train_log / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fold] n_proteins={len(split_items)} | " f"train={sum(x['split']['n_train'] for x in split_items)} | " f"test={sum(x['split']['n_test'] for x in split_items)}")

    model, mlp = build_model(args, split_items[0]["geo_neighbor"], split_items[0]["epi_neighbor"])
    model, mlp = model.to(device), mlp.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(mlp.parameters()),
        lr=args.lr,
        eps=1e-6,
    )
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
                T_max=args.max_epochs - args.warmup_epochs,
                eta_min=args.min_lr,
            ),
        ],
        milestones=[args.warmup_epochs],
    )

    records = []

    for epoch in range(args.max_epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        random.shuffle(split_items)

        losses = []
        spearmans = []

        for item in split_items:
            set_model_neighbors(model, item["geo_neighbor"], item["epi_neighbor"])
            data = to_gpu(load_features(item["assay_dir"]), device)
            loss, spearman = train_one_protein(model, mlp, optimizer, data, item["split"], args, device)
            losses.append(loss)
            spearmans.append(spearman)
            del data

        record = {
            "epoch": epoch,
            "lr": lr_now,
            "mean_train_loss": np.mean(losses),
            "mean_train_spearman": np.mean(spearmans),
        }
        records.append(record)

        print(f"[epoch] fold={fold} | {epoch + 1}/{args.max_epochs} | " f"lr={lr_now:.3e} | loss={record['mean_train_loss']:.6f} | " f"spearman={record['mean_train_spearman']:.6f}")
        scheduler.step()

    pd.DataFrame(records).to_csv(fold_dir / "loss.csv", index=False)

    torch.save(
        {
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "mlp": {k: v.cpu() for k, v in mlp.state_dict().items()},
            "epoch": args.max_epochs - 1,
            "fold": fold,
            "args": vars(args),
        },
        fold_dir / "final_model.pt",
    )

    return model, mlp


def test_one_fold(fold, model, mlp, split_items, pred_dfs, device, train_log):
    fold_dir = train_log / f"fold_{fold}"

    test_spearmans = []

    for item in split_items:
        split = item["split"]
        assay_id = item["assay_id"]

        set_model_neighbors(model, item["geo_neighbor"], item["epi_neighbor"])
        data = to_gpu(load_features(item["assay_dir"]), device)
        pred_raw = predict_one_protein(model, mlp, data, split, device)
        del data

        test_spearmans.append(
            spearman_corr(
                torch.as_tensor(pred_raw, dtype=torch.float32),
                torch.as_tensor(split["test_y_raw"], dtype=torch.float32),
            ).item()
        )
        pred_dfs[assay_id].loc[pred_dfs[assay_id]["fold_id"] == fold, "pred"] = pred_raw

    final_record = {
        "fold": fold,
        "n_proteins": len(split_items),
        "n_train": sum(x["split"]["n_train"] for x in split_items),
        "n_test": sum(x["split"]["n_test"] for x in split_items),
        "test_spearman": np.mean(test_spearmans),
    }

    pd.DataFrame([final_record]).to_csv(fold_dir / "final_test_metrics.csv", index=False)
    print(f"[final-test] fold={fold} | PROTEIN_MEAN | " f"n_proteins={len(split_items)} | spearman={final_record['test_spearman']:.6f}")


def run_train(args):
    set_seed_everywhere(args.seed)
    device = torch.device(args.device)

    pred_output = Path(args.pred_output_root)
    train_log = Path(args.train_log_root)
    pred_output.mkdir(parents=True, exist_ok=True)
    train_log.mkdir(parents=True, exist_ok=True)

    assays, pred_dfs = load_assays(args)

    for fold in range(args.k_folds):
        split_items = [{**assay, "split": make_split(assay["cv_df"], fold)} for assay in assays]

        model, mlp = train_one_fold(fold, split_items, args, device, train_log)
        test_one_fold(fold, model, mlp, split_items, pred_dfs, device, train_log)
        del model, mlp

    for assay_id, df in pred_dfs.items():
        df[["mutation_name", "label", "fold_id", "pred"]].to_csv(
            pred_output / f"{assay_id}.csv",
            index=False,
        )

    print(f"\n[Done] prediction csvs -> {pred_output}")
    print(f"[Done] training logs -> {train_log}")


def build_parser():
    parser = argparse.ArgumentParser("Joint cDNA mutation-holdout training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_root", default=str(proj_root / "data" / "1M+_to_1M+"))
    parser.add_argument("--pred_output_root", default=str(base_dir / "output" / "joint_cdna_mutation_holdout"))
    parser.add_argument("--train_log_root", default=str(base_dir / "training_log" / "joint_cdna_mutation_holdout"))
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--adj_dim", type=int, default=32)
    parser.add_argument("--rankH", type=int, default=320)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--geo_neighbor", type=float, default=-1.0)
    parser.add_argument("--epi_neighbor", type=float, default=-1.0)
    return parser


def main():
    run_train(build_parser().parse_args())


if __name__ == "__main__":
    main()
