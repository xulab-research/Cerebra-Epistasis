import torch
import numpy as np
import pandas as pd

import argparse
import itertools
from pathlib import Path


from cerebra_epistasis.model import SE3Transformer
from utils.metrics import spearman_corr
from utils.utils_func import (
    read_wt_idx_from_fasta,
    to_gpu,
    max_mut_from_list,
    set_seed_everywhere,
    compute_twobody_epistasis_labels,
    load_features,
)
from utils.calculate_nbodys_mutation_effect import (
    calculate_batch_prediction_mlp,
    EpistasisMLP,
)

base_dir = Path(__file__).resolve().parent
proj_root = base_dir.parent


def load_assay(assay_dir):
    df = pd.read_csv(assay_dir / "data.csv").dropna(subset=["mutation_name", "label", "fold_id"]).copy()
    df["mutation_name"] = df["mutation_name"].astype(str).str.strip()
    df["label"] = df["label"].astype(float)
    df["fold_id"] = df["fold_id"].astype(int)

    train_df = df[df["fold_id"] == 0].reset_index(drop=True)
    test_df = df[df["fold_id"] == 1].reset_index(drop=True)

    protein_length = len(read_wt_idx_from_fasta(assay_dir / "wt.fasta"))
    geo_neighbor, epi_neighbor = (1.0 / 3.0, 0.0) if protein_length > 200 else (0.5, 1.0 / 3.0)

    print(f"[assay] {assay_dir.name} | train={len(train_df)} | test={len(test_df)}")

    return {
        "assay_id": assay_dir.name,
        "data_cpu": load_features(assay_dir),
        "train_df": train_df,
        "test_df": test_df,
        "max_mut": int(max_mut_from_list(df["mutation_name"].tolist())),
        "geo_neighbor": geo_neighbor,
        "epi_neighbor": epi_neighbor,
    }


def random_train_val_split(train_df, val_fraction, seed):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(train_df))
    n_val = int(round(len(train_df) * val_fraction))
    val_idx = indices[:n_val]
    subtrain_idx = indices[n_val:]
    return train_df.iloc[subtrain_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True), val_idx


def make_training_split(train_df, eval_df):
    train_mean = float(train_df["label"].mean())
    train_denom = float(train_df["label"].std(ddof=0)) + 1e-8

    train_y = ((train_df["label"].to_numpy(dtype=float) - train_mean) / train_denom).astype(np.float32)
    eval_y = ((eval_df["label"].to_numpy(dtype=float) - train_mean) / train_denom).astype(np.float32)

    epi_indices, epi_labels = compute_twobody_epistasis_labels(
        train_df=train_df,
        train_mean=train_mean,
        train_std=train_denom,
    )

    return {
        "train_mut_list": train_df["mutation_name"].tolist(),
        "eval_mut_list": eval_df["mutation_name"].tolist(),
        "train_y": train_y,
        "eval_y": eval_y,
        "train_mean": train_mean,
        "train_denom": train_denom,
        "epi_indices": epi_indices,
        "epi_labels": epi_labels,
        "n_train_epi": len(epi_indices),
        "n_train": len(train_df),
        "n_eval": len(eval_df),
    }


def build_model(rankH, args, unit, device):
    model = SE3Transformer(
        depth=1,
        hidden_fiber_dict={0: 320, 1: 32},
        out_fiber_dict={0: 128, 1: 32},
        adj_dim=args.adj_dim,
        rankH=rankH,
        geo_neighbor=unit["geo_neighbor"],
        epi_neighbor=unit["epi_neighbor"],
    ).to(device)

    mlp_model = EpistasisMLP(
        input_dim=rankH,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.mlp_dropout,
    ).to(device)

    return model, mlp_model


def train_one_epoch(model, mlp_model, optimizer, data, train_y, split, max_mut, args, device):
    model.train()
    mlp_model.train()
    optimizer.zero_grad(set_to_none=True)

    single_pred, high_delta = model(data)

    preds, pred_epi_all = calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=split["train_mut_list"],
        U=high_delta,
        mlp_model=mlp_model,
        max_mut=max_mut,
        device=str(device),
        return_epi=True,
        sqrt_scale=False,
    )

    fitness_loss = torch.nn.functional.smooth_l1_loss(preds.float(), train_y.float(), beta=args.huber_delta)

    if args.lambda_epi > 0 and split["epi_indices"].numel() > 0:
        epi_loss = torch.nn.functional.smooth_l1_loss(
            pred_epi_all[split["epi_indices"]].float(),
            split["epi_y"].float(),
            beta=args.huber_delta,
        )
        loss = fitness_loss + args.lambda_epi * epi_loss
    else:
        epi_loss = fitness_loss.new_zeros(())
        loss = fitness_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(mlp_model.parameters()), args.clip_grad)
    optimizer.step()

    return float(loss.item()), float(fitness_loss.item()), float(epi_loss.item()), float(spearman_corr(preds.detach(), train_y.detach()).item())


@torch.no_grad()
def evaluate(model, mlp_model, data, mut_list, y, max_mut, device):
    model.eval()
    mlp_model.eval()

    single_pred, high_delta = model(data)
    pred = calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=mut_list,
        U=high_delta,
        mlp_model=mlp_model,
        max_mut=max_mut,
        device=str(device),
        return_epi=False,
        sqrt_scale=False,
    )

    return float(spearman_corr(pred.detach(), y.detach()).item()), pred


def train_model(args, unit, split, epochs, lr, rankH, device, log_path=None):
    set_seed_everywhere(args.seed)
    model, mlp_model = build_model(rankH, args, unit, device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(mlp_model.parameters()),
        lr=lr,
        eps=1e-6,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.min_lr)

    data = to_gpu(unit["data_cpu"], device)
    train_y = torch.as_tensor(split["train_y"], dtype=torch.float32, device=device)
    eval_y = torch.as_tensor(split["eval_y"], dtype=torch.float32, device=device)

    split["epi_indices"] = torch.as_tensor(split["epi_indices"], dtype=torch.long, device=device)
    split["epi_y"] = torch.as_tensor(split["epi_labels"], dtype=torch.float32, device=device)

    records = []

    for epoch in range(epochs):
        total_loss, fitness_loss, epi_loss, train_spearman = train_one_epoch(model, mlp_model, optimizer, data, train_y, split, unit["max_mut"], args, device)
        scheduler.step()

        if log_path is not None:
            records.append(
                {
                    "epoch": epoch,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "train_total_loss": total_loss,
                    "train_fitness_loss": fitness_loss,
                    "train_epi_loss": epi_loss,
                    "train_spearman": train_spearman,
                }
            )

    if log_path is not None:
        pd.DataFrame(records).to_csv(log_path, index=False)

    return model, mlp_model, data, eval_y


def run_validation_sweep(args, unit, assay_out, device):
    subtrain_df, val_df, val_idx = random_train_val_split(unit["train_df"], args.val_fraction, args.seed)

    split_record = unit["train_df"].copy()
    split_record["validation_split"] = "subtrain"
    split_record.loc[val_idx, "validation_split"] = "validation"
    split_record.to_csv(assay_out / "validation_split.csv", index=False)

    split = make_training_split(subtrain_df, val_df)
    combinations = list(itertools.product(args.epochs, args.lr, args.rankH))

    print(f"[validation split] subtrain={len(subtrain_df)} | validation={len(val_df)} | combinations={len(combinations)}")

    results = []

    for combo_id, (epochs, lr, rankH) in enumerate(combinations, 1):
        model, mlp_model, data, val_y = train_model(args, unit, split, epochs, lr, rankH, device)

        val_spearman, _ = evaluate(
            model,
            mlp_model,
            data,
            split["eval_mut_list"],
            val_y,
            unit["max_mut"],
            device,
        )

        results.append(
            {
                "combo_id": combo_id,
                "epochs": epochs,
                "lr": lr,
                "min_lr": args.min_lr,
                "adj_dim": args.adj_dim,
                "rankH": rankH,
                "mlp_hidden_dim": args.mlp_hidden_dim,
                "mlp_dropout": args.mlp_dropout,
                "huber_delta": args.huber_delta,
                "clip_grad": args.clip_grad,
                "lambda_epi": args.lambda_epi,
                "n_subtrain": split["n_train"],
                "n_validation": split["n_eval"],
                "n_train_epi": split["n_train_epi"],
                "val_spearman": val_spearman,
            }
        )

        print(f"[combo {combo_id:02d}/{len(combinations)}] epochs={epochs} | lr={lr:g} | rankH={rankH} | val_spearman={val_spearman:.6f}")

        del model, mlp_model, data, val_y
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    results_df.to_csv(assay_out / "sweep_results.csv", index=False)

    best = results_df.loc[results_df["val_spearman"].idxmax()].to_dict()
    pd.DataFrame([best]).to_csv(assay_out / "best_hyperparameters.csv", index=False)

    print(f"[best] combo={int(best['combo_id'])} | epochs={int(best['epochs'])} | lr={best['lr']:g} | rankH={int(best['rankH'])} | val_spearman={best['val_spearman']:.6f}")

    return best


def final_retrain_and_test(args, unit, assay_out, pred_csv_path, best, device):
    split = make_training_split(unit["train_df"], unit["test_df"])

    epochs = int(best["epochs"])
    lr = float(best["lr"])
    rankH = int(best["rankH"])

    final_out = assay_out / "final_retrain"
    final_out.mkdir(parents=True, exist_ok=True)

    model, mlp_model, data, test_y = train_model(
        args,
        unit,
        split,
        epochs,
        lr,
        rankH,
        device,
        log_path=final_out / "train_log.csv",
    )

    test_spearman, test_pred_z = evaluate(
        model,
        mlp_model,
        data,
        split["eval_mut_list"],
        test_y,
        unit["max_mut"],
        device,
    )

    test_mse = float(torch.nn.functional.mse_loss(test_pred_z.float(), test_y.float()).item())

    pd.DataFrame(
        [
            {
                "assay": unit["assay_id"],
                "epochs": epochs,
                "lr": lr,
                "rankH": rankH,
                "test_mse": test_mse,
                "test_spearman": test_spearman,
            }
        ]
    ).to_csv(final_out / "final_test_metrics.csv", index=False)

    torch.save(
        {
            "epoch": epochs - 1,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "mlp_state": {k: v.detach().cpu() for k, v in mlp_model.state_dict().items()},
            "best_epochs": epochs,
            "best_lr": lr,
            "best_rankH": rankH,
            "args": vars(args),
            "train_mean": split["train_mean"],
            "train_denom": split["train_denom"],
            "n_train_epi": split["n_train_epi"],
        },
        final_out / "last.pt",
    )

    test_pred_raw = test_pred_z.detach().cpu().numpy().reshape(-1) * split["train_denom"] + split["train_mean"]

    pred_df = unit["test_df"][["mutation_name", "label", "fold_id"]].copy()
    pred_df["pred"] = test_pred_raw
    pred_df.to_csv(pred_csv_path, index=False)

    print(f"[final-test] assay={unit['assay_id']} | mse={test_mse:.6f} | spearman={test_spearman:.6f}")
    print(f"[save] {pred_csv_path}")


def run(args):
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    log_root = Path(args.train_log)
    pred_root = Path(args.pred_output)

    log_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    assay_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if args.only_assay:
        assay_dirs = [p for p in assay_dirs if p.name == args.only_assay]

    for assay_dir in assay_dirs:
        unit = load_assay(assay_dir)
        assay_out = log_root / unit["assay_id"]
        assay_out.mkdir(parents=True, exist_ok=True)

        best = run_validation_sweep(args, unit, assay_out, device)
        final_retrain_and_test(
            args,
            unit,
            assay_out,
            pred_root / f"{unit['assay_id']}.csv",
            best,
            device,
        )

        print(f"[done] assay={unit['assay_id']}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_root", default=str(proj_root / "data" / "wet_benchmark"))
    parser.add_argument("--only_assay", default="")
    parser.add_argument("--train_log", default=str(base_dir / "training_log" / "wet_benchmark"))
    parser.add_argument("--pred_output", default=str(base_dir / "output" / "wet_benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, nargs="+", default=[150, 200, 250])
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-4, 5e-4])
    parser.add_argument("--rankH", type=int, nargs="+", default=[64, 128, 320])
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--adj_dim", type=int, default=32)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--clip_grad", type=float, default=2.0)
    parser.add_argument("--lambda_epi", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
