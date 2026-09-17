import sys
import re
import gc
import random
import argparse
import itertools
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

base_dir = Path(__file__).resolve().parent
proj_root = base_dir.parent
sys.path[:0] = [str(base_dir), str(proj_root), str(base_dir / "utils")]

from model.cerebra_epistasis.model import SE3Transformer
from metrics import spearman_corr
from utils_func import read_wt_idx_from_fasta, to_gpu, max_mut_from_list, set_seed_everywhere, load_features, compute_twobody_epistasis_labels
from calculate_nbodys_mutation_effect import calculate_batch_prediction_mlp, EpistasisMLP


def set_model_neighbors(model, geo_neighbor, epi_neighbor):
    for module in model.modules():
        if hasattr(module, "geo_neighbor"):
            module.geo_neighbor = float(geo_neighbor)
        if hasattr(module, "epi_neighbor"):
            module.epi_neighbor = float(epi_neighbor)


def auto_neighbors(protein_dir, args):
    length = len(read_wt_idx_from_fasta(protein_dir / "wt.fasta"))
    auto_geo, auto_epi = (1.0 / 3.0, 0.0) if length > 200 else (0.5, 1.0 / 3.0)
    return auto_geo if args.geo_neighbor < 0 else args.geo_neighbor, auto_epi if args.epi_neighbor < 0 else args.epi_neighbor


def pairwise_identity(nwalign_bin, fasta1, fasta2):
    def run(a, b):
        output = subprocess.check_output([str(nwalign_bin), str(a), str(b)], text=True, stderr=subprocess.STDOUT)
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

    names = [item["assay_id"] for item in assays]
    fasta_paths = [item["assay_dir"] / "wt.fasta" for item in assays]
    identity = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    pairs = list(itertools.combinations(range(len(names)), 2))
    print(f"[cluster] build nwalign identity matrix | n={len(names)} | pairs={len(pairs)}")

    for step, (i, j) in enumerate(pairs, 1):
        identity.iat[i, j] = identity.iat[j, i] = pairwise_identity(args.nwalign_bin, fasta_paths[i], fasta_paths[j])
        if step == 1 or step % args.cluster_print_every == 0 or step == len(pairs):
            print(f"[cluster] pairs {step}/{len(pairs)}")

    identity.to_csv(identity_csv)
    distance_array = (1.0 - identity).to_numpy(copy=True)
    np.fill_diagonal(distance_array, 0.0)
    distance = pd.DataFrame(distance_array, index=identity.index, columns=identity.columns)
    distance.to_csv(distance_csv)
    linkage = hierarchy.linkage(squareform(distance.values, checks=False), method=args.linkage_method)
    cluster_index = hierarchy.fcluster(linkage, args.k_folds, criterion="maxclust")
    if len(np.unique(cluster_index)) != args.k_folds:
        cluster_index = hierarchy.cut_tree(linkage, n_clusters=args.k_folds).reshape(-1) + 1

    fold_map = {cluster: fold for fold, cluster in enumerate(sorted(np.unique(cluster_index)))}
    fold_df = pd.DataFrame({"assay_id": names, "cluster_index": cluster_index.astype(int), "protein_fold": [fold_map[x] for x in cluster_index]})
    fold_df.to_csv(cluster_csv, index=False)
    print(f"[cluster] saved identity -> {identity_csv}")
    print(f"[cluster] saved distance -> {distance_csv}")
    print(f"[cluster] saved folds -> {cluster_csv}")
    return fold_df[["assay_id", "protein_fold"]]


def make_split(df, compute_epi=False):
    df = df[["mutation_name", "label"]].copy().reset_index(drop=True)
    df["mutation_name"] = df["mutation_name"].astype(str).str.strip()
    df["label"] = df["label"].astype(float)
    mut_list = df["mutation_name"].tolist()
    labels = df["label"].to_numpy(dtype=np.float32)
    epi_indices, epi_y = compute_twobody_epistasis_labels(df, 0.0, 1.0) if compute_epi else ([], [])
    return {"mut_list": mut_list, "labels": labels, "epi_indices": epi_indices, "epi_y": epi_y, "max_mut": int(max_mut_from_list(mut_list)), "n": len(df), "n_epi": len(epi_indices)}


def make_prediction_split(df):
    mut_list = df["mutation_name"].astype(str).str.strip().tolist()
    return {"mut_list": mut_list, "max_mut": int(max_mut_from_list(mut_list)), "n": len(df)}


def load_assays(args):
    assays = []
    assay_dirs = sorted(path for path in Path(args.data_root).iterdir() if path.is_dir() and path.name.startswith(args.assay_prefix))
    for assay_dir in assay_dirs:
        cv_df = pd.read_csv(assay_dir / "cv_data.csv")
        geo_neighbor, epi_neighbor = auto_neighbors(assay_dir, args)
        assays.append({"assay_id": assay_dir.name, "assay_dir": assay_dir, "cv_df": cv_df, "geo_neighbor": geo_neighbor, "epi_neighbor": epi_neighbor})
    print(f"[cDNA] n_assays={len(assays)}")
    return assays


def find_S461_csv(protein_dir):
    for name in ("data.csv", "cv_data.csv"):
        path = protein_dir / name
        if path.exists():
            return path
    csv_files = sorted(protein_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    raise FileNotFoundError(f"Cannot uniquely determine the mutation CSV in {protein_dir}")


def load_S461(args):
    items = []
    for protein_dir in sorted(path for path in Path(args.S461_data_root).iterdir() if path.is_dir()):
        csv_path = find_S461_csv(protein_dir)
        mutation_df = pd.read_csv(csv_path)
        if len(mutation_df) < args.S461_min_mutations:
            print(f"[S461-skip] {protein_dir.name} | n={len(mutation_df)}")
            continue
        geo_neighbor, epi_neighbor = auto_neighbors(protein_dir, args)
        items.append({"protein_id": protein_dir.name, "protein_dir": protein_dir, "mutation_df": mutation_df, "split": make_prediction_split(mutation_df), "geo_neighbor": geo_neighbor, "epi_neighbor": epi_neighbor})
    print(f"[S461] retained_proteins={len(items)} | min_mutations={args.S461_min_mutations}")
    return items


def build_models(args, geo_neighbor, epi_neighbor, device):
    model = SE3Transformer(depth=1, hidden_fiber_dict={0: 320, 1: 32}, out_fiber_dict={0: 128, 1: 32}, adj_dim=args.adj_dim, rankH=args.rankH, geo_neighbor=geo_neighbor, epi_neighbor=epi_neighbor).to(device)
    mlp_model = EpistasisMLP(input_dim=args.rankH, hidden_dim=args.mlp_hidden_dim, dropout=args.mlp_dropout).to(device)
    return model, mlp_model


def train_one_protein(model, mlp_model, optimizer, data, split, args, device):
    model.train()
    mlp_model.train()
    optimizer.zero_grad(set_to_none=True)

    target = torch.as_tensor(split["labels"], dtype=torch.float32, device=device)
    single_pred, high_delta = model(data)
    use_epi = args.lambda_epi > 0 and split["n_epi"] > 0
    output = calculate_batch_prediction_mlp(single_mut_matrix=single_pred, mut_name_list=split["mut_list"], U=high_delta, mlp_model=mlp_model, max_mut=split["max_mut"], device=str(device), return_epi=use_epi)
    preds, pred_epi_all = output if use_epi else (output, None)
    fitness_loss = F.smooth_l1_loss(preds.float(), target, beta=args.huber_delta)

    if use_epi:
        epi_indices = torch.as_tensor(split["epi_indices"], dtype=torch.long, device=device)
        epi_y = torch.as_tensor(split["epi_y"], dtype=torch.float32, device=device)
        epi_loss = F.smooth_l1_loss(pred_epi_all[epi_indices].float(), epi_y, beta=args.huber_delta)
    else:
        epi_loss = fitness_loss.new_tensor(0.0)

    loss = fitness_loss + args.lambda_epi * epi_loss
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(mlp_model.parameters()), args.clip_grad)
    optimizer.step()

    return {"n_train": split["n"], "n_epi": split["n_epi"], "train_total_loss": float(loss.detach()), "train_fitness_loss": float(fitness_loss.detach()), "train_epi_loss": float(epi_loss.detach()), "train_spearman": float(spearman_corr(preds.detach(), target.detach()).detach()), "train_grad_norm": float(grad_norm)}


@torch.no_grad()
def predict_one_protein(model, mlp_model, data, split, device):
    model.eval()
    mlp_model.eval()
    single_pred, high_delta = model(data)
    pred = calculate_batch_prediction_mlp(single_mut_matrix=single_pred, mut_name_list=split["mut_list"], U=high_delta, mlp_model=mlp_model, max_mut=split["max_mut"], device=str(device), return_epi=False)
    return pred.detach().cpu().numpy().reshape(-1)


@torch.no_grad()
def validate(model, mlp_model, val_items, args, device):
    records = []
    for item in val_items:
        set_model_neighbors(model, item["geo_neighbor"], item["epi_neighbor"])
        data = to_gpu(load_features(item["assay_dir"]), device)
        pred = predict_one_protein(model, mlp_model, data, item["split"], device)
        split = item["split"]
        val_loss = F.smooth_l1_loss(torch.as_tensor(pred, dtype=torch.float32), torch.as_tensor(split["labels"], dtype=torch.float32), beta=args.huber_delta).item()
        records.append({"assay_id": item["assay_id"], "n_val": split["n"], "val_loss": val_loss, "val_spearman": float(spearmanr(split["labels"], pred).correlation)})
        del data

    val_df = pd.DataFrame(records)
    return float(val_df["val_loss"].mean()), float(val_df["val_spearman"].mean()), val_df


def save_model(path, model, mlp_model, epoch, fold, val_spearman, args):
    torch.save({"model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "mlp": {key: value.detach().cpu().clone() for key, value in mlp_model.state_dict().items()}, "epoch": epoch, "fold": fold, "val_spearman": val_spearman, "args": vars(args)}, path)


def train_one_fold(fold, train_items, val_items, test_items, args, device, train_log):
    print(f"\n========== fold {fold}/{args.k_folds - 1} ==========")
    fold_dir = train_log / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fold] train_proteins={len(train_items)} | val_proteins={len(val_items)} | test_proteins={len(test_items)} | train_variants={sum(x['split']['n'] for x in train_items)} | val_variants={sum(x['split']['n'] for x in val_items)} | test_variants={sum(x['split']['n'] for x in test_items)}")

    model, mlp_model = build_models(args, train_items[0]["geo_neighbor"], train_items[0]["epi_neighbor"], device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(mlp_model.parameters()), lr=args.lr, eps=1e-6)
    best_path = fold_dir / "final_model.pt"
    best_epoch, best_val_spearman, bad_epochs = -1, -np.inf, 0
    loss_records = []

    for epoch in range(args.max_epochs):
        random.shuffle(train_items)
        train_records = []

        for item in train_items:
            set_model_neighbors(model, item["geo_neighbor"], item["epi_neighbor"])
            data = to_gpu(load_features(item["assay_dir"]), device)
            train_records.append(train_one_protein(model, mlp_model, optimizer, data, item["split"], args, device))
            del data

        train_df = pd.DataFrame(train_records)
        val_loss, val_spearman, val_df = validate(model, mlp_model, val_items, args, device)

        if best_epoch == -1 or val_spearman > best_val_spearman + args.min_delta:
            best_epoch, best_val_spearman, bad_epochs = epoch, val_spearman, 0
            save_model(best_path, model, mlp_model, epoch, fold, val_spearman, args)
            val_df.to_csv(fold_dir / "best_validation_metrics.csv", index=False)
        else:
            bad_epochs += 1

        lr_reduced = False
        if bad_epochs > 0 and bad_epochs < args.early_stop_patience and bad_epochs % args.lr_patience == 0:
            old_lr = float(optimizer.param_groups[0]["lr"])
            new_lr = max(old_lr * args.lr_factor, args.min_lr)
            if new_lr < old_lr:
                for group in optimizer.param_groups:
                    group["lr"] = new_lr
                lr_reduced = True
                print(f"[lr-reduce] fold={fold} | epoch={epoch + 1} | {old_lr:.3e} -> {new_lr:.3e}")

        lr_now = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch + 1,
            "lr": lr_now,
            "lr_reduced": int(lr_reduced),
            "n_train_proteins": len(train_items),
            "n_val_proteins": len(val_items),
            "total_train_variants": int(train_df["n_train"].sum()),
            "total_epi_samples": int(train_df["n_epi"].sum()),
            "mean_train_total_loss": float(train_df["train_total_loss"].mean()),
            "mean_train_fitness_loss": float(train_df["train_fitness_loss"].mean()),
            "mean_train_epi_loss": float(train_df["train_epi_loss"].mean()),
            "mean_train_spearman": float(train_df["train_spearman"].mean()),
            "mean_train_grad_norm": float(train_df["train_grad_norm"].mean()),
            "mean_val_loss": val_loss,
            "mean_val_spearman": val_spearman,
            "best_epoch": best_epoch + 1,
            "best_val_spearman": best_val_spearman,
            "bad_epochs": bad_epochs,
            "lambda_epi": args.lambda_epi,
        }
        loss_records.append(record)
        pd.DataFrame(loss_records).to_csv(fold_dir / "loss.csv", index=False)
        print(f"[epoch] fold={fold} | {epoch + 1}/{args.max_epochs} | lr={lr_now:.3e} | train_loss={record['mean_train_total_loss']:.6f} | train_rho={record['mean_train_spearman']:.6f} | val_loss={val_loss:.6f} | val_rho={val_spearman:.6f} | bad={bad_epochs}/{args.early_stop_patience}")

        if epoch + 1 >= args.min_epochs and bad_epochs >= args.early_stop_patience:
            print(f"[early-stop] fold={fold} | stop_epoch={epoch + 1} | best_epoch={best_epoch + 1} | best_val_rho={best_val_spearman:.6f}")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    mlp_model.load_state_dict(checkpoint["mlp"])
    print(f"[best] fold={fold} | epoch={checkpoint['epoch'] + 1} | val_rho={checkpoint['val_spearman']:.6f}")
    return model, mlp_model


def test_one_fold(fold, model, mlp_model, train_items, val_items, test_items, pred_dfs, device, train_log, pred_output):
    protein_metrics = []

    for item in test_items:
        set_model_neighbors(model, item["geo_neighbor"], item["epi_neighbor"])
        data = to_gpu(load_features(item["assay_dir"]), device)
        pred = predict_one_protein(model, mlp_model, data, item["split"], device)
        test_spearman = float(spearmanr(item["split"]["labels"], pred).correlation)
        protein_metrics.append({"assay_id": item["assay_id"], "fold": fold, "n_test": item["split"]["n"], "test_spearman": test_spearman})
        pred_dfs[item["assay_id"]]["pred"] = pred
        del data

    protein_df = pd.DataFrame(protein_metrics)
    final_metrics = pd.DataFrame(
        [
            {
                "assay_id": "PROTEIN_MEAN",
                "fold": fold,
                "n_train_proteins": len(train_items),
                "n_val_proteins": len(val_items),
                "n_test_proteins": len(test_items),
                "n_train": int(sum(item["split"]["n"] for item in train_items)),
                "n_val": int(sum(item["split"]["n"] for item in val_items)),
                "n_test": int(protein_df["n_test"].sum()),
                "n_epi": int(sum(item["split"]["n_epi"] for item in train_items)),
                "test_spearman": float(protein_df["test_spearman"].mean()),
            }
        ]
    )
    final_metrics.to_csv(train_log / f"fold_{fold}" / "final_test_metrics.csv", index=False)
    print(f"[final-test] fold={fold} | n_proteins={len(protein_df)} | spearman={final_metrics.loc[0, 'test_spearman']:.6f}")

    for assay_id, df in pred_dfs.items():
        df[["mutation_name", "label", "fold_id", "pred"]].to_csv(pred_output / f"{assay_id}.csv", index=False)


def predict_S461_one_fold(fold, model, mlp_model, S461_items, S461_predictions, device):
    print(f"[S461] predict with fold_{fold} best model")
    for item in S461_items:
        set_model_neighbors(model, item["geo_neighbor"], item["epi_neighbor"])
        data = to_gpu(load_features(item["protein_dir"]), device)
        pred = predict_one_protein(model, mlp_model, data, item["split"], device)
        S461_predictions[item["protein_id"]][f"pred_fold_{fold + 1}"] = pred

        del data


def save_S461_predictions(S461_predictions, output_dir, k_folds):
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_csv in output_dir.glob("*.csv"):
        old_csv.unlink()

    pred_columns = [f"pred_fold_{fold + 1}" for fold in range(k_folds)]
    for protein_id, columns in S461_predictions.items():
        output_df = pd.DataFrame({"label": columns["label"], **{column: columns[column] for column in pred_columns}})
        output_df["pred_mean"] = output_df[pred_columns].mean(axis=1)
        output_df.to_csv(output_dir / f"{protein_id}.csv", index=False)

    print(f"[S461] saved {len(S461_predictions)} protein CSVs -> {output_dir}")


def run_train(args):
    set_seed_everywhere(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pred_output = Path(args.pred_output_root)
    train_log = Path(args.train_log_root)
    pred_S461_output = Path(args.pred_S461_output_root)
    pred_output.mkdir(parents=True, exist_ok=True)
    train_log.mkdir(parents=True, exist_ok=True)
    pred_S461_output.mkdir(parents=True, exist_ok=True)

    assays = load_assays(args)
    S461_items = load_S461(args)
    S461_predictions = {item["protein_id"]: {"label": item["mutation_df"]["label"].astype(float).to_numpy()} for item in S461_items}
    fold_df = build_or_load_folds(assays, args, train_log)
    fold_map = dict(zip(fold_df["assay_id"].astype(str), fold_df["protein_fold"].astype(int)))
    pred_dfs = {}

    for assay in assays:
        assay["protein_fold"] = fold_map[assay["assay_id"]]
        cv_df = assay["cv_df"]
        pred_dfs[assay["assay_id"]] = pd.DataFrame({"mutation_name": cv_df["mutation_name"].astype(str), "label": cv_df["label"].astype(float), "fold_id": assay["protein_fold"], "pred": np.nan})

    print("[fold-count]")
    print(pd.Series([item["protein_fold"] for item in assays]).value_counts().sort_index().to_string())

    for fold in range(args.k_folds):
        test_fold = fold
        val_fold = (fold + 1) % args.k_folds
        train_folds = [x for x in range(args.k_folds) if x not in {test_fold, val_fold}]
        train_raw = [item for item in assays if item["protein_fold"] in train_folds]
        val_raw = [item for item in assays if item["protein_fold"] == val_fold]
        test_raw = [item for item in assays if item["protein_fold"] == test_fold]
        print(f"[split] test_fold={test_fold} | val_fold={val_fold} | train_folds={train_folds}")

        train_items = [{**assay, "split": make_split(assay["cv_df"], compute_epi=args.lambda_epi > 0)} for assay in train_raw]
        val_items = [{**assay, "split": make_split(assay["cv_df"])} for assay in val_raw]
        test_items = [{**assay, "split": make_split(assay["cv_df"])} for assay in test_raw]

        model, mlp_model = train_one_fold(fold, train_items, val_items, test_items, args, device, train_log)
        test_one_fold(fold, model, mlp_model, train_items, val_items, test_items, pred_dfs, device, train_log, pred_output)
        predict_S461_one_fold(fold, model, mlp_model, S461_items, S461_predictions, device)

        del model, mlp_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_S461_predictions(S461_predictions, pred_S461_output, args.k_folds)
    print(f"\n[Done] cluster files -> {train_log / 'protein_similarity_folds'}")
    print(f"[Done] cDNA prediction CSVs -> {pred_output}")
    print(f"[Done] S461 prediction CSVs -> {pred_S461_output}")
    print(f"[Done] training logs -> {train_log}")


def build_parser():
    parser = argparse.ArgumentParser("Joint cDNA nwalign protein-holdout training with S461 zero-shot inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default=str(proj_root / "data" / "1M+_to_1M+"))
    parser.add_argument("--S461_data_root", type=str, default=str(proj_root / "data" / "S461"))
    parser.add_argument("--pred_output_root", type=str, default=str(base_dir / "output" / "joint_cdna_nwalign_5fold_earlystop"))
    parser.add_argument("--pred_S461_output_root", type=str, default=str(base_dir / "output" / "joint_cdna_nwalign_5fold_earlystop_S461"))
    parser.add_argument("--train_log_root", type=str, default=str(base_dir / "training_log" / "joint_cdna_nwalign_5fold_earlystop"))
    parser.add_argument("--S461_min_mutations", type=int, default=10)
    parser.add_argument("--assay_prefix", type=str, default="cdna")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--nwalign_bin", type=str, default="/mydata/xulab/yuzimu/NW-align/NWalign")
    parser.add_argument("--recompute_clusters", action="store_true")
    parser.add_argument("--linkage_method", type=str, default="complete")
    parser.add_argument("--cluster_print_every", type=int, default=500)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--lambda_epi", type=float, default=0.0)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--clip_grad", type=float, default=2.0)
    parser.add_argument("--adj_dim", type=int, default=32)
    parser.add_argument("--rankH", type=int, default=320)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--geo_neighbor", type=float, default=-1.0)
    parser.add_argument("--epi_neighbor", type=float, default=-1.0)
    return parser


if __name__ == "__main__":
    run_train(build_parser().parse_args())
