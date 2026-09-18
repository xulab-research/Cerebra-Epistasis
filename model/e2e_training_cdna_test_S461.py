"""Train Cerebra-Epistasis on cDNA ddG and evaluate on S461."""

import torch
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

base_dir = Path(__file__).resolve().parent
proj_root = base_dir.parent

from cerebra_epistasis.model import SE3Transformer
from utils.metrics import spearman_corr
from utils.utils_func import read_wt_idx_from_fasta, to_gpu, max_mut_from_list, set_seed_everywhere, compute_twobody_epistasis_labels
from utils.calculate_nbodys_mutation_effect import calculate_batch_prediction_mlp, EpistasisMLP
from utils.Cerebra_Seq_utils import *
from utils.structure_utils import residue_constants as rc
from utils.structure_utils.atom_geometry import hu_model_pred_to_atom14_pos, make_atom14_masks
from utils.structure_utils.mmcif_labels import FEATURE_FILENAME

CEREBRA_REVISION = "aab7318429599d6efc711049e1ef614374102d69"


def load_features(assay_dir: Path) -> Dict[str, torch.Tensor]:
    sequence = read_fasta_sequence(assay_dir / "wt.fasta")
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


def build_training_optimizer(cerebra_model, model, mlp_model, args):
    groups = []
    if args.training_mode == "e2e":
        groups.append({"params": cerebra_model.parameters(), "lr": args.cerebra_lr})
    groups.extend([{"params": model.parameters(), "lr": args.lr}, {"params": mlp_model.parameters(), "lr": args.lr}])
    return torch.optim.Adam(groups, eps=1e-6, weight_decay=0.0)


def build_cerebra_model(args, device):
    cerebra_model = load_cerebra_model(
        device,
        checkpoint="model1",
        revision=args.cerebra_revision,
        cache_dir=args.hf_cache_dir,
        training=args.training_mode == "e2e",
    )
    configure_cerebra_training(cerebra_model, args.training_mode)
    return cerebra_model


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
                    return outputs, batch

                prevs = [m_prev.detach(), z_prev.detach(), x_prev.detach()]



def pdb_feats_write(outputs, batch, main_anchor_id=8):
    device = batch["target_feat"].device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch = make_atom14_masks(batch)

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


def apply_cerebra_features(data, cerebra_model, mode="train"):
    outputs, batch = run_cycle(data, cerebra_model, mode)
    pdb_feats = pdb_feats_write(outputs, batch)

    data["atom14_coords"] = pdb_feats["final_atom_positions"].float()
    data["atom14_masks"] = pdb_feats["final_atom_mask"].float()
    data["node_embedding"] = pdb_feats["node_embedding"].float()
    data["edge_embedding"] = pdb_feats["edge_embedding"].permute(1, 2, 0).float()
    return data


def set_neighbors(model, geo_neighbor, epi_neighbor):
    for module in model.modules():
        if hasattr(module, "geo_neighbor"):
            module.geo_neighbor = float(geo_neighbor)
        if hasattr(module, "epi_neighbor"):
            module.epi_neighbor = float(epi_neighbor)


def get_neighbors(assay_dir, args):
    L = len(read_wt_idx_from_fasta(assay_dir / "wt.fasta"))

    if L > 200:
        geo_neighbor = 1.0 / 3.0
        epi_neighbor = 0.0
    else:
        geo_neighbor = 0.5
        epi_neighbor = 1.0 / 3.0

    if args.geo_neighbor >= 0:
        geo_neighbor = args.geo_neighbor
    if args.epi_neighbor >= 0:
        epi_neighbor = args.epi_neighbor

    return geo_neighbor, epi_neighbor


def prepare_train_data(df):
    df = df.copy()
    df["mutation_name"] = df["mutation_name"].astype(str)
    df["label"] = df["label"].astype(float)

    y_raw = df["label"].to_numpy(dtype=float)
    mut_list = df["mutation_name"].tolist()

    # Keep both fitness and epistasis targets in their original label scale.
    epi_indices, epi_y = compute_twobody_epistasis_labels(
        df,
        0.0,
        1.0,
    )

    return {
        "mut_list": mut_list,
        "y_raw": y_raw.astype(np.float32),
        "epi_indices": epi_indices,
        "epi_y": epi_y,
        "max_mut": int(max_mut_from_list(mut_list)),
        "n_mutants": len(df),
        "n_epi": len(epi_indices),
    }


def load_cdna_train_data(args):
    train_root = Path(args.train_data_root)

    train_assays = []

    for assay_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        df = pd.read_csv(assay_dir / args.train_csv_name)

        geo_neighbor, epi_neighbor = get_neighbors(assay_dir, args)
        train_data = prepare_train_data(df)

        train_assays.append(
            {
                "name": assay_dir.name,
                "dir": assay_dir,
                "geo_neighbor": geo_neighbor,
                "epi_neighbor": epi_neighbor,
                "train_data": train_data,
            }
        )

    print(f"[train] assays={len(train_assays)} | " f"mutants={sum(x['train_data']['n_mutants'] for x in train_assays)} | " f"epi={sum(x['train_data']['n_epi'] for x in train_assays)}")

    return train_assays


def load_zeroshot_data(args):
    root = Path(args.s461_root)
    dataset_name = "S461"
    assays = []

    for assay_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        df = pd.read_csv(assay_dir / args.zeroshot_csv_name)

        if len(df) < args.min_test_variants:
            continue

        df = df.copy()
        df["mutation_name"] = df["mutation_name"].astype(str)
        df["label"] = df["label"].astype(float)

        geo_neighbor, epi_neighbor = get_neighbors(assay_dir, args)

        mut_list = df["mutation_name"].tolist()

        assays.append(
            {
                "dataset": dataset_name,
                "name": assay_dir.name,
                "dir": assay_dir,
                "df": df,
                "mut_list": mut_list,
                "label": df["label"].to_numpy(dtype=float),
                "max_mut": int(max_mut_from_list(mut_list)),
                "n_mutants": len(df),
                "geo_neighbor": geo_neighbor,
                "epi_neighbor": epi_neighbor,
            }
        )

    print(f"[zeroshot] {dataset_name} | " f"assays={len(assays)} | " f"mutants={sum(x['n_mutants'] for x in assays)}")

    return assays


def build_model_optimizer_scheduler(args, first_assay, device):
    model = SE3Transformer(
        depth=1,
        hidden_fiber_dict={0: 320, 1: 32},
        out_fiber_dict={0: 128, 1: 32},
        adj_dim=args.adj_dim,
        rankH=args.rankH,
        geo_neighbor=first_assay["geo_neighbor"],
        epi_neighbor=first_assay["epi_neighbor"],
    ).to(device)

    mlp_model = EpistasisMLP(
        input_dim=args.rankH,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.mlp_dropout,
    ).to(device)

    cerebra_model = build_cerebra_model(args, device)

    optimizer = build_training_optimizer(cerebra_model, model, mlp_model, args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    return cerebra_model, model, mlp_model, optimizer, scheduler


def train_one_assay(cerebra_model, model, mlp_model, optimizer, assay, args, device):
    configure_cerebra_training(cerebra_model, args.training_mode)
    model.train()
    mlp_model.train()

    set_neighbors(
        model,
        assay["geo_neighbor"],
        assay["epi_neighbor"],
    )

    data = to_gpu(load_features(assay["dir"]), device)
    train_data = assay["train_data"]

    optimizer.zero_grad(set_to_none=True)

    y_true = torch.as_tensor(
        train_data["y_raw"],
        dtype=torch.float32,
        device=device,
    )

    if args.training_mode == "e2e":
        data = apply_cerebra_features(data, cerebra_model, mode="train")
    else:
        with torch.no_grad():
            data = apply_cerebra_features(data, cerebra_model, mode="eval")
    single_pred, high_delta = model(data)

    y_pred, pred_epi_all = calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=train_data["mut_list"],
        U=high_delta,
        mlp_model=mlp_model,
        max_mut=train_data["max_mut"],
        device=str(device),
        return_epi=True,
    )

    fitness_loss = torch.nn.functional.smooth_l1_loss(
        y_pred.float(),
        y_true.float(),
        beta=args.huber_delta,
    )

    if len(train_data["epi_indices"]) > 0:
        epi_indices = torch.as_tensor(
            train_data["epi_indices"],
            dtype=torch.long,
            device=device,
        )
        epi_y = torch.as_tensor(
            train_data["epi_y"],
            dtype=torch.float32,
            device=device,
        )

        epi_loss = torch.nn.functional.smooth_l1_loss(
            pred_epi_all[epi_indices].float(),
            epi_y.float(),
            beta=args.huber_delta,
        )
    else:
        epi_loss = fitness_loss.new_tensor(0.0)

    loss = fitness_loss + args.lambda_epi * epi_loss
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        max_norm=args.clip_grad,
    )

    optimizer.step()

    train_spearman = float(
        spearman_corr(
            y_pred.detach(),
            y_true.detach(),
        )
        .detach()
        .item()
    )

    del data, single_pred, high_delta, y_pred, pred_epi_all

    return {
        "n_mutants": train_data["n_mutants"],
        "n_epi": train_data["n_epi"],
        "loss": float(loss.detach().item()),
        "fitness_loss": float(fitness_loss.detach().item()),
        "epi_loss": float(epi_loss.detach().item()),
        "train_spearman": train_spearman,
        "grad_norm": float(grad_norm),
    }


@torch.no_grad()
def test_zeroshot_dataset(cerebra_model, model, mlp_model, assays, device, epoch, save_dir=None):
    cerebra_model.eval()
    model.eval()
    mlp_model.eval()

    rows = []

    for assay in assays:
        set_neighbors(
            model,
            assay["geo_neighbor"],
            assay["epi_neighbor"],
        )

        data = to_gpu(
            load_features(assay["dir"]),
            device,
        )

        data = apply_cerebra_features(data, cerebra_model, mode="eval")
        single_pred, high_delta = model(data)

        pred = calculate_batch_prediction_mlp(
            single_mut_matrix=single_pred,
            mut_name_list=assay["mut_list"],
            U=high_delta,
            mlp_model=mlp_model,
            max_mut=assay["max_mut"],
            device=str(device),
            return_epi=False,
        )

        pred = pred.detach().cpu().numpy().reshape(-1)

        spearman = float(pd.Series(assay["label"]).corr(pd.Series(pred), method="spearman"))

        rows.append(
            {
                "epoch": epoch,
                "dataset": assay["dataset"],
                "assay": assay["name"],
                "n_mutants": assay["n_mutants"],
                "spearman": spearman,
            }
        )

        if save_dir is not None:
            out_dir = Path(save_dir) / assay["dataset"]
            out_dir.mkdir(parents=True, exist_ok=True)

            pred_df = assay["df"].copy()
            pred_df["pred"] = pred
            pred_df.to_csv(out_dir / f"{assay['name']}.csv", index=False)

        del data, single_pred, high_delta, pred

    return pd.DataFrame(rows)


def collect_model_states(cerebra_model, model, mlp_model):
    return {
        name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for name, module in (("cerebra_model", cerebra_model), ("model", model), ("mlp", mlp_model))
    }


def resolve_output_paths(args):
    experiment_name = f"{args.training_mode}_cdna_all_zeroshot_S461"
    if args.train_log_root is None:
        args.train_log_root = str(base_dir / "training_log" / experiment_name)
    if args.pred_output_root is None:
        args.pred_output_root = str(base_dir / "cerebra_outputs" / experiment_name)


def train_and_test(args):
    resolve_output_paths(args)
    set_seed_everywhere(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device}, but CUDA is unavailable")

    log_dir = Path(args.train_log_root)
    pred_dir = Path(args.pred_output_root)

    log_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    train_assays = load_cdna_train_data(args)
    s461_assays = load_zeroshot_data(args)

    validate_feature_files(train_assays, s461_assays)

    cerebra_model, model, mlp_model, optimizer, scheduler = build_model_optimizer_scheduler(
        args=args,
        first_assay=train_assays[0],
        device=device,
    )

    loss_rows = []
    zeroshot_rows = []

    for epoch in range(1, args.epochs + 1):
        lr_now = float(optimizer.param_groups[0]["lr"])

        random.shuffle(train_assays)

        train_rows = []

        for assay in train_assays:
            row = train_one_assay(
                cerebra_model=cerebra_model,
                model=model,
                mlp_model=mlp_model,
                optimizer=optimizer,
                assay=assay,
                args=args,
                device=device,
            )

            train_rows.append(row)

        train_df = pd.DataFrame(train_rows)

        epoch_row = {
            "epoch": epoch,
            "lr": lr_now,
            "n_cdna_assays": len(train_df),
            "total_train_mutants": int(train_df["n_mutants"].sum()),
            "total_epi_samples": int(train_df["n_epi"].sum()),
            "mean_loss": float(train_df["loss"].mean()),
            "mean_fitness_loss": float(train_df["fitness_loss"].mean()),
            "mean_epi_loss": float(train_df["epi_loss"].mean()),
            "mean_train_spearman": float(train_df["train_spearman"].mean()),
            "mean_grad_norm": float(train_df["grad_norm"].mean()),
            "lambda_epi": float(args.lambda_epi),
        }

        print_parts = []
        do_eval = epoch % args.ckpt_every == 0 or epoch == args.epochs

        if do_eval:
            save_dir = pred_dir if epoch == args.epochs else None

            test_df = test_zeroshot_dataset(
                cerebra_model=cerebra_model,
                model=model,
                mlp_model=mlp_model,
                assays=s461_assays,
                device=device,
                epoch=epoch,
                save_dir=save_dir,
            )

            zeroshot_rows.extend(test_df.to_dict("records"))

            mean_spearman = float(test_df["spearman"].mean())
            n_assays = int(len(test_df))
            n_valid_assays = int(test_df["spearman"].notna().sum())
            n_mutants = int(test_df["n_mutants"].sum())

            epoch_row["S461_mean_spearman"] = mean_spearman
            epoch_row["S461_n_assays"] = n_assays
            epoch_row["S461_n_valid_assays"] = n_valid_assays
            epoch_row["S461_n_mutants"] = n_mutants

            print_parts.append(f"S461={mean_spearman:.6f}({n_valid_assays}/{n_assays})")

            ckpt_path = log_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    **collect_model_states(cerebra_model, model, mlp_model),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                    "zeroshot_eval": {key: value for key, value in epoch_row.items() if key.endswith("_mean_spearman") or key.endswith("_n_assays") or key.endswith("_n_valid_assays") or key.endswith("_n_mutants")},
                },
                ckpt_path,
            )
            print_parts.append(f"ckpt={ckpt_path.name}")

        loss_rows.append(epoch_row)

        pd.DataFrame(loss_rows).to_csv(log_dir / "loss.csv", index=False)
        if zeroshot_rows:
            pd.DataFrame(zeroshot_rows).to_csv(log_dir / "zeroshot_by_assay.csv", index=False)

        print(f"[epoch] {epoch}/{args.epochs} | " f"lr={lr_now:.3e} | " f"loss={epoch_row['mean_loss']:.6f} | " f"fitness={epoch_row['mean_fitness_loss']:.6f} | " f"epi={epoch_row['mean_epi_loss']:.6f} | " f"train_rho={epoch_row['mean_train_spearman']:.6f} | " f"zeroshot: {' | '.join(print_parts) if do_eval else 'skip'}")

        scheduler.step()

    torch.save(
        {
            **collect_model_states(cerebra_model, model, mlp_model),
            "epoch": args.epochs,
            "args": vars(args),
        },
        log_dir / "final_model.pt",
    )

    print(f"\n[Done] loss -> {log_dir / 'loss.csv'}")
    print(f"[Done] zeroshot by assay -> {log_dir / 'zeroshot_by_assay.csv'}")
    print(f"[Done] model -> {log_dir / 'final_model.pt'}")
    print(f"[Done] final predictions -> {pred_dir}")


def validate_feature_files(train_assays, s461_assays):
    if not train_assays:
        raise ValueError("No training assays found")
    missing = []
    for assay in train_assays + s461_assays:
        for path in (
            assay["dir"] / "embedding_ESM2_650M_for_Cerebra_Epistasis.pt",
            assay["dir"] / FEATURE_FILENAME,
        ):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required .pt feature files. First 10: " + "; ".join(missing[:10]))


def build_parser():
    parser = argparse.ArgumentParser("Joint Cerebra cDNA training and zeroshot validation on S461")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--training_mode",
        type=str,
        default="e2e",
        choices=["e2e", "downstream"],
        help="Training mode: end-to-end joint training or downstream-only training with Cerebra-Seq frozen.",
    )
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--cerebra_revision", default=CEREBRA_REVISION)

    parser.add_argument("--train_data_root", type=str, default=str(proj_root / "data" / "cdna_ddG_data"))
    parser.add_argument("--train_csv_name", type=str, default="data.csv")

    parser.add_argument(
        "--s461_root",
        type=str,
        default=str(proj_root / "data" / "S461"),
    )
    parser.add_argument("--zeroshot_csv_name", type=str, default="data.csv")
    parser.add_argument("--min_test_variants", type=int, default=10)

    parser.add_argument("--train_log_root", type=str, default=None, help="Default: training_log/<training_mode>_cdna_all_test_S461")
    parser.add_argument("--pred_output_root", type=str, default=None, help="Default: cerebra_outputs/<training_mode>_cdna_all_test_S461")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=1)
    parser.add_argument("--cerebra_lr", type=float, default=5e-6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lambda_epi", type=float, default=0.0)
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
    train_and_test(args)


if __name__ == "__main__":
    main()
