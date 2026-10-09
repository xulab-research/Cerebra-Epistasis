from pathlib import Path

import pandas as pd
import torch
from clize import run

from cerebra_epistasis.model import SE3Transformer
from utils.calculate_nbodys_mutation_effect import (
    EpistasisMLP,
    calculate_batch_prediction_mlp,
)
from utils.metrics import spearman_corr
from utils.utils_func import (
    compute_twobody_epistasis_labels,
    load_features,
    max_mut_from_list,
    set_seed_everywhere,
    to_gpu,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def load_data(data_root, device):
    df = pd.read_csv(data_root / "data.csv", usecols=["mutation_name", "label", "fold_id"])
    train_df = df[df["fold_id"] == 0].reset_index(drop=True)
    test_df = df[df["fold_id"] == 1].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("data.csv must contain training rows (fold_id=0) and test rows (fold_id=1).")

    mean = float(train_df["label"].mean())
    std = float(train_df["label"].std(ddof=0)) + 1e-8
    train_y = torch.as_tensor((train_df["label"].to_numpy() - mean) / std, dtype=torch.float32, device=device)
    epi_indices, epi_labels = compute_twobody_epistasis_labels(train_df=train_df, train_mean=mean, train_std=std)
    data = to_gpu(load_features(data_root), device)
    seq_len = len(data["wt_idx"])
    geo_neighbor, epi_neighbor = (1.0 / 3.0, 0.0) if seq_len > 200 else (0.5, 1.0 / 3.0)
    return {
        "data": data,
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


def train(
    unit,
    device,
    *,
    seed,
    epochs,
    lr,
    min_lr,
    adj_dim,
    rankH,
    mlp_hidden_dim,
    mlp_dropout,
    huber_delta,
    clip_grad,
    lambda_epi,
    train_log_dir,
):
    set_seed_everywhere(seed)
    model = SE3Transformer(
        hidden_fiber_dict={0: 320, 1: 32},
        out_fiber_dict={0: 128, 1: 32},
        adj_dim=adj_dim,
        rankH=rankH,
        geo_neighbor=unit["geo_neighbor"],
        epi_neighbor=unit["epi_neighbor"],
    ).cuda()
    mlp = EpistasisMLP(input_dim=rankH, hidden_dim=mlp_hidden_dim, dropout=mlp_dropout).cuda()
    parameters = list(model.parameters()) + list(mlp.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    mut_list = unit["train_df"]["mutation_name"].tolist()
    logs = []

    model.train()
    mlp.train()

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        single_pred, high_delta = model(unit["data"])
        preds, pred_epi = calculate_batch_prediction_mlp(
            single_mut_matrix=single_pred,
            mut_name_list=mut_list,
            U=high_delta,
            mlp_model=mlp,
            max_mut=unit["max_mut"],
            device=device,
            return_epi=True,
            sqrt_scale=True,
        )
        fitness_loss = torch.nn.functional.smooth_l1_loss(preds, unit["train_y"], beta=huber_delta)
        epi_loss = fitness_loss.new_zeros(())
        if lambda_epi > 0 and unit["epi_indices"].numel() > 0:
            epi_loss = torch.nn.functional.smooth_l1_loss(pred_epi[unit["epi_indices"]], unit["epi_y"], beta=huber_delta)
        loss = fitness_loss + lambda_epi * epi_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        optimizer.step()

        train_spearman = spearman_corr(preds.detach(), unit["train_y"]).item()
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

        scheduler.step()

    train_log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv(train_log_dir / "train_log.csv", index=False)
    torch.save(
        {
            "epoch": epochs - 1,
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "mlp_state": {k: v.cpu() for k, v in mlp.state_dict().items()},
            "train_mean": unit["mean"],
            "train_denom": unit["std"],
            "n_train_epi": unit["epi_indices"].numel(),
        },
        train_log_dir / "last.pt",
    )
    return model, mlp


@torch.no_grad()
def predict_test(unit, model, mlp, device, *, pred_output_dir: Path):
    model.eval()
    mlp.eval()
    single_pred, high_delta = model(unit["data"])
    test_df = unit["test_df"].copy()
    pred_z = calculate_batch_prediction_mlp(
        single_mut_matrix=single_pred,
        mut_name_list=test_df["mutation_name"].tolist(),
        U=high_delta,
        mlp_model=mlp,
        max_mut=unit["max_mut"],
        device=device,
        return_epi=False,
        sqrt_scale=True,
    )
    test_df["pred"] = pred_z.cpu().numpy() * unit["std"] + unit["mean"]
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(pred_output_dir / "predictions.csv", index=False)
    print(f"[saved] {pred_output_dir / 'predictions.csv'} ({len(test_df)} test variants)")


def main(
    *,
    data_root: Path = PROJECT_ROOT / "data",
    train_log_dir: Path = BASE_DIR / "training_log",
    pred_output_dir: Path = BASE_DIR / "output",
    seed: int = 42,
    epochs: int = 150,
    lr: float = 1e-4,
    min_lr: float = 1e-6,
    adj_dim: int = 32,
    rankH: int = 64,
    mlp_hidden_dim: int = 256,
    mlp_dropout: float = 0.2,
    huber_delta: float = 1.0,
    clip_grad: float = 2.0,
    lambda_epi: float = 2.0,
):
    device = torch.device("cuda")
    unit = load_data(data_root, device)
    model, mlp = train(
        unit,
        device,
        seed=seed,
        epochs=epochs,
        lr=lr,
        min_lr=min_lr,
        adj_dim=adj_dim,
        rankH=rankH,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_dropout=mlp_dropout,
        huber_delta=huber_delta,
        clip_grad=clip_grad,
        lambda_epi=lambda_epi,
        train_log_dir=train_log_dir,
    )
    predict_test(unit, model, mlp, device, pred_output_dir=pred_output_dir)


if __name__ == "__main__":
    run(main)
