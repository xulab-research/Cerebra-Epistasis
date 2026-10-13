import json
from pathlib import Path

import pandas as pd
import torch

from cerebra_epistasis.model import SE3Transformer
from utils.calculate_nbodys_mutation_effect import EpistasisMLP, calculate_batch_prediction_mlp
from utils.metrics import spearman_corr
from utils.utils_func import compute_twobody_epistasis_labels, load_features, max_mut_from_list, set_seed_everywhere, to_gpu

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent / "data"
TRAIN_LOG_DIR = BASE_DIR / "training_log"


def load_data(data_root, device):
    df = pd.read_csv(data_root / "data.csv", usecols=["mutation_name", "label", "fold_id"])
    train_df = df[df["fold_id"] == 0].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("data.csv must contain training rows (fold_id=0).")

    mean = float(train_df["label"].mean())
    std = float(train_df["label"].std(ddof=0)) + 1e-8
    train_y = torch.as_tensor((train_df["label"].to_numpy() - mean) / std, dtype=torch.float32, device=device)
    epi_indices, epi_labels = compute_twobody_epistasis_labels(train_df=train_df, train_mean=mean, train_std=std)
    data = to_gpu(load_features(data_root))
    seq_len = len(data["wt_idx"])
    geo_neighbor, epi_neighbor = (1.0 / 3.0, 0.0) if seq_len > 200 else (0.5, 1.0 / 3.0)

    return {"data": data, "train_df": train_df, "train_y": train_y, "epi_indices": torch.as_tensor(epi_indices, dtype=torch.long, device=device), "epi_y": torch.as_tensor(epi_labels, dtype=torch.float32, device=device), "max_mut": max_mut_from_list(df["mutation_name"].tolist()), "geo_neighbor": geo_neighbor, "epi_neighbor": epi_neighbor, "mean": mean, "std": std}


with open("config.json", encoding="utf-8") as file:
    config = json.load(file)

seed = config["seed"]
epochs = config["epochs"]
lr = config["lr"]
min_lr = config["min_lr"]
adj_dim = config["adj_dim"]
rankH = config["rankH"]
mlp_hidden_dim = config["mlp_hidden_dim"]
mlp_dropout = config["mlp_dropout"]
huber_delta = config["huber_delta"]
clip_grad = config["clip_grad"]
lambda_epi = config["lambda_epi"]

device = torch.device("cuda")
unit = load_data(DATA_ROOT, device)
set_seed_everywhere(seed)

model_config = {"heads": 4, "dim_head": 64, "depth": 1, "input_degrees": 2, "num_degrees": 2, "attend_self": True, "differentiable_coors": False, "fourier_encode_dist": True, "rel_dist_num_fourier_features": 4, "adj_dim": adj_dim, "dim_in": (320, 14), "splits": 4, "hidden_fiber_dict": {0: 320, 1: 32}, "out_fiber_dict": {0: 128, 1: 32}, "rankH": rankH, "geo_neighbor": unit["geo_neighbor"], "epi_neighbor": unit["epi_neighbor"], "esm_dropout": 0.3}
mlp_config = {"input_dim": rankH, "hidden_dim": mlp_hidden_dim, "dropout": mlp_dropout}

model = SE3Transformer(**model_config).cuda()
mlp = EpistasisMLP(**mlp_config).cuda()

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
    preds, pred_epi = calculate_batch_prediction_mlp(single_mut_matrix=single_pred, mut_name_list=mut_list, U=high_delta, mlp_model=mlp, max_mut=unit["max_mut"], device=device, return_epi=True, sqrt_scale=True)

    fitness_loss = torch.nn.functional.smooth_l1_loss(preds, unit["train_y"], beta=huber_delta)
    epi_loss = fitness_loss.new_zeros(())
    if lambda_epi > 0 and unit["epi_indices"].numel() > 0:
        epi_loss = torch.nn.functional.smooth_l1_loss(pred_epi[unit["epi_indices"]], unit["epi_y"], beta=huber_delta)

    loss = fitness_loss + lambda_epi * epi_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
    optimizer.step()

    train_spearman = spearman_corr(preds.detach(), unit["train_y"]).item()

    logs.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_total_loss": loss.item(), "train_fitness_loss": fitness_loss.item(), "train_epi_loss": epi_loss.item(), "train_spearman": train_spearman})
    print(f"[epoch {epoch:03d}] loss={loss.item():.6f} train_spearman={train_spearman:.6f}")

    scheduler.step()

TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(logs).to_csv(TRAIN_LOG_DIR / "train_log.csv", index=False)

torch.save({"epoch": epochs - 1, "model_state": {k: v.cpu() for k, v in model.state_dict().items()}, "mlp_state": {k: v.cpu() for k, v in mlp.state_dict().items()}, "model_config": model_config, "mlp_config": mlp_config, "train_mean": unit["mean"], "train_denom": unit["std"], "n_train_epi": unit["epi_indices"].numel()}, TRAIN_LOG_DIR / "last.pt")
