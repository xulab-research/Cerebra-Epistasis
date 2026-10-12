from pathlib import Path

import pandas as pd
import torch

from cerebra_epistasis.model import SE3Transformer
from utils.calculate_nbodys_mutation_effect import (
    EpistasisMLP,
    calculate_batch_prediction_mlp,
)
from utils.utils_func import (
    load_features,
    max_mut_from_list,
    to_gpu,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent / "data"
CHECKPOINT_PATH = BASE_DIR / "training_log" / "last.pt"
PRED_OUTPUT_DIR = BASE_DIR / "output"


def main():
    device = torch.device("cuda")

    df = pd.read_csv(DATA_ROOT / "data.csv", usecols=["mutation_name", "label", "fold_id"])
    test_df = df[df["fold_id"] == 1].reset_index(drop=True)
    if test_df.empty:
        raise ValueError("data.csv must contain test rows (fold_id=1).")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)

    model = SE3Transformer(**checkpoint["model_config"]).cuda()
    mlp = EpistasisMLP(**checkpoint["mlp_config"]).cuda()
    model.load_state_dict(checkpoint["model_state"])
    mlp.load_state_dict(checkpoint["mlp_state"])

    model.eval()
    mlp.eval()

    data = to_gpu(load_features(DATA_ROOT))
    with torch.no_grad():
        single_pred, high_delta = model(data)
        pred_z = calculate_batch_prediction_mlp(single_mut_matrix=single_pred, mut_name_list=test_df["mutation_name"].tolist(), U=high_delta, mlp_model=mlp, max_mut=max_mut_from_list(df["mutation_name"].tolist()), device=device, return_epi=False, sqrt_scale=True)

    test_df["pred"] = pred_z.cpu().numpy() * checkpoint["train_denom"] + checkpoint["train_mean"]

    PRED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(PRED_OUTPUT_DIR / "predictions.csv", index=False)
    print(f"[saved] {PRED_OUTPUT_DIR / 'predictions.csv'} ({len(test_df)} test variants)")


if __name__ == "__main__":
    main()
