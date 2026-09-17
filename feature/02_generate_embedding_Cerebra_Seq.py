import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# Allow direct execution from any working directory.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.utils.Cerebra_Seq_utils import (
    AnchorFrameConsensus,
    cerebra_autocast,
    clear_cuda_cache,
    ensure_model_on_device,
    hu_model_pred_to_atom14_pos,
    make_atom14_masks,
    move_batch_to_model,
    rc,
    select_anchor_indices,
    sequence_to_hhblits_ids,
)

DATA_ROOT = PROJECT_ROOT / "data"

CEREBRA_REPO = "Gonglab/Cerebra_Seq"
ESMC_REPO = "biohub/ESMC-600M-hf"
ESM3_REPO = "Synthyra/ESM3_small"

ESMC_REVISION = "0fb34e7e5fe1f85d0abaa3d35e2671107c0b458c"
ESM3_REVISION = "7e94ea4a31369bcbb5b9130a20d7e6fa00ebfbed"

SEQUENCE_FILENAME = "wt.fasta"
OUTPUT_FILENAME = "embedding_Cerebra_Seq_for_Cerebra_Epistasis.pt"


def autocast_context(device):
    device = torch.device(device)
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def read_fasta(path):
    sequence = "".join(line.strip() for line in path.read_text().splitlines() if not line.startswith(">"))
    if not sequence:
        raise ValueError(f"Empty FASTA: {path}")
    return sequence.upper()


def load_models(args):
    device = torch.device(args.device)

    esmc_model = AutoModel.from_pretrained(
        ESMC_REPO,
        revision=ESMC_REVISION,
        cache_dir=args.hf_cache_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()

    esm3_model = AutoModel.from_pretrained(
        ESM3_REPO,
        revision=ESM3_REVISION,
        cache_dir=args.hf_cache_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()

    if device.type == "cuda":
        esm3_model = esm3_model.to(dtype=torch.bfloat16)

    esmc_tokenizer = AutoTokenizer.from_pretrained(
        ESMC_REPO,
        revision=ESMC_REVISION,
        cache_dir=args.hf_cache_dir,
        trust_remote_code=True,
    )

    cerebra_model = (
        AutoModel.from_pretrained(
            CEREBRA_REPO,
            trust_remote_code=True,
            checkpoint=args.checkpoint,
            device=args.device,
            revision=args.revision,
            cache_dir=args.hf_cache_dir,
        )
        .eval()
        .float()
    )

    return esm3_model, esmc_model, cerebra_model, esmc_tokenizer


@torch.inference_mode()
def residue_embedding(model, inputs, device):
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with autocast_context(device):
        output = model(**inputs, return_dict=True)
    return output.last_hidden_state[:, 1:-1].float()


@torch.inference_mode()
def generate_sequence_features(sequence, esm3_model, esmc_model, esmc_tokenizer, device):
    embeddings = {}

    ensure_model_on_device(esmc_model, device)
    esmc_inputs = esmc_tokenizer(sequence, return_tensors="pt")
    embeddings["esmc"] = residue_embedding(esmc_model, esmc_inputs, device)[0]
    esmc_model.cpu()
    clear_cuda_cache()

    ensure_model_on_device(esm3_model, device)
    esm3_inputs = esm3_model.tokenize_sequences([sequence], device=device)
    embeddings["esm3"] = residue_embedding(esm3_model, esm3_inputs, device)[0]
    esm3_model.cpu()
    clear_cuda_cache()

    length = len(sequence)
    return {
        "X1D_esm_c": embeddings["esmc"].unsqueeze(0),
        "X1D_esm3": embeddings["esm3"].unsqueeze(0),
        "target_feat": sequence_to_hhblits_ids(sequence, device).unsqueeze(0),
        "residue_index": torch.arange(1, length + 1, dtype=torch.long, device=device).unsqueeze(0),
    }


@torch.inference_mode()
def run_cerebra(batch, model, conf_anchor_chunk_size=24, num_iters=4):
    model_device = next(model.parameters()).device
    batch = move_batch_to_model(batch, model)
    anchors = select_anchor_indices(batch["X1D_esm_c"].shape[1])

    feats = {
        "seq_mask": torch.ones(batch["X1D_esm_c"].shape[:2], device=model_device, dtype=next(model.parameters()).dtype),
        "residue_index": batch["residue_index"],
        "X1D_esm_c": batch["X1D_esm_c"],
        "X1D_esm3": batch["X1D_esm3"],
        "target_feat": batch["target_feat"],
    }

    prevs = [None, None, None]
    outputs = None

    for cycle in range(num_iters):
        recycle = cycle < num_iters - 1
        with cerebra_autocast(model_device, True):
            m_1_prev, z_prev, x_prev, outputs = model(
                feats,
                prevs,
                anchors,
                _recycle=recycle,
                return_aux=not recycle,
                reduce_plddt=True,
                conf_anchor_chunk_size=conf_anchor_chunk_size,
                compress_recycle=True,
                keep_structure_all=False,
            )

        if recycle:
            prevs = [m_1_prev.detach(), z_prev.detach(), x_prev.detach()]
        clear_cuda_cache()

    return outputs, anchors, batch


@torch.inference_mode()
def extract_cerebra_epistasis_features(outputs, anchors, batch):
    device = outputs["translation"][-1].device

    residue_map = torch.as_tensor(
        rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE,
        dtype=torch.long,
        device=device,
    )

    atom_batch = {"aatype": residue_map[batch["target_feat"].to(device)]}
    atom_batch = make_atom14_masks(atom_batch)

    pred_q, pred_t = AnchorFrameConsensus(outputs, len(anchors) // 2)
    _, angles = outputs["angles"]

    atom14_coords = hu_model_pred_to_atom14_pos(
        pred_q,
        pred_t,
        angles.to(device),
        atom_batch["aatype"],
    )

    return {
        "node_embedding": outputs["x1D"][0].detach().float().cpu(),
        "edge_embedding": outputs["x2D"][0].detach().float().cpu(),
        "atom14_coords": atom14_coords[0].detach().float().cpu(),
        "atom14_masks": atom_batch["atom14_atom_exists"][0].detach().float().cpu(),
    }


def protein_directories(args):
    task_dirs = [args.data_root / args.task] if args.task else sorted(p for p in args.data_root.iterdir() if p.is_dir())

    for task_dir in task_dirs:
        protein_dirs = [task_dir / args.protein] if args.protein else sorted(p for p in task_dir.iterdir() if p.is_dir())
        for protein_dir in protein_dirs:
            yield protein_dir


def run(args):
    device = torch.device(args.device)
    proteins = []

    for protein_dir in protein_directories(args):
        fasta_path = protein_dir / SEQUENCE_FILENAME
        output_path = protein_dir / OUTPUT_FILENAME

        if output_path.exists() and not args.overwrite:
            print(f"[skip] {protein_dir.parent.name}/{protein_dir.name}")
            continue

        proteins.append((protein_dir, read_fasta(fasta_path)))

    if not proteins:
        print("No pending proteins.")
        return

    esm3_model, esmc_model, cerebra_model, esmc_tokenizer = load_models(args)

    for protein_dir, sequence in proteins:
        print(f"[run] {protein_dir.parent.name}/{protein_dir.name} (L={len(sequence)})")

        batch = generate_sequence_features(
            sequence,
            esm3_model,
            esmc_model,
            esmc_tokenizer,
            device,
        )

        outputs, anchors, batch = run_cerebra(
            batch,
            cerebra_model,
            conf_anchor_chunk_size=args.conf_anchor_chunk_size,
        )

        features = extract_cerebra_epistasis_features(outputs, anchors, batch)
        torch.save(features, protein_dir / OUTPUT_FILENAME)

        print(f"[done] {protein_dir.parent.name}/{protein_dir.name}")
        clear_cuda_cache()


def build_parser():
    parser = argparse.ArgumentParser(description="Generate Cerebra-Seq features for Cerebra-Epistasis")
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--task", default="")
    parser.add_argument("--protein", default="")
    parser.add_argument("--checkpoint", choices=[f"model{i}" for i in range(1, 6)], default="model1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--conf_anchor_chunk_size", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
