"""Generate ESMC/ESM3 embeddings and optional WT-aligned mmCIF labels in .pt format for Cerebra_Seq structure training"""

import csv
import sys
import torch
import argparse
from pathlib import Path
from contextlib import nullcontext
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from model.utils.structure_utils.mmcif_labels import FEATURE_FILENAME, read_sequence, structure_features
from model.utils.Cerebra_Seq_utils import ensure_model_on_device, clear_cuda_cache

ESMC_REPO = "biohub/ESMC-600M-hf"
ESM3_REPO = "Synthyra/ESM3_small"
ESMC_REVISION = "0fb34e7e5fe1f85d0abaa3d35e2671107c0b458c"
ESM3_REVISION = "7e94ea4a31369bcbb5b9130a20d7e6fa00ebfbed"


def load_models(args):
    options = dict(cache_dir=args.hf_cache_dir, trust_remote_code=True, dtype=torch.float32, attn_implementation="sdpa")
    esmc = AutoModel.from_pretrained(ESMC_REPO, revision=ESMC_REVISION, **options)
    esm3 = AutoModel.from_pretrained(ESM3_REPO, revision=ESM3_REVISION, **options)
    if torch.device(args.device).type == "cuda":
        esm3 = esm3.to(dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(
        ESMC_REPO,
        revision=ESMC_REVISION,
        cache_dir=args.hf_cache_dir,
        trust_remote_code=True,
    )
    return esmc, esm3, tokenizer


@torch.inference_mode()
def generate_embeddings(sequence, models, device):
    device = torch.device(device)
    esmc, esm3, tokenizer = models
    result = {}
    for key, model in (("esmc", esmc), ("esm3", esm3)):
        ensure_model_on_device(model, device)
        try:
            inputs = tokenizer(sequence, return_tensors="pt") if key == "esmc" else model.tokenize_sequences([sequence], device=device)
            inputs = {name: value.to(device) for name, value in inputs.items()}
            context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with context:
                output = model(**inputs, return_dict=True)
            embedding = output.last_hidden_state[0, 1:-1].float().cpu().contiguous()
            expected = (len(sequence), 1152 if key == "esmc" else 1536)
            if tuple(embedding.shape) != expected or not torch.isfinite(embedding).all():
                raise ValueError(f"Invalid {key} embedding: {tuple(embedding.shape)}, expected {expected}")
            result[key] = embedding
            del output, inputs
        finally:
            model.cpu()
            clear_cuda_cache()
    return result


def protein_directories(args):
    tasks = [args.data_root / args.task] if args.task else sorted(p for p in args.data_root.iterdir() if p.is_dir())
    for task in tasks:
        proteins = [task / args.protein] if args.protein else sorted(p for p in task.iterdir() if p.is_dir())
        for protein in proteins:
            yield task.name, protein


def read_manifest(path, mmcif_root):
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"assay", "mmcif_file", "chain_id"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest requires assay, mmcif_file and chain_id columns; optional task")
        rows = {}
        for row in reader:
            key = (row.get("task", "").strip(), row["assay"].strip())
            if key in rows:
                raise ValueError(f"Duplicate manifest entry: {key}")
            filename = row["mmcif_file"].strip()
            chain_id = row["chain_id"].strip()
            if not key[1] or not filename or not chain_id:
                raise ValueError(f"Empty assay, mmcif_file or chain_id in manifest: {row}")
            if filename == "-" or chain_id == "-":
                if filename != "-" or chain_id != "-":
                    raise ValueError(f"Missing structures require both mmcif_file and chain_id to be -: {row}")
                rows[key] = (None, None)
                continue
            mmcif_path = Path(filename)
            mmcif_path = mmcif_path if mmcif_path.is_absolute() else mmcif_root / mmcif_path
            if not mmcif_path.resolve().is_relative_to(mmcif_root.resolve()):
                raise ValueError(f"mmCIF file is outside --mmcif_root: {mmcif_path}")
            if mmcif_path.suffix.lower() not in (".cif", ".mmcif"):
                raise ValueError(f"Expected a .cif or .mmcif file: {mmcif_path}")
            rows[key] = (mmcif_path, " " if chain_id == "." else chain_id)
        return rows


def run(args):
    if (args.mmcif_root is None) != (args.mmcif_manifest is None):
        raise ValueError("Provide --mmcif_root and --mmcif_manifest together, or omit both for embeddings only")
    manifest = None
    if args.mmcif_root is not None:
        if not args.mmcif_root.is_dir():
            raise NotADirectoryError(args.mmcif_root)
        manifest = read_manifest(args.mmcif_manifest, args.mmcif_root)
    failure_count = 0
    models = None
    for task, directory in protein_directories(args):
        output = directory / FEATURE_FILENAME
        try:
            sequence = read_sequence(directory / "wt.fasta")
            source = (None, None) if manifest is None else manifest.get((task, directory.name), manifest.get(("", directory.name)))
            if source is None:
                raise ValueError("No entry in mmCIF manifest")
            mmcif_path, chain_id = source
            if output.exists() and not args.overwrite:
                existing = torch.load(output, map_location="cpu", weights_only=True)
                required = {"sequence", "esmc", "esm3"}
                if mmcif_path is not None:
                    required.update(("all_atom_positions", "all_atom_mask"))
                matching_structure = (mmcif_path is not None) == ("all_atom_positions" in existing)
                if required <= existing.keys() and existing["sequence"] == sequence and matching_structure:
                    print(f"[skip] {task}/{directory.name}: complete output exists")
                    continue
            feature = {"sequence": sequence}
            if mmcif_path is not None:
                feature.update(
                    structure_features(
                        mmcif_path,
                        sequence,
                        chain_id,
                        args.model_index,
                    )
                )
            if models is None:
                models = load_models(args)
            feature.update(generate_embeddings(sequence, models, args.device))
            temporary = output.with_suffix(".pt.tmp")
            torch.save(feature, temporary)
            temporary.replace(output)
            print(f"[done] {task}/{directory.name}: {output.name}")
        except Exception as error:
            failure_count += 1
            print(f"[error] {task}/{directory.name}: {error}")
    if failure_count:
        raise SystemExit(f"{failure_count} proteins failed; see errors above")


def build_parser():
    parser = argparse.ArgumentParser(description="Generate ESMC/ESM3 embeddings and optional mmCIF structure labels")
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--task", default="")
    parser.add_argument("--protein", default="")
    parser.add_argument("--mmcif_root", type=Path, default=None, help="Optional mmCIF directory; provide together with --mmcif_manifest")
    parser.add_argument("--mmcif_manifest", type=Path, default=None, help="Optional TSV: assay, mmcif_file, chain_id; optional task. Omit both structure arguments for embeddings only.")
    parser.add_argument("--model_index", type=int, default=0, help="Zero-based model index, default first model")
    parser.add_argument("--device", default="cuda:0", help="Device shared by ESMC and ESM3; models run sequentially")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
