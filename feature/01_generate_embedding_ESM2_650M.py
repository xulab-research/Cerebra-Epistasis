import argparse
from pathlib import Path

import torch
from tqdm import tqdm

base_dir = Path(__file__).resolve().parent
proj_root = base_dir.parent
data_root = proj_root / "data"

sequence_filename = "wt.fasta"
embedding_filename = "embedding_ESM2_650M_for_Cerebra_Epistasis.pt"


def read_fasta_sequence(fasta_path):
    sequence = ""
    with open(fasta_path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                sequence += line.strip()
    return sequence


@torch.no_grad()
def generate_embedding(sequence, model, batch_converter):
    _, _, tokens = batch_converter([("", sequence)])
    results = model(tokens.cuda(), repr_layers=[33], return_contacts=False)
    return results["representations"][33].squeeze(0)[1:-1]


def process_protein(protein_dir, model, batch_converter):
    seq_file = protein_dir / sequence_filename
    embedding_file = protein_dir / embedding_filename

    if embedding_file.exists():
        print(f"[skip] {protein_dir.name}")
        return

    sequence = read_fasta_sequence(seq_file)
    embedding = generate_embedding(sequence, model, batch_converter)
    torch.save(embedding.cpu(), embedding_file)

    print(f"[done] {protein_dir.name}")


def run(args):
    model, alphabet = torch.hub.load(
        "facebookresearch/esm:main",
        "esm2_t33_650M_UR50D",
    )
    model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    if args.task:
        task_dirs = [data_root / args.task]
    else:
        task_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())

    for task_dir in task_dirs:
        print(f"\n[task] {task_dir.name}")

        if args.protein:
            protein_dirs = [task_dir / args.protein]
        else:
            protein_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir())

        for protein_dir in tqdm(protein_dirs, desc=task_dir.name):
            process_protein(
                protein_dir,
                model,
                batch_converter,
            )


def build_parser():
    parser = argparse.ArgumentParser("Generate ESM2-650M embeddings")
    parser.add_argument("--task", default="")
    parser.add_argument("--protein", default="")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
