from pathlib import Path

import torch
from Bio import SeqIO
from transformers import AutoModel, AutoTokenizer

data_root = Path(__file__).resolve().parents[1] / "data"
sequence = str(SeqIO.read(data_root / "wt.fasta", "fasta").seq)

repo = "facebook/esm2_t33_650M_UR50D"

tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModel.from_pretrained(repo).eval().cuda()

with torch.inference_mode():
    inputs = tokenizer(sequence, return_tensors="pt").cuda()
    embedding = model(**inputs).last_hidden_state[0, 1:-1].float().cpu()

torch.save(embedding, data_root / "embedding_ESM2_650M_for_Cerebra_Epistasis.pt")
