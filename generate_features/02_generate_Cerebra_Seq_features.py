import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

data_root = Path(__file__).resolve().parents[1] / "data"
sequence = "".join(line.strip() for line in (data_root / "wt.fasta").read_text().splitlines() if not line.startswith(">"))

device = "cuda"
L = len(sequence)


@torch.inference_mode()
def esm_features(repo, sequence, esm3=False):
    model = (
        AutoModel.from_pretrained(
            repo,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        .eval()
        .to(device)
    )

    if esm3:
        inputs = model.tokenize_sequences([sequence], device=device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        inputs = tokenizer(sequence, return_tensors="pt").to(device)

    return model(**inputs, return_dict=True).last_hidden_state[:, 1:-1].float().cpu()


esm3 = esm_features("Synthyra/ESM3_small", sequence, esm3=True)
esmc = esm_features("biohub/ESMC-600M-hf", sequence)

aa_ids = dict(zip("ACDEFGHIKLMNPQRSTVWY", range(20)))
aa_ids.update(B=2, Z=3, U=1, X=20, J=20, O=20)

feats = {
    "target_feat": torch.tensor([[aa_ids.get(aa, 20) for aa in sequence]], device=device),
    "residue_index": torch.arange(1, L + 1, device=device)[None],
    "seq_mask": torch.ones(1, L, device=device),
    "X1D_esm_c": esmc.to(device),
    "X1D_esm3": esm3.to(device),
}

n = next((count for limit, count in [(96, 18), (224, 24), (656, 32), (900, 24)] if L <= limit), 24)
anchors = torch.tensor([int(i * (L - 8) / n) + 5 for i in range(n)]).clamp(2, L - 2)
if n + 8 > L:
    anchors = torch.arange(2, L - 2)

model = (
    AutoModel.from_pretrained(
        "GongLab-THU/Cerebra-Seq",
        trust_remote_code=True,
        checkpoint="model1",
        device=device,
    )
    .eval()
    .float()
)

prevs = [None, None, None]
with torch.inference_mode():
    for cycle in range(4):
        m, z, x, outputs = model(
            feats,
            prevs,
            anchors,
            _recycle=cycle < 3,
            return_aux=cycle == 3,
            return_dist=False,
            return_pae=False,
            reduce_plddt=True,
            conf_anchor_chunk_size=24,
            compress_recycle=True,
            keep_structure_all=False,
        )
        if cycle < 3:
            prevs = [m, z, x]

features = model.extract_features(outputs, anchors, feats)
torch.save(features, data_root / "embedding_Cerebra_Seq_for_Cerebra_Epistasis.pt")
