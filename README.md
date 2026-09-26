# Cerebra-Epistasis

[![Python >= 3.10](https://img.shields.io/badge/Python-%E2%89%A5%203.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue?logo=apache&logoColor=white)](https://github.com/xulab-research/Cerebra-Epistasis/blob/main/LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Cerebra--Seq-yellow)](https://huggingface.co/GongLab-THU/Cerebra-Seq)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.22899137-1682D4?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.22899137)
[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.09.24.753701-blue?logo=doi&logoColor=white)](https://doi.org/10.64898/2026.09.24.753701)


Cerebra-Epistasis is a composable, structure- and epistasis-aware framework for protein fitness landscape modeling. It encodes a wild-type protein once to construct a reusable mutation atlas, from which single-mutation effects and epistatic representations can be assembled to predict arbitrary-order mutant fitness.

**Preprint**: https://doi.org/10.64898/2026.09.06.749687

<p align="center">
  <a href="assets/overview.svg">
    <img src="assets/overview.svg" width="100%">
  </a>
</p>

## Overview

Cerebra-Epistasis is a structure- and epistasis-aware framework for protein fitness landscape modeling. It encodes a wild-type protein once using sequence and structural representations to construct a reusable mutation atlas, enabling efficient prediction of fitness and epistatic effects for arbitrary-order mutants.

The framework integrates ESM-2 sequence representations with Cerebra-Seq-derived residue, pairwise, and atomic-coordinate features through an SE(3)-equivariant network. It explicitly separates mutation-specific contributions from epistatic interactions and assembles the corresponding mutation representations without repeatedly encoding each candidate variant.

### Key Capabilities

- **Structure-aware fitness prediction**: Integrate ESM-2 sequence representations with Cerebra-Seq-derived residue, pairwise, and side-chain-resolved structural features for mutation-effect prediction.
- **Explicit epistasis modeling**: Decompose multi-mutant fitness into mutation-specific contributions and a learned epistatic component, with optional pairwise epistasis supervision during training.
- **Reusable mutation atlas**: Encode the wild-type protein once to construct mutation-indexed single-mutation scores and epistasis representations covering all amino-acid substitutions.
- **Arbitrary-order mutant assembly**: Retrieve and combine precomputed mutation-specific representations through permutation-invariant assembly to predict single- and multi-mutant fitness.
- **Efficient landscape-scale inference**: Reuse the wild-type mutation atlas across candidate variants, avoiding repeated sequence and structure encoding during large-scale mutant evaluation.

## Installation

```bash
git clone https://github.com/xulab-research/Cerebra-Epistasis.git
cd Cerebra-Epistasis
pip install -e .
```

## Repository structure

```text
Cerebra-Epistasis/
├── data/
│   ├── data.csv
│   └── wt.fasta
├── generate_features/
│   ├── 01_generate_ESM2_650M_embedding.py
│   └── 02_generate_Cerebra_Seq_features.py
├── model/
│   ├── cerebra_epistasis/
│   ├── utils/
│   └── train.py
├── assets/
├── environment.yml
├── LICENSE
└── README.md
```

## Quick Start

### Input data

Place `data.csv` and `wt.fasta` under `data/`.

`data.csv` must contain:

```text
mutation_name
label
fold_id
```

`mutation_name` specifies the amino-acid substitution(s) using 0-based residue indexing, with multiple substitutions separated by commas. `label` contains the experimentally measured fitness value. `fold_id=0` is used for training and `fold_id=1` for testing.

### Feature generation

Cerebra-Epistasis uses **ESM-2 650M** sequence representations and **Cerebra-Seq** structural representations derived from the wild-type sequence.

```bash
cd generate_features
python 01_generate_ESM2_650M_embedding.py
python 02_generate_Cerebra_Seq_features.py
```

The generated files are:

```text
data/embedding_ESM2_650M_for_Cerebra_Epistasis.pt
data/embedding_Cerebra_Seq_for_Cerebra_Epistasis.pt
```

Cerebra-Seq is available on [Hugging Face](https://huggingface.co/GongLab-THU/Cerebra-Seq).

### Training and prediction

```bash
cd model
python train.py
```

Training logs and checkpoints are written to `model/training_log/`, and predictions to `model/output/`.

Use `python train.py --help` to view available arguments.

### Resources

| Resource | Location |
|---|---|
| Cerebra-Epistasis source code | [GitHub](https://github.com/xulab-research/Cerebra-Epistasis) |
| Cerebra-Seq | [Hugging Face](https://huggingface.co/GongLab-THU/Cerebra-Seq) |
| Cerebra-Epistasis code, benchmark datasets, precomputed features, and prediction outputs | [Zenodo](https://doi.org/10.5281/zenodo.22899137) |


## Citation

If you use Cerebra-Epistasis in your research, please cite:

```bibtex
@article{cerebra_epistasis,
  title   = {From single-sequence structure prediction to protein fitness landscape through a composable, epistasis-aware mutation atlas},
  author  = {Weizhe Wang, Zimu Yu, Endi Yang, Ziyu Shi, Shize Yu, Jian Hu, Yunxin Xu, Haipeng Gong},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.24.753701},
  url     = {https://doi.org/10.64898/2026.09.24.753701}
}
```

## License

This project is licensed under the Apache License 2.0.

Unless otherwise stated, the source code, model architecture, training scripts,
inference scripts, and released model weights/checkpoints are licensed under
Apache-2.0.

Datasets used in this project may be subject to their original licenses and
terms of use. Please refer to the corresponding dataset sources for details.

This software is provided for research purposes and is not intended for clinical
diagnosis, medical decision-making, or direct therapeutic use.
