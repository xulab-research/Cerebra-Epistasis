# Cerebra-Epistasis

[![Python version badge](https://img.shields.io/badge/Python-%E2%89%A5%203.11-3776AB?logo=python&logoColor=white)](https://github.com/xulab-research/Cerebra-Epistasis)
[![License badge](https://img.shields.io/badge/License-Apache_2.0-blue?logo=apache&logoColor=white)](https://github.com/xulab-research/Cerebra-Epistasis/blob/main/LICENSE)
[![Hugging Face badge](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Cerebra--Seq-yellow)](https://huggingface.co/GongLab-THU/Cerebra-Seq)
[![Zenodo badge](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.22899137-1682D4?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.22899137)
[![DOI badge](https://img.shields.io/badge/DOI-10.64898%2F2026.09.24.753701-blue?logo=doi&logoColor=white)](https://doi.org/10.64898/2026.09.24.753701)

## Overview

Cerebra-Epistasis is a composable, structure- and epistasis-aware framework for protein fitness landscape modeling. It encodes a wild-type protein once to construct a reusable mutation atlas, from which single-mutation effects and epistatic representations can be assembled to predict arbitrary-order mutant fitness.

**Paper**: https://doi.org/10.64898/2026.09.24.753701

<p align="center">
  <a href="assets/overview.svg">
    <img src="assets/overview.svg" width="100%">
  </a>
</p>

## Installation

```bash
pip install \
    torch==2.12.0 \
    pandas==3.0.3 \
    transformers==5.17.0 \
    einops==0.8.2 \
    filelock==3.29.0 \
    clize==5.0.2 \
    biopython==1.88 \
    iterative-stratification==0.1.9
```

The dependency versions are pinned to those used in our tested environment to ensure reproducibility. Newer versions may also be compatible.

## Repository structure

```text
Cerebra-Epistasis/
├── data/                                      # Example input data for training and inference
│   ├── data.csv                               # Mutation-fitness dataset
│   └── wt.fasta                               # Wild-type protein sequence
├── generate_features/                         # Scripts for generating sequence and structure features
│   ├── 01_generate_ESM2_650M_embedding.py     # Generate ESM-2 650M residue-level embeddings
│   └── 02_generate_Cerebra_Seq_features.py    # Generate Cerebra-Seq structure-aware features
├── model/                                     # Cerebra-Epistasis model and training code
│   ├── cerebra_epistasis/                     # Core model architecture
│   ├── utils/                                 # Utility functions and helper modules
│   └── run.py                                 # Training and prediction script
├── assets/                                    # Figures and other assets used in the documentation
├── LICENSE                                    # Software license
└── README.md                                  # Project overview, installation, and usage instructions
```

## Features

- **SE(3)-equivariant structure-aware fitness prediction**: Integrates protein sequence representations with predicted 3D structural information through an SE(3)-equivariant encoder to capture sequence–structure–fitness relationships.

- **Composable mutation atlas**: Represents individual substitutions with additive fitness effects and latent epistatic embeddings, allowing mutations to be flexibly assembled into multi-mutant combinations.

- **Explicit epistasis modeling**: Models non-additive interactions among mutations through a shared nonlinear epistasis module, with direct supervision from experimentally derived epistatic effects when available.

- **Arbitrary-order mutant prediction**: Supports fitness prediction for single and higher-order mutants without explicitly parameterizing every pairwise or higher-order interaction term.

- **Efficient combinatorial inference**: Encodes the wild-type protein once and reuses the resulting mutation atlas to rapidly evaluate large combinatorial mutation spaces.

## Quick Start

The following workflow trains a Cerebra-Epistasis model on a single assay and predicts fitness for its held-out variants. Start from the repository root and run the commands in order. File paths shown below are relative to the repository root.

### Input data

Example input files are provided in `data/`. To use your own data, place `data.csv` and `wt.fasta` for one assay and its corresponding wild-type protein in this directory.

`data.csv` must contain the following columns:

| Column | Description |
|---|---|
| `mutation_name` | Amino-acid substitutions using 0-based residue indexing, such as `H25N` or `H25N,M27K`. Separate multiple substitutions with commas. |
| `label` | Numerical mutation-effect measurement for the assay. |
| `fold_id` | Train/test assignment: `0` for training and `1` for testing. |

### Feature generation

Generate ESM-2 650M sequence embeddings and Cerebra-Seq structural features for the wild-type protein. These features are reused across all variants derived from the same wild-type sequence; individual mutant sequences do not need to be encoded separately.

The feature-generation scripts use CUDA.

```bash
cd generate_features
python 01_generate_ESM2_650M_embedding.py
python 02_generate_Cerebra_Seq_features.py
```

The scripts save the following files:

```text
data/embedding_ESM2_650M_for_Cerebra_Epistasis.pt
data/embedding_Cerebra_Seq_for_Cerebra_Epistasis.pt
```

Cerebra-Seq is available on [Hugging Face](https://huggingface.co/GongLab-THU/Cerebra-Seq).

### Training and prediction

Train the downstream model using the precomputed features and records with `fold_id=0`, then predict fitness for records with `fold_id=1`:

```bash
cd ../model
python run.py
```

Training logs and checkpoints are written to `model/training_log/`, and predictions to `model/output/`.

Use `python run.py --help` to view available arguments.

## Resources

| Resource | Location |
|---|---|
| Cerebra-Epistasis source code | [GitHub](https://github.com/xulab-research/Cerebra-Epistasis) |
| Cerebra-Seq | [Hugging Face](https://huggingface.co/GongLab-THU/Cerebra-Seq) |
| Cerebra-Epistasis code, benchmark datasets, precomputed features, and prediction outputs | [Zenodo](https://doi.org/10.5281/zenodo.22899137) |

## Citation

If you use Cerebra-Epistasis in your research, please cite:

```bibtex
@article{Cerebra-Epistasis,
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
