# Cerebra-Epistasis

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue?logo=apache&logoColor=white)](LICENSE)

Cerebra-Epistasis is a composable, structure- and epistasis-aware framework for protein fitness landscape modeling. It encodes a wild-type protein once to construct a reusable mutation atlas, from which single-mutation effects and epistatic representations can be assembled to predict arbitrary-order mutant fitness.

<p align="center">
  <a href="assets/overview.svg">
    <img src="assets/overview.svg" width="100%">
  </a>
</p>

<p align="center">
  <i>Overview of the Cerebra-Epistasis framework.</i>
</p>

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
└── README.md
```

## Installation

```bash
conda env create -f environment.yml
conda activate cerebra-epistasis
```

## Input data

Place `data.csv` and `wt.fasta` under `data/`.

`data.csv` must contain:

```text
mutation_name
label
fold_id
```

`mutation_name` specifies the amino-acid substitution(s), with multiple substitutions separated by commas. `label` contains the experimentally measured fitness value. `fold_id=0` is used for training and `fold_id=1` for testing.

## Feature generation

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

## Training and prediction

```bash
cd model
python train.py
```

Training logs and checkpoints are written to `model/training_log/`, and predictions to `model/output/`.

Use `python train.py --help` to view available arguments.

## Resources

| Resource | Location |
|---|---|
| Cerebra-Epistasis source code | [GitHub](https://github.com/xulab-research/Cerebra-Epistasis) |
| Cerebra-Seq | [Hugging Face](https://huggingface.co/GongLab-THU/Cerebra-Seq) |
| Benchmark datasets and precomputed features | [Zenodo](https://doi.org/10.5281/zenodo.22899138) |
| Cerebra-Epistasis and baseline predictions | [Zenodo](https://doi.org/10.5281/zenodo.22899138) |

**Zenodo:** coming soon.

## Citation

If you find Cerebra-Epistasis useful, please cite:

```bibtex
@article{cerebra_epistasis,
  title   = {From single-sequence structure prediction to protein fitness landscapes through a composable, epistasis-aware mutation atlas},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

## License

This project is licensed under the Apache License 2.0.

Datasets and external pretrained models remain subject to their respective licenses and terms of use.
