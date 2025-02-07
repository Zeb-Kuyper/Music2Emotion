<div align="center">

# Music2Emo: Towards Unified Music Emotion Recognition across Dimensional and Categorical Models

[![arXiv](https://img.shields.io/badge/arXiv-2311.00968-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2502.03979)

</div>

This repository contains the code accompanying the paper "Towards Unified Music Emotion Recognition across Dimensional and Categorical Models" by Dr. Jaeyong Kang and Prof. Dorien Herremans.

<div align="center">
  <img src="m2e.png" width="300"/>
</div>

## Introduction

We present a unified multitask learning framework for Music Emotion Recognition (MER) that integrates categorical and dimensional emotion labels, enabling training across multiple datasets. Our approach combines musical features (key and chords) with MERT embeddings and employs knowledge distillation to enhance generalization. Evaluated on MTG-Jamendo, DEAM, PMEmo, and EmoMusic, our model outperforms state-of-the-art methods, including the best-performing model from the MediaEval 2021 competition.

![](framework.png)


## Change Log

## Quickstart Guide

## Installation

## Dataset

## Directory Structure

## Training

## Inference

## Evaluation

| **Model** | **PR-AUC** ↑ | **ROC-AUC** ↑ |
|--------------------|:-----------:|:----------:|
| lileonardo | 0.1508 | 0.7747 |
| SELAB-HCMUS | 0.1435 | 0.7599 |
| Mirable | 0.1356 | 0.7687 |
| UIBK-DBIS | 0.1087 | 0.7046 |
| Hasumi et al. | 0.0730 | 0.7750 |
| Greer et al. | 0.1082 | 0.7354 |
| MERT-95M | 0.1340 | 0.7640 |
| MERT-330M | 0.1400 | 0.7650 |
| **Proposed (Ours)** | **0.1543** | **0.7810** |

## TODO

- [ ] 

## Citation


## Citation

If you find this resource useful, [please cite the original work](https://doi.org/10.48550/arXiv.2502.03979): 

```bibtex
@misc{kang2025unifiedmusicemotionrecognition,
      title={Towards Unified Music Emotion Recognition across Dimensional and Categorical Models}, 
      author={Jaeyong Kang and Dorien Herremans},
      year={2025},
      eprint={2502.03979},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2502.03979}, 
}
```

Kang, J. & Herremans, D. (2025). Towards Unified Music Emotion Recognition across Dimensional and Categorical Models, arXiv.

## Acknowledgements

