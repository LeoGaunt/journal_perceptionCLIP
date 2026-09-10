# Contextual Conditioning in Contrastive Vision-Language Models: A Systematic Empirical Study

Official code for the paper:

> **Contextual Conditioning in Contrastive Vision-Language Models: A Systematic Empirical Study**
> Leo Gaunt, Alaa Alahmadi
> *Submitted to the International Journal of Computer Vision (IJCV), 2026.*
> [Paper link — TODO after publish] · [DOI — TODO after archive]

This repository contains everything needed to reproduce the paper's experiments: a systematic empirical study of contextual conditioning (PerceptionCLIP-style inference) across **seven pretrained contrastive vision-language models** and **nine datasets**, with a three-layer statistical evidence framework (Δp effect sizes, bootstrap confidence intervals, and McNemar's tests.

The codebase is a fork and substantial extension of [PerceptionCLIP](https://github.com/umd-huang-lab/perceptionCLIP) (An et al., ICLR 2024) — see [Attribution](#attribution--licensing).

---

## Models evaluated

All models are loaded through a unified `VLMEncoder` (`src/models/modeling.py`) that abstracts over the `openai/CLIP` and `open_clip` backends. The study is organised as four controlled comparison groups sharing **OpenCLIP-ViT-B/16** as a common baseline:

| Group | Model string | Backbone / pretrained tag | Varies |
|---|---|---|---|
| Baseline (all groups) | `OpenCLIP-ViT-B-16` | ViT-B-16 / `laion2b_s34b_b88k` | — |
| Scale | `OpenCLIP-ViT-L-14` | ViT-L-14 / `laion2b_s32b_b82k` | Model size |
| Scale | `OpenCLIP-ViT-H-14` | ViT-H-14 / `laion2b_s32b_b79k` | Model size |
| Architecture | `ConvNeXt-B` | convnext_base_w / `laion2b_s13b_b82k` | Image encoder |
| Objective | `SigLIP-ViT-B-16` | ViT-B-16-SigLIP / `webli` | Training loss |
| Data | `MetaCLIP-ViT-B-16` | ViT-B-16 / `metaclip_400m` | Training data |
| Data | `DataComp-ViT-B-16` | ViT-B-16 / `datacomp_xl_s13b_b90k` | Training data |

> **Important:** always tokenise via `encoder.tokenize(texts)`, never `clip.tokenize()` directly, so the correct vocabulary is used for each backend.

> **Note on SigLIP2:** SigLIP2 model strings remain in the registry for completeness but were **excluded from the published study** due to compounding confounds (multiple simultaneous changes relative to the baseline). See the paper's experimental design section.

Original OpenAI CLIP strings (`ViT-B/16`, `ViT-L/14@336px`, `RN50x4`, …) are retained for backward compatibility with the preceding dissertation study but are not part of the published results.

## Datasets

Nine datasets are used. Two carry legacy internal names that differ from the paper:

| Paper name | Code name (`--dataset`) | Role |
|---|---|---|
| EuroSAT | `EuroSAT` | Remote sensing |
| UCM | `UCM` | Remote sensing |
| Fish | `Fish` | Fine-grained natural |
| Sea Animals | `Marine` | Fine-grained natural |
| Flowers-102 | `Flowers102` | Fine-grained natural |
| CUB-200 | `CUB200` | Fine-grained natural |
| Oxford-IIIT Pets | `OxfordPets` | Fine-grained natural |
| COCO-Transport | `Dissertation` | Ceiling-effect control |
| ImageNet | `ImageNet` | Generalisation benchmark |

Expected directory layouts are documented in [`DATA.md`](DATA.md). Download/formatting scripts for each dataset are in [`importers/`](importers/) (`bash importers/run_all.sh`). **ImageNet must be obtained separately** subject to its terms of access and is not downloaded automatically.

## Installation

```bash
git clone https://github.com/LeoGaunt/journal_perceptionCLIP.git
cd journal_perceptionCLIP
pip install -r requirements.txt
```
Requirements are pinned to the versions used for the paper. The `open_clip_torch` version matters: pretrained tags are resolved by the installed version, so an incompatible version may silently load different weights.

A CUDA-capable GPU is required. All published results were produced on a single NVIDIA A100 (Google Colab); the full suite takes roughly 12–18 hours.

## Reproducing the paper

### Everything in one command

```bash
bash scripts/run_all_groups.sh
```

This runs all four comparison groups (scale → architecture → objective → data) and writes per-condition results to `./results/{scale,architecture,objective,data}/`. Groups can also be run independently (`scripts/run_scale_group.sh`, etc.) to parallelise across sessions.

### A single dataset/model combination

`run_experiment.py` runs the full Simple → Domain → +Z sequence for one combination, saves per-image outputs, and computes Δp statistics automatically:

```bash
python run_experiment.py \
    --dataset EuroSAT \
    --data_location ./datasets/data \
    --model OpenCLIP-ViT-B-16 \
    --factors condition,source \
    --simple_template simple_template \
    --main_template eurosat_main_template \
    --factor_templates eurosat_factor_templates \
    --save_path ./results/eurosat \
    --save_name eurosat_OpenCLIP-ViT-B-16
```

Each run writes a `.csv` summary and a `.json` containing per-image `preds`, `labels`, and `correct` arrays (required for McNemar/bootstrap), per-class and macro F1, and the confusion matrix.

## Repository structure

```
├── run_experiment.py        # unified runner + statistics
├── check_datasets.py        # pre-flight data loading verification
├── requirements.txt         # pinned dependencies
├── DATA.md                  # dataset layouts and name mapping
├── importers/               # dataset download & formatting scripts
├── scripts/                 # experiment group runners
├── visualizations/          # where figures are created and data is analysed
├── clip/                    # vendored OpenAI CLIP (MIT, see clip/README.md)
└── src/
    ├── datasets/            # dataset classes and dataloaders
    ├── templates/           # prompt templates and contextual factors
    ├── models/              # unified VLMEncoder and model utilities
    ├── evaluation/          # metrics, bootstrap CIs, McNemar, FDR
    └── zero_shot_inference/ # PerceptionCLIP one/two-step inference
```

## Method summary

Following PerceptionCLIP (An et al., 2024), classification is conditioned on inferred contextual factors using **soft ClassAttr marginalisation** (not hard argmax) in a two-step procedure: the model first infers a distribution over contextual factor values for the image, then class scores are marginalised over that distribution. The paper measures the change in accuracy (Δp) that this conditioning produces relative to a simple-template baseline, and asks how the effect varies with model scale, architecture, training objective, and training-data curation.

## Citation

If you use this code, please cite:

```bibtex
@article{gaunt2026contextual,
  title   = {Contextual Conditioning in Contrastive Vision-Language Models:
             A Systematic Empirical Study},
  author  = {Gaunt, Leo and Alahmadi, Alaa},
  journal = {International Journal of Computer Vision},
  year    = {2026},
  note    = {Under review}  % TODO: update on acceptance
}
```

Please also cite the original PerceptionCLIP paper:

```bibtex
@inproceedings{an2024perceptionclip,
  title     = {More Context, Less Distraction: Visual Classification by
               Inferring and Conditioning on Contexts},
  author    = {An, Bang and Zhu, Sicheng and Panaitescu-Liess, Michael-Andrei
               and Mummadi, Chaithanya Kumar and Huang, Furong},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}
```

## Attribution & licensing

- This repository is a fork of [PerceptionCLIP](https://github.com/umd-huang-lab/perceptionCLIP) by Bang An, Sicheng Zhu, Michael-Andrei Panaitescu-Liess, Chaithanya Kumar Mummadi, and Furong Huang. Their method and original implementation are described in the [PerceptionCLIP paper](https://arxiv.org/abs/2308.01313).
- `clip/` contains vendored code from [OpenAI CLIP](https://github.com/openai/CLIP), MIT licensed.
- New components in this repository (unified `VLMEncoder`, `run_experiment.py`, `src/evaluation/`, dataset importers, and experiment scripts) are © 2026 Leo Gaunt, released under the MIT License. See [`LICENSE`](LICENSE).

## Contact

Questions about the code or paper: open an issue, or contact Leo Gaunt <l.gaunt2@ncl.ac.uk>
