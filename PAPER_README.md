# Context-Conditioned Zero-Shot Classification — TPAMI Codebase

Refactored from the BSc dissertation implementation of PerceptionCLIP
(An et al., ICLR 2024) for the expanded empirical study.

## What changed from the dissertation version

### 1. `src/models/modeling.py` — unified `VLMEncoder`
Previously `CLIPEncoder` hardcoded two backends inconsistently.
Now a single `VLMEncoder` class handles:

| Model string           | Backend     | Pretrained weights    |
|------------------------|-------------|-----------------------|
| `ViT-B/16`             | openai/CLIP | OpenAI                |
| `ViT-L/14@336px`       | openai/CLIP | OpenAI                |
| `RN50x4`               | openai/CLIP | OpenAI                |
| `OpenCLIP-ViT-H-14`    | open_clip   | laion2b_s32b_b79k     |
| `OpenCLIP-ViT-L-14`    | open_clip   | laion2b_s32b_b82k     |
| `OpenCLIP-ViT-B-16`    | open_clip   | laion2b_s34b_b88k     |
| `MetaCLIP-ViT-B-16`    | open_clip   | metaclip_400m         |
| `MetaCLIP-ViT-L-14`    | open_clip   | metaclip_fullcc       |
| `MetaCLIP-ViT-H-14`    | open_clip   | metaclip_fullcc       |
| `SigLIP-ViT-B-16`      | open_clip   | webli                 |
| `SigLIP-ViT-L-16-384`  | open_clip   | webli                 |

**Critical**: always call `encoder.tokenize(texts)` — never
`clip.tokenize()` directly — so the correct vocabulary is used.

### 2. `src/zero_shot_inference/utils.py` — model-agnostic tokenisation
Every `clip.tokenize()` call replaced with `clip_model.tokenize()`.
No other logic changed — existing templates and factor definitions
work without modification.

### 3. `src/zero_shot_inference/perceptionclip_two_step.py` — enriched output
Now saves a `.json` file alongside the existing `.csv`:
- `preds`, `labels`, `correct` — per-image arrays (needed for McNemar/bootstrap)
- `f1_per_class`, `f1_macro` — per-class and macro F1
- `conf_matrix` — raw confusion matrix

### 4. `src/evaluation/statistical.py` — statistical analysis
- `bootstrap_ci(correct)` — 95% CI on accuracy via resampling (N=2000)
- `mcnemar_test(correct_a, correct_b)` — significance of classifier difference
- `delta_p_with_stats(baseline, context)` — Δp with CI + McNemar in one call

### 5. `src/evaluation/metrics.py` — classification metrics
- `per_class_f1(preds, labels, n_classes)`
- `confusion_matrix_report(preds, labels, classnames)` — formatted text table

### 6. `run_experiment.py` — unified runner
Replaces individual bash scripts. Runs Simple → Domain → +Z in sequence,
then computes Δp statistics automatically.

```bash
python run_experiment.py \
    --dataset EuroSAT \
    --data_location ./datasets/data \
    --model OpenCLIP-ViT-H-14 \
    --factors condition,source \
    --simple_template eurosat_template \
    --main_template eurosat_main_template \
    --factor_templates eurosat_factor_templates \
    --save_path ./results/eurosat \
    --save_name eurosat_OpenCLIP-ViT-H-14
```

## Installation

```bash
pip install open_clip_torch scipy
# existing requirements (clip, torch, torchvision) unchanged
```

## Running the full cross-architecture study

```bash
bash scripts/run_all_models_eurosat.sh
bash scripts/run_all_models_fish.sh
# add similar scripts for UCM, SeaAnimals, Flowers, COCO
```

## Statistical analysis after all runs

```python
import json
from src.evaluation.statistical import delta_p_with_stats, summarise_results

with open("results/eurosat/eurosat_ViT-B-16_simple.json") as f:
    simple = json.load(f)
with open("results/eurosat/eurosat_ViT-B-16_plus_z.json") as f:
    plus_z = json.load(f)

stats = delta_p_with_stats(simple["correct"], plus_z["correct"])
print(summarise_results(stats))
# Baseline:  51.43%  [49.81, 53.05]
# Context:   58.84%  [57.27, 60.41]
# Δp:        +7.41%  [+5.22, +9.60]  (se=1.12)
# McNemar:   χ²=47.3  p<0.0001  b=312  c=514  ✓ significant (α=0.05)
```
