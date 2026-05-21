"""
perceptionclip_two_step.py — PerceptionCLIP two-step zero-shot inference.

Key changes from the dissertation version
------------------------------------------
1. Uses VLMEncoder / tokenize() instead of clip.tokenize() directly,
   so any registered model (CLIP, OpenCLIP, MetaCLIP, SigLIP) works.
2. Returns per-image (pred, label) arrays alongside aggregate accuracy,
   enabling McNemar's test and bootstrap CIs in post-processing.
3. Saves a structured JSON results file in addition to the CSV, so the
   evaluation module can consume it without reparsing.
4. Reports per-class F1 and confusion matrix in the JSON output.
"""

import torch
import argparse
import csv
import json
import os
import numpy as np

import src.datasets as datasets
import src.templates as templates
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.zero_shot_inference.utils import (
    get_zeroshot_classifier_flat_advance,
    get_zeroshot_classifier_puretext_advance,
    generate_composite_factors,
    compose_template,
    template_convert,
    group_accuracy,
)
from src.models import utils as model_utils
from src.models.modeling import VLMEncoder


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_location", type=str, default="./datasets/data/")
    parser.add_argument("--dataset",        type=str, default="EuroSAT")
    parser.add_argument("--template",       type=str, default="simple_template")
    parser.add_argument("--model",          type=str, default="ViT-B/16",
                        help="Any model string supported by VLMEncoder.")
    parser.add_argument("--batch_size",     type=int, default=128)
    parser.add_argument("--workers",        type=int, default=2)
    parser.add_argument("--eval_augmentation",   type=str, default="None")
    parser.add_argument("--eval_augmentation_2", type=str, default="None")
    parser.add_argument("--eval_augmentation_param", type=int, default=1)
    parser.add_argument("--eval_trainset",  type=bool, default=False)
    parser.add_argument("--cache_dir",      type=str, default=None)
    parser.add_argument("--save_name",      type=str, default="tmp")
    parser.add_argument("--save_path",      type=str,
                        default="./results/zero_shot_inference/eval_acc_ours")
    parser.add_argument("--finetuned_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_mode", type=int, default=0)
    parser.add_argument("--infer_mode",     type=int, choices=[0, 1], default=0,
                        help="0: infer z with y  |  1: infer z without y")
    parser.add_argument("--convert_text",   type=str, default="object")
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--num_attrs",      type=int, default=2)
    parser.add_argument("--num_labels",     type=int, default=2)
    parser.add_argument("--eval_group",     type=bool, default=False)
    parser.add_argument("--factors",        default=None,
                        type=lambda x: x.split(","))
    parser.add_argument("--main_template",  type=str, default="main_template")
    parser.add_argument("--factor_templates", type=str, default="factor_templates")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Load model via unified VLMEncoder
    model = VLMEncoder(args, keep_lang=True)
    print(f"Model: {args.model}  |  device: {args.device}")

    if args.finetuned_checkpoint is not None:
        ckpt = torch.load(args.finetuned_checkpoint)
        if args.checkpoint_mode == 0:
            model.load_state_dict(ckpt["model_state_dict"])
        elif args.checkpoint_mode == 1:
            model.load(args.finetuned_checkpoint)
        print("Finetuned checkpoint loaded.")

    model = model.to(args.device)

    # Load dataset
    dataset_class = getattr(datasets, args.dataset)
    dataset = dataset_class(
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print(f"Dataset: {args.dataset}  |  classes: {len(dataset.classnames)}")

    # Build template list
    if args.factors is None:
        raise ValueError("Please provide --factors")

    main_template    = _get_template(args.main_template)
    factor_templates = _get_template(args.factor_templates)
    composite        = generate_composite_factors(factor_templates,
                                                  selected_factors=args.factors)
    template_list    = compose_template(main_template, composite)

    # Build classification head
    classification_head = get_zeroshot_classifier_flat_advance(
        args, model.model, dataset.classnames, template_list
    )
    classification_head = classification_head.to(args.device)
    args.num_factor_value = len(template_list)

    factor_head = None
    if args.infer_mode == 1:
        template_list_woy = template_convert(template_list, args.convert_text)
        factor_head = get_zeroshot_classifier_puretext_advance(
            args, model.model, template_list_woy
        ).to(args.device)

    # Run inference
    results = classify(model, classification_head, dataset, args, factor_head,
                       classnames=dataset.classnames)

    # Save outputs
    _save_results(args, results)
    return results


# ---------------------------------------------------------------------------
# Inference loop — returns rich results dict
# ---------------------------------------------------------------------------

def classify(model, classification_head, dataset, args,
             factor_head=None, classnames=None):
    model.eval()
    classification_head.eval()

    dataloader = get_dataloader(dataset, is_train=args.eval_trainset,
                                args=args, image_encoder=None)
    device = args.device

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            # --- infer z ---
            if args.infer_mode == 0:
                logits = model_utils.get_logits(x, model, classification_head)
                factor_probs = (logits / args.temperature).exp()
                factor_probs = factor_probs.view(
                    factor_probs.shape[0], -1, args.num_factor_value
                ).sum(dim=1)
                factor_probs = factor_probs / factor_probs.sum(dim=1, keepdim=True)
            else:
                factor_logits = model_utils.get_logits(x, model, factor_head)
                factor_probs  = torch.softmax(factor_logits / args.temperature, dim=1)
                logits = model_utils.get_logits(x, model, classification_head)

            # --- infer y | x, z ---
            probs = logits.exp()
            probs = probs.view(probs.shape[0], -1, args.num_factor_value)
            probs = probs / probs.sum(dim=1, keepdim=True)
            probs = (probs * factor_probs.unsqueeze(1)).sum(dim=2)

            pred = probs.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    correct = all_preds.eq(all_labels)
    acc     = correct.float().mean().item() * 100.0

    print(f"Accuracy: {acc:.2f}%")

    # Per-class metrics
    n_classes  = len(classnames) if classnames else (all_labels.max().item() + 1)
    f1_scores, conf_matrix = _compute_per_class_metrics(
        all_preds.numpy(), all_labels.numpy(), n_classes
    )

    return {
        "accuracy":   acc,
        "preds":      all_preds.numpy().tolist(),
        "labels":     all_labels.numpy().tolist(),
        "correct":    correct.numpy().tolist(),
        "f1_per_class": f1_scores.tolist(),
        "f1_macro":   float(np.mean(f1_scores)),
        "conf_matrix": conf_matrix.tolist(),
        "classnames": classnames,
    }


# ---------------------------------------------------------------------------
# Per-class F1 and confusion matrix
# ---------------------------------------------------------------------------

def _compute_per_class_metrics(preds, labels, n_classes):
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for p, l in zip(preds, labels):
        conf[l, p] += 1

    f1 = np.zeros(n_classes)
    for c in range(n_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1[c] = (2 * tp / denom) if denom > 0 else 0.0

    return f1, conf


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def _save_results(args, results):
    os.makedirs(args.save_path, exist_ok=True)

    # --- CSV (backward compatible) ---
    csv_path = os.path.join(args.save_path, args.save_name + ".csv")
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([args.factors, args.temperature, results["accuracy"]])

    # --- JSON (rich output for statistical analysis) ---
    json_path = os.path.join(args.save_path, args.save_name + ".json")
    payload = {
        "model":      args.model,
        "dataset":    args.dataset,
        "factors":    args.factors,
        "temperature": args.temperature,
        "accuracy":   results["accuracy"],
        "f1_macro":   results["f1_macro"],
        "f1_per_class": results["f1_per_class"],
        "conf_matrix":  results["conf_matrix"],
        "classnames":   results["classnames"],
        # per-image arrays for McNemar / bootstrap
        "preds":   results["preds"],
        "labels":  results["labels"],
        "correct": results["correct"],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Results saved → {csv_path}")
    print(f"             → {json_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    main(args)
