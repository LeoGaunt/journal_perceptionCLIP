"""
run_experiment.py — Unified experiment runner for the TPAMI paper.

Runs Simple → Domain → +Z for a given dataset/model combination in one
call, writes JSON per condition, then computes Δp with bootstrap CIs
and McNemar's test between Simple and +Z.

Usage
-----
    python run_experiment.py \\
        --dataset EuroSAT \\
        --data_location ./datasets/data \\
        --model ViT-B/16 \\
        --factors condition,source \\
        --simple_template eurosat_template \\
        --domain_template eurosat_template \\
        --main_template eurosat_main_template \\
        --factor_templates eurosat_factor_templates \\
        --save_path ./results/eurosat \\
        --save_name eurosat_ViT-B16

Supported --model values (see src/models/modeling.py for the full list)
    Original CLIP   : ViT-B/16  ViT-L/14@336px  RN50x4
    OpenCLIP LAION  : OpenCLIP-ViT-H-14  OpenCLIP-ViT-L-14  OpenCLIP-ViT-B-16
    MetaCLIP        : MetaCLIP-ViT-B-16  MetaCLIP-ViT-L-14  MetaCLIP-ViT-H-14
    SigLIP          : SigLIP-ViT-B-16    SigLIP-ViT-L-16-384
"""

import argparse
import json
import os
import types
import torch

import types as _types

import src.datasets as datasets
import src.templates as templates
from src.models.modeling import VLMEncoder


def _get_template(name: str):
    """
    Safely retrieve a template variable from src.templates by name.

    When __init__.py does `from .eurosat_template import *`, Python sets
    templates.eurosat_template to the submodule object as a side effect of
    the import, even though a same-named list variable is also exported.
    This helper detects that and fetches the variable from inside the module.
    """
    obj = getattr(templates, name)
    if isinstance(obj, _types.ModuleType):
        obj = getattr(obj, name)
    return obj
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models import utils as model_utils
from src.zero_shot_inference.utils import (
    get_zeroshot_classifier,
    get_zeroshot_classifier_flat_advance,
    generate_composite_factors,
    compose_template,
)
from src.zero_shot_inference.perceptionclip_two_step import classify as classify_twostep
from src.evaluation.statistical import delta_p_with_stats, summarise_results
from src.evaluation.metrics import confusion_matrix_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",          type=str, required=True)
    p.add_argument("--data_location",    type=str, default="./datasets/data")
    p.add_argument("--model",            type=str, default="ViT-B/16")
    p.add_argument("--batch_size",       type=int, default=128)
    p.add_argument("--workers",          type=int, default=2)
    p.add_argument("--factors",          type=lambda x: x.split(","), required=True)
    p.add_argument("--simple_template",  type=str, default="simple_template")
    p.add_argument("--domain_template",  type=str, default=None,
                   help="Defaults to simple_template if not set.")
    p.add_argument("--main_template",    type=str, required=True)
    p.add_argument("--factor_templates", type=str, required=True)
    p.add_argument("--temperature",      type=float, default=1.0)
    p.add_argument("--infer_mode",       type=int,   default=0, choices=[0, 1])
    p.add_argument("--save_path",        type=str,   default="./results")
    p.add_argument("--save_name",        type=str,   default="experiment")
    p.add_argument("--n_bootstrap",      type=int,   default=2000)
    p.add_argument("--skip_simple",      action="store_true",
                   help="Skip Simple condition (e.g. results already saved).")
    p.add_argument("--skip_domain",      action="store_true")
    args = p.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.domain_template is None:
        args.domain_template = args.simple_template
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _args_stub(args, **overrides):
    """Return a simple namespace inheriting from args with overrides applied."""
    stub = types.SimpleNamespace(**vars(args))
    for k, v in overrides.items():
        setattr(stub, k, v)
    return stub


def _run_condition(model, dataset, args, template_name, condition_label):
    """
    Standard zero-shot inference for Simple or Domain conditions.
    template_name: the template variable name to look up (e.g. 'simple_template'
                   or 'eurosat_main_template').
    condition_label: 'simple' or 'domain', used only for the print line.
    """
    from src.zero_shot_inference.utils import get_zeroshot_classifier
    tmpl = _get_template(template_name)
    head = get_zeroshot_classifier(args, model, dataset.classnames, tmpl)
    head = head.to(args.device)

    all_preds, all_labels = [], []
    model.eval()
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    with torch.no_grad():
        for data in dataloader:
            data = maybe_dictionarize(data)
            x = data["images"].to(args.device)
            y = data["labels"].to(args.device)
            logits = model_utils.get_logits(x, model, head)
            pred   = logits.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())

    import numpy as np
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    correct = (preds == labels).astype(int).tolist()
    acc = float(np.mean(correct)) * 100.0
    print(f"[{condition_label}] Accuracy: {acc:.2f}%")

    n_classes = len(dataset.classnames)
    report    = confusion_matrix_report(preds, labels, dataset.classnames)

    return {
        "condition": condition_label,
        "model":     args.model,
        "dataset":   args.dataset,
        "accuracy":  acc,
        "f1_macro":  report["f1_macro"],
        "f1_per_class": report["f1_per_class"].tolist(),
        "conf_matrix":  report["matrix_raw"].tolist(),
        "classnames":   dataset.classnames,
        "preds":    preds.tolist(),
        "labels":   labels.tolist(),
        "correct":  correct,
    }


def _run_plus_z(model, dataset, args):
    """PerceptionCLIP +Z condition."""
    stub = _args_stub(args, num_attrs=2, num_labels=len(dataset.classnames),
                      eval_trainset=False, eval_group=False)
    main_tmpl    = _get_template(args.main_template)
    factor_tmpl  = _get_template(args.factor_templates)
    composite    = generate_composite_factors(factor_tmpl, selected_factors=args.factors)
    template_list = compose_template(main_tmpl, composite)
    stub.num_factor_value = len(template_list)

    head = get_zeroshot_classifier_flat_advance(
        stub, model, dataset.classnames, template_list
    ).to(args.device)

    return classify_twostep(model, head, dataset, stub,
                            factor_head=None, classnames=dataset.classnames)


def _save_condition(results, save_path, save_name, condition):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f"{save_name}_{condition}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Model   : {args.model}")
    print(f"  Factors : {args.factors}")
    print(f"{'='*60}\n")

    # Load model once
    model = VLMEncoder(args, keep_lang=True).to(args.device)

    # Load dataset
    dataset_class = getattr(datasets, args.dataset)
    dataset = dataset_class(
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    collected = {}

    # --- Simple ---
    if not args.skip_simple:
        print("\n[1/3] Simple (baseline CLIP)")
        simple_results = _run_condition(model, dataset, args,
                                         args.simple_template, "simple")
        simple_results["condition"] = "simple"
        _save_condition(simple_results, args.save_path, args.save_name, "simple")
        collected["simple"] = simple_results

    # --- Domain ---
    if not args.skip_domain:
        print("\n[2/3] Domain (main_template without factors)")
        domain_results = _run_condition(model, dataset, args,
                                        args.main_template, "domain")
        domain_results["condition"] = "domain"
        _save_condition(domain_results, args.save_path, args.save_name, "domain")
        collected["domain"] = domain_results

    # --- +Z ---
    print("\n[3/3] +Z (PerceptionCLIP contextual)")
    plus_z_results = _run_plus_z(model, dataset, args)
    plus_z_results["condition"] = "+Z"
    _save_condition(plus_z_results, args.save_path, args.save_name, "plus_z")
    collected["+Z"] = plus_z_results

    # --- Statistical summary ---
    if "simple" in collected:
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY  (Simple vs +Z)")
        print("="*60)
        summary = delta_p_with_stats(
            collected["simple"]["correct"],
            collected["+Z"]["correct"],
            n_bootstrap=args.n_bootstrap,
        )
        print(summarise_results(summary))

        # Confusion matrix report
        print("\nConfusion matrix (+Z condition):")
        report = confusion_matrix_report(
            collected["+Z"]["preds"],
            collected["+Z"]["labels"],
            dataset.classnames,
        )
        print(report["text"])

        # Save combined summary
        combined = {
            "model":   args.model,
            "dataset": args.dataset,
            "factors": args.factors,
            "simple_acc": collected["simple"]["accuracy"],
            "plus_z_acc": collected["+Z"]["accuracy"],
            "delta_p_stats": summary,
            "f1_macro_simple": collected["simple"]["f1_macro"],
            "f1_macro_plus_z": collected["+Z"]["f1_macro"],
        }
        summary_path = os.path.join(
            args.save_path, f"{args.save_name}_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
