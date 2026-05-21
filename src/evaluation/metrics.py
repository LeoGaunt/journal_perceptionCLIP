"""
metrics.py — Per-class F1 and confusion matrix utilities.

Works directly on the JSON output of perceptionclip_two_step.py.

Typical usage
-------------
    from src.evaluation.metrics import confusion_matrix_report, per_class_f1
    import json

    with open("results/eurosat_ViT-B16_z.json") as f:
        results = json.load(f)

    report = confusion_matrix_report(
        results["preds"], results["labels"], results["classnames"]
    )
    print(report["text"])
"""

import numpy as np
from typing import List, Optional


def per_class_f1(preds, labels, n_classes: int):
    """
    Compute per-class F1 scores and macro-average F1.

    Parameters
    ----------
    preds, labels : array-like of int
    n_classes : total number of classes

    Returns
    -------
    f1_per_class : np.ndarray, shape (n_classes,)
    f1_macro     : float
    """
    preds  = np.asarray(preds,  dtype=int)
    labels = np.asarray(labels, dtype=int)

    f1 = np.zeros(n_classes)
    for c in range(n_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        denom = 2 * tp + fp + fn
        f1[c] = (2 * tp / denom) if denom > 0 else 0.0

    return f1, float(f1.mean())


def confusion_matrix_report(
    preds,
    labels,
    classnames: Optional[List[str]] = None,
    normalise: bool = True,
):
    """
    Build a confusion matrix and produce a text report.

    Parameters
    ----------
    preds, labels  : array-like of int
    classnames     : list of class name strings (optional)
    normalise      : if True, rows sum to 1.0 (recall per class)

    Returns
    -------
    dict with keys:
        matrix_raw    – raw counts, shape (n_classes, n_classes)
        matrix_norm   – normalised (row = true class), or None if not normalised
        f1_per_class  – np.ndarray
        f1_macro      – float
        classnames    – list of strings
        text          – formatted text report
    """
    preds  = np.asarray(preds,  dtype=int)
    labels = np.asarray(labels, dtype=int)
    n      = max(preds.max(), labels.max()) + 1

    if classnames is None:
        classnames = [str(i) for i in range(n)]

    n_classes = len(classnames)

    # Raw confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, l in zip(preds, labels):
        if 0 <= l < n_classes and 0 <= p < n_classes:
            cm[l, p] += 1

    # Normalised confusion matrix (rows = true class = recall)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    f1, f1_macro = per_class_f1(preds, labels, n_classes)

    # Format text report
    col_w = max(max(len(c) for c in classnames), 8)
    header = " " * col_w + "  " + "  ".join(c[:col_w].rjust(col_w) for c in classnames)
    lines  = [header]
    for i, row_name in enumerate(classnames):
        if normalise:
            row_vals = "  ".join(f"{v:.2f}".rjust(col_w) for v in cm_norm[i])
        else:
            row_vals = "  ".join(str(v).rjust(col_w) for v in cm[i])
        lines.append(f"{row_name[:col_w].rjust(col_w)}  {row_vals}  F1={f1[i]:.3f}")

    lines.append(f"\nMacro F1: {f1_macro:.4f}")

    return {
        "matrix_raw":   cm,
        "matrix_norm":  cm_norm if normalise else None,
        "f1_per_class": f1,
        "f1_macro":     f1_macro,
        "classnames":   classnames,
        "text":         "\n".join(lines),
    }
