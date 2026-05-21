"""
statistical.py — Bootstrap confidence intervals and McNemar's test.

All functions operate on plain Python lists or numpy arrays of per-image
correct/incorrect flags (1/0), which are saved in the JSON output of
perceptionclip_two_step.py.

Typical usage
-------------
    from src.evaluation.statistical import bootstrap_ci, mcnemar_test, delta_p_with_stats

    # Load saved per-image results
    with open("results/simple.json") as f:  simple = json.load(f)
    with open("results/plus_z.json") as f:  plus_z = json.load(f)

    ci_simple = bootstrap_ci(simple["correct"])
    ci_plusZ  = bootstrap_ci(plus_z["correct"])

    stat, p = mcnemar_test(simple["correct"], plus_z["correct"])

    summary = delta_p_with_stats(simple["correct"], plus_z["correct"])
    print(summary)
"""

import numpy as np
from scipy.stats import chi2


def bootstrap_ci(correct, n_bootstrap: int = 2000, alpha: float = 0.05, seed: int = 42):
    """
    Compute a bootstrap confidence interval for classification accuracy.

    Parameters
    ----------
    correct : array-like of int (1 = correct, 0 = wrong), length N
    n_bootstrap : number of resampling iterations
    alpha : significance level (default 0.05 → 95% CI)
    seed : random seed for reproducibility

    Returns
    -------
    dict with keys:
        mean   – point estimate (%)
        lower  – lower bound of CI (%)
        upper  – upper bound of CI (%)
        se     – bootstrap standard error (%)
    """
    rng     = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    n       = len(correct)

    # Point estimate
    mean_acc = correct.mean() * 100.0

    # Bootstrap distribution
    boot_accs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        indices      = rng.integers(0, n, size=n)
        boot_accs[i] = correct[indices].mean() * 100.0

    lo = np.percentile(boot_accs, 100 * (alpha / 2))
    hi = np.percentile(boot_accs, 100 * (1 - alpha / 2))
    se = boot_accs.std(ddof=1)

    return {"mean": mean_acc, "lower": lo, "upper": hi, "se": se}


def mcnemar_test(correct_a, correct_b, continuity_correction: bool = True):
    """
    McNemar's test for paired nominal data.

    Tests whether the disagreements between two classifiers (A and B)
    evaluated on the same images are symmetric — i.e. whether A and B
    differ significantly.

    Parameters
    ----------
    correct_a, correct_b : array-like of int (1 = correct, 0 = wrong)
        Per-image correctness from condition A and condition B.
        Must be the same length and correspond to the same images.
    continuity_correction : apply Edwards's continuity correction (recommended
        when b+c < 25).

    Returns
    -------
    stat : float  — chi-squared statistic
    p    : float  — p-value (two-sided)
    b    : int    — A correct, B wrong
    c    : int    — A wrong, B correct
    """
    a = np.asarray(correct_a, dtype=int)
    b_arr = np.asarray(correct_b, dtype=int)

    if len(a) != len(b_arr):
        raise ValueError("correct_a and correct_b must have the same length.")

    # Contingency cells
    b = int(((a == 1) & (b_arr == 0)).sum())   # A right, B wrong
    c = int(((a == 0) & (b_arr == 1)).sum())   # A wrong, B right

    if continuity_correction:
        stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    else:
        stat = (b - c) ** 2 / (b + c) if (b + c) > 0 else 0.0

    # One degree of freedom chi-squared
    p = 1.0 - chi2.cdf(stat, df=1)

    return stat, p, b, c


def delta_p_with_stats(
    correct_baseline,
    correct_context,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
):
    """
    Compute Δp (context gain) with bootstrap CI and McNemar significance test.

    Parameters
    ----------
    correct_baseline : per-image correct flags for Simple/Domain condition
    correct_context  : per-image correct flags for +Z condition

    Returns
    -------
    dict with full statistical summary, ready for inclusion in results tables.
    """
    ci_base = bootstrap_ci(correct_baseline, n_bootstrap=n_bootstrap,
                           alpha=alpha, seed=seed)
    ci_ctx  = bootstrap_ci(correct_context,  n_bootstrap=n_bootstrap,
                           alpha=alpha, seed=seed + 1)

    delta = ci_ctx["mean"] - ci_base["mean"]

    # Bootstrap CI on Δp directly
    rng  = np.random.default_rng(seed + 2)
    base = np.asarray(correct_baseline, dtype=float)
    ctx  = np.asarray(correct_context,  dtype=float)
    n    = len(base)
    boot_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = (ctx[idx].mean() - base[idx].mean()) * 100.0

    delta_lo = np.percentile(boot_deltas, 100 * (alpha / 2))
    delta_hi = np.percentile(boot_deltas, 100 * (1 - alpha / 2))

    stat, p, b, c = mcnemar_test(correct_baseline, correct_context)

    return {
        "baseline_acc":   ci_base["mean"],
        "baseline_ci":    (ci_base["lower"], ci_base["upper"]),
        "context_acc":    ci_ctx["mean"],
        "context_ci":     (ci_ctx["lower"],  ci_ctx["upper"]),
        "delta_p":        delta,
        "delta_p_ci":     (delta_lo, delta_hi),
        "delta_p_se":     float(boot_deltas.std(ddof=1)),
        "mcnemar_stat":   stat,
        "mcnemar_p":      p,
        "significant":    p < alpha,
        "mcnemar_b":      b,   # baseline right, context wrong
        "mcnemar_c":      c,   # baseline wrong, context right
        "alpha":          alpha,
    }


def summarise_results(results_dict: dict, alpha: float = 0.05) -> str:
    """
    Format a delta_p_with_stats result dict as a human-readable string.
    """
    r = results_dict
    sig = "✓ significant" if r["significant"] else "✗ not significant"
    return (
        f"Baseline:  {r['baseline_acc']:.2f}%  "
        f"[{r['baseline_ci'][0]:.2f}, {r['baseline_ci'][1]:.2f}]\n"
        f"Context:   {r['context_acc']:.2f}%  "
        f"[{r['context_ci'][0]:.2f}, {r['context_ci'][1]:.2f}]\n"
        f"Δp:        {r['delta_p']:+.2f}%  "
        f"[{r['delta_p_ci'][0]:+.2f}, {r['delta_p_ci'][1]:+.2f}]  "
        f"(se={r['delta_p_se']:.2f})\n"
        f"McNemar:   χ²={r['mcnemar_stat']:.3f}  p={r['mcnemar_p']:.4f}  "
        f"b={r['mcnemar_b']}  c={r['mcnemar_c']}  {sig} (α={alpha})"
    )
