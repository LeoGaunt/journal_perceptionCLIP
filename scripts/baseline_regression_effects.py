"""Baseline accuracy vs contextual-conditioning gain, with checkpoint and dataset effects.

Revision analysis for the baseline-accuracy subsection (\\label{subsec:baseline}).
The 56 checkpoint-dataset pairs (7 checkpoints x 8 datasets, COCO-Transport
excluded) reuse the same checkpoints and datasets, so the pooled OLS p-value
treats dependent observations as independent. This script reports:

  1. the pooled OLS fit drawn in the figure (descriptive only);
  2. OLS with checkpoint and dataset fixed effects (the model reported in the text);
  3. robustness: standard errors clustered by dataset / by checkpoint, and a
     model with crossed random intercepts for checkpoint and dataset;
  4. a two-way bootstrap of the pooled slope (resampling checkpoints and datasets);
  5. diagnostics quoted in the text (variance share between datasets, pooled
     slope without Fish and EuroSAT).

Inputs are transcribed from the paper: Delta p from Table tab:deltap and Simple
accuracy from Tables tab:accuracies-a/b (both rounded to 2 dp). Unrounded values
from the released result files can be substituted; rounding does not change any
reported figure at the precision quoted.

Requires: numpy, pandas, scipy, statsmodels.

Optional small-sample check in R (lme4 + lmerTest, Kenward-Roger df):
    library(lmerTest)
    m <- lmer(dp ~ base + (1|ckpt) + (1|ds), data = d)   # d = the 56 rows below
    summary(m, ddf = "Kenward-Roger")
    # slope -0.065, SE 0.033, df 10.8, p = 0.075, 95% CI -0.137 to 0.008
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

CKPTS = ["OC-B16", "OC-L14", "OC-H14", "CnXt-B", "SigLIP", "MC-B16", "DC-B16"]

# Delta p (pp), Simple -> +Z; rows = dataset, columns = CKPTS (Table tab:deltap)
DELTA_P = {
    "Fish":        [7.01, 2.50, 9.64, 2.82, 4.72, 1.32, 3.30],
    "EuroSAT":     [1.83, 11.47, 10.15, 5.53, 7.49, -1.07, 6.66],
    "UCM":         [-0.62, 4.95, 0.90, 1.24, 0.10, 0.00, 4.71],
    "Sea Animals": [0.94, 1.41, 1.41, 0.55, 1.51, 2.28, 2.55],
    "Flowers102":  [2.70, 1.12, 0.70, 0.41, 2.83, 2.39, -0.10],
    "OxfordPets":  [0.71, 2.15, 2.48, 1.96, 0.98, 1.80, 2.02],
    "CUB200":      [-0.67, -0.24, -0.19, -0.66, -0.78, 0.76, -0.35],
    "ImageNet":    [0.05, -0.07, 0.36, -0.17, -0.01, 0.17, 0.11],
}

# Simple-template top-1 accuracy (%); rows = dataset, columns = CKPTS
# (Tables tab:accuracies-a/b)
SIMPLE = {
    "Fish":        [28.87, 31.44, 39.37, 28.50, 47.65, 35.16, 40.67],
    "EuroSAT":     [49.64, 53.27, 63.00, 48.74, 36.20, 56.29, 47.03],
    "UCM":         [70.19, 70.43, 77.43, 71.05, 62.71, 68.57, 67.62],
    "Sea Animals": [78.17, 84.03, 86.30, 79.61, 84.74, 82.64, 83.44],
    "Flowers102":  [69.56, 75.57, 80.09, 69.57, 83.64, 71.20, 75.57],
    "OxfordPets":  [89.34, 91.41, 91.82, 90.11, 93.16, 88.53, 90.38],
    "CUB200":      [70.47, 75.53, 81.96, 71.95, 62.29, 69.74, 78.46],
    "ImageNet":    [69.26, 74.56, 76.82, 70.27, 75.68, 69.97, 72.98],
}

N_BOOT = 2000
SEED = 0


def load() -> pd.DataFrame:
    rows = [
        dict(ckpt=c, ds=d, base=SIMPLE[d][i], dp=DELTA_P[d][i])
        for d in DELTA_P
        for i, c in enumerate(CKPTS)
    ]
    return pd.DataFrame(rows)


def slope_ci(fit, name="base"):
    lo, hi = fit.conf_int().loc[name]
    return fit.params[name], lo, hi, fit.pvalues[name]


def clustered(formula, df, group):
    """CR1 cluster-robust SEs with t(G-1) reference distribution."""
    codes = pd.factorize(df[group])[0]
    fit = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": codes})
    g = df[group].nunique()
    b, se = fit.params["base"], fit.bse["base"]
    q = stats.t.ppf(0.975, g - 1)
    return b, b - q * se, b + q * se, 2 * stats.t.sf(abs(b / se), g - 1)


def crossed_random_intercepts(df):
    """dp ~ base + (1|ckpt) + (1|ds), REML; Wald CI."""
    d = df.assign(one=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(
            "dp ~ base", d, groups="one", re_formula="0",
            vc_formula={"ckpt": "0 + C(ckpt)", "ds": "0 + C(ds)"},
        ).fit(reml=True, method=["lbfgs"])
    b, se = fit.params["base"], fit.bse["base"]
    return b, b - 1.96 * se, b + 1.96 * se, 2 * stats.norm.sf(abs(b / se))


def pooled_slope(x, y):
    return np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)


def two_way_bootstrap(df, n_boot=N_BOOT, seed=SEED):
    """Resample checkpoints and datasets independently, keep all their pairs."""
    rng = np.random.default_rng(seed)
    tab = df.set_index(["ckpt", "ds"])
    ck, ds = df.ckpt.unique(), df.ds.unique()
    out = []
    for _ in range(n_boot):
        c, d = rng.choice(ck, len(ck)), rng.choice(ds, len(ds))
        sub = tab.loc[[(a, b) for a in c for b in d]]
        if sub.base.nunique() > 1:
            out.append(pooled_slope(sub.base.values, sub.dp.values))
    return np.percentile(out, [2.5, 97.5])


def fmt(label, b, lo, hi, p=None):
    ptxt = "" if p is None else f"  p = {p:.3g}"
    print(f"  {label:<44s} slope {b:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]{ptxt}")


def main():
    df = load()
    assert len(df) == 56 and df.ckpt.nunique() == 7 and df.ds.nunique() == 8

    print("1. Pooled OLS (figure line; descriptive only)")
    pooled = smf.ols("dp ~ base", df).fit()
    r = np.corrcoef(df.base, df.dp)[0, 1]
    print(f"  slope {pooled.params['base']:+.4f}, intercept {pooled.params['Intercept']:.2f}, "
          f"r = {r:.3f}, R^2 = {pooled.rsquared:.3f}, residual SD = {np.sqrt(pooled.scale):.2f}")
    fmt("treating 56 pairs as independent", *slope_ci(pooled))

    print("\n2. Checkpoint and dataset fixed effects (reported in text)")
    fe_formula = "dp ~ base + C(ckpt) + C(ds)"
    fmt("OLS + checkpoint FE + dataset FE", *slope_ci(smf.ols(fe_formula, df).fit()))

    print("\n3. Robustness (all intervals should include zero)")
    fmt("  same, SEs clustered by dataset (t, 7 df)", *clustered(fe_formula, df, "ds"))
    fmt("  same, SEs clustered by checkpoint (t, 6 df)", *clustered(fe_formula, df, "ckpt"))
    fmt("  crossed random intercepts (REML, Wald)", *crossed_random_intercepts(df))

    print("\n4. Two-way bootstrap of the pooled slope (figure caption)")
    lo, hi = two_way_bootstrap(df)
    print(f"  {N_BOOT} resamples of checkpoints and datasets: 95% CI [{lo:+.3f}, {hi:+.3f}]")

    print("\n5. Diagnostics quoted in the text")
    tot = ((df.base - df.base.mean()) ** 2).sum()
    btw = ((df.groupby("ds").base.transform("mean") - df.base.mean()) ** 2).sum()
    print(f"  share of baseline-accuracy variance between datasets: {btw / tot:.0%}")
    sub = df[~df.ds.isin(["Fish", "EuroSAT"])]
    s = smf.ols("dp ~ base", sub).fit().params["base"]
    print(f"  pooled slope without Fish and EuroSAT (n = {len(sub)}): {s:+.3f}")

    print("\n(context, not reported) Regression on the 8 dataset means")
    means = df.groupby("ds")[["base", "dp"]].mean()
    r, p = stats.pearsonr(means.base, means.dp)
    print(f"  all 8: r = {r:+.2f}, p = {p:.3f}")
    for drop in ["Fish", "EuroSAT"]:
        r, p = stats.pearsonr(means.drop(drop).base, means.drop(drop).dp)
        print(f"  without {drop}: r = {r:+.2f}, p = {p:.3f}")


if __name__ == "__main__":
    main()
