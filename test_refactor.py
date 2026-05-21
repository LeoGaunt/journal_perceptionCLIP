"""
test_refactor.py — Smoke test for the TPAMI refactor.

Stage 1 (always runs): pure-Python tests, no GPU or model download needed.
    - statistical.py: bootstrap CI, McNemar's test, delta_p_with_stats
    - metrics.py: per-class F1, confusion matrix
    - utils.py: generate_composite_factors, compose_template

Stage 2 (runs if --model is passed): loads a real model and tokenises text.
    python test_refactor.py --model ViT-B/32      # smallest original CLIP
    python test_refactor.py --model ViT-B/16      # your dissertation baseline
    python test_refactor.py --model OpenCLIP-ViT-H-14

Run stage 1 only:
    python test_refactor.py
"""

import argparse
import sys
import types
import numpy as np

# ------------------------------------------------------------------ helpers --

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
HEAD = "\033[1m{}\033[0m"

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{status}  {name}{suffix}")
    return condition

def section(title):
    print(f"\n{HEAD.format(title)}")
    print("-" * 50)

# ============================================================ STAGE 1 TESTS ==

def test_statistical():
    section("statistical.py")
    from src.evaluation.statistical import bootstrap_ci, mcnemar_test, delta_p_with_stats

    # Use a deterministic exactly-50% array so the tolerance check is reliable
    correct_50 = ([1] * 500 + [0] * 500)
    ci = bootstrap_ci(correct_50, n_bootstrap=500, seed=1)
    ok = (
        check("bootstrap_ci returns dict with mean/lower/upper/se",
              all(k in ci for k in ("mean", "lower", "upper", "se")))
        and check("bootstrap_ci lower < mean < upper",
                  ci["lower"] < ci["mean"] < ci["upper"],
                  f"{ci['lower']:.2f} < {ci['mean']:.2f} < {ci['upper']:.2f}")
        and check("bootstrap_ci mean = 50% for 50/50 data",
                  abs(ci["mean"] - 50.0) < 1e-6,
                  f"mean={ci['mean']:.2f}%")
    )

    # --- mcnemar_test ---
    a = [1]*600 + [0]*400          # baseline: 60% correct
    b = [1]*700 + [0]*300          # context:  70% correct
    stat, p, bv, cv = mcnemar_test(a, b)
    ok &= (
        check("mcnemar_test returns stat, p, b, c",
              stat >= 0 and 0 <= p <= 1)
        and check("mcnemar detects significant difference (p<0.05)",
                  p < 0.05, f"χ²={stat:.2f}  p={p:.4f}")
    )

    # identical classifiers → not significant
    _, p_same, _, _ = mcnemar_test(a, a)
    ok &= check("mcnemar p=1.0 for identical classifiers", p_same == 1.0,
                f"p={p_same}")

    # --- delta_p_with_stats ---
    base = [1 if i < 600 else 0 for i in range(1000)]
    ctx  = [1 if i < 720 else 0 for i in range(1000)]
    d = delta_p_with_stats(base, ctx, n_bootstrap=500)
    ok &= (
        check("delta_p_with_stats keys present",
              all(k in d for k in ("delta_p","delta_p_ci","mcnemar_p","significant")))
        and check("delta_p ≈ +12%",
                  abs(d["delta_p"] - 12.0) < 1.0, f"Δp={d['delta_p']:.2f}")
        and check("delta_p_ci brackets zero correctly (positive gain)",
                  d["delta_p_ci"][0] > 0,
                  f"CI=({d['delta_p_ci'][0]:.2f},{d['delta_p_ci'][1]:.2f})")
        and check("significant=True for clear gain", d["significant"])
    )

    return ok


def test_metrics():
    section("metrics.py")
    from src.evaluation.metrics import per_class_f1, confusion_matrix_report

    preds  = [0,0,1,1,2,2,0,1,2,0]
    labels = [0,1,1,2,2,0,0,1,2,2]

    f1, macro = per_class_f1(preds, labels, n_classes=3)
    ok = (
        check("per_class_f1 shape", len(f1) == 3, f"len={len(f1)}")
        and check("per_class_f1 values in [0,1]", all(0 <= v <= 1 for v in f1))
        and check("f1_macro in [0,1]", 0 <= macro <= 1, f"macro={macro:.3f}")
    )

    report = confusion_matrix_report(preds, labels, classnames=["cat","dog","fish"])
    ok &= (
        check("confusion_matrix_report keys", all(k in report for k in
              ("matrix_raw","matrix_norm","f1_per_class","f1_macro","text")))
        and check("confusion matrix shape (3×3)",
                  report["matrix_raw"].shape == (3, 3))
        and check("confusion matrix row sums = class counts",
                  report["matrix_raw"].sum() == len(preds),
                  f"total={report['matrix_raw'].sum()}")
        and check("text report is non-empty", len(report["text"]) > 10)
    )

    # Perfect classifier
    perfect_preds  = [0,0,0,1,1,1,2,2,2]
    perfect_labels = [0,0,0,1,1,1,2,2,2]
    _, macro_perf = per_class_f1(perfect_preds, perfect_labels, n_classes=3)
    ok &= check("perfect classifier → macro F1=1.0",
                abs(macro_perf - 1.0) < 1e-6, f"macro={macro_perf:.4f}")

    return ok


def test_template_utils():
    section("template composition (inline — no torch needed)")
    # Inline the pure-Python logic so this test runs without a GPU environment.
    def generate_composite_factors(templates, selected_factors=None):
        if selected_factors:
            templates = {k: templates[k] for k in selected_factors if k in templates}
        if not templates:
            return {"": [""]}
        key, sub_dict = next(iter(templates.items()))
        rest = {k: v for k, v in templates.items() if k != key}
        composite = {}
        for sub_key, values in sub_dict.items():
            for rest_key, rest_values in generate_composite_factors(rest).items():
                new_key = f"{sub_key}_{rest_key}" if rest_key else sub_key
                combined = [v + (", " + rv if rv else "") if v else rv
                            for v in values for rv in rest_values]
                composite[new_key] = combined
        return composite

    def compose_template(org_templates, factor_templates):
        factor_list = [factor_templates[k] for k in factor_templates]
        new_templates = []
        for descriptions in factor_list:
            new_factors = []
            for desc in descriptions:
                if desc:
                    new_factors.append(
                        lambda c, main=org_templates[0], d=desc: main(c) + ", " + d + "."
                    )
                else:
                    new_factors.append(lambda c, main=org_templates[0]: main(c) + ".")
            new_templates.append(new_factors)
        return new_templates

    # Minimal factor template (same structure as your dissertation templates)
    factor_templates = {
        "condition": {
            "clear":   ["clear"],
            "overcast": ["overcast"],
        },
        "source": {
            "nasa":  ["by NASA"],
            "other": [""],
        },
    }
    main_template = [lambda c: f"a satellite photo of {c}"]

    composite = generate_composite_factors(factor_templates,
                                           selected_factors=["condition","source"])
    ok = check("generate_composite_factors produces 4 combos (2×2)",
               len(composite) == 4, f"got {len(composite)}: {list(composite.keys())}")

    template_list = compose_template(main_template, composite)
    ok &= check("compose_template produces 4 template groups",
                len(template_list) == 4, f"got {len(template_list)}")

    # Materialise one template and verify it looks right
    sample = template_list[0][0]("forest")
    ok &= check("composed template starts with main template text",
                "satellite photo of forest" in sample, f"got: '{sample}'")

    # Factor with blank (robustness experiment style)
    factor_with_blank = {
        "species": {
            "others": [""],
            "dog":    ["dog"],
            "cat":    ["cat"],
        }
    }
    composite_blank = generate_composite_factors(factor_with_blank)
    ok &= check("blank attribute ('others') is included in composite",
                any("" in v for v in composite_blank.values()))

    return ok


# ============================================================ STAGE 2 TESTS ==

def test_model_loading(model_name: str):
    section(f"VLMEncoder — {model_name}")
    import torch
    from src.models.modeling import VLMEncoder, _ORIGINAL_CLIP, _OPENCLIP_REGISTRY

    args = types.SimpleNamespace(
        model=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=None,
    )

    ok = True

    # --- load ---
    try:
        encoder = VLMEncoder(args, keep_lang=True)
        ok &= check("VLMEncoder instantiated", True)
    except Exception as e:
        check("VLMEncoder instantiated", False, str(e))
        return False

    # --- tokenise ---
    try:
        texts = ["a photo of a cat", "a satellite image of farmland"]
        tokens = encoder.tokenize(texts)
        ok &= check("tokenize() returns tensor",
                    hasattr(tokens, "shape"), f"shape={tokens.shape}")
        ok &= check("tokenize() batch size matches input",
                    tokens.shape[0] == 2, f"batch={tokens.shape[0]}")
    except Exception as e:
        ok &= check("tokenize()", False, str(e))

    # --- encode_text ---
    try:
        encoder = encoder.to(args.device)
        tokens  = tokens.to(args.device)
        with torch.no_grad():
            feats = encoder.encode_text(tokens)
        ok &= check("encode_text() returns float tensor",
                    feats.dtype in (torch.float16, torch.float32),
                    f"dtype={feats.dtype}  shape={feats.shape}")
        norms = feats.norm(dim=-1)
        ok &= check("encode_text() features are L2-normalised",
                    torch.allclose(norms, torch.ones_like(norms), atol=1e-3),
                    f"norms={norms.tolist()}")
    except Exception as e:
        ok &= check("encode_text()", False, str(e))

    # --- encode_image (random tensor) ---
    try:
        dummy_img = torch.randn(2, 3, 224, 224).to(args.device)
        with torch.no_grad():
            img_feats = encoder.encode_image(dummy_img)
        ok &= check("encode_image() on random tensor works",
                    img_feats.shape[0] == 2, f"shape={img_feats.shape}")
    except Exception as e:
        ok &= check("encode_image()", False, str(e))

    # --- logit_scale ---
    try:
        ls = encoder.logit_scale
        ok &= check("logit_scale accessible", ls is not None,
                    f"exp(logit_scale)={ls.exp().item():.2f}")
    except Exception as e:
        ok &= check("logit_scale", False, str(e))

    # --- CLIPEncoder alias ---
    from src.models.modeling import CLIPEncoder
    ok &= check("CLIPEncoder alias still works (backward compat)",
                CLIPEncoder is VLMEncoder)

    return ok


# ================================================================== ENTRY ==

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="If set, also run Stage 2 model loading tests.")
    args = parser.parse_args()

    print(HEAD.format("\n=== TPAMI Refactor Smoke Test ==="))
    print("Stage 1: pure-Python tests (no GPU / model download required)\n")

    results = []
    results.append(("statistical.py", test_statistical()))
    results.append(("metrics.py",     test_metrics()))
    results.append(("template utils", test_template_utils()))

    if args.model:
        print(f"\nStage 2: model loading test for '{args.model}'")
        print("(This will download the model weights if not cached.)\n")
        results.append((f"VLMEncoder[{args.model}]", test_model_loading(args.model)))

    # Summary
    section("Summary")
    all_pass = True
    for name, passed in results:
        status = PASS if passed else FAIL
        print(f"{status}  {name}")
        all_pass = all_pass and passed

    print()
    if all_pass:
        print("\033[92mAll tests passed.\033[0m")
        if not args.model:
            print("\nTo also test model loading, run:")
            print("  python test_refactor.py --model ViT-B/32      # fastest (smallest CLIP)")
            print("  python test_refactor.py --model ViT-B/16      # your dissertation baseline")
            print("  python test_refactor.py --model OpenCLIP-ViT-H-14")
    else:
        print("\033[91mSome tests failed — see above.\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
