"""
check_datasets.py — Verify every dataset loads correctly before running experiments.

Instantiates each dataset class, pulls one batch from the test loader, and
reports success or failure with a clear summary. No model weights are downloaded
— only the data loading pipeline is tested.

Usage
-----
    python check_datasets.py                         # use default ./datasets/data
    python check_datasets.py --root ./datasets/data  # explicit path

Run this before starting any experiment group to catch path or file
permission issues early.
"""

import argparse
import os
import sys
import traceback
import time

import torchvision.transforms as T
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Minimal transform — no model needed, just checks data can be loaded
# ---------------------------------------------------------------------------
_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# ---------------------------------------------------------------------------
# Dataset configs: name → (class_name, location, extra_kwargs)
# ---------------------------------------------------------------------------
def _dataset_configs(root: str):
    return [
        {
            "name":       "EuroSAT",
            "class":      "EuroSAT",
            "location":   root,
            "note":       "torchvision managed — expects eurosat/ subfolder",
        },
        {
            "name":       "Flowers102",
            "class":      "Flowers102",
            "location":   root,
            "note":       "torchvision managed — expects flowers-102/ subfolder",
        },
        {
            "name":       "OxfordPets",
            "class":      "OxfordPets",
            "location":   root,
            "note":       "torchvision managed — expects oxford-iiit-pet/ subfolder",
        },
        {
            "name":       "CUB200",
            "class":      "CUB200",
            "location":   root,
            "note":       "ImageFolder — expects CUB_200_2011/ subfolder",
        },
        {
            "name":       "Fish",
            "class":      "Fish",
            "location":   os.path.join(root, "fish"),
            "note":       "ImageFolder — location IS the fish folder",
        },
        {
            "name":       "Marine (Sea Animals)",
            "class":      "Marine",
            "location":   os.path.join(root, "marine"),
            "note":       "ImageFolder — location IS the marine folder",
        },
        {
            "name":       "UCM",
            "class":      "UCM",
            "location":   os.path.join(root, "ucm"),
            "note":       "ImageFolder — location IS the ucm folder",
        },
        {
            "name":       "Dissertation (Curated COCO Transport)",
            "class":      "Dissertation",
            "location":   os.path.join(root, "dissertation"),
            "note":       "ImageFolder — location IS the dissertation folder",
        },
    ]


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):  return f"{GREEN}  PASS{RESET}  {msg}"
def fail(msg):return f"{RED}  FAIL{RESET}  {msg}"
def warn(msg):return f"{YELLOW}  WARN{RESET}  {msg}"


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def check_dataset(cfg: dict) -> dict:
    """Try to instantiate and load one batch. Returns a result dict."""
    import src.datasets as ds_module

    name       = cfg["name"]
    class_name = cfg["class"]
    location   = cfg["location"]

    result = {
        "name":       name,
        "class":      class_name,
        "location":   location,
        "note":       cfg.get("note", ""),
        "success":    False,
        "n_images":   None,
        "n_classes":  None,
        "classnames": None,
        "error":      None,
        "elapsed":    None,
    }

    t0 = time.time()
    try:
        # 1. Instantiate
        cls     = getattr(ds_module, class_name)
        dataset = cls(
            preprocess=_PREPROCESS,
            location=location,
            batch_size=8,
            num_workers=0,   # 0 workers for checker — avoids multiprocessing issues
        )

        # 2. Pull one batch from test loader
        batch = next(iter(dataset.test_loader))

        # 3. Basic sanity checks
        if isinstance(batch, dict):
            images = batch["images"]
            labels = batch["labels"]
        else:
            images, labels = batch[0], batch[1]

        assert images.shape[0] > 0,    "Empty batch returned"
        assert images.ndim == 4,        f"Expected 4D image tensor, got {images.ndim}D"
        assert images.shape[1] == 3,   f"Expected 3 channels, got {images.shape[1]}"

        # 4. Count total images
        n_images = len(dataset.test_dataset)

        result["success"]    = True
        result["n_images"]   = n_images
        result["n_classes"]  = len(dataset.classnames)
        result["classnames"] = dataset.classnames
        result["batch_shape"]= tuple(images.shape)

    except Exception as e:
        result["error"] = traceback.format_exc()

    result["elapsed"] = time.time() - t0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./datasets/data",
                        help="Root directory containing all dataset subfolders")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full tracebacks for failed datasets")
    args = parser.parse_args()

    root = os.path.abspath(args.root)

    print(f"\n{BOLD}Dataset Loader Checker{RESET}")
    print(f"Root: {root}")
    print("="*65)

    configs = _dataset_configs(root)
    results = []

    for cfg in configs:
        name = cfg["name"]
        print(f"\nChecking {BOLD}{name}{RESET} ...", end=" ", flush=True)
        r = check_dataset(cfg)
        results.append(r)

        if r["success"]:
            print(f"{GREEN}OK{RESET}  ({r['elapsed']:.1f}s)")
            print(f"         images={r['n_images']}  classes={r['n_classes']}  "
                  f"batch={r['batch_shape']}")
        else:
            print(f"{RED}FAILED{RESET}  ({r['elapsed']:.1f}s)")
            first_line = r["error"].strip().split("\n")[-1] if r["error"] else "unknown error"
            print(f"         {RED}{first_line}{RESET}")
            if args.verbose and r["error"]:
                print()
                for line in r["error"].strip().split("\n"):
                    print(f"         {line}")

    # --- Summary ---
    passed = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'='*65}")
    print(f"{BOLD}Summary{RESET}: {len(passed)}/{len(results)} datasets loaded successfully\n")

    for r in passed:
        print(ok(f"{r['name']:<40} {r['n_images']:>6} images  {r['n_classes']:>3} classes"))

    if failed:
        print()
        for r in failed:
            first_line = r["error"].strip().split("\n")[-1] if r["error"] else "unknown"
            print(fail(f"{r['name']:<40} {first_line}"))
        print(f"\n{YELLOW}Run with --verbose for full tracebacks.{RESET}")
        print(f"\nCommon fixes:")
        print(f"  Path not found  → check datasets/data/ folder names match expected paths")
        print(f"  Permission error → check file permissions on dataset folders")
        print(f"  Missing files   → re-run the relevant data downloader script")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}All datasets ready. You can now run the experiment scripts.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
