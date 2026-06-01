#!/bin/bash
# importers/imagenet.sh
#
# Downloads and prepares the ILSVRC2012 validation split (and optionally train)
# from image-net.org using your registered username and access key.
#
# Usage:
#   bash importers/imagenet.sh <username> <access_key>           # val only
#   bash importers/imagenet.sh <username> <access_key> --train   # val + train
#
# Or via environment variables (safer in shared/logged environments):
#   export IMAGENET_USER=your_username
#   export IMAGENET_KEY=your_access_key
#   bash importers/imagenet.sh
#
# Output layout (matches src/datasets/imagenet.py):
#   datasets/data/imagenet/
#   └── val/
#       ├── n01440764/   (tench, 50 images)
#       ├── n01443537/   (goldfish, 50 images)
#       └── ...          (1000 synset dirs, 50k images total)
#
# MD5 checksums (image-net.org official):
#   val:   29b22e2961454d5413ddabcf34fc5622
#   train: 1d675b47d978889d74fa0da5fadfb00e

set -e

# ── Credentials ─────────────────────────────────────────────────────────────
IMAGENET_USER="${IMAGENET_USER:-$1}"
IMAGENET_KEY="${IMAGENET_KEY:-$2}"
DOWNLOAD_TRAIN="${3:-}"

if [ -z "$IMAGENET_USER" ] || [ -z "$IMAGENET_KEY" ]; then
    echo "ERROR: credentials required."
    echo ""
    echo "Usage:   bash importers/imagenet.sh <username> <access_key>"
    echo "         bash importers/imagenet.sh <username> <access_key> --train"
    echo ""
    echo "Or set environment variables:"
    echo "         export IMAGENET_USER=... && export IMAGENET_KEY=..."
    exit 1
fi

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="$ROOT_DIR/datasets/data/imagenet"
VAL_DIR="$DATASET_DIR/val"
TRAIN_DIR="$DATASET_DIR/train"
TMP_DIR="$DATASET_DIR/.tmp"

VAL_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
TRAIN_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
SYNSET_LABELS_URL="https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt"

VAL_MD5="29b22e2961454d5413ddabcf34fc5622"
TRAIN_MD5="1d675b47d978889d74fa0da5fadfb00e"

mkdir -p "$DATASET_DIR" "$TMP_DIR"

# ── Helper: verify MD5 ───────────────────────────────────────────────────────
check_md5() {
    local file="$1"
    local expected="$2"
    local actual
    if command -v md5sum &>/dev/null; then
        actual=$(md5sum "$file" | awk '{print $1}')
    else
        actual=$(md5 -q "$file")   # macOS
    fi
    if [ "$actual" != "$expected" ]; then
        echo "  MD5 MISMATCH for $file"
        echo "  Expected: $expected"
        echo "  Got:      $actual"
        exit 1
    fi
    echo "  MD5 OK"
}

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION SET
# ═══════════════════════════════════════════════════════════════════════════════
if [ -d "$VAL_DIR" ] && [ "$(ls "$VAL_DIR" | wc -l)" -ge 1000 ]; then
    echo "Val already set up — skipping download (delete $VAL_DIR to re-run)"
else
    echo "──────────────────────────────────────────"
    echo " Downloading validation set (6.3 GB)..."
    echo "──────────────────────────────────────────"
    wget \
        --user="$IMAGENET_USER" \
        --password="$IMAGENET_KEY" \
        --progress=bar:force \
        --tries=3 \
        --continue \
        -O "$TMP_DIR/ILSVRC2012_img_val.tar" \
        "$VAL_URL"

    echo "Verifying MD5..."
    check_md5 "$TMP_DIR/ILSVRC2012_img_val.tar" "$VAL_MD5"

    echo "Extracting 50,000 images..."
    mkdir -p "$VAL_DIR"
    tar -xf "$TMP_DIR/ILSVRC2012_img_val.tar" -C "$VAL_DIR"
    rm "$TMP_DIR/ILSVRC2012_img_val.tar"

    echo "Downloading synset labels..."
    curl -sL "$SYNSET_LABELS_URL" -o "$TMP_DIR/val_labels.txt"

    echo "Organising images into per-class folders..."
    VAL_DIR="$VAL_DIR" LABELS_FILE="$TMP_DIR/val_labels.txt" python3 - <<'PYEOF'
import os, shutil

val_dir     = os.environ["VAL_DIR"]
labels_file = os.environ["LABELS_FILE"]

images = sorted(f for f in os.listdir(val_dir) if f.endswith(".JPEG"))
labels = open(labels_file).read().strip().split("\n")

assert len(images) == 50000, f"Expected 50000 images, found {len(images)}"
assert len(labels) == 50000, f"Expected 50000 labels, found {len(labels)}"

for img, synset in zip(images, labels):
    dst = os.path.join(val_dir, synset)
    os.makedirs(dst, exist_ok=True)
    shutil.move(os.path.join(val_dir, img), os.path.join(dst, img))

classes = len(set(labels))
print(f"  Done — {classes} classes, 50 images each")
PYEOF

    rm -f "$TMP_DIR/val_labels.txt"
    echo "✅ Val ready: $VAL_DIR"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN SET (optional — only needed if you add fine-tuning experiments)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$DOWNLOAD_TRAIN" = "--train" ]; then
    if [ -d "$TRAIN_DIR" ] && [ "$(ls "$TRAIN_DIR" | wc -l)" -ge 1000 ]; then
        echo "Train already set up — skipping download"
    else
        echo ""
        echo "──────────────────────────────────────────"
        echo " Downloading training set (138 GB)..."
        echo " This will take a long time."
        echo "──────────────────────────────────────────"
        wget \
            --user="$IMAGENET_USER" \
            --password="$IMAGENET_KEY" \
            --progress=bar:force \
            --tries=3 \
            --continue \
            -O "$TMP_DIR/ILSVRC2012_img_train.tar" \
            "$TRAIN_URL"

        echo "Verifying MD5..."
        check_md5 "$TMP_DIR/ILSVRC2012_img_train.tar" "$TRAIN_MD5"

        echo "Extracting train archives (1000 synset tars inside the outer tar)..."
        mkdir -p "$TRAIN_DIR"
        tar -xf "$TMP_DIR/ILSVRC2012_img_train.tar" -C "$TRAIN_DIR"
        rm "$TMP_DIR/ILSVRC2012_img_train.tar"

        echo "Extracting per-class tars..."
        find "$TRAIN_DIR" -name "*.tar" | while read -r synset_tar; do
            synset=$(basename "$synset_tar" .tar)
            mkdir -p "$TRAIN_DIR/$synset"
            tar -xf "$synset_tar" -C "$TRAIN_DIR/$synset"
            rm "$synset_tar"
        done

        echo "✅ Train ready: $TRAIN_DIR"
    fi
fi

# ── Cleanup ──────────────────────────────────────────────────────────────────
rmdir "$TMP_DIR" 2>/dev/null || true

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " ImageNet setup complete"
echo "══════════════════════════════════════════"
echo " Val:   $(ls "$VAL_DIR" | wc -l | tr -d ' ') class folders → $VAL_DIR"
[ -d "$TRAIN_DIR" ] && echo " Train: $(ls "$TRAIN_DIR" | wc -l | tr -d ' ') class folders → $TRAIN_DIR"
echo ""
echo " Verify the loader:"
echo "   python check_datasets.py"
echo ""
echo " Run an experiment:"
echo "   python run_experiment.py --dataset ImageNet --model ViT-B/16 \\"
echo "     --data_location ./datasets/data/imagenet"
