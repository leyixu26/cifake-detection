#!/usr/bin/env bash
# Verify/acquire the CIFAKE dataset.
# CIFAKE expects this layout under data/cifake/:
#   data/cifake/train/REAL  (~50000 jpg)
#   data/cifake/train/FAKE  (~50000 jpg)
#   data/cifake/test/REAL   (~10000 jpg)
#   data/cifake/test/FAKE   (~10000 jpg)
# Override with `CIFAKE_DATA=/abs/path/to/cifake`.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${CIFAKE_DATA:-$REPO_ROOT/data/cifake}"

echo "=== checking CIFAKE at $DATA_ROOT ==="
ok=true
for split in train test; do
  for cls in REAL FAKE; do
    d="$DATA_ROOT/$split/$cls"
    if [ -d "$d" ]; then
      n=$(ls "$d" 2>/dev/null | wc -l | tr -d ' ')
      echo "  $split/$cls: $n images"
    else
      echo "  MISSING: $d"
      ok=false
    fi
  done
done

if ! $ok; then
  cat <<'INSTR'

CIFAKE not found. Choose one path to obtain it:

  Option A — Kaggle CLI (recommended):
    pip install kaggle
    # Drop your kaggle.json into ~/.kaggle/ first (kaggle.com -> Profile -> Create New API Token)
    kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
    unzip cifake-real-and-ai-generated-synthetic-images.zip -d data/cifake/

  Option B — manual download:
    https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
    Extract train/ and test/ under data/cifake/

After acquisition, rerun this script to confirm.
INSTR
  exit 1
fi

echo "=== CIFAKE OK ==="
