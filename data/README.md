# `data/` — acquisition instructions only

**No actual data is checked into this repo.** This directory contains
instructions for obtaining the two datasets we use.

## 1. CIFAKE (training + sealed test)

CIFAKE is a 120k-image dataset of 32×32 photographs (real CIFAR-10 + Stable
Diffusion 1.4 generated synthetics).

### Layout expected by every script in this repo

```
data/cifake/
├── train/REAL/   (50,000 .jpg, CIFAR-10 originals)
├── train/FAKE/   (50,000 .jpg, SD-1.4 generated)
├── test/REAL/    (10,000 .jpg)
└── test/FAKE/    (10,000 .jpg)
```

### Acquisition (Kaggle)

```bash
# Install the Kaggle CLI and drop your API token in ~/.kaggle/kaggle.json
pip install kaggle
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
unzip cifake-real-and-ai-generated-synthetic-images.zip -d data/cifake/
```

Then verify:

```bash
bash scripts/setup_data.sh
```

### Alternative location

If you keep CIFAKE elsewhere on disk, point `CIFAKE_DATA` at it:

```bash
export CIFAKE_DATA=/path/to/cifake
bash scripts/setup_data.sh
```

## 2. Cross-generator OOD set (sd-turbo)

A controlled OOD test set: 2000 images (1000 sd-turbo generated FAKEs +
1000 CIFAR-10 REALs, all 32×32 JPEG, both re-encoded through CIFAKE's
exact quantization tables to control compression confound).

See `data/ood_sdturbo/README.md` for regeneration instructions. You can
also ship the OOD set with the repo if you don't mind ~5 MB of JPEGs; it's
gitignored by default.

The methodology is documented in `docs/methodology/ood_methodology.md`.
