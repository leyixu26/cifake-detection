# `data/ood_sdturbo/` — sd-turbo cross-generator OOD set

A controlled cross-generator OOD test set built specifically for this project.
See `docs/methodology/ood_methodology.md` for the full design rationale.

## Quick facts

| | |
|---|---|
| Real side | 1000 CIFAR-10 originals (sampled from CIFAKE `test/REAL`) |
| Fake side | 1000 sd-turbo generated images (100 per CIFAR-10 class) |
| Resolution | 32×32 RGB JPEG |
| JPEG quant tables | identical to CIFAKE (luma 1858, chroma 2780) |
| The variable that differs from CIFAKE | the generator process: sd-turbo (single-step distilled diffusion) vs SD-1.4 |

## How to regenerate

```bash
python scripts/generate_ood.py
```

This:
1. Samples 1000 REAL images from `data/cifake/test/REAL/` into `data/ood_sdturbo/REAL/`
2. Loads `stabilityai/sd-turbo` via diffusers, generates 100 images per
   CIFAR-10 class prompt at 256×256, Lanczos-resizes to 32×32, saves as JPEG
3. Re-encodes BOTH classes through CIFAKE's exact JPEG quantization tables
   to control the compression confound

Total runtime: ~10 min on MPS, ~5 min on CUDA. Disk: ~5 MB.

## Layout expected by every script in this repo

```
data/ood_sdturbo/
├── REAL/   1000 .jpg (CIFAR-10 originals, qt-matched)
└── FAKE/   1000 .jpg (sd-turbo, qt-matched)
```

The names are alphabetised so that the harness reliably orders REAL before
FAKE (labels: REAL=0, FAKE=1).

## Override location

```bash
export CIFAKE_OOD=/some/other/path/ood_sdturbo
python scripts/evaluate.py --model resnet18_nathan   # picks it up
```
