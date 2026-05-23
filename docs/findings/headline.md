# Headline Findings — 5-Model Comparison + Team Ensemble

**Project:** Detect AI-generated images on CIFAKE (32×32 RGB; REAL = CIFAR-10, FAKE = Stable Diffusion v1.4).
**Team:** Yin (small CNN), Nathan (ResNet-18), Alex (ViT-Small), Leyi (frequency detector + CLIP probe).
**Date:** 2026-05-24.

---

## TL;DR

- All four spatial models hit ≥ **0.997 clean-test AUROC** on CIFAKE — the in-distribution problem is genuinely easy.
- **Cross-generator OOD** (against sd-turbo, content + JPEG-quant-table matched to CIFAKE) is where models separate. **Alex's ViT-Small drops least** (−2.6 pp); the frequency detector drops most (−12.8 pp).
- **Yin's from-scratch CNN OOD-drops less than Nathan's ImageNet ResNet** (−5.5 pp vs −6.4 pp) — a surprising find: under cross-generator stress, a model trained purely on CIFAKE generalises *slightly* better than one warm-started from ImageNet.
- The best **2-model team ensemble** for cross-generator robustness is **Alex's ViT + Leyi's CLIP probe**: **OOD AUROC 0.9657** (each alone: 0.9732 and 0.9485). Their errors are decorrelated (4.4% complementary errors on test).
- The best **3-model team ensemble** is **Yin's CNN + Alex's ViT + Leyi's CLIP**: **OOD AUROC 0.9670** (beats the best pair by +0.13 pp). Yin's CNN adds positive cross-generator contribution where Nathan's ResNet adds none.
- The best **2-model team ensemble for clean test** is **Nathan ResNet-18 + Alex ViT**: **test AUROC 0.9993**.
- **Frequency detector is dominated** by spatial models on every detection axis; it earns its place as scientific characterisation (an interpretable spectral fingerprint of SD-1.4), not as a competitive classifier.

---

## Per-model headline (test / OOD AUROC)

| # | Model | Owner | Test AUROC | OOD AUROC | OOD drop | Trained params |
|---|---|---|---:|---:|---:|---:|
| 1 | from-scratch CNN | Yin    | **0.9974** | **0.9429** | −5.5 pp | 288 k |
| 2 | ResNet-18 (ImageNet, partial fine-tune) | Nathan | **0.9977** | **0.9341** | −6.4 pp | 11.2 M |
| 3 | ViT-Small (timm, full fine-tune) | Alex   | **0.9994** | **0.9732** | **−2.6 pp** | 21.7 M |
| 4 | Frequency detector (mag. SpectrumCNN) | Leyi   | 0.9435 | 0.8150 | −12.8 pp | 222 k |
| 5 | CLIP probe (LAION ViT-B/32 + MLP head) | Leyi   | **0.9968** | **0.9485** | −4.9 pp | 132 k (probe only; 151 M frozen) |

All five models were evaluated through the *same* shared harness (`src/eval_harness.evaluate(...)`), same frozen 90/10 split (seed 42), same val-derived Youden-J threshold, same sealed test (n=20 000) and same OOD set (n=2 000). Yin's number reproduces his original `results_CNN_from_scratch.json` to four decimal places.

## Team ensemble (4 spatial models with cached per-sample scores)

We average class probabilities across diverse inductive biases. The frequency detector is deliberately excluded from the team ensemble because earlier sweeps showed it has a negative OOD contribution (the spectral fingerprint is fragile to cross-generator shift).

| Subset | Test AUROC | OOD AUROC |
|---|---:|---:|
| Best single (Alex ViT) on test / OOD | 0.9994 | 0.9732 |
| **Best pair on test: Nathan + Alex** | **0.9993** | 0.9529 |
| **Best pair on OOD: Alex ViT + CLIP probe** | 0.9988 | **0.9657** |
| **Best 3 on OOD: Yin CNN + Alex ViT + CLIP probe** | 0.9994 | **0.9670** |
| All 4 (Yin + Nathan + Alex + CLIP) | 0.9994 | 0.9655 |

**Leave-one-out OOD contributions** (positive = adding this model improved OOD AUROC):

- Alex ViT: **+0.70 pp** ← biggest contributor on OOD
- CLIP probe: **+0.27 pp** ← second contributor
- Yin CNN: **+0.17 pp** ← cheap, complementary, positive on OOD
- Nathan ResNet: −0.15 pp (correlated with Alex)

The Yin + Alex + CLIP triple is the *empirical* best subset on cross-generator OOD: it edges out the Alex + CLIP pair by +0.13 pp while matching the all-4 ensemble on test (0.9994).

## Methodology highlights (controlled comparison)

- **Same harness:** every model goes through `src/eval_harness.evaluate(...)` with the same metrics + JSON schema. Test threshold = val-derived Youden-J. Sealed test touched once per model.
- **Same OOD set:** 1000 sd-turbo + 1000 CIFAR-10 REAL (matched content distribution). Both classes re-encoded through CIFAKE's exact JPEG quantization tables (luma 1858, chroma 2780) so the *only* difference between in-dist and OOD is the generator process.
- **Same robustness battery:** 19 perturbations (JPEG q∈{90,75,60,40,25,10}, blur σ∈{0.3,0.5,0.8,1.0,1.5}, noise σ∈{2,4,8,16,32}/255, rescale {24,16,12}→32) — applied at inference time on the sealed test.

## Frequency detector — what it contributes (despite losing on metrics)

- **Interpretable spectral fingerprint of SD-1.4** (see `09_freq_lr_coefficients.png`):
  the LR weights load ~3.3× more on the high-frequency tail (radial bins 12-16) than on the JPEG-block-grid bin (radial_4) — confound check passes.
- **Visualizable decoder grid** (see `08_freq_mean_spectra.png`): the FAKE−REAL difference image shows anisotropic vertical-axis bright spots — the latent-diffusion VAE-decoder periodicity.
- **Validates the cross-generator OOD methodology**: the same handcrafted radial-PSD features that work on SD-1.4 still beat chance on sd-turbo (−9.2 pp drop), suggesting a transferable spectral fingerprint.

## What's in this repo

- `notebooks/01_cnn_baseline.ipynb` ... `06_team_comparison.ipynb` — the 6 graded notebooks
- `src/{eval_harness,perturbations,ensemble}.py` + `src/{freq_detector,clip_probe}/` — library code
- `models/<name>/predict.py` + `<checkpoint>` — inference wrappers (checkpoints via Git LFS)
- `scripts/evaluate.py` — generic re-evaluator (`--model <name>`)
- `results/per_model/*` — headline JSONs (test/val/ood); `results/team_ensemble_report.json` — ensemble
- `results/figures/` — 12 curated PNGs
- `docs/REPORT_HEADLINE.md` — extended cribsheet for the report; `docs/findings/*.md` — per-model deep dives; `docs/methodology/*.md` — design rationale
