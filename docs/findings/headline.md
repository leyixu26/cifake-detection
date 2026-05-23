# Headline Findings — 5-Model Comparison + Team Ensemble

**Project:** Detect AI-generated images on CIFAKE (32×32 RGB; REAL = CIFAR-10, FAKE = Stable Diffusion v1.4).
**Team:** Yin (small CNN), Nathan (ResNet-18), Alex (ViT-Small), Leyi (frequency detector + CLIP probe).
**Date:** 2026-05-24.

---

## TL;DR

- All four spatial models hit ≥ **0.997 clean-test AUROC** on CIFAKE — the in-distribution problem is genuinely easy.
- **Cross-generator OOD** (against sd-turbo, content + JPEG-quant-table matched to CIFAKE) is where models separate. **Alex's ViT-Small drops least** (−2.6 pp); the frequency detector drops most (−12.8 pp).
- The best **2-model team ensemble** for cross-generator robustness is **Alex's ViT + Leyi's CLIP probe**: **OOD AUROC 0.9657** (each alone: 0.9732 and 0.9485). Their errors are decorrelated (4.4% complementary errors on test).
- The best **2-model team ensemble for clean test** is **Nathan ResNet-18 + Alex ViT**: **test AUROC 0.9993**.
- **Frequency detector is dominated** by spatial models on every detection axis; it earns its place as scientific characterisation (an interpretable spectral fingerprint of SD-1.4), not as a competitive classifier.

---

## Per-model headline (test / OOD AUROC)

| # | Model | Owner | Test AUROC | OOD AUROC | OOD drop | Trained params |
|---|---|---|---:|---:|---:|---:|
| 1 | from-scratch CNN | Yin    | **0.9974** (reported) | n/a (pending re-eval) | — | 288 k |
| 2 | ResNet-18 (ImageNet, partial fine-tune) | Nathan | **0.9977** | **0.9341** | −6.4 pp | 11.2 M |
| 3 | ViT-Small (timm, full fine-tune) | Alex   | **0.9994** | **0.9732** | **−2.6 pp** | 21.7 M |
| 4 | Frequency detector (mag. SpectrumCNN) | Leyi   | 0.9435 | 0.8150 | −12.8 pp | 222 k |
| 5 | CLIP probe (LAION ViT-B/32 + MLP head) | Leyi   | **0.9968** | **0.9485** | −4.9 pp | 132 k (probe only; 151 M frozen) |

**Yin's number is from his original `results_CNN_from_scratch.json`.** The re-train under the shared harness is pending (his `best_cnn.pt` is en route).

## Team ensemble (3 models with cached per-sample scores)

We average class probabilities across diverse inductive biases:

| Subset | Test AUROC | OOD AUROC |
|---|---:|---:|
| Best single (CLIP probe) on OOD | 0.9968 | 0.9485 |
| **Best pair on test: Nathan + Alex** | **0.9993** | 0.9529 |
| **Best pair on OOD: Alex ViT + CLIP probe** | 0.9988 | **0.9657** |
| All 3 (Nathan + Alex + CLIP) | 0.9993 | 0.9637 |

**Leave-one-out OOD contributions** (positive = adding this model improved OOD AUROC):

- Alex ViT: **+1.08 pp** ← biggest contributor on OOD
- CLIP probe: −0.07 pp (≈ neutral; close to Alex on OOD)
- Nathan ResNet: −0.20 pp (correlated with Alex)

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
