# `results/figures/` — captioned figure inventory

Twelve curated figures used in the team presentation. All are auto-regenerable
via `python scripts/generate_figures.py` (cross-model figures 01-03) and from
the per-model notebooks (model-specific figures 04-12).

| File | Caption |
|---|---|
| `01_clean_vs_ood.png` | All 5 models — clean test AUROC (CIFAKE) vs cross-generator OOD AUROC (sd-turbo). The Δ overlay reads as the OOD drop in percentage points. **Headline figure for the report.** |
| `02_robustness_curves.png` | Test AUROC vs perturbation strength for each model across JPEG re-compression / Gaussian blur / additive noise / downscale-upscale. Shows CLIP probe dominates blur/rescale; spatial CNNs slightly win on heavy noise; frequency detector collapses on low-pass perturbations. |
| `03_team_ensemble.png` | Pairwise + 3-way + all-model ensemble AUROCs (test and OOD panels). Best pair on test = Nathan + Alex = 0.9993; best pair on OOD = Alex + CLIP probe = 0.9657. |
| `04_clip_capacity_ladder.png` | CLIP capacity-push ablation: baseline ViT-B/32-OpenAI + LR → swap LAION weights (+1.7 pp OOD) → add flip-TTA (+0.2 pp) → MLP head (+1.0 pp). Establishes which lever buys what. |
| `05_clip_vs_spatial_agreement.png` | 2×2 of `(CLIP correct/wrong) × (spatial CNN correct/wrong)` on test. Shows 4.4% complementary errors — the empirical basis for ensembling. |
| `06_clip_calibration.png` | Reliability diagram for the final CLIP MLP probe on sealed test. Curve sits essentially on the diagonal — well-calibrated. |
| `07_freq_radial_psd.png` | Class-mean radial PSD profiles (REAL vs FAKE) with ±1σ bands. Shows the Durall/Dzanic-style HF-tail divergence at 32×32. |
| `08_freq_mean_spectra.png` | Per-class mean log-magnitude FFT spectra + FAKE−REAL difference image. The anisotropic vertical-axis bright spots in the difference are the latent-diffusion VAE-decoder fingerprint. |
| `09_freq_lr_coefficients.png` | Variant-A logistic-regression coefficients by feature group (radial bins, azimuthal sectors, scalar summaries). JPEG-block bin (radial_4) marked — confound check passes (HF/JPEG-bin weight ratio = 3.33×). |
| `10_freq_cnn_saliency.png` | Grad-CAM-style saliency on the SpectrumCNN; aligns with the LR coefficient story (high-radial-frequency band). |
| `11_freq_robustness_battery.png` | Frequency variants under the perturbation battery — quantifies the "spectral fingerprint fragility to low-pass" claim. |
| `12_freq_ood_drop.png` | Cross-generator OOD (sd-turbo) AUROC for freq variants. The handcrafted Variant A drops *least* (−9.2 pp) — partially supports the inductive-bias hypothesis. |

## Regeneration

```
# Cross-model: 01 02 03  (loads from results/per_model/*)
python scripts/generate_figures.py

# Per-model: 04-06 (CLIP)  -> notebooks/05_clip_probe.ipynb
# Per-model: 07-12 (freq)  -> notebooks/04_freq_detector.ipynb
```
