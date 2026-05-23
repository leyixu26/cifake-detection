# Methodology — Frequency-Domain Detector (Model 4)

A spectrum-input detector for AI-generated images. Tests a fundamentally
different inductive bias from the team's spatial models (CNN/ViT/CLIP): no
pixel-space features, just the Fourier/DCT magnitude. The cost is on
accuracy; the win is a falsifiable physical claim about the generator's
fingerprint that no spatial model can produce.

## Scientific motivation

Generative models leave **spectral fingerprints** — systematic deviations in
the Fourier/DCT magnitude spectrum that are weak in pixel space but
structured in frequency space. Key literature:
- **Zhang 2019 (AmpNet)** — GAN upsampling produces periodic high-frequency peaks.
- **Frank et al. 2020** — DCT artifacts from upsampling, simple classifier on DCT coefficients separates real/fake with >99% in-distribution.
- **Durall et al. 2020** — GANs fail to reproduce the azimuthally-averaged radial PSD of natural images; deviations especially in the HF tail.
- **Dzanic et al. 2020** — quantifies that HF decay rate differs between real and generated; a few spectral statistics suffice.
- **Corvi et al. 2023** — diffusion models (incl. SD) leave spectral fingerprints too, *different* from GANs (subtler VAE-decoder periodicity), and *more fragile* to post-processing.

## Pipeline

```
   image (32×32×3)
      │
      ▼
  ┌─────────────────────────┐
  │ luminance or per-RGB    │
  │ Hann window (optional)  │
  │ FFT2 + fftshift         │
  │ |·|² → log magnitude    │
  │ zero DC                 │
  └────┬──────────────┬─────┘
       │              │
       ▼              ▼
  ┌─────────────┐  ┌─────────────────────────┐
  │ radial PSD  │  │  full 2-D log|FFT|      │
  │ + azimuthal │  │  (3, 32, 32)            │
  │ + scalars   │  └────────────┬────────────┘
  │ → 29-d      │               │
  └────┬────────┘               ▼
       │             ┌─────────────────────┐
       ▼             │ SpectrumCNN          │
  ┌──────────┐       │ ~220k params         │
  │ LR/MLP   │       │ 3 conv blocks + GAP  │
  │ (4k pp)  │       └─────────────────────┘
  └──────────┘
   Variant A           Variant B
```

## Two variants

**Variant A — handcrafted radial PSD + shallow classifier:**
- Compute radial PSD (16 bins), azimuthal PSD (8 sectors), 5 scalar summaries → 29-d vector
- Standardise with train-only mean/std (leakage-safe)
- Logistic regression / shallow MLP / SVM
- 4k params, interpretable, ~0.90 test AUROC

**Variant B — log-magnitude spectrum CNN:**
- (3, 32, 32) log|FFT| input
- Small CNN (~220k params, matches teammate's small CNN scale)
- ~0.94 test AUROC

Variant A is the principled core (directly tests the Durall hypothesis).
Variant B is the empirical complement (a CNN may find 2-D structure radial
averaging discards). Honest caveat: applying spatial convolution to a Fourier
magnitude image lacks the usual translation-equivariance justification —
neighbouring units are *frequencies*, not space.

## The 32×32 constraint

A 32×32 FFT yields only ~16 usable radial bins (Nyquist at 16). Most
spectral-fingerprint papers used 128-1024 px where upsampling-grid peaks are
well-resolved. At 32×32 those peaks may alias into a few bins.

We validated up front (`notebooks/04_freq_detector.ipynb` Phase M0) that a
discriminative signal *does* exist at 32×32: max |Cohen's d| = 1.41 at
radial bin 15, and 5-fold-CV AUROC of logistic regression on the 16-d
radial PSD = 0.86 on a diagnostic sample. So the fingerprint survives the
low-resolution regime — barely.

## Confound controls (these matter for the report)

1. **JPEG quantization tables verified equal** between REAL and FAKE in CIFAKE
   (luma 1858, chroma 2780). The classic "different JPEG quality per class"
   leak is largely absent.
2. **LR coefficient inspection** (figure `09_freq_lr_coefficients.png`):
   |w| at JPEG block bin (radial_4) = 0.082 vs mean |w| over HF tail = 0.272.
   HF/JPEG-bin ratio = **3.33×** — LR is not keying on JPEG block artifacts.
3. **JPEG-q90 equalised retrain**: re-encoded every image through one
   identical fresh JPEG-q90 pipeline. LR test AUROC = 0.8725 vs baseline
   0.8797 (Δ = 0.007). Signal survives compression equalization; it is
   **not** a pure JPEG-history artefact.

## Where the freq detector loses (honestly)

| Axis | Result |
|---|---|
| Clean test AUROC | 0.944 (best variant). Trails the spatial models (0.997-0.999) by ~5 pp. |
| OOD AUROC | 0.815. Bigger drop than CLIP (0.949) or spatial CNNs (0.93-0.97). |
| Blur σ=0.8 | 0.508 (CNN); collapses because blur is low-pass and the HF fingerprint is what gets killed. |
| Rescale to 16×16 | 0.568; same mechanism. |
| Ensemble contribution | NEGATIVE on both test and OOD. The 2-model ensemble is better without it. |

## Where the freq detector earns its place (in the report)

- **Interpretability.** The LR coefficient bar chart over radial / azimuthal
  / scalar features is a *physical statement* about SD-1.4's fingerprint
  (high-frequency tail + anisotropic azimuthal structure). No spatial model
  can produce this kind of falsifiable, model-agnostic claim about the
  generator.
- **Mean-spectra visualisation** (`08_freq_mean_spectra.png`): the FAKE−REAL
  difference image directly shows the latent-diffusion VAE-decoder
  fingerprint as anisotropic vertical-axis bright spots — a publication-
  quality figure that ties our findings to Corvi 2023.
- **Methodological rigour demonstration**: the freq detector's
  vulnerability to blur/rescale (in the robustness battery) is the
  cleanest illustration that the spectral signal is real but fragile —
  matches the Corvi 2023 prediction precisely.

The report should frame Model 4 as an **interpretability companion** to the
spatial models, not as a competing classifier. Its job is to characterise
*what* artifact the spatial detectors latch onto, not to win on AUROC.
