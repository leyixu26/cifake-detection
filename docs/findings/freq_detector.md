# Frequency-Domain Detector — Final Results

**Dataset:** CIFAKE (32×32, REAL = CIFAR-10, FAKE = Stable Diffusion v1.4).
Train pool 100k → frozen 90/10 stratified split (seed 42) → 90k train / 10k val.
Sealed test 20k. Positive class = FAKE.

## Headline numbers

| Variant | Model | Val AUROC | Test AUROC | Test Acc | Test F1 |
|---|---|---|---|---|---|
| A – handcrafted 29-d → shallow | LogReg (C=1) | 0.880 | – | – | – |
| A – handcrafted 29-d → shallow | LinearSVM | 0.881 | – | – | – |
| A – handcrafted 29-d → shallow | RBF-SVM (15k subsample) | 0.892 | – | – | – |
| A – handcrafted 29-d → shallow | **ShallowMLP** (4k params) | **0.905** | **0.900** | 0.819 | 0.820 |
| B – log-magnitude spectrum CNN | **SpectrumCNN** (221k params) | **0.944** | **0.944** | 0.872 | 0.874 |

Val→test generalization is essentially perfect (0.905→0.900, 0.944→0.944) —
no val-overfitting from the ablation grid (all decisions on val, single sealed-test pass).

## M0 — does the fingerprint survive at 32×32? Yes.

`results/figures/m0_*.png`. Diagnostic on 5k REAL + 5k FAKE train images:

- **Max \|Cohen's d\| = 1.41 at radial bin 15/16** (HF tail) — exactly the
  Durall/Dzanic prediction.
- **5-fold CV AUROC of LR on the 16-d radial PSD alone = 0.86** on the diagnostic sample.
- The FAKE − REAL mean log-spectrum difference is **structured, not noise**:
  excess HF energy at the corners plus anisotropic bright spots on the vertical
  axis — consistent with latent-diffusion VAE-decoder upsampling artifacts
  (Corvi 2023).
- The JPEG **quantization tables are byte-identical** between REAL and FAKE
  (luma sum 1858, chroma sum 2780). The classic "different JPEG quality per
  class" confound is largely absent in CIFAKE.

→ Decision: **PROCEED CONFIDENTLY** (recorded in `M0_findings.md`).

## M4 — Ablation grid (Variant A LogReg, val AUROC)

| Config | Dim | Val AUROC | Note |
|---|---:|---:|---|
| **DCT instead of FFT** | 29 | **0.8842** | Frank et al. signal confirmed |
| strong low-freq mask (dc=3) | 29 | 0.8820 | low-freq carries little info |
| RGB (per-channel) | 29 | 0.8800 | luma is enough |
| baseline luma / FFT / Hann / dc=1 | 29 | 0.8797 | reference |
| DC retained | 29 | 0.8795 | DC is not used (good — no brightness leak) |
| radial + azimuthal | 24 | 0.8781 | scalars add little |
| radial + scalars | 21 | 0.8653 | azimuthal carries real info |
| radial only | 16 | 0.8598 | even pure 16-d radial almost matches the full feature set |
| no Hann window | 29 | 0.8588 | boundary leakage matters |
| **HIGH-PASS only (radial ≥ 6, 11 dims)** | 11 | **0.8582** | the signal *is* in the HF tail |
| LOW-PASS only (radial < 6) | 5 | 0.6769 | low frequencies alone are weak |

Two clean architectural-justification results: **DCT marginally beats FFT** and
the **high-pass-only model (11 features) is within 0.022 of the full 29-d
baseline**, while low-pass-only nearly collapses.

## M5 — Robustness battery on sealed test (`figures/m5_robustness.png`)

Inference-time perturbations, both models frozen. All numbers test AUROC.

| Perturbation @ strength | A_mlp | B_cnn |
|---|---:|---:|
| clean | 0.900 | 0.944 |
| JPEG q=75 | 0.900 | 0.943 |
| JPEG q=40 | 0.812 | 0.886 |
| JPEG q=10 | 0.690 | 0.756 |
| Blur σ=0.5 | 0.837 | 0.918 |
| Blur σ=0.8 | 0.654 | **0.508** |
| Blur σ=1.5 | 0.364 | 0.410 |
| Noise σ=8/255 | 0.876 | 0.808 |
| Noise σ=16/255 | **0.804** | 0.707 |
| Noise σ=32/255 | **0.739** | 0.576 |
| Rescale 16×16 | 0.474 | 0.568 |
| Rescale 24×24 | 0.706 | 0.811 |

**Findings**

1. **Blur is the predicted weak point** — it is a low-pass filter and the HF
   fingerprint is exactly what gets killed. The CNN collapses faster than the
   handcrafted MLP (0.508 vs 0.654 at σ=0.8).
2. **Noise reverses the leaderboard:** handcrafted radial-PSD features beat the
   CNN at every noise level σ ≥ 8/255 (e.g. 0.804 vs 0.707 at σ=16). Radial
   *averaging* smooths over additive noise; the CNN keys on fine 2-D patterns
   that noise destroys. This is the interpretable-vs-learned tradeoff in pure form.
3. **Rescale to 16×16** drops both near chance — the fingerprint is destroyed
   by the implied low-pass, just like blur.
4. **JPEG re-compression** degrades both gracefully and the CNN keeps its lead,
   which doubles as a compression-confound probe (a pure-JPEG detector would
   collapse to chance under recompression — both models clearly do not).

## M6 — Interpretability + confound control

`figures/m6_lr_coefficients.png`, `m6_permutation_importance.png`, `m6_cnn_gradcam.png`.

- **LR coefficient confound check:** |w| at the JPEG 8×8 block bin (radial_4)
  = 0.082 vs mean |w| over the HF tail (radial_11..16) = 0.272. The LR loads
  **3.33× more on the HF tail than on the JPEG bin** → it is not keying on
  compression.
- **Largest signed weights** (interpretable): `radial_1` strongly positive
  (FAKE has excess low-freq energy beyond DC — consistent with the latent
  diffusion VAE bottleneck producing slightly oversmoothed but energetically
  dense low-frequency content), `radial_15/16` positive (the HF tail), and the
  **azimuthal sectors carry the largest weights of any group** (`azim_4` +1.7,
  `azim_7` −2.7) — capturing the **angular structure** of the VAE-decoder grid
  that radial averaging would otherwise discard.
- **Permutation importance top-5:** `azim_7, radial_1, azim_4, centroid,
  azim_0` — angular features dominate, exactly matching the M0 observation
  that the FAKE−REAL difference image has vertical-axis bright spots.
- **Confound-control retrain:** AUROC after re-encoding every image through
  one identical fresh JPEG-q90 pipeline = **0.8725** vs baseline 0.8797
  (Δ = 0.007). The signal is **not** a pure JPEG-history artifact.
- **CNN saliency:** input-gradient magnitudes are concentrated at corners and
  edges (high spatial frequency), with the FAKE−REAL saliency difference
  showing the same anisotropic vertical structure that drives the LR azimuthal
  coefficients — two independent models agree on the artifact.

## Narrative for the 4-model comparison

The frequency detector is the only one of the four models whose architecture is
**justified by the physics of the artifact**: the radial PSD encodes the Durall
inductive bias by construction; the azimuthal sectors capture the VAE-decoder
grid orientation; the CNN exploits the residual 2-D structure. Variant A's
~29-dim feature vector and ~4k-param MLP reach 0.900 test AUROC; the CNN reaches
0.944. The spatial models will almost certainly beat both in-distribution
(orders of magnitude more parameters), but the robustness battery shows that
the handcrafted detector is **more noise-robust than the CNN** and the
discriminative signal is concentrated in the HF tail rather than the JPEG block
bin — an interpretable, principled story none of the spatial models can tell.

True cross-generator generalization (the strongest test of the inductive-bias
hypothesis) is impossible on CIFAKE alone (single generator) and is the
deferred M8 stretch goal: inference-only on the frozen models against a non-SD
dataset, droppable without affecting the core.

## Limitations honestly stated

- **Single generator** — see above. M8 is the answer; it is not yet run.
- **Low resolution** — at 32×32 there are only ~16 usable radial bins. Many
  literature findings come from 128–1024 px; what survives here is the
  coarsest version of the fingerprint.
- **Augmentation off by default** — the frequency model gives up the standard
  spatial augmentation toolbox (random crop/translation destroys the spectrum).
  Robustness-aware training (JPEG/blur as augmentation) is an obvious extension.
- **Conv-on-spectrum** for Variant B has weak architectural justification —
  neighbouring units in a Fourier magnitude image are *frequencies*, not space.
  Variant A is the principled model; Variant B is the empirical "can a CNN do
  better" complement. The polar-resampled-spectrum CNN is an open follow-up.
