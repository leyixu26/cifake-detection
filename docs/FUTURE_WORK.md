# Future Work

This document consolidates the natural next steps for the project. Each item is
grounded in a specific limitation we ran into in this round, with a pointer to
the existing doc where the limitation is discussed in more detail.

---

## 1. Evaluation: more generators, more architectures

**What we did.** A single cross-generator OOD probe (sd-turbo) with content,
resolution, and JPEG quantization tables held constant. n = 2 000.

**What's missing.** Per `docs/methodology/ood_methodology.md` § "Honest
limitations", one generator is one data point. The clean way to call this a
generalization study is to evaluate against a panel of generators that spans
two axes:

| Axis | What we tested | What's missing |
|---|---|---|
| Generator family | Multi-step diffusion (SD-1.4) → single-step distilled diffusion (sd-turbo) | Architecturally different generators (GANs, autoregressive) |
| Generator scale | SD-1.4 (CIFAKE source) → sd-turbo (same SD family) | Different families: DALL-E 3, Midjourney v6, Imagen, Flux, GAN variants |

**Concrete next steps.**
- Add 3–5 generators from the **ForenSynths benchmark** (Wang et al. 2020) for
  GAN coverage and the **GenImage** benchmark (Zhu et al. 2023) for modern
  diffusion coverage.
- Cross-architecture probe: evaluate detectors trained on diffusion against a
  pure-GAN OOD set. The literature reports 2–3× larger OOD drops on this axis
  than within the diffusion family — would put a hard bound on transfer.
- Report bootstrap confidence intervals for AUROC differences at n ≥ 2 000.
  The current per-model OOD drops are point estimates; a 1-pp difference at
  n = 2 000 is on the edge of significance.

## 2. Robustness: adversarial perturbations

**What we did.** A 19-perturbation inference-time robustness battery (JPEG,
blur, noise, rescale) on the sealed test split. Documented in
`src/perturbations.py` (`BATTERY` dict) and `scripts/evaluate.py`.

**What's missing.** Per `docs/LITERATURE.md` § "Adversarial-robustness gap",
all current foundation-model-based detectors — including the CLIP probe in
this project — are vulnerable to small adversarial perturbations that do not
meaningfully change the image content but flip the detection prediction.

**Concrete next steps.**
- Run FGSM and PGD attacks (ε = 1/255, 2/255, 4/255 on pixel scale) against
  every model in the lineup and report AUROC under attack.
- Adversarial training for the spatial models (mix clean + ε-PGD samples
  during fine-tuning). The CLIP encoder is frozen so adversarial training is
  bounded to the MLP head — limited but worth quantifying.
- Add an attack-detection layer (e.g., input-gradient anomaly score) and
  evaluate the "detect and reject" deployment mode rather than try to make
  the classifier itself robust.

## 3. Resolution: lift the 32×32 ceiling

**What we did.** Trained and evaluated at 32×32 (CIFAKE's native resolution).

**What's missing.** Per `docs/methodology/frequency_detector.md` § "The 32×32
constraint", a 32×32 FFT has only ~16 usable radial bins, which compresses
upsampling-grid artifacts into a few bins and makes them aliased. The
spectral-fingerprint literature (Frank 2020, Durall 2020, Dzanic 2020) works
at 128–1024 px where these peaks are well-resolved. The CLIP probe is also
under-served by the resolution: CLIP was pretrained at 224×224 and we feed
it 32→224 bicubic upsamples, throwing away a 7× linear factor of information
that the encoder was trained to use.

**Concrete next steps.**
- Re-evaluate all five models on higher-resolution variants of CIFAKE
  (CIFAKE-256, CIFAKE-512 if available; otherwise create one from the same
  CIFAR-10 + SD-1.4 prompt pipeline at 256×256).
- For the frequency detector specifically: 256×256 gives 128 usable radial
  bins; expect the AUROC gap vs. the spatial models to narrow.
- For the CLIP probe: at native 224×224 input, expect the OOD lift to grow
  (the bicubic-from-32 path is currently the bottleneck).

## 4. Architecture & scaling

**What we did.** Five architectures spanning four inductive-bias families.

**What's missing.** We did not test the obvious foundation-model alternative
to CLIP, nor scaling within the families we did pick.

**Concrete next steps.**
- **DINOv2 backbone.** Oquab et al. TMLR 2024 — a self-supervised vision
  foundation model that matches or exceeds CLIP on most transfer benchmarks
  and beats CLIP on fine-grained tasks. A natural substitution for the
  frozen-encoder probe; reported in `docs/LITERATURE.md` § 3 as the obvious
  next backbone to try.
- **Larger ViTs.** ViT-Base, ViT-Large from timm. Marginal returns on test
  AUROC at our resolution but the OOD axis is unmapped at scale.
- **Multi-foundation-model ensemble.** Recent literature (cited in
  `docs/LITERATURE.md` § 5 — "Foundation model ensembles") shows that
  ensembling CLIP + DINOv2 + SAM features gives +1–3 pp AUROC over any
  single-encoder probe. Our four-model ensemble already shows the
  inductive-bias-diversity payoff; foundation diversity is the next axis.

## 5. Methodology hardening

**What we did.** Point-estimate AUROC and pp-drop reporting at n = 20 000
(test) and n = 2 000 (OOD). Per-sample scores persisted for ensemble +
leave-one-out reproducibility.

**What's missing.**
- **Bootstrap confidence intervals on AUROC and AUROC differences.** Most
  comparisons in the headline (e.g., "Yin's from-scratch CNN drops 0.9 pp
  less than Nathan's ImageNet-pretrained ResNet") are at the edge of what
  n = 2 000 can resolve. A 2 000-sample bootstrap on the AUROC differences
  would put numerical bounds on which gaps are real and which are noise.
- **Stacking ensemble**, not just probability averaging. We use probability
  averaging because it doesn't need a held-out meta-training set, but with
  more held-out data we could fit a logistic stacker on per-model scores and
  measure the marginal lift.
- **Calibration.** We report AUROC and PR-AUC (rank-based metrics) but the
  CLIP probe's calibration plot exists (`results/figures/06_clip_calibration.png`)
  for only one model. A reliability diagram per model + Brier score would
  let downstream applications use the predicted probabilities directly
  rather than re-fitting a threshold.

## 6. Applications and deployment

**What we did.** Sealed test on a synthetic balanced benchmark; one OOD probe.

**What's missing.** This is a research benchmark. Production deployment would
need:
- **Per-domain validation.** Faces (DeepFakeDetection, FaceForensics++),
  landscapes, document images — each has different generator artifacts.
- **Real-world image pipelines.** Social-media compression chains, screen
  capture, third-party recompression. Our robustness battery covers single
  perturbations; production sees compositions.
- **Open-set detection.** The current setup is a binary classifier; in
  production the "REAL" class has unbounded support and the "FAKE" class is
  ever-growing. A one-class or open-set formulation (e.g., density estimation
  on real images, anomaly score for fake) would generalize better as new
  generators appear.

---

## Priority ranking for a next round

If we had one quarter to extend this:

1. **Multi-generator OOD evaluation** with bootstrap CIs (§ 1 + § 5
   first bullet). Closes the most-asked methodological gap.
2. **Higher-resolution re-run** (§ 3). Unlocks the CLIP probe's design
   intent and gives the frequency detector a fair fight.
3. **DINOv2 backbone substitution** (§ 4 first bullet). Single experiment,
   directly comparable to CLIP, settles whether contrastive image-text
   pretraining is specifically what helps, or any large self-supervised
   encoder will do.
4. **Adversarial robustness** (§ 2). Required before any production use;
   currently a documented open gap.
