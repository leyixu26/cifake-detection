# Assumptions

This project takes a number of things for granted. This page lists them
honestly, with the test in the repo that supports each one and the
counter-evidence that would make us wrong.

---

## About the data

**1. CIFAKE's labels are correct.** Every image in `train/REAL` and
`test/REAL` is a real CIFAR-10 photograph; every image in `train/FAKE` and
`test/FAKE` is a Stable-Diffusion-1.4 generation. We did not re-verify
labels — we inherit them from the published dataset.
- *Where we'd be wrong:* if Bird (the CIFAKE author) mis-attributed any
  subset, our entire framing collapses. We accept this exposure because
  CIFAKE is the standard benchmark used in the literature.

**2. CIFAKE's class balance is 50 / 50, and our stratified split preserves
it.** Train, val, and test are all naturally balanced. Our 90/10 stratified
split (seed 42, `src/freq_detector/datasets.py::make_splits`) keeps the
ratio inside each fold.
- *Where we'd be wrong:* if the dataset isn't actually balanced, the
  Youden-J threshold derivation in `src/eval_harness.best_threshold` would
  shift, and class-weighted metrics would be more appropriate than the ones
  we report. We checked the split distribution at dev time and it is 50 / 50.

**3. CIFAR-10 originals are representative of "real images" for the
purpose of this task.** CIFAR-10 is a narrow domain — 10 object classes,
internet-scraped, 32 × 32. We use it as our REAL class because it's what
CIFAKE provides.
- *Where we'd be wrong:* a detector trained against CIFAR-10 REAL might
  fail when the real input is a face, a document, a screenshot, or
  high-resolution photography. We do not test this. Flagged explicitly in
  `docs/FUTURE_WORK.md` § "Applications and deployment".

**4. SD-1.4 fingerprints are detectable at 32 × 32.** The published
spectral-fingerprint literature (Frank 2020, Durall 2020) works at
128–1024 px, where upsampling-grid peaks are well-resolved. At 32 × 32
those peaks are likely aliased.
- *Why we think it's defensible:* the M0 diagnostic in
  `docs/findings/freq_detector.md` § "M0" measures Cohen's d = 1.41 on the
  highest-frequency radial bin in a 5 000-sample held-out probe. So *some*
  signal survives the low-resolution regime — barely.
- *Where we'd be wrong:* at 64 × 64 or higher we would expect a much
  cleaner separation for spectral methods. The frequency detector being
  the weakest model on AUROC is partly a consequence of this assumption.

**5. JPEG quantization tables are matched between CIFAKE's REAL and FAKE
classes.** This is the standard confound for any frequency-domain
detector — a model could trivially separate the two classes if they have
different compression histories.
- *Verified in `docs/methodology/frequency_detector.md` § "Confound
  controls"*: REAL and FAKE both use the same (luma 1858, chroma 2780)
  quantization tables. We re-encode the OOD set through these same tables
  on both sides (`scripts/generate_ood.py`) for the same reason.

---

## About modeling and training

**6. The sealed test split is statistically independent of train and val.**
We use one frozen split (seed 42, 90/10) for the entire team. Every model
is trained on the same 90 k train images, tuned on the same 10 k val
images, and evaluated once on the same 20 k test images.
- *Where we'd be wrong:* if any of us touched the test set during
  development — including peeking at test metrics to choose a hyperparameter
  — the reported numbers would be optimistic. The shared
  `src/eval_harness.evaluate(...)` call only fires from `scripts/evaluate.py`,
  which makes test access auditable.

**7. Architectures and hyperparameters were chosen a priori, not by
peeking at test results.** Model depths, LR schedules, augmentation
choices, and decision thresholds (val-derived Youden-J) were all fixed
before the sealed test was opened.

**8. Freezing the CLIP encoder isolates the pretraining contribution.**
Model 5 freezes the OpenCLIP ViT-B/32 encoder and trains only the 132 k
MLP head. The interpretation that "any OOD lift from the pretraining-data
swap (OpenAI → LAION-2B) is attributable to the pretraining data, not to
fine-tuning" depends on this freeze being absolute.
- *Verified:* `src/clip_probe/pipeline.py` sets `requires_grad = False` on
  every encoder parameter before optimizer setup.
- *Where we'd be wrong:* if the encoder were unfrozen, we couldn't
  attribute the +1.7 pp OOD lift specifically to the pretraining-data
  diversity. This is precisely why we chose to freeze in the first place.

---

## About evaluation

**9. AUROC, PR-AUC, F1, and accuracy are appropriate metrics for this
balanced binary problem.** With 50 / 50 class balance, AUROC and PR-AUC
will agree closely and accuracy is interpretable as a simple percent
correct.
- *Where this would change:* an unbalanced deployment domain (e.g., 1 %
  FAKE prevalence) would make AUROC less actionable and require
  recall-at-low-FPR or PR-AUC-at-low-recall as the primary metric.

**10. The 19-perturbation robustness battery represents realistic
deployment-time degradations.** JPEG re-compression, blur, Gaussian noise,
and downsample-upsample cover the most common image-pipeline corruptions.
- *Where we'd be wrong:* this battery does *not* include adversarial
  perturbations (FGSM/PGD). All current foundation-model-based detectors,
  including the CLIP probe in this project, are known to be vulnerable to
  these. Flagged in `docs/FUTURE_WORK.md` § "Robustness".

**11. sd-turbo is far enough from SD-1.4 to count as a cross-generator
probe.** Both are diffusion models from the same source organization, but
sd-turbo uses single-step adversarial score distillation with a different
sampler and a different VAE decoder.
- *Where we'd be wrong:* if a reader argues sd-turbo is "too close" to
  SD-1.4 for the result to count, the right response is that this is a
  *directional* probe — see assumption 12 — and the future-work doc lists
  multi-generator and cross-architecture extensions as the next step.

**12. The OOD evaluation is a directional probe, not a generalization
study.** With one OOD generator and n = 2 000, we can rank models by drop
magnitude but we cannot put tight confidence intervals on the drops or
claim "these models generalize to AI-generated images in general".
- We say this explicitly in the OOD methodology slide and in
  `docs/methodology/ood_methodology.md` § "Honest limitations".

---

## About the ensemble

**13. Probability averaging is a fair ensemble baseline.** It does not
need a held-out meta-training set, which matters when n = 2 000 OOD makes
stacking high-variance. Other ensemble rules (logit averaging,
temperature-weighted averaging, stacking with a logistic meta-learner)
would likely move the ensemble AUROC by less than the difference between
"best pair" and "best triple" — but we did not measure this.

**14. The leave-one-out attribution of model contributions is fair.** When
we say "the frequency detector's contribution to the ensemble is −2.05 pp
on OOD", we mean specifically that removing it from the five-model
probability average improves OOD AUROC by 2.05 pp. This is a marginal
metric, not a causal claim about whether the model is "useful" in some
broader sense — the frequency detector is still the only model in the
lineup that yields per-band interpretability.

---

## What we are *not* assuming

A few things worth stating explicitly because audiences sometimes assume
they're implicit:

- We do **not** assume the five-model lineup is exhaustive. Many obvious
  alternatives are listed in `docs/FUTURE_WORK.md` § "Architecture & scaling"
  (DINOv2, larger ViTs, multi-foundation-model ensembles).
- We do **not** assume that in-distribution AUROC predicts cross-generator
  AUROC. The headline finding of this project is precisely that it does not.
- We do **not** assume that the most accurate model is the most useful
  model. The frequency detector is the weakest classifier in the lineup
  but earns its slot because it produces interpretable per-band
  attributions.
