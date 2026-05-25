# Training Choices — Regularization & Generalization Checks

This doc consolidates per-model regularization choices and how we checked for
overfitting / underfitting. Detailed per-model training schedules live in the
notebooks; this page is the auditable summary.

## Shared protocol (every model)

| Concern | Defense | Source |
|---|---|---|
| Train/test contamination | Frozen 90/10 stratified split, seed = 42; sealed 20 k test touched once per model | `src/freq_detector/datasets.py::make_splits` |
| Threshold overfit | Decision threshold is val-derived Youden-J, never test-tuned | `src/eval_harness.best_threshold` |
| Reporting bias | Every model goes through the same harness with the same metrics and the same sealed split | `src/eval_harness.evaluate(...)` |
| Class imbalance | CIFAKE is naturally balanced (50/50 in train, val, test); stratified split preserves the balance | `data/README.md` |

## Per-model regularization + overfit check

### CNN (from scratch)

- **Regularization.** Spatial-augmentation pair from the standard CIFAR-10
  recipe — `RandomHorizontalFlip` + `RandomCrop(32, padding=4)` — plus
  AdamW weight decay = 1e-4 and 25-epoch cosine LR decay. No dropout.
- **Overfit check (numeric).** From the full training log persisted at
  `results/per_model/cnn_baseline_yin/training_report_yin.json`:
  - Final epoch train loss = 0.0503, val loss = 0.0670 → **gap ≈ 0.017**, small.
  - Best val accuracy = 0.9766 reached at epoch 23 of 25; the model was
    still improving on val, not over-fitting.
  - Reported test AUROC 0.9974 vs val AUROC 0.9991 → **gap = 0.0017**, well
    within the noise of n = 20 000 vs n = 10 000.

### ResNet-18 (ImageNet transfer)

- **Regularization.** Partial freeze of early layers (conv1, bn1, layer1,
  layer2) — only layer3, layer4, and the new 2-way head receive gradient
  updates. Limits the effective parameter count to ~7 M of the 11.2 M total
  and prevents over-adaptation to CIFAKE's specific style.
- **Overfit check.** Test AUROC 0.9977 vs OOD AUROC 0.9341 → −6.4 pp drop.
  The in-distribution number is essentially at ceiling; the OOD drop is
  large enough that further in-distribution training would not have helped.

### ViT-Small (full fine-tune)

- **Regularization.** Standard timm augmentation. Test AUROC 0.9994 vs the
  next-best in-distribution model at 0.9977 — the 0.0017 spread says all
  spatial models are near-ceiling on in-distribution test, so additional
  capacity is not the bottleneck.

### Frequency detector (Variants A + B)

- **Regularization.** Variant A is a 29-d feature vector fed into a 4 k-param
  shallow MLP — capacity is structurally bounded by the input dimensionality.
  Variant B is a 222 k-param CNN; we use BCE-with-logits and a
  weight-decay-only AdamW (no dropout). No spatial augmentation — the input
  is a spectrum, where horizontal flips don't make physical sense.
- **Overfit check.** From `docs/findings/freq_detector.md` § "M4 — Ablation
  grid": Variant A val AUROC = 0.905 → test AUROC = 0.900 (Δ = 0.005),
  Variant B val AUROC = 0.944 → test AUROC = 0.944 (Δ = 0.000). Both
  variants generalize cleanly from val to test.

### CLIP probe (frozen encoder + 132 k MLP head)

- **Regularization.** The frozen encoder is the strongest form of
  regularization in the lineup — the only trainable parameters are 132 k
  on top of a 151 M-param frozen backbone. The trainable head also includes
  dropout = 0.3 between its two layers. Training uses AdamW with weight
  decay 1e-4, BCE-with-logits loss, and val-AUROC early stopping (patience
  = 6 epochs).
- **Underfit / overfit check.** Test AUROC 0.9968 vs OOD AUROC 0.9485. The
  small head reaches the spatial-CNN tier on test, which means it is not
  underfitting; the OOD gap is on par with the spatial models, which means
  it is not overfitting either.

## What the headline numbers tell us about over- and underfitting

The cleanest joint diagnostic across all five models:

| Model | Test AUROC | OOD AUROC | OOD drop | Reading |
|---|---:|---:|---:|---|
| CNN (from scratch) | 0.9974 | 0.9429 | −5.5 pp | Tight |
| ResNet-18 | 0.9977 | 0.9341 | −6.4 pp | Tight |
| ViT-Small | 0.9994 | 0.9732 | −2.6 pp | Tight; generalizes best |
| Frequency detector | 0.9435 | 0.8150 | −12.8 pp | Constrained by design — and the OOD drop shows it |
| CLIP probe | 0.9968 | 0.9485 | −4.9 pp | Frozen encoder gives the most stable test↔OOD profile per trained parameter |

Every model has a positive drop, meaning every model is at least somewhat
generator-specific (overfit to SD-1.4). None of the models is *under*fit —
all clear 0.94 in-distribution AUROC. The cross-model spread on OOD (10.3 pp
between ViT and freq) is the real diagnostic — different inductive biases
overfit by different amounts.

## Selection rationale (model + ensemble)

| Goal | Choice | Justification |
|---|---|---|
| Maximum in-distribution test AUROC | Best 2-model ensemble: Nathan ResNet + Alex ViT → 0.9993 | Numerically best (see `results/team_ensemble_report.json`) |
| Maximum cross-generator OOD AUROC | Best 3-model ensemble: Yin CNN + Alex ViT + CLIP probe → 0.9670 | +0.13 pp over best pair, +0.21 pp over single best model |
| Most stable single model | Alex ViT-Small | Smallest OOD drop in the lineup (−2.6 pp) |
| Most interpretable single model | Frequency detector | Per-band attribution (3.3× HF/JPEG weight ratio); no other model decomposes this way |

The frequency detector is **excluded from the ensemble pool** by an
empirical leave-one-out check: from `results/five_model_loo.json`, removing
the frequency detector from the five-model ensemble *improves* test AUROC
by +0.04 pp and OOD AUROC by **+2.05 pp**. It earns its slide as a
characterization model, not a competitive classifier.
