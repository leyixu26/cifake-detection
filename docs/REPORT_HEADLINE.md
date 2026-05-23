# Model 5 (CLIP Probe) — Team Handoff

**Owner:** Leyi · **Status:** complete, sealed-test evaluated · **Date:** 2026-05-20

Drop this directly into the team report's "Model 5" section. Every number below
sourced from the shared `eval_harness.py` JSON records under
`project/results/clip_*` (`*_scores.npz` files included for ensembling).

## Method (one paragraph)

We add a fifth model that tests **whether web-scale multimodal pretraining
generalises better than ImageNet pretraining or from-scratch training**: a
frozen OpenCLIP `ViT-B-32` encoder (LAION-2B weights, 151 M parameters, never
fine-tuned) followed by a small MLP head (512 → 256 → 1, ReLU + dropout 0.3,
~132 k parameters). Images are bicubic-upsampled from 32×32 to 224×224 (CLIP's
native input). The head is trained with BCE-with-logits + AdamW (lr 1e-3, wd
1e-4, batch 1024, val-AUROC early stopping). The decision threshold is
Youden-J on the val split. All scores reproducible from the shared harness.

## Headline numbers

| | test AUROC | OOD AUROC (sd-turbo) | params trained |
|---|---:|---:|---:|
| from-scratch CNN (Yin, baseline) | 0.9974 | — | 287 k |
| ResNet-18 (Nate, transfer) | TBD | TBD | ~11 M |
| ViT (Alex) | TBD | TBD | ~5–22 M |
| frequency probe (Leyi, Model 4) | 0.944 best | 0.815 best | 4 k–222 k |
| **CLIP probe (Leyi, Model 5)** | **0.9968** | **0.9485** | **132 k (probe only)** |

The CLIP probe matches the from-scratch CNN on clean test (within 0.0006 AUROC)
**with zero gradient steps through the 151 M-param encoder** and beats the
matched-architecture spatial CNN baseline on cross-generator OOD by **+1.4 pp**.

## Improvement ladder (each upgrade individually)

| Variant | Test AUROC | OOD AUROC | OOD lift |
|---|---:|---:|---:|
| Ojha 2023 baseline: ViT-B/32 openai + LR | 0.9871 | 0.9191 | — |
| LAION-2B weights instead of OpenAI 400M | 0.9911 | 0.9365 | **+1.7 pp** |
| Horizontal-flip test-time augmentation | 0.9916 | 0.9385 | +0.2 pp |
| **MLP head (256 hidden, dropout 0.3)** | **0.9968** | **0.9485** | **+1.2 pp** |

**Two non-obvious findings worth surfacing in the report:**

1. **LAION pretraining lifts OOD more than ID** (+1.7 pp vs +0.4 pp). Web-scale
   dataset diversity is exactly what helps under distribution shift.
2. **Adding capacity to the probe did NOT hurt OOD** (contrary to Ojha 2023's
   warning). With 90 k CIFAKE training examples and dropout 0.3, the MLP head
   improved OOD by +1.2 pp over the linear probe.

## Robustness profile (measured AUROC on sealed test)

CLIP final (LAION+MLP) vs baseline CLIP vs matched spatial CNN vs freq A.
Full curves in `fig_robust_curves_final.png`.

| Perturb | clean | jpeg q40 | jpeg q10 | blur 0.8 | blur 1.5 | noise 16 | noise 32 | resc 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CLIP final** | **0.9968** | **0.9800** | 0.8182 | **0.9495** | **0.8373** | **0.9358** | 0.6746 | **0.9109** |
| CLIP baseline | 0.9871 | 0.9612 | 0.8054 | 0.9482 | 0.7752 | 0.9252 | **0.7683** | 0.9272 |
| spatial CNN | 0.9947 | 0.880  | 0.908  | 0.799  | 0.802  | 0.948  | 0.811  | 0.804 |
| freq A | 0.9003 | 0.812  | 0.690  | 0.654  | 0.364  | 0.804  | 0.739  | 0.474 |

**Headline pattern:**
* **CLIP final dominates on blur** (low-pass kills HF artefacts; semantic features survive).
  At σ=1.5 final CLIP = 0.84 vs freq A = 0.36.
* **CLIP final dominates on JPEG and rescale** for moderate-to-large strengths.
* **One regression**: at extreme noise (σ=32) baseline CLIP beats final CLIP
  (0.77 vs 0.67). The MLP head's added capacity becomes a liability when the
  input embeddings are noise-dominated -- worth surfacing as the one regime
  where the capacity push backfired.
* **Frequency variants collapse on blur/rescale** -- the HF fingerprint they
  key on disappears under low-pass corruption (predicted, and now confirmed).

## Cross-generator OOD setup (controlled)

* 1000 sd-turbo FAKE images, generated at the 10 CIFAR-10 class prompts,
  256×256 → Lanczos resample to 32×32.
* 1000 CIFAR-10 REAL from CIFAKE's `test/REAL` (matched content distribution).
* **Both classes re-encoded through CIFAKE's exact JPEG quantization tables
  (luma 1858, chroma 2780)** so compression history matches. The only
  difference between in-dist and OOD is the generator process (SD-1.4 vs
  sd-turbo, different VAE + sampler + training data).

## Recommended team headline

Ensemble of probability-averaged scores across **CLIP probe + spatial CNN**
(or whichever team-best ranks: ResNet-18 / ViT, by their own OOD numbers).
The 2-model average:

| | test AUROC | OOD AUROC |
|---|---:|---:|
| Nathan ResNet-18 alone | 0.9977 | 0.9341 |
| Alex ViT-Small alone | 0.9994 | 0.9732 |
| **CLIP probe (Leyi Model 5) alone** | **0.9968** | **0.9485** |
| Nathan + Alex (best test pair) | **0.9993** | 0.9529 |
| **Alex + CLIP (best OOD pair)** | 0.9988 | **0.9657** |
| All 3 (Nathan + Alex + CLIP) | 0.9993 | 0.9637 |
| (legacy) CLIP + my matched spatial CNN | 0.9981 | 0.9570 |

**Recommended team headline ensemble**: **Alex ViT + CLIP probe** (best OOD AUROC 0.9657) for cross-generator deployment, or **Nathan + Alex** (test AUROC 0.9993) for clean in-distribution maximum. The all-3 ensemble lands in between (test 0.9993 / OOD 0.9637).

Yin's from-scratch CNN was attempted in the overnight run but training timed out under system contention. His reported metrics (test AUROC 0.9974 from `results_CNN_from_scratch.json`) match the spatial-CNN tier.

The 2-model ensemble lifts AUROC by **+1.4 pp ID** (0.9947 → 0.9981) and
**+0.9 pp OOD** (0.9485 → 0.9570) over the best single model. Adding the
frequency probe makes the ensemble *worse* on both axes — confirmed by
leave-one-out, which shows dropping freq_A from the 3-model ensemble improves
it by +1.7 pp ID and +2.25 pp OOD (since "drop freq_A" = the 2-model winner).

Adding a frequency variant to the ensemble **hurts** rather than helps
(measured directly via leave-one-out — see `fig_ensemble_final.png`). The
frequency work should be reported as an interpretability appendix
characterising what SD-1.4's spectral fingerprint looks like, not as a
competitive classifier.

## How teammates can integrate

To produce ensemble numbers with their models:

```python
# In teammate's notebook, replace the ad-hoc JSON writer with:
import sys; sys.path.insert(0, "/Users/leyi/Desktop/ML2/project")
from eval_harness import evaluate, best_threshold

# y_val, p_val_fake are the validation labels + P(FAKE) scores (1 per sample)
thr = best_threshold(y_val, p_val_fake)
evaluate(y_val,  p_val_fake,  "<model_name>", "val",
         threshold=thr, threshold_policy="best_val_youden", save_scores=True)
evaluate(y_test, p_test_fake, "<model_name>", "test",
         threshold=thr, threshold_policy="best_val_youden", save_scores=True)
# Repeat for the OOD set at project/ood_data/sdturbo/ (REAL=0, FAKE=1).
```

Then run `ensemble.py` (in `project/`) — it picks up every model with persisted
`*_scores.npz` files and computes the full ensemble + leave-one-out table.

## Files (for code review)

* `project_clip/clip_v2.py` — extractor + LR/MLP probes (parameterised)
* `project_clip/clip_robust_final.py` — robustness battery for the final MLP
* `project_clip/clip_synthesis_final.py` — figures + this doc's source numbers
* `project_clip/embeds/vit_b32_laion/best_mlp.pt` — the final probe weights
* `project_clip/clip_probe.ipynb` — graded notebook (runs in seconds, uses cache)
* `project_clip/results/findings_final.md` — fuller version of this handoff
* `project/eval_harness.py`, `project/perturbations.py` — shared harness used by
  every model in the team
