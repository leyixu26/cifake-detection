# Leyi — slide content (3 slides)

**Owner:** Leyi
**Total target duration:** ~4 min (90 s + 90 s + 60 s, with a 30 s buffer for slide 2 transitions)
**Position in deck:** slides 7, 8a, 8b in the team-agreed ordering. Slide 1 here = team slide 7 (Model 4). Slides 2-3 here = team slide 8, split for breathing room.

---

## Slide 1 — Frequency detector: a spectrum-input baseline (~90 s)

### Title on slide
**Model 4 — Frequency detector**
*Spectrum-only inputs; lowest test AUROC of the five models.*

### Bullets on slide
- **Input:** log |FFT| of the 32×32 RGB image. No pixel-domain features.
- **Two variants evaluated.** Variant A: 29-dim handcrafted features (radial PSD + azimuthal sectors + scalar summaries) → shallow MLP (~4 k params). Variant B: full 2-D log-magnitude → small CNN (~222 k params). Headline numbers below are Variant B.
- **In-distribution test AUROC: 0.9435**, the lowest of the five models. Input is restricted to spectral magnitude.
- **Why include it:** weights are attributable to named frequency bands, which is not true of the other four models.

### Primary figure on slide
`results/figures/09_freq_lr_coefficients.png`

**One-line caption beneath the figure:** Variant-A logistic-regression weights by feature group. The high-frequency tail (radial bins 12–16) carries 3.3× the average weight of the JPEG-block-grid bin (radial 4), consistent with the detector responding to generator artifacts rather than JPEG compression.

### What Leyi says (speaker notes, not on slide)
> "Model 4 takes the FFT magnitude of the image as input, so it cannot see pixel-domain features. In-distribution AUROC is 0.9435, the lowest of the five models. The reason we include it is that its weights map onto specific spectral bands. The figure shows the logistic-regression coefficients grouped by feature family. The high-frequency tail carries 3.3× the average weight of the JPEG-block bin, which is the standard confound check for frequency-based detectors. That ratio is the evidence behind the claim that the detector responds to generator artifacts rather than to JPEG compression history."

### Hand-off line to slide 2
> "Now I'll show what happens to all five models — including this one — when we evaluate against a different generator."

### Anticipated questions
- **Why use spectral features when spatial CNNs are higher AUROC?** → "Spatial CNNs report higher AUROC, but their decisions do not decompose into named frequency bands. This model is included for the per-feature attribution, not as a competing classifier." Cite: `docs/methodology/frequency_detector.md`.
- **At 32×32, is there enough frequency content to learn from?** → "There are ~16 usable radial bins at 32×32. Cohen's d at the highest-frequency bin is 1.41 on a held-out diagnostic sample. That is the discriminative signal the model uses." Cite: `docs/methodology/frequency_detector.md` § "The 32×32 constraint".

### Cite-back data
| Metric | Value | Source |
|---|---:|---|
| Test AUROC (Variant B, headline) | 0.9435 | `results/per_model/freq_detector/test.json` |
| Test AUROC (Variant A, ShallowMLP) | 0.900 | `docs/findings/freq_detector.md` § "Two variants" (ShallowMLP row) |
| HF / JPEG-bin coefficient ratio | 3.3× | `docs/methodology/frequency_detector.md` § "Confound controls" (computed as 0.272 / 0.082) |
| Variant B parameter count | ~222 k | `docs/findings/clip_probe.md` (model-card row); `models/README.md` |
| Cross-generator OOD AUROC | 0.8150 | `results/per_model/freq_detector/ood_sdturbo.json` |

---

## Slide 2 — Cross-generator OOD evaluation (~90 s, the longest slide)

### Title on slide
**Cross-generator OOD evaluation**
*Same task, unseen generator: train on SD-1.4, evaluate on sd-turbo.*

### Bullets on slide
- **OOD set.** 1 000 sd-turbo generations (100 per CIFAR-10 class prompt) + 1 000 CIFAR-10 originals. n = 2 000.
- **Controls held constant.** Resolution (32×32), content distribution (matched class prompts), JPEG quantization tables (both classes re-encoded through CIFAKE's exact luma/chroma tables). Only the generator differs.
- **Per-model drops** (in-dist → OOD AUROC):
  - Alex ViT-Small: 0.9994 → 0.9732 (−2.6 pp)
  - CLIP probe: 0.9968 → 0.9485 (−4.9 pp)
  - Yin from-scratch CNN: 0.9974 → 0.9429 (−5.5 pp)
  - Nathan ResNet-18: 0.9977 → 0.9341 (−6.4 pp)
  - Frequency detector: 0.9435 → 0.8150 (−12.8 pp)
- **Note.** Yin's from-scratch CNN drops 0.9 pp less than Nathan's ImageNet-pretrained ResNet. At n = 2 000 we report this as a point estimate; no significance test was run.

### Primary figure on slide
`results/figures/01b_ood_reveal.png`

**One-line caption beneath the figure:** In-distribution test AUROC (muted bars) and cross-generator OOD AUROC (hatched bars) for the five models, with the per-model drop annotated below each pair.

### What Leyi says (speaker notes)
> "Every spatial model cleared 0.997 AUROC on the CIFAKE test split. That number alone does not tell us whether they detect Stable Diffusion in general or just this one generator. To separate those, we built a second test set using sd-turbo, a different diffusion model. Resolution, content distribution, and JPEG quantization tables are held constant — only the generator process changes between in-distribution and OOD. The drops range from 2.6 percentage points for Alex's ViT to 12.8 for the frequency detector. One thing to note: Yin's CNN trained from scratch on CIFAKE drops a little less than Nathan's ImageNet-pretrained ResNet, by 0.9 pp. At n = 2 000 we report this as a point estimate and have not run a significance test."

### Hand-off line to slide 3
> "These five models have different drops because they have different inductive biases. The next slide is what we can do with that diversity."

### Anticipated questions
- **How do we know the JPEG quantization tables actually match?** → "`scripts/generate_ood.py` extracts CIFAKE's quantization tables and re-encodes every OOD image through them. The frequency detector is the natural confound check: a pure JPEG-history detector would *gain* on the JPEG-equalized version; ours doesn't (Δ = 0.007 AUROC after re-encoding)." Cite: `docs/methodology/ood_methodology.md` § "JPEG quantization-table control"; `docs/methodology/frequency_detector.md` § "Confound controls".
- **Why sd-turbo specifically?** → "Same diffusion family as SD-1.4 (so content quality is comparable), single-step distilled (so the sampler is different from the multi-step DDPM that produced CIFAKE), publicly available. Multi-generator and cross-architecture extensions are listed in the future-work slide." Cite: `data/ood_sdturbo/README.md`.
- **Is −5.5 vs −6.4 pp statistically significant?** → "At n = 2 000 we have not bootstrapped confidence intervals. We report point estimates only. A bootstrap CI for AUROC differences would be the right next step before publishing this observation." (No source needed — this is honest scoping.)

### Cite-back data
| Metric | Value | Source |
|---|---:|---|
| OOD set composition | 1 000 REAL + 1 000 FAKE | `data/ood_sdturbo/README.md` |
| CIFAKE qtable (luma / chroma) | 1858 / 2780 | `docs/methodology/ood_methodology.md` |
| Per-model OOD AUROCs | see bullet list above | `results/per_model/<model>/ood_sdturbo.json` |
| Frequency OOD drop | −12.8 pp | `results/per_model/freq_detector/ood_sdturbo.json` |
| JPEG-equalized retrain Δ (freq) | 0.007 AUROC | `docs/methodology/frequency_detector.md` § "Confound controls" |

---

## Slide 3 — CLIP probe and the team ensemble (~60 s, brisk close)

### Title on slide
**A frozen-encoder probe and the team ensemble**

### Bullets on slide
- **Model 5 — CLIP probe.** OpenCLIP ViT-B/32 with LAION-2B weights, encoder frozen (151 M params, no gradient updates). A 132 k-param MLP head (512 → 256 → 1, ReLU, dropout 0.3) is trained on the encoder outputs. Test AUROC 0.9968, OOD AUROC 0.9485.
- **Ensembles** are probability averages over the four spatial models. The frequency detector is excluded from the ensemble pool: in a five-model leave-one-out, removing it changes test AUROC by +0.04 pp and OOD AUROC by +2.05 pp.
- **Best 2-model subset by test AUROC:** Nathan + Alex = 0.9993.
- **Best 2-model subset by OOD AUROC:** Alex + CLIP = 0.9657.
- **Best 3-model subset by OOD AUROC:** Yin + Alex + CLIP = 0.9670. Lift over the best pair is +0.13 pp; significance not tested at n = 2 000.

### Primary figure on slide
`results/figures/03_team_ensemble.png`

**One-line caption beneath the figure:** Per-model and ensemble AUROCs on the sealed test (left) and the sd-turbo OOD set (right). Bar colour encodes group: single models (blue), pairs (light green), triples (medium green), all-four ensemble (dark green).

### What Leyi says (speaker notes)
> "Two things on this slide. First, the CLIP probe — model 5. The encoder is a 151-million-parameter CLIP backbone with LAION-2B weights, frozen. We train only a 132-thousand-parameter MLP head on top of its outputs. The probe matches Yin's from-scratch CNN on the sealed test to within 0.0006 AUROC, and exceeds Nathan's ResNet by 1.4 percentage points on OOD. Second, the ensembles. We average the predicted FAKE probabilities across the four spatial models. The frequency detector is excluded from this pool: in a five-model leave-one-out, removing it improves test AUROC by 0.04 pp and OOD AUROC by 2.05 pp. The best two-model subset on test is Nathan plus Alex. The best two-model subset on OOD is Alex plus CLIP. The best three-model subset on OOD adds Yin's CNN, gaining 0.13 pp over the pair. We have not run a significance test on that lift; n is 2 000 and we report point estimates."

### Hand-off line to next slide (Alex's future work)
> "Alex will close with what we'd test next to put these observations on firmer statistical ground."

### Anticipated questions
- **Why frozen encoder rather than fine-tuned?** → "The frozen encoder follows the Ojha 2023 recipe. We also wanted to isolate one variable: whether the pretraining data alone (LAION-2B vs OpenAI 400M, same architecture) changes OOD behaviour. The capacity-ladder figure (`04_clip_capacity_ladder.png`) shows that swapping OpenAI weights for LAION-2B — same encoder, same MLP head — accounts for +1.7 pp OOD. That is what we attribute to pretraining-data diversity rather than to fine-tuning." Cite: `docs/findings/clip_probe.md` § "Capacity ladder".
- **Why probability average rather than stacking or logit averaging?** → "Probability averaging is the simplest combination rule that does not require a held-out meta-training set. At n = 2 000 OOD, fitting a stacker on those scores would have high variance. We report the simplest rule that uses all four signals."
- **Why is the frequency detector excluded from the ensemble pool?** → "In a five-model leave-one-out, removing the frequency detector improves test AUROC by 0.04 pp and OOD AUROC by 2.05 pp. Including it lowers both ensemble numbers." Cite: computed from `results/per_model/*/test_scores.npz` and `ood_sdturbo_scores.npz` (the per-sample scores persisted by `scripts/evaluate.py`); the four-model ensemble in `results/team_ensemble_report.json` already reflects the exclusion.

### Cite-back data
| Metric | Value | Source |
|---|---:|---|
| CLIP probe test AUROC | 0.9968 | `results/per_model/clip_mlp_vit_b32_laion/test.json` |
| CLIP probe OOD AUROC | 0.9485 | `results/per_model/clip_mlp_vit_b32_laion/ood_sdturbo.json` |
| Capacity-ladder lifts (OOD pp, incremental) | LAION +1.7, TTA +0.2, MLP head +1.0 | `docs/findings/clip_probe.md` § "Capacity ladder" |
| Best test pair AUROC | 0.9993 (Nathan + Alex) | `results/team_ensemble_report.json` |
| Best OOD pair AUROC | 0.9657 (Alex + CLIP) | `results/team_ensemble_report.json` |
| Best OOD triple AUROC | 0.9670 (Yin + Alex + CLIP) | `results/team_ensemble_report.json` |
| 5-model LOO: drop freq, test AUROC | +0.04 pp (0.9990 → 0.9994) | `results/five_model_loo.json` |
| 5-model LOO: drop freq, OOD AUROC | +2.05 pp (0.9449 → 0.9655) | `results/five_model_loo.json` |
| Trained CLIP-probe params | 132 k (151 M encoder frozen) | `models/README.md` |

---

## Style notes for the deck

- Use the data colors already in the figures (Yin green, Nathan red, Alex purple, freq blue, CLIP violet). Don't recolor for the deck.
- Numbers on the slide should be exactly as listed above (4-dp AUROC; signed pp drops). Do not round to 3 dp on the slide if the body uses 4 dp.
- Slide 2's figure (`01b_ood_reveal.png`) is the only figure where the in-dist bars are muted and the OOD bars are emphasized — that styling is deliberate; do not "fix" the alpha in a slide-tool edit.
- If `03_team_ensemble.png` looks crowded on the projector, the cite-back table on slide 3 has the same numbers in larger type as a fallback talk-track.

## Figures referenced by this slide block

All paths relative to repo root.

| Slide | Figure | Used for |
|---|---|---|
| 1 | `results/figures/09_freq_lr_coefficients.png` | LR weight bar chart (HF tail vs JPEG bin) |
| 2 | `results/figures/01b_ood_reveal.png` | 5-model in-dist vs OOD reveal |
| 3 | `results/figures/03_team_ensemble.png` | Per-model + pair + triple + all-4 ensembles, test and OOD panels |
| 3 (backup) | `results/figures/04_clip_capacity_ladder.png` | CLIP capacity ladder (for the "why frozen / why LAION" question) |
| 3 (backup) | `results/figures/05_clip_vs_spatial_agreement.png` | 4.4% complementary errors (for the "why does ensemble help" question) |
