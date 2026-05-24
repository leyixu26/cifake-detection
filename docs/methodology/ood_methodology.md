# Methodology — Cross-Generator OOD Evaluation

## What "OOD" means here, specifically

OOD = out-of-distribution. For **AI-image detection** the OOD that matters
is **cross-generator**: train your detector on images from generator X, test
on images from generator Y the detector has never seen.

This is different from:
- standard test set (in-dist; samples from same generator)
- generic domain shift (different sensor / domain)

**Why cross-generator matters in deployment:** by the time an AI-image
detector is in production, new generators have been released that weren't
in its training data. A detector that only works against generators it saw
during training is useless.

## The core principle: change one thing at a time

A naive OOD test would confound the generator change with content
differences, resolution differences, codec differences, etc. **Our OOD set
holds every variable constant except the generator process.**

## Our specific OOD set (`data/ood_sdturbo/`)

| Variable | In-distribution (CIFAKE) | OOD (sd-turbo) | Same? |
|---|---|---|---|
| Real-image source | CIFAR-10 | CIFAR-10 (sampled from CIFAKE `test/REAL`) | ✓ |
| Content distribution | 10 CIFAR classes | Same 10 classes (via class-name prompts) | ✓ |
| Image resolution | 32×32 | 32×32 (Lanczos resize from 256×256 generation) | ✓ |
| JPEG quantization tables | luma 1858, chroma 2780 | **Same tables** (re-encoded explicitly to match) | ✓ |
| Generator | Stable Diffusion 1.4 | sd-turbo (different VAE + sampler + training data) | **✗ ← the only difference** |

## The compression-confound control (the methodologically subtle part)

The first attempt at generating the OOD set saved the sd-turbo images with
PIL's default JPEG settings (quantization tables sum to ~927). But CIFAKE
uses a different encoder (qt sum 4638). That mismatch within the OOD set
would let a frequency-based detector exploit the JPEG block-grid difference
instead of the generator difference.

**Fix:** read CIFAKE's exact quantization tables off a sample image, then
re-save every OOD image (both REAL and FAKE) through those tables. Both
classes now have byte-identical compression history; the only signal a
detector can use is the generator itself. See
`scripts/generate_ood.py::reencode_with_cifake_qtables`.

This is the kind of confound that looks like a detail but determines whether
your result means anything. A figure summarising the qtable check is a good
candidate for the deck's methodology slide.

## Threshold discipline

The val-derived Youden-J threshold is **reused unchanged** on OOD. We never
tune the threshold on OOD — that would constitute peeking. Same threshold
across test, OOD, and every robustness perturbation.

## Interpretation — magnitude bands

Rules of thumb from Wang 2020, Ojha 2023, Corvi 2023:

| AUROC drop (in-dist → OOD) | Interpretation |
|---|---|
| < 5 pp | Small. Within normal generalisation variance; doesn't distinguish models. |
| 5-15 pp | Moderate. Real OOD difficulty; most spatial detectors hit this band on cross-architecture (GAN ↔ diffusion). |
| > 15 pp | Large. Detector substantially overfit to its training generator. |

Applied to our results:

| Model | OOD drop |
|---|---:|
| Alex ViT-Small | **−2.6 pp** (smallest) |
| CLIP probe (LAION+MLP) | −4.9 pp |
| Nathan ResNet-18 | −6.4 pp |
| Frequency detector (handcrafted) | −9.2 pp |
| Frequency detector (mag. CNN) | −12.8 pp |

## Honest limitations

- **Single OOD generator.** sd-turbo is one specific Stable-Diffusion-family
  model. A multi-generator test (e.g. ForenSynths' 10 generators or
  GenImage) would be stronger.
- **Same generator family.** sd-turbo is distilled latent diffusion — close
  to SD-1.4. A cross-architecture test (diffusion → GAN) would likely
  separate models more.
- **Matched content.** Both classes are conditioned on CIFAR-10 class names;
  a real adversary could generate arbitrary content.
- **Low resolution.** 32×32 limits the resolvable spectral fingerprints;
  findings may not transfer to high-resolution deployment.

## Why this matters for the deck

Three concrete recommendations:

1. **Include a "OOD set construction" slide** showing the matched-everything-
   except-generator table above. Reviewers will ask how you isolated the
   generator change; this slide answers preemptively.
2. **Mention the JPEG quantization-table control explicitly.** A single
   sentence is enough: *"both OOD classes were re-encoded through CIFAKE's
   exact JPEG quantization tables to control for compression confounds."*
3. **Acknowledge the single-generator limitation.** Strong scientific
   communication includes honest scope statements; "multi-generator and
   cross-architecture evaluation are obvious next steps" is much stronger
   than overclaiming.
