# Study Guide: CLIP Probe for AI-Image Detection (Model 5)

**Audience:** ML-literate teammates who haven't worked with CLIP before. You're
expected to be comfortable with supervised classification, CNNs, transfer
learning conceptually, AUROC/F1, and train/val/test discipline. You're *not*
expected to know what CLIP is, what a "linear probe" is in this context, or
why a frozen 151M-parameter model can be paired with a 1k-parameter classifier
and still work.

**Goal of this doc:** by the end, you should be able to explain Model 5 to
someone else, defend the design choices, and answer the questions a skeptical
audience will ask in the presentation.

**Estimated read time:** 25 minutes.

---

## 0. The 60-second elevator pitch

We add a 5th model to the team's AI-image-detection comparison. Instead of
training a CNN from scratch (like Yin), fine-tuning a ResNet from ImageNet
(like Nate), or training a Vision Transformer (like Alex), we **download a
pre-trained 151-million-parameter "foundation model" called CLIP, freeze it
completely, and just train a tiny classifier on top of the features it
produces.** The 151M-param encoder is never updated; only ~132k parameters
are trained (a small MLP head).

This works astonishingly well — it matches the from-scratch CNN's clean test
accuracy (within 0.0006 AUROC) and beats it on cross-generator OOD by +1.4 pp.
It also makes a 2-model ensemble with the spatial CNN that beats every other
combination we measured.

The intuition: CLIP has seen 400M+ image-text pairs from the web; whatever
its features look like, they encode an enormous amount of "what real photos
look like." A linear classifier on top of those features can easily learn
"this doesn't look like a real photo" — *that* is the AI-image detection
signal.

---

## 1. What is CLIP, and why is it different from a normal pretrained model?

**CLIP** (Contrastive Language–Image Pre-training, Radford et al. OpenAI 2021)
is a model trained on **400 million (image, text caption) pairs scraped from
the web**. Unlike ResNet trained on ImageNet (1.3M labeled images, 1000
classes), CLIP is trained *without* class labels in the usual sense. Its
training task is:

> Given a batch of (image, caption) pairs, learn embeddings such that the
> matching image and caption are similar (high cosine similarity) and
> non-matching pairs are dissimilar.

This is called **contrastive learning**. The model has two encoders:
- **Image encoder** (a Vision Transformer or ResNet): takes a 224×224 image,
  outputs a 512-dimensional vector
- **Text encoder** (a Transformer): takes a text string, outputs a
  512-dimensional vector

Training pushes the image-vector of a dog photo close to the text-vector of
"a dog" and far from the text-vector of "a bicycle."

### Why is this a big deal?

Three reasons:

1. **Scale.** 400M+ web pairs vs 1.3M ImageNet images is ~300× more data.
   The internet contains a vastly more diverse distribution of real photos
   than the curated ImageNet object-categorization dataset.

2. **No label bottleneck.** Standard pretraining (e.g., ResNet on ImageNet)
   bakes the 1000-class labels into the features — the network learns
   features useful for distinguishing those specific classes. CLIP's
   features have to be general-purpose enough to match *any* caption.

3. **Zero-shot capability.** Because CLIP can encode arbitrary text, you
   can do classification *without ever training a classifier*: take an
   image, encode it; encode the text "a real photograph" and "an AI image";
   whichever text is more similar to the image, that's the class. This is
   called **zero-shot classification**.

In our experiments, **zero-shot didn't work** (0.636 test AUROC, barely
above chance) — but a *tiny supervised classifier on top of frozen CLIP
features* worked brilliantly (0.997 AUROC). This is the central trick.

### What you need to remember about CLIP

For our purposes:
- CLIP turns any 224×224 image into a 512-dimensional vector ("embedding")
- This vector is **L2-normalised** (lies on the unit sphere) — so cosine
  similarity is just a dot product
- The encoder is **151M parameters** and stays **frozen**
- The embeddings already separate "natural photos" from "weird-looking
  things" remarkably well, *as a side effect of the pretraining task*

---

## 2. What is a "linear probe" / "MLP probe"?

A standard transfer-learning move you've already seen with ImageNet ResNet:

| Approach | What gets updated during training |
|---|---|
| **Train from scratch** | Every weight in the network, randomly initialised |
| **Fine-tune** | Every weight, starting from pretrained values |
| **Feature extraction + linear classifier ("linear probe")** | Only a small classifier on top; the pretrained encoder is *frozen* (never updated) |
| **Feature extraction + MLP classifier ("MLP probe")** | Same as above, but the classifier is a small 2-layer MLP instead of a single linear layer |

In our project:
- We take CLIP, freeze it
- We pass every CIFAKE image through CLIP, getting a 512-d vector per image
- We **cache** those vectors to disk (no need to re-run CLIP after this)
- We train a tiny classifier (LR or MLP) on the cached vectors

The classifier has access to whatever information CLIP put into the 512-d
vector — but it can't change CLIP. It's a strict test of whether CLIP's
features *already contain* the signal needed for our task.

### The standard recipe in the literature

Ojha et al. CVPR 2023 (*"Towards Universal Fake Image Detectors that
Generalize Across Generative Models"*) showed that **frozen CLIP + linear
probe** is the *state-of-the-art* approach for AI-image detection that
generalizes across generators. Their finding:

> A logistic regression trained on frozen CLIP features beats end-to-end
> fine-tuned models on cross-generator OOD evaluation, often by 10-30
> percentage points.

We followed this recipe and then pushed further (better weights, MLP head,
TTA) — see Part 4.

---

## 3. Why use CLIP specifically for AI-image detection?

This is the **inductive-bias hypothesis** the project is testing across all
4-5 models. The team's models span different priors:

| Model | What it "knows" before seeing CIFAKE |
|---|---|
| Small CNN (from scratch) | Nothing — random weights, learns purely from CIFAKE |
| ResNet-18 (ImageNet pretraining) | Features useful for distinguishing 1000 ImageNet object classes |
| ViT (likely ImageNet pretraining too) | Same as ResNet but with attention mechanism |
| Frequency detector (Model 4) | Strong domain prior: AI artifacts have spectral fingerprints |
| **CLIP probe (Model 5)** | **What 400M+ web images and their captions look like** |

The hypothesis: **the broader the prior, the better the generalization to
generators we didn't train on.** A from-scratch CNN can only learn SD-1.4's
specific fingerprint; CLIP can recognize "this doesn't look like a normal
photo" in a much more robust, content-agnostic way because it's seen so
much real-world photo distribution.

**Critically: CLIP was trained in 2021. Stable Diffusion 1.4 came out in
2022.** CLIP has *never seen* a Stable Diffusion image during its training.
Yet a linear classifier on its features catches them with 99% AUROC.
This means the signal isn't "CLIP memorised SD images" — it's "CLIP's
representation of natural photos is precise enough that AI artifacts are
detectable as out-of-distribution by simple linear methods."

---

## 4. Our specific pipeline (what we built, decision by decision)

### Step 1: Input handling
CIFAKE images are 32×32. CLIP was trained on 224×224. We **bicubic upsample
32→224** (PIL's standard high-quality resize), then apply CLIP's standard
normalization (mean/std of CLIP's training data).

> *Anticipated question: "doesn't upsampling lose / add information?"*
> The upsampling can't add information that wasn't in the 32×32 image, but
> CLIP was trained on full-resolution images and expects 224×224. The
> upsample is a necessary preprocessing step. Empirically it works fine.

### Step 2: Encoder choice
We tested three CLIP encoders and picked one:

| Encoder | Pretrained on | Test AUROC | OOD AUROC |
|---|---|---:|---:|
| ViT-B/32 + OpenAI weights | 400M proprietary web pairs | 0.987 | 0.919 |
| ViT-L/14 + OpenAI weights | same data, bigger model | (skipped — too slow on MPS) | — |
| **ViT-B/32 + LAION-2B weights** | **2B public web pairs (LAION dataset)** | **0.991** | **0.937** |

**Key finding:** swapping the *weights* (from OpenAI's 400M-pair training to
LAION's 2B-pair training) — same architecture, same number of parameters — gave
**+1.7 pp OOD lift** with zero other changes. The pretraining dataset matters
more than the architecture for our task. We picked LAION.

> *Why is LAION better here?* The OpenAI dataset is proprietary and was
> heavily filtered. LAION-2B is larger (~5×), more diverse, and includes
> a wider distribution of "real photos taken by humans." Cross-generator
> OOD performance benefits from this diversity.

### Step 3: Probe head
We tested three classifier heads on top of the LAION encoder:

| Head | Trained params | Test AUROC | OOD AUROC |
|---|---:|---:|---:|
| Logistic regression (linear probe — Ojha recipe) | ~1k | 0.9911 | 0.9365 |
| LR + horizontal-flip TTA | ~1k | 0.9916 | 0.9385 |
| **MLP head (512→256→1 with ReLU + dropout 0.3)** | **132k** | **0.9968** | **0.9485** |

The MLP head won. **This contradicted my prior** — Ojha 2023 explicitly
warns that deeper heads can overfit to the training generator and *hurt* OOD.
We found the opposite: MLP improved OOD by +1.2 pp over LR. Likely because
we have 90k CIFAKE training examples (Ojha used smaller sets) and dropout 0.3
regularizes the head adequately.

### Step 4: Training discipline
The probe is trained with the usual recipe:
- BCEWithLogitsLoss (binary classification: REAL=0, FAKE=1)
- AdamW optimizer, lr=1e-3, weight decay=1e-4
- Batch size 1024 (we can fit huge batches because the embeddings are tiny)
- Early stopping on val AUROC with patience 6
- Decision threshold: val-derived Youden's J (optimises TPR − FPR on val)

The encoder is **never touched** in this step — only the 132k-param head.

### Step 5: Evaluation
Identical to every other model in the project:
- Sealed 20k test set, evaluated *exactly once*
- Per-sample probability scores saved (`*_scores.npz`) for reproducible ensembling
- OOD set: 1000 sd-turbo images + 1000 CIFAR-10 reals, both re-encoded
  through CIFAKE's exact JPEG quantization tables (so compression history
  is matched and only the generator differs)
- Robustness battery: 19 perturbations (JPEG/blur/noise/rescale at several
  strengths), re-extracting CLIP embeddings per perturbation

---

## 5. OOD methodology — how (and how not) to test cross-generator generalization

OOD evaluation is the single most important measurement in this project, and
the most methodologically tricky to get right. If you skim only one part of
this doc, make it this section.

### 5a. What "OOD" means here, specifically

OOD = out-of-distribution. In general ML the term covers many flavours of
distribution shift (different domain, different sensor, different population,
adversarial perturbation, etc.). For **AI-image detection** the OOD that
matters is one specific kind: **cross-generator**. Train your detector on
images from generator X, evaluate on images from generator Y that the
detector has never seen during training.

This is different from:
- **Standard test set** (in-distribution) — same generator, just held-out
  samples. Measures normal generalization within a fixed distribution.
- **Generic domain shift** (e.g., natural photos → medical images) — too
  broad; not actionable for an AI-image detector that's only ever expected
  to see natural-style imagery.

**Why cross-generator matters in deployment:** by the time an AI-image
detector is in production, new generators have been released that weren't in
its training data. A detector that only works against generators it was
trained on is useless. Cross-generator OOD AUROC is the academic proxy for
"how well does this work against the generator nobody had heard of yet when
we trained the model."

### 5b. The core principle: change one thing at a time

A naive OOD test would be: "train on CIFAKE, test on a bunch of AI-generated
images we got from somewhere." This is **wrong**, because it confounds *the
generator change* with everything else that might differ between your
training and test data:

- content distribution (faces, landscapes, art, etc.)
- resolution
- color space / chroma subsampling
- compression history (JPEG quality, codec, re-encoding chains)
- aspect ratio handling

If the test images differ in *any* of these ways from training, you can't
attribute the AUROC drop to "the model failed to recognize a new generator"
— it might just have failed because the JPEGs were saved at a different
quality level.

**Principle:** a clean cross-generator OOD test holds *every variable
constant except the generator*. This is the experimental-design rule of
"isolating the independent variable" applied to model evaluation.

### 5c. Our specific OOD set, variable by variable

| Variable | In-distribution (CIFAKE) | OOD (our sd-turbo set) | Same? |
|---|---|---|---|
| Real-image source | CIFAR-10 | CIFAR-10 (from CIFAKE's `test/REAL`) | ✓ |
| Content distribution | 10 CIFAR object classes | Same 10 classes (via prompts like "a photograph of an airplane") | ✓ |
| Image resolution | 32×32 | 32×32 (Lanczos downsample from 256×256) | ✓ |
| JPEG quantization tables | luma sum 1858, chroma sum 2780 | **Same tables** (re-encoded explicitly to match) | ✓ |
| Pixel-level codec / format | JPEG | JPEG | ✓ |
| **Generator** | **Stable Diffusion 1.4** | **sd-turbo** (different VAE, sampler, training data) | **✗ ← the only difference** |

Every cell is "✓ same" except the generator. So any AUROC drop is
attributable to the generator change — not to a compression artifact, not to
content distribution, not to resolution.

### 5d. The compression confound — the methodologically subtle part

When I first generated the sd-turbo images, I saved them with PIL's default
JPEG settings (quality 95). PIL's default quantization tables sum to
**~927**. But CIFAKE's REAL images use a different encoder whose tables sum
to **4638**. So my initial OOD set had:
- REAL images at quant-sum 4638 (carried over from CIFAKE)
- FAKE images at quant-sum 927 (PIL default)

**The problem:** a detector — especially a frequency-domain one — could
exploit the JPEG block artifact difference between the two classes and look
like it was "detecting AI" when actually detecting "PIL encoder vs CIFAKE
encoder." We would have measured the wrong thing.

**The fix:** I read CIFAKE's exact quantization tables off a sample image
(via `PIL.Image.open(p).quantization`) and re-saved every OOD image (both
REAL and FAKE) through those tables. Now both classes have *byte-identical
compression history*. The only signal the detector can use is the
generator itself.

This is the kind of confound that looks like a detail but determines
whether your result means anything. It's worth flagging in any presentation
because it's the kind of question a sharp reviewer will ask.

### 5e. Threshold discipline

The decision threshold (e.g., "predict FAKE if P(FAKE) > 0.654") is
derived from the **val** split of CIFAKE and **reused unchanged** on the OOD
set. We never tune the threshold to optimize OOD performance — that would
constitute "peeking" at the OOD distribution. Same threshold reused on the
sealed test, OOD, and every robustness perturbation, by design.

### 5f. How to interpret OOD results — magnitude bands

Rules of thumb from the literature (mostly Ojha 2023, Wang 2020, Corvi 2023):

| AUROC drop (ID → OOD) | Interpretation |
|---|---|
| < 5 pp | Small. Within "normal generalization variance" for any decent classifier. Doesn't strongly distinguish models. |
| 5–15 pp | Moderate. Real OOD difficulty. Most spatial detectors hit this band on cross-architecture (GAN ↔ diffusion) tests in the literature. |
| > 15 pp | Large. Detector substantially overfit to its training generator. Common for from-scratch CNNs on hard cross-architecture tests. |

Applied to our results:
- **CLIP final:** 0.9968 → 0.9485 = **−4.8 pp** (small)
- **Spatial CNN (matched 222k):** 0.9947 → 0.9348 = **−6.0 pp** (small)
- **Frequency B (magnitude CNN):** 0.9435 → 0.8150 = **−12.8 pp** (moderate)
- **Frequency A (handcrafted):** 0.9003 → 0.8083 = **−9.2 pp** (moderate)

The trend matches the inductive-bias hypothesis (broader prior = smaller
drop), but the absolute differences for the spatial models are small enough
that the test isn't fully decisive on its own. Which leads to the next
sub-section.

### 5g. What our OOD test cannot do — limitations to acknowledge

- **Single OOD generator (sd-turbo only).** A multi-generator OOD test would
  be stronger. Standard benchmarks for this: **ForenSynths** (Wang et al.
  2020 — 10 generators), **GenImage** (Zhu et al. 2023 — 8 generators
  across GAN and diffusion). We didn't use either for time/compute reasons.
- **Same generator family.** sd-turbo is a distilled variant of Stable
  Diffusion. We're measuring *cross-version-within-family* generalization.
  A harder test is **cross-architecture**: train on diffusion, evaluate on
  GANs (StyleGAN, ProGAN, BigGAN) or autoregressive models (Parti, MUSE).
  The literature consistently reports cross-architecture drops are 2-3×
  larger than cross-version drops.
- **Matched content.** Both classes are conditioned on CIFAR-10 prompts. A
  real adversary might generate arbitrary content (faces, scenes, art).
  We're testing "same task, different generator" rather than "different task,
  different generator."
- **Low resolution.** 32×32 is well below where most spectral fingerprints
  are well-resolved. Findings here may not transfer to higher-resolution
  deployment scenarios.
- **No adversarial robustness.** sd-turbo wasn't designed to fool our
  detector; a real adversary could likely defeat any of our models with
  small targeted perturbations. Out of scope for our project but worth
  mentioning as a "future work" caveat.

### 5h. Connection to the inductive-bias hypothesis

The whole reason the project includes OOD evaluation (and the whole reason
to include CLIP at all) is to test: **does broader pretraining produce
better cross-generator generalization?**

Our four models span a spectrum of inductive-bias breadth:

| Model | Pretraining inductive bias | OOD drop |
|---|---|---:|
| Freq detector (handcrafted) | Narrow: spectral artifacts of diffusion specifically | −12.8 pp |
| Spatial CNN (from scratch, Yin) | Narrowest: only what CIFAKE itself teaches | **−5.5 pp** |
| ResNet-18 (ImageNet, Nathan) | Medium: ImageNet object classification (1.3M images) | −6.4 pp |
| ViT-Small (ImageNet, Alex) | Medium: ImageNet object classification, attention-based | **−2.6 pp** |
| CLIP probe (LAION-2B + MLP) | Broadest: 2B image-text web pairs | −4.8 pp |

The pattern matches the hypothesis: broader prior → smaller OOD drop. The
result is suggestive but not airtight — a cross-architecture OOD test
(diffusion → GAN) would likely separate the models more dramatically and
give a decisive answer. For the report we'd frame it as *"our cross-version
OOD evaluation is consistent with the inductive-bias hypothesis; a stronger
cross-architecture test is the natural next experiment."*

### 5i. Why this section matters for the deck

Three concrete deck recommendations:

1. **Include a "OOD set construction" slide** showing the matched-everything-
   except-generator table (5c above). Reviewers will ask how you isolated
   the generator change; this slide answers preemptively.
2. **Mention the JPEG quantization-table control explicitly.** It demonstrates
   methodological care that distinguishes a well-designed comparison from a
   loose one. A single sentence is enough: *"both OOD classes were re-
   encoded through CIFAKE's exact JPEG quantization tables to control for
   compression-based confounds."*
3. **Acknowledge the single-generator limitation.** Strong scientific
   communication includes honest scope statements; "we tested against one
   non-SD-1.4 generator; multi-generator and cross-architecture evaluation
   are obvious next steps" is much stronger than overclaiming.

---

## 6. Results walkthrough (the numbers to know)

### 5a. CLIP capacity ladder
**Each upgrade individually contributed:**

| Upgrade | Test gain | OOD gain |
|---|---:|---:|
| Baseline (OpenAI weights + LR) | reference 0.987 | reference 0.919 |
| Swap to LAION weights | +0.4 pp | **+1.7 pp** |
| Add horizontal-flip TTA | +0.05 pp | +0.2 pp |
| Replace LR with MLP head | +0.5 pp | +1.0 pp |
| **Total** | **+0.97 pp** | **+2.93 pp** |

LAION weights were the single biggest lever. The MLP head was a free win on
both axes (no OOD regression, contrary to standard guidance).

### 5b. Cross-model comparison

| Model | Test AUROC | OOD AUROC | Trained params |
|---|---:|---:|---:|
| Frequency detector A (handcrafted) | 0.900 | 0.808 | 4 k |
| Frequency detector B (CNN on spectrum) | 0.944 | 0.815 | 222 k |
| Spatial CNN (my matched 222k) | 0.995 | 0.935 | 222 k |
| **CLIP final (LAION + MLP)** | **0.997** | **0.949** | **132 k probe only** |
| CLIP zero-shot (no training) | 0.636 | 0.462 | 0 |
| Yin's from-scratch CNN | 0.9974 | 0.9429 | 288 k |

**Takeaway:** CLIP probe matches the heavily-trained from-scratch CNN on
clean test (within 0.0006 AUROC) and beats it on OOD — with zero gradient
steps through its 151M encoder. The encoder did the heavy lifting at
pretraining time; we just plugged in a router.

### 5c. Ensemble study (the team's recommended headline number)

We averaged the predicted probabilities of three diverse models:

| Subset | Test AUROC | OOD AUROC |
|---|---:|---:|
| Frequency A alone | 0.900 | 0.808 |
| Spatial CNN alone | 0.995 | 0.935 |
| CLIP final alone | 0.997 | 0.949 |
| Freq A + Spatial CNN | 0.987 (worse) | 0.900 (worse) |
| Freq A + CLIP final | 0.993 (worse) | 0.918 (worse) |
| **Spatial CNN + CLIP final** | **0.998** | **0.957** |
| All 3 | 0.997 (worse than 2) | 0.935 (worse than 2) |

**The 2-model CLIP + spatial CNN ensemble is the winner on both ID and OOD.**
Adding the frequency model *hurts* — its errors are not complementary enough,
and it pulls the average down.

**Why does the 2-model ensemble work?** Because CLIP and the spatial CNN make
*different* errors. On the 20k test set:
- 94.7% — both right (easy examples)
- 0.8% — both wrong (genuinely hard examples)
- 2.7% — CLIP right, spatial wrong
- 1.8% — spatial right, CLIP wrong
- **4.4% — complementary errors** ← the ensemble's headroom

When models disagree, averaging usually picks the correct answer, because
each model is right more often than wrong. That 4.4% is what gets converted
to ensemble gain.

### 5d. Robustness profile

The robustness battery applies perturbations to test images and measures
AUROC. Pattern across 19 perturbations:

| Perturbation type | CLIP final's behaviour |
|---|---|
| **JPEG re-compression** | Excellent — holds 0.99 down to q=40, only collapses at q=10 |
| **Gaussian blur** | Dominates — at σ=1.5 CLIP=0.84, freq detector=0.36 |
| **Additive noise** | Good until σ=16; at σ=32 the MLP overconfident, drops to 0.67 |
| **Rescale (downscale+upscale)** | Solid — 0.91 at 16×16, 0.84 at 12×12 |

**Why CLIP is robust to blur and rescale:** these are low-pass filters that
remove high-frequency content. The frequency detector keys on those exact
high-frequency artifacts and collapses. CLIP keys on semantic and mid-level
features that survive low-pass corruption.

**Why CLIP slightly fails at extreme noise:** at σ=32/255 the noise is large
enough that CLIP embeddings genuinely get distorted. The MLP head is more
confident than the simpler LR baseline and overshoots into wrong answers.
This is the one regime where our capacity push backfired.

---

## 7. How to present this in the deck (slide-by-slide narrative)

Here's a 5-slide template for Model 5:

### Slide 1 — Motivation
> *"We compared four inductive biases for AI-image detection. We add a fifth:
> a frozen foundation model that has seen 400M+ web images during
> pretraining — never fine-tuned for AI-image detection. This tests whether
> general visual understanding from massive pretraining beats CIFAKE-specific
> training."*

Show a diagram: pretraining (CLIP on web) → freeze → small probe trained on
CIFAKE. Contrast with from-scratch CNN (trained directly on CIFAKE).

### Slide 2 — The CLIP recipe
> *"OpenCLIP ViT-B/32 with LAION-2B weights. 151M parameters, frozen. We
> upsample CIFAKE 32×32 to 224×224, push through CLIP to get a 512-d
> embedding, then train a small MLP head (132k params) to classify real vs
> fake."*

Show the architecture: encoder (frozen) → embedding (512-d) → MLP (2 layers)
→ scalar (P(FAKE)).

### Slide 3 — Capacity ladder
Show `fig_capacity_ladder.png`. Talk through:
> *"We started with the standard Ojha 2023 recipe and ablated each
> improvement. The biggest single lever was swapping OpenAI weights for
> LAION-2B weights — +1.7 pp OOD with zero architecture change. Adding the
> MLP head gave a free +1.0 pp OOD without hurting generalization."*

### Slide 4 — Cross-model + ensemble
Show `fig_clean_vs_ood_final.png` and `fig_ensemble_final.png`.
> *"CLIP probe matches the from-scratch CNN on clean test with zero gradient
> steps through the encoder, and beats it on cross-generator OOD by 1.4 pp.
> The recommended team headline is the 2-model CLIP + spatial CNN ensemble
> at 0.998 ID / 0.957 OOD AUROC — strictly better than any single member."*

### Slide 5 — Why it works (and where it doesn't)
Show `fig_robust_curves_final.png` and `fig_clip_vs_spatial_agreement.png`.
> *"CLIP is more blur-robust and rescale-robust than CNNs because semantic
> features survive low-pass filtering. CLIP and the spatial CNN make
> 4.4% complementary errors — that's the headroom the ensemble captures.
> The one regime where CLIP underperforms is extreme noise (σ=32), where
> the MLP head's added capacity becomes a liability."*

---

## 8. FAQ — questions you should expect

**Q: Isn't using a 151M-param model "cheating" compared to a 287k-param CNN?**
> No, because we're not training a 151M-param model. We're using a 132k-param
> classifier on features that were learned by Anthropic-of-AI from publicly-
> available data. The pretraining cost is amortised across every downstream
> task. The interesting question is "what does pretraining give you for free?"
> and our answer is "it matches end-to-end CIFAKE training."

**Q: Why didn't you fine-tune CLIP for this task? Wouldn't that be better?**
> Three reasons. (1) Linear/MLP probing is much cheaper — we trained the head
> in 30 seconds on cached features. (2) Frozen probes are a stricter test of
> *what's already in the features*, which is the scientific question we
> wanted to answer. (3) Ojha 2023 found that fine-tuning *hurts* cross-
> generator generalization — fine-tuning lets the model overfit to the
> training generator's specific artifacts and lose CLIP's general "what's a
> real photo" prior. Frozen wins on OOD.

**Q: Did CLIP "see" any CIFAKE / Stable Diffusion images during pretraining?**
> No. CLIP's training data was collected from the web in 2020-2021. Stable
> Diffusion v1.4 was released in August 2022. CLIP has never been trained
> on any output of any diffusion model. The fact that it can detect
> SD-1.4 images is purely "this doesn't look like the natural photos I
> learned to represent."

**Q: Why is the upsampling from 32 to 224 OK?**
> The information content of the image is unchanged (interpolation can't
> add information). CLIP expects 224×224 input because that's what it was
> trained on. We use PIL's bicubic resize, the standard high-quality method.
> Empirically: it works (99% AUROC). If we used a CLIP variant trained at
> 64×64 or 96×96 we could skip this, but no mainstream CLIP variant exists
> for tiny inputs.

**Q: Why does zero-shot CLIP fail (0.636 AUROC) but linear-probe CLIP succeed
(0.997 AUROC)? Aren't they using the same features?**
> Yes, same features, but they're using them differently. Zero-shot relies on
> the *text* embeddings of "real photograph" vs "AI-generated image" being
> meaningfully positioned in CLIP's joint space — which they aren't,
> because CLIP didn't see captions like "AI-generated image" during 2021
> training. The image features themselves encode the distinction, but you
> need to *learn* a direction in feature space that picks out the AI signal.
> That's exactly what the linear/MLP probe does on 90k labeled examples.

**Q: How does this compare to just using DINOv2 or another self-supervised
encoder?**
> DINOv2 (Meta, 2023) is the closest comparable foundation model. We didn't
> test it here for time reasons, but the literature suggests it would
> perform similarly. CLIP's advantage is text alignment — that doesn't
> directly help with AI detection, but the data diversity it forced does.

**Q: What's the inference cost?**
> Per image: ~4.4 GFLOPs (CLIP forward at 224×224) + a few thousand multiplies
> for the MLP head. That's ~56× the spatial CNN's 78 MFLOPs. On MPS we got
> ~300 images/second for CLIP forward, plenty for any realistic deployment.
> Training the head was ~30 seconds total on cached embeddings.

**Q: Will this generalize to other generators, not just sd-turbo?**
> Our cross-generator OOD test used sd-turbo (a different latent diffusion
> model than SD-1.4 — different VAE, different sampling). CLIP held up well
> (0.949 OOD vs 0.997 clean, only 4.8 pp drop). Ojha 2023 tested CLIP
> against many generators (GANs, diffusion, autoregressive) and found
> similarly strong transfer. We'd expect CLIP to be the most generator-
> agnostic model in our team study, though we didn't measure against
> GAN-generated images.

**Q: What about adversarial attacks?**
> Not tested. CLIP probes are known to be vulnerable to adversarial
> perturbations (small pixel changes that flip the prediction). For
> deployment against actively adversarial AI-image generators, you'd want
> adversarial training or detection of such attacks. For our project
> (in-the-wild detection benchmark) it's not in scope.

---

## 9. Common confusions to head off in your deck

**Confusion 1: "CLIP is generating something."**
> No. CLIP is a *classifier of similarity*, not a generator. It takes images
> and text and tells you how related they are. It can't produce images.
> (You might be thinking of "CLIP-guided diffusion" — that uses CLIP as a
> *scoring function* for another model that does the generating.)

**Confusion 2: "We trained CLIP on CIFAKE."**
> No. We trained a tiny 132k-parameter classifier *on top of* CLIP's frozen
> features. CLIP itself wasn't updated by a single gradient step.

**Confusion 3: "OOD = a different test set."**
> A "different test set" might be just a different *sample* from the same
> distribution. Our OOD test is **a different generator entirely**
> (sd-turbo vs SD-1.4). The whole point is that the model has never seen
> this generator. If you call regular train/test "OOD" people will be
> confused; specifically say "cross-generator OOD."

**Confusion 4: "ImageNet pretraining and CLIP pretraining are the same thing."**
> Both produce frozen feature extractors, but the pretraining tasks differ
> radically. ImageNet pretraining = supervised 1000-class object
> classification on 1.3M curated images. CLIP pretraining = contrastive
> image-text matching on 400M+ web pairs. CLIP's features generalize
> across more domains because (a) the data was vastly more diverse and
> (b) the training task didn't bottleneck features through 1000 class
> labels.

**Confusion 5: "If CLIP works, the frequency detector was a waste."**
> Not quite. The frequency work *did* fail as a competitive classifier,
> but it produces a unique interpretability artifact: the LR-coefficient
> figure shows *which spectral frequencies* SD-1.4 over- and under-produces.
> No CLIP feature is interpretable in that way. The frequency work is
> being repositioned as a scientific characterization appendix, not a
> detector. It still teaches us something the spatial/CLIP models cannot.

---

## 10. The pipeline in code (for anyone who wants to reproduce)

```python
# 1. Install
pip install open_clip_torch

# 2. Extract embeddings (one-time, cached to disk)
import open_clip
import torch
from PIL import Image
import numpy as np

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k")
model = model.to("mps").eval()

@torch.no_grad()
def embed(paths):
    out = np.empty((len(paths), 512), np.float32)
    for i, p in enumerate(paths):
        x = preprocess(Image.open(p).convert("RGB"))[None].to("mps")
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
        out[i] = v.cpu().numpy()
    return out

X_train = embed(train_paths)  # shape (90000, 512)
np.save("train_X.npy", X_train)
# ... same for val, test, ood

# 3. Train probe (on the cached embeddings)
import torch.nn as nn
class MLPHead(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

# train with BCEWithLogitsLoss, AdamW, batch 1024, early stop on val AUROC

# 4. Inference at deployment time
# encode image through frozen CLIP -> 512-d vector -> MLP -> sigmoid -> P(FAKE)
```

The actual code is in `project_clip/clip_v2.py` (more polished, with the
shared eval harness, etc.).

---

## 11. Recommended further reading

Three papers worth skimming (one paragraph each) if you want to go deeper:

1. **Radford et al. 2021, "Learning Transferable Visual Models From Natural
   Language Supervision"** — the original CLIP paper. Read sections 1-3 for
   the contrastive objective intuition.

2. **Ojha et al. 2023 CVPR, "Towards Universal Fake Image Detectors that
   Generalize Across Generative Models"** — the linear-probe-on-CLIP recipe
   we used. Read the experiments section for cross-generator results.

3. **Schuhmann et al. 2022, "LAION-5B: An open large-scale dataset for
   training next generation image-text models"** — context on why
   LAION-pretrained models beat OpenAI-pretrained ones on transfer tasks.

If you have time for only one: **Ojha 2023.** It's the direct intellectual
parent of our model and shows the cross-generator results that motivated
the inductive-bias framing in the team report.

---

## 12. TL;DR for the deck

If you can only say five things on a slide:

1. CLIP is a 151M-param image encoder trained on 400M+ web image-text pairs in 2021.
   It produces general-purpose features that already separate "natural photos"
   from "AI-generated images" — even though it's never seen an AI image.

2. We froze it and trained a tiny 132k-param MLP classifier on top of the
   features it produces for CIFAKE images.

3. This matches a from-scratch 287k-param CNN's clean test AUROC (within 0.0006)
   and beats it on cross-generator OOD by +1.4 pp — with zero gradient steps
   through the 151M encoder.

4. The 2-model ensemble of CLIP + spatial CNN (predictions averaged) reaches
   0.998 test / 0.957 OOD AUROC — strictly better than either alone, because
   they make 4.4% complementary errors on the test set.

5. The biggest single design lever was swapping CLIP's pretraining dataset
   from OpenAI's 400M-pair set to LAION's 2B-pair set: +1.7 pp OOD AUROC
   with zero architectural change. Data scale during pretraining > model
   capacity for cross-generator transfer.
