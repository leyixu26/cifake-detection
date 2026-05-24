# CIFAKE AI-Image Detection — 5-Model Team Study

**ADSP 31018 Machine Learning II · Final Project · Spring 2026**
**Team:** Yin (small CNN) · Nathan (ResNet-18) · Alex (ViT-Small) · Leyi (frequency detector + CLIP probe)

We study five models with **fundamentally different inductive biases** for
detecting AI-generated images in the CIFAKE benchmark (32×32 RGB; REAL =
CIFAR-10, FAKE = Stable Diffusion 1.4). Every model is evaluated through a
**shared harness** for in-distribution test, **cross-generator OOD** against
sd-turbo, and a **19-perturbation robustness battery**. We then build a
**team ensemble** to identify which combinations transfer best.

## Headline numbers

| # | Model | Owner | Test AUROC | OOD AUROC | OOD drop | Trained params |
|---|---|---|---:|---:|---:|---:|
| 1 | from-scratch CNN | Yin | **0.9974** | **0.9429** | −5.5 pp | 288 k |
| 2 | ResNet-18 (ImageNet) | Nathan | **0.9977** | **0.9341** | −6.4 pp | 11.2 M |
| 3 | ViT-Small (timm) | Alex | **0.9994** | **0.9732** | **−2.6 pp** | 21.7 M |
| 4 | Frequency detector | Leyi | 0.9435 | 0.8150 | −12.8 pp | 222 k |
| 5 | CLIP probe (LAION + MLP) | Leyi | **0.9968** | **0.9485** | −4.9 pp | 132 k (probe only; 151 M frozen) |
| - | **Best 2-model ensemble (Alex ViT + CLIP) on OOD** | — | 0.9988 | **0.9657** | — | — |
| - | **Best 3-model ensemble (Yin CNN + Alex ViT + CLIP) on OOD** | — | **0.9994** | **0.9670** | — | — |

All five models were evaluated through the same shared harness — Yin's
0.9974 reproduces her original `results_CNN_from_scratch.json` to four
decimal places. The cross-generator OOD column is a new finding for
Yin's model (her original report was in-distribution only); her
from-scratch CNN drops *less* than Nathan's ImageNet ResNet under
cross-generator stress.

See `docs/findings/headline.md` for the full story.

## Repo layout

```
cifake-detection/
├── notebooks/    01-06       graded notebooks (one per model + team comparison)
├── src/                       library code: eval_harness, perturbations, ensemble, freq_detector/, clip_probe/
├── models/                    per-model predict.py + checkpoint (Git LFS)
├── scripts/                   evaluate.py / generate_ood.py / run_team_ensemble.py / generate_figures.py
├── results/                   metrics JSONs + 12 curated figures + team_ensemble_report.json
├── docs/                      methodology + findings + report cribsheet + literature
├── data/                      acquisition instructions (no actual data shipped)
├── README.md                  this file
├── requirements.txt           pinned Python deps
├── .gitattributes             Git LFS rules for *.pt *.pth
└── .gitignore                 caches, embeds, *_scores.npz, robust_*.json
```

## Quick start

```bash
# 1. Install
pip install -r requirements.txt
git lfs install                    # fetches checkpoints

# 2. Acquire CIFAKE
bash scripts/setup_data.sh         # prints kaggle CLI commands if missing

# 3. (Optional) Regenerate the cross-generator OOD set (~10 min)
python scripts/generate_ood.py

# 4. Open any notebook (notebooks/04_freq_detector.ipynb is a good place to start)
jupyter notebook notebooks/

# 5. (Reproduction) Re-run any single model's evaluation end-to-end
PYTHONPATH=. python scripts/evaluate.py --model resnet18_nathan
PYTHONPATH=. python scripts/evaluate.py --model vit_small_alex
PYTHONPATH=. python scripts/evaluate.py --model cnn_baseline_yin       # once Yin's ckpt is dropped in
# CLIP probe has its own end-to-end pipeline:
PYTHONPATH=. python -c "from src.clip_probe import extract_embeddings, fit_mlp, run_robust_battery; \
                        extract_embeddings(); fit_mlp(); run_robust_battery()"

# 6. Run team ensemble + regenerate cross-model figures
PYTHONPATH=. python scripts/run_team_ensemble.py
PYTHONPATH=. python scripts/generate_figures.py
```

## The five models, one paragraph each

**Model 1 (Yin) — small CNN from scratch.** 3 VGG-style ConvBlocks → GAP →
Linear, 288 k parameters, trained end-to-end on CIFAKE. Tests "what can you
learn purely from CIFAKE without any prior?". Achieves 0.9974 test AUROC
in-distribution (matches her original training report exactly), 0.9429
cross-generator OOD — drops only −5.5 pp under sd-turbo, less than Nathan's
ImageNet-warm-started ResNet (−6.4 pp). Notebook: `01_cnn_baseline.ipynb`.

**Model 2 (Nathan) — ResNet-18 ImageNet transfer.** ResNet-18 with ImageNet
weights, conv1/bn1/layer1/layer2 frozen, layer3/layer4/fc fine-tuned. Tests
"what does ImageNet pretraining buy you?". 0.9977 test, 0.9341 OOD.
Notebook: `02_resnet18.ipynb`.

**Model 3 (Alex) — ViT-Small full fine-tune.** timm `vit_small_patch16_224`
pretrained on ImageNet-1k, fully fine-tuned for 25 epochs. Tests "does
attention pick up different artifacts than convolutions?". **Smallest OOD
drop of any model (−2.6 pp).** Notebook: `03_vit_small.ipynb`.

**Model 4 (Leyi) — frequency detector.** Spectrum input only — log|FFT| →
radial PSD + azimuthal + scalar features (29-d) for the principled Variant A,
or 2-D log-magnitude → 220k-param CNN for Variant B. Dominated on every
detection metric but **uniquely interpretable** (we can point to the
specific spectral bands SD-1.4 over-/under-produces). Notebook:
`04_freq_detector.ipynb`.

**Model 5 (Leyi) — CLIP probe.** OpenCLIP ViT-B/32 with LAION-2B weights,
**frozen**, plus a 132k-param MLP head trained on top. Tests "does
web-scale multimodal pretraining beat ImageNet pretraining?". Within
0.0006 of Yin's reported clean-test AUROC with zero gradient steps through
the encoder; beats Nathan's ResNet on OOD. Notebook: `05_clip_probe.ipynb`.

## The team ensemble story

After running per-sample predictions for every model through the same
harness, `scripts/run_team_ensemble.py` computes probability-averaged
ensembles for every pair / triple / full set + leave-one-out contributions.

The empirical winner depends on what you care about:
- **Maximum clean-test AUROC** → Nathan ResNet + Alex ViT (test 0.9993)
- **Maximum cross-generator OOD AUROC** (best pair) → Alex ViT + CLIP probe (OOD 0.9657)
- **Maximum cross-generator OOD AUROC** (best triple) → Yin CNN + Alex ViT + CLIP probe (OOD **0.9670**)

The complementary errors between CLIP and the spatial CNN (4.4% of test
images — see `results/figures/05_clip_vs_spatial_agreement.png`) are the
empirical basis for the ensemble lift. Yin's small CNN earns its slot in
the OOD-optimal triple by contributing +0.17 pp under leave-one-out —
the diversity benefit of an architecture trained without any prior
beyond CIFAKE itself.

## Documentation

| Doc | What it's for |
|---|---|
| [`docs/findings/headline.md`](docs/findings/headline.md) | One-page summary of 5-model + ensemble findings — read this first |
| [`docs/REPORT_HEADLINE.md`](docs/REPORT_HEADLINE.md) | Longer report cribsheet for the team write-up |
| [`docs/STUDY_GUIDE_CLIP.md`](docs/STUDY_GUIDE_CLIP.md) | CLIP background for teammates who haven't worked with it |
| [`docs/LITERATURE.md`](docs/LITERATURE.md) | Citations: Ojha 2023 (CLIP), Wang 2020 (cross-generator), Durall 2020 (spectral), Corvi 2023 (diffusion fingerprints) |
| [`docs/methodology/shared_harness.md`](docs/methodology/shared_harness.md) | Why we built a shared eval harness + JSON schema |
| [`docs/methodology/ood_methodology.md`](docs/methodology/ood_methodology.md) | The cross-generator OOD design + JPEG-quant-table confound control |
| [`docs/methodology/frequency_detector.md`](docs/methodology/frequency_detector.md) | Spectral-fingerprint narrative + Variant A/B design |
| [`docs/findings/freq_detector.md`](docs/findings/freq_detector.md) | Full frequency-detector findings (ablations + interpretability) |
| [`docs/findings/clip_probe.md`](docs/findings/clip_probe.md) | Full CLIP probe findings (capacity ladder + ensemble) |
| [`models/README.md`](models/README.md) | The `predict.py` contract |
| [`scripts/README.md`](scripts/README.md) | Run order + reproduction recipe |

## Reproducibility

- Frozen split: stratified 90/10 train/val, seed 42 (`src/freq_detector/datasets.py::make_splits`)
- Sealed test: touched exactly once per model via `scripts/evaluate.py`
- Decision threshold: val-derived Youden-J (`src/eval_harness.best_threshold`)
- Cross-generator OOD: 1000 sd-turbo + 1000 CIFAR-10 REAL, both re-encoded through
  CIFAKE's exact JPEG quantization tables (so only the generator differs)
- All metrics through `src/eval_harness.evaluate(..., save_scores=True)`
- Per-sample probability scores persisted in `*_scores.npz` files (gitignored)
  so ensemble computations are reproducible from cached predictions alone

## Status of all five models

All five models are integrated and reproducible end-to-end:

- `cnn_baseline_yin/best_cnn.pt` — Yin's `SimpleCNN` (288 k params).
  Reproduces her 0.9974 test AUROC to four decimal places. **New**:
  cross-generator OOD AUROC = 0.9429 (smaller drop than Nathan's
  ImageNet ResNet).
- `resnet18_nathan/best_resnet18.pth` — Nathan's ResNet-18 (45 MB).
- `vit_small_alex/best_vit.pt` — Alex's ViT-Small fine-tune (87 MB).
- `clip_probe_leyi/best_mlp.pt` — Leyi's 132 k-param MLP probe (516 KB);
  encoder loaded on demand from OpenCLIP LAION-2B.
- Frequency detector: lives in `src/freq_detector/` + Variant-A/B
  checkpoints baked into the notebook outputs (the artifact for the
  paper is the interpretability figure, not a saved classifier).

To re-evaluate any model end-to-end:

```bash
PYTHONPATH=. python scripts/evaluate.py --model cnn_baseline_yin
# (or resnet18_nathan, vit_small_alex; CLIP uses src.clip_probe.pipeline)
PYTHONPATH=. python scripts/run_team_ensemble.py
PYTHONPATH=. python scripts/generate_figures.py
```

## License

MIT — see `LICENSE`. This is academic coursework; no commercial use restrictions.
