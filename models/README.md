# `models/` — per-model artifacts

Each subdirectory contains everything needed to run inference for one model:
the **`predict.py`** wrapper (the only file the evaluator imports) and the
**trained checkpoint** (tracked via Git LFS).

## The `predict.py` contract

Every wrapper exposes a single public function:

```python
def predict_fake_probability(paths: list[str]) -> np.ndarray
    """Return shape (N,) array of P(FAKE) in [0, 1] for the given image paths."""
```

Conventions (must hold for every model in this repo):

| Convention | Value |
|---|---|
| Positive class | `FAKE = 1`, `REAL = 0` (so `y_score = P(FAKE)`) |
| Class index in softmax | varies per model; the wrapper translates internally |
| Input | absolute paths to 32×32 RGB JPEGs (CIFAKE format) |
| Output dtype | `np.float32`, no NaNs, all values in `[0, 1]` |

The wrapper **does its own checkpoint loading + preprocessing** so the eval
harness can stay model-agnostic. It must also be safe to call repeatedly —
the model is loaded lazily on first call and cached in a module-level global.

## The four models

| Folder | Owner | Architecture | Trained params | Checkpoint (LFS) |
|---|---|---|---:|---|
| `cnn_baseline_yin/`  | Yin    | 3-block VGG-style CNN, native 32×32 | 288 k  | `best_cnn.pt` ⚠ pending |
| `resnet18_nathan/`   | Nathan | ResNet-18 (ImageNet pretrained, partial fine-tune) | 11.2 M | `best_resnet18.pth` (45 MB) |
| `vit_small_alex/`    | Alex   | timm `vit_small_patch16_224` (pretrained, full fine-tune) | 21.7 M | `best_vit.pt` (87 MB) |
| `clip_probe_leyi/`   | Leyi   | OpenCLIP ViT-B/32-LAION-2B (FROZEN) + MLP head (256, dropout 0.3) | 132 k probe (151 M frozen) | `best_mlp.pt` (516 KB) |

**`models/cnn_baseline_yin/best_cnn.pt`** — Yin's from-scratch CNN
checkpoint, integrated and reproducing her reported metrics to four
decimal places. See `models/cnn_baseline_yin/README.md` for the
reproduction recipe.

`models/clip_probe_leyi/` has only the trained head; the encoder is loaded
on demand from OpenCLIP. The inference pipeline lives at
`src/clip_probe/pipeline.py` (no per-model `predict.py` because it shares
the trained head with the embedding extractor).

## Adding a new model

1. Create `models/<name>/`
2. Drop a `predict.py` implementing `predict_fake_probability(paths)`
3. Drop the checkpoint as `models/<name>/<whatever>.pt` (LFS auto-tracks)
4. Add `<name>` to the list in `scripts/evaluate.py` (or call directly via `--model <name>`)
5. Run `python scripts/evaluate.py --model <name>` to populate `results/per_model/<name>/`

The eval pipeline (`scripts/evaluate.py` → `src/eval_harness.evaluate`) will:
- Reproduce the model's reported clean-test AUROC (sanity check)
- Run cross-generator OOD evaluation on `data/ood_sdturbo/`
- Optionally run the 19-perturbation robustness battery
- Persist per-sample probability scores so `scripts/run_team_ensemble.py` can include the new model
