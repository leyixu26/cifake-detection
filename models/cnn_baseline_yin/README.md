# Yin — from-scratch CNN baseline

Owner: Yin · Architecture: 3 VGG-style ConvBlocks + GAP + Linear → 288 k params · Input: 32×32 native (no resize)

## Status

**INTEGRATED & REPRODUCED.** Yin's `best_cnn.pt` checkpoint is in place and
the shared harness reproduces his original training metrics:

| Metric | Yin's original (`results_CNN_from_scratch.json`) | Repo (`scripts/evaluate.py`) | Match |
|---|---:|---:|---|
| Test AUROC | 0.99741 | **0.9974** | ✓ to 4 dp |
| Test accuracy | 0.9766 | **0.9762** | ✓ within 0.0004 |
| Test F1 (macro) | 0.9766 | **0.9763** | ✓ within 0.0003 |

**New finding from the shared harness** (not in Yin's original report —
in-distribution only): cross-generator OOD AUROC = **0.9429** on sd-turbo.
That's a −5.5 pp drop, *smaller* than Nathan's ImageNet-pretrained
ResNet-18 (−6.4 pp) — a surprising result documented in
`docs/findings/headline.md`.

## Reproduce end-to-end

```bash
PYTHONPATH=. python scripts/evaluate.py --model cnn_baseline_yin
```

That single command writes:
- `results/per_model/cnn_baseline_yin/val.json` (n=10000, AUROC 0.9991)
- `results/per_model/cnn_baseline_yin/test.json` (n=20000, AUROC 0.9974)
- `results/per_model/cnn_baseline_yin/ood_sdturbo.json` (n=2000, AUROC 0.9429)
- 19 `results/per_model/cnn_baseline_yin/robust_*.json` (gitignored)
- Companion `*_scores.npz` files for ensemble computation (gitignored)

Runtime: ~10 s on Apple Silicon MPS for val+test+OOD; +~2.5 min for the
full robustness battery.

## Architecture (one paragraph)

3 sequential ConvBlocks (each = 2×(Conv3x3 → BN → ReLU) + MaxPool2x2):
32→64→128 channels, 32×32 → 16×16 → 8×8 → 4×4 spatial. Then AdaptiveAvgPool
to (B, 128) and a single Linear → 2 logits. Loss: CrossEntropy. Training:
AdamW (lr 1e-3, wd 1e-4), CosineAnnealingLR, batch 128, 25 epochs, mixed
precision on T4. Standard CIFAR augmentation (RandomCrop with padding=4 +
RandomHorizontalFlip). Best val-loss epoch = 24 (val 0.0670).

## Class index convention

Yin's training used `ImageFolder` alphabetical ordering: **`{FAKE: 0, REAL: 1}`**
— *opposite* of the repo-wide convention (`{REAL: 0, FAKE: 1}`). The
`predict.py` wrapper transparently handles this — it returns `softmax(...)[:, 0]`
(the FAKE logit under Yin's ordering) as `P(FAKE)`, matching the harness's
`y_score = P(FAKE)` contract. **No upstream caller needs to know.**
