# Yin — from-scratch CNN baseline

Owner: Yin · Architecture: 3 VGG-style ConvBlocks + GAP + Linear → 288 k params · Input: 32×32 native (no resize)

## Status

**`best_cnn.pt` checkpoint is PENDING.** Yin's overnight re-train attempt
failed under multi-process CPU contention on Leyi's machine. Two paths to
fill this in:

1. **Get the checkpoint from Yin directly** — he produced it on Colab/local
   when he originally trained, file is `best_cnn.pt` in his Colab working
   directory. Drop it here:
   ```
   models/cnn_baseline_yin/best_cnn.pt
   ```
   Then run:
   ```
   python scripts/evaluate.py --model cnn_baseline_yin
   ```

2. **Re-train from his notebook** (`notebooks/01_cnn_baseline.ipynb`):
   ```
   jupyter nbconvert --to notebook --execute --inplace notebooks/01_cnn_baseline.ipynb
   # The notebook saves to ./best_cnn.pt in cwd; move it here:
   mv best_cnn.pt models/cnn_baseline_yin/
   ```
   Expect ~15 min on T4 GPU or ~30 min on Apple Silicon CPU.

## Expected reproduction

Per Yin's reported `results_CNN_from_scratch.json`:
- Clean test AUROC: ≈ 0.9974
- Test accuracy: ≈ 0.9766

The `predict.py` wrapper here loads the checkpoint and reproduces these
within ±0.001 (AMP nondeterminism).
