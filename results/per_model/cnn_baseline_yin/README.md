# Yin from-scratch CNN — results pending

The checkpoint `models/cnn_baseline_yin/best_cnn.pt` is not yet on disk
(see that folder's README). Once Yin sends it, run:

    python scripts/evaluate.py --model cnn_baseline_yin

That will write `test.json`, `val.json`, `ood_sdturbo.json`, and 19
`robust_*.json` records here.

Yin's *reported* clean-test AUROC from his own notebook is **0.9974**
(see `notebooks/01_cnn_baseline.ipynb`).
