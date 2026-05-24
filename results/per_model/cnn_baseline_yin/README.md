# Yin from-scratch CNN — sealed-harness results

Headline metrics under the shared evaluation harness:

| split | n | AUROC | accuracy | F1 |
|---|---:|---:|---:|---:|
| val          | 10 000 | 0.9991 | 0.9870 | 0.9870 |
| test         | 20 000 | 0.9974 | 0.9762 | 0.9763 |
| ood_sdturbo  |  2 000 | 0.9429 | 0.8120 | 0.7767 |

The test AUROC matches Yin's original `results_CNN_from_scratch.json`
(her own notebook output, included here as `training_report_yin.json` for
reference) to four decimal places. The OOD column is a new measurement
under the shared harness; her original report was in-distribution only.

To reproduce these JSONs end-to-end (also writes 19 `robust_*.json` files
that are gitignored):

```
PYTHONPATH=. python scripts/evaluate.py --model cnn_baseline_yin
```
