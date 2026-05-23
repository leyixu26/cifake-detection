# Methodology — Shared Evaluation Harness

All five team models route through the same evaluation pipeline so the
numbers are directly comparable. This doc describes the contract, schema,
and threshold-discipline conventions.

## Why a shared harness?

Without a shared harness, each model's "0.998 AUROC" would be incomparable
because different teammates would (silently) be using:
- different test-set splits (random seed drift)
- different positive-class conventions
- different decision thresholds (argmax vs Youden vs manual)
- different metric implementations (sklearn vs custom)
- different JSON field names → impossible to aggregate

`src/eval_harness.py` enforces all of these. Every model that goes through
`evaluate(...)` produces a record that the team aggregator (`src/ensemble.py`)
can ingest with no per-model special-casing.

## The contract

| Field | Value | Why |
|---|---|---|
| Positive class | **FAKE = 1** (REAL = 0) | AI-detector convention; metrics break if inverted |
| `y_score` | probability of FAKE, in [0,1] | softmax index varies per model; the wrapper handles it |
| Decision threshold | val-derived Youden-J | apples-to-apples per-class P/R/F1; AUROC is threshold-free |
| Split seed | 42, stratified 90/10 | every model uses the same fixed seed via `make_splits()` |
| Test discipline | sealed; touched exactly once per model | no peeking → no ablation-driven test overfitting |

## JSON schema (every record under `results/per_model/<model>/`)

```json
{
  "model_name":     "resnet18_nathan",
  "split":          "test" | "val" | "ood:sdturbo" | "robust:<pname>@<level>",
  "n":              20000,
  "positive_label": "FAKE(AI-generated)=1",
  "threshold":      {"value": 0.528, "policy": "best_val_youden"},
  "metrics":        {"accuracy", "auroc", "f1", "precision", "recall"},
  "confusion":      {"tn", "fp", "fn", "tp"},
  "curves":         {"roc": {"fpr","tpr"}, "pr": {"recall","precision"}},
  "provenance":     {"timestamp", "seed", "config", "libs"}
}
```

When `save_scores=True` is passed (default in `scripts/evaluate.py`), the
harness also writes `<key>_scores.npz` next to the JSON with `y_true` and
`y_score` arrays — needed for ensemble computation downstream.

## The `predict.py` contract for each model

Each model in `models/<name>/predict.py` exposes one function:

```python
def predict_fake_probability(paths: list[str]) -> np.ndarray:
    """Return shape (N,) array of P(FAKE) in [0,1]."""
```

The model is loaded lazily inside the function (cached in a module-level
global) so the harness can call `predict()` repeatedly across splits without
re-initialising.

Importantly: **the wrapper internally handles its own class indexing.**
Different teammates use different softmax indices (Yin/Nathan/Alex got FAKE=0
from `ImageFolder` alphabetical order); the wrapper translates to a probability
of FAKE in [0,1] regardless of the underlying convention.

## End-to-end flow

```
                 ┌──────────────┐
                 │  predict.py  │  (model-specific)
                 │  load ckpt   │
                 │  forward()   │
                 │  -> P(FAKE)  │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────────────┐
                 │  scripts/evaluate.py │  (model-agnostic)
                 │  loops over splits   │
                 │  calls predict()     │
                 │  calls evaluate()    │
                 └──────┬───────────────┘
                        │
                        ▼
                 ┌──────────────────────┐
                 │  src/eval_harness.py │
                 │  computes metrics,   │
                 │  writes JSON+npz     │
                 └──────────────────────┘
                        │
                        ▼
                 results/per_model/<name>/  ⇨  scripts/run_team_ensemble.py
                                                 → results/team_ensemble_report.json
```

## Why this matters for the report

Reviewers immediately notice when models have different evaluation axes.
A 4-model comparison where one model has `roc_auc=0.998` and another has
`auroc=0.998` and a third has `macro_f1=0.97` reads as "the team didn't
coordinate." Our shared harness produces uniform records, which makes the
cross-model table in `headline.md` directly defensible.
