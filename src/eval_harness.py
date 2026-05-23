"""SHARED evaluation harness for the 4-model AI-image-detection project.

All four models (small-CNN, ResNet-18, ViT, frequency-detector) call
``evaluate(...)`` so the numbers are computed identically and written to a
common JSON schema, then aggregated by the final comparison notebook.

Positive class = FAKE (AI-generated) = 1.

JSON schema (results/<model_name>/<record_key>.json)
----------------------------------------------------
{
  "model_name": str,
  "split":      str,     # "val" | "test" | "ood:<name>" | "robust:<pert>@<lvl>"
  "n":          int,
  "threshold":  {"value": float, "policy": str},
  "metrics":    {"accuracy","auroc","f1","precision","recall"},
  "confusion":  {"tn","fp","fn","tp"},
  "curves":     {"roc":{"fpr","tpr"}, "pr":{"recall","precision"}},  # downsampled
  "provenance": {"timestamp","seed","norm_hash","config","libs"}
}
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Default results root. Resolve relative to repo root so the harness works
# regardless of where the repo is cloned. Override with CIFAKE_RESULTS env var.
import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parent.parent
RESULTS_ROOT = os.environ.get(
    "CIFAKE_RESULTS",
    str(_REPO_ROOT / "results" / "per_model"),
)


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Youden's J optimum on the VALIDATION set (use this thr. on test)."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return float(thr[np.argmax(tpr - fpr)])


def _downsample(x: np.ndarray, n: int = 200) -> list:
    if len(x) <= n:
        return [float(v) for v in x]
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return [float(v) for v in np.asarray(x)[idx]]


def evaluate(
    y_true,
    y_score,
    model_name: str,
    split: str,
    threshold: float | None = None,
    threshold_policy: str = "0.5",
    config: dict | None = None,
    norm_hash: str | None = None,
    seed: int = 42,
    save: bool = True,
    record_key: str | None = None,
    save_scores: bool = False,
) -> dict:
    """Compute the standardized metrics dict and (optionally) persist it.

    ``threshold`` None -> use 0.5. For test, pass the val-derived
    ``best_threshold`` and set threshold_policy="best_val_youden".
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    thr = 0.5 if threshold is None else float(threshold)
    y_pred = (y_score >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    rec_out = {
        "model_name": model_name,
        "split": split,
        "n": int(len(y_true)),
        "positive_label": "FAKE(AI-generated)=1",
        "threshold": {"value": thr, "policy": threshold_policy},
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "auroc": float(roc_auc_score(y_true, y_score)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        },
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "curves": {
            "roc": {"fpr": _downsample(fpr), "tpr": _downsample(tpr)},
            "pr": {"recall": _downsample(rec), "precision": _downsample(prec)},
        },
        "provenance": {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "norm_hash": norm_hash,
            "config": config or {},
            "libs": _lib_versions(),
        },
    }

    if save:
        out_dir = os.path.join(RESULTS_ROOT, model_name)
        os.makedirs(out_dir, exist_ok=True)
        key = record_key or split.replace(":", "_").replace("@", "_")
        with open(os.path.join(out_dir, f"{key}.json"), "w") as f:
            json.dump(rec_out, f, indent=2)
        if save_scores:
            np.savez(os.path.join(out_dir, f"{key}_scores.npz"),
                     y_true=y_true, y_score=y_score)
    return rec_out


def _lib_versions() -> dict:
    import numpy
    import sklearn
    import torch

    return {
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
    }


def summarize(rec: dict) -> str:
    m = rec["metrics"]
    return (f"[{rec['model_name']}/{rec['split']}] n={rec['n']} "
            f"AUROC={m['auroc']:.4f} acc={m['accuracy']:.4f} "
            f"F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
            f"(thr={rec['threshold']['value']:.3f}/{rec['threshold']['policy']})")
