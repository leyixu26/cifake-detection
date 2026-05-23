"""Generic per-model evaluation: replaces the four per-teammate evaluate_*.py
scripts that used to live in the old project/ tree.

Each model is expected to have a `predict.py` under `models/<name>/` that
exposes one function:

    predict_fake_probability(paths: list[str]) -> np.ndarray  in [0,1]

This script:

  1. Loads the frozen split (seed=42, 90/10 stratified) and the sealed test
     and OOD images.
  2. Calls predict() on val to derive a Youden-J threshold.
  3. Calls predict() on test, OOD, and (optional) the full 19-perturbation
     robustness battery.
  4. Emits one JSON + scores.npz per split via the shared eval harness,
     under results/per_model/<name>/.

Usage:
    python scripts/evaluate.py --model resnet18_nathan
    python scripts/evaluate.py --model vit_small_alex --skip-robust
    python scripts/evaluate.py --model cnn_baseline_yin --splits test ood
"""
from __future__ import annotations
import argparse
import glob
import importlib.util
import os
import pathlib
import sys
import tempfile
import time

import numpy as np
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval_harness import best_threshold, evaluate, summarize  # noqa: E402
from src.perturbations import BATTERY                              # noqa: E402
from src.freq_detector.datasets import make_splits, load_image     # noqa: E402

OOD_ROOT = pathlib.Path(os.environ.get(
    "CIFAKE_OOD", REPO_ROOT / "data" / "ood_sdturbo"))


def load_predict(model_name: str):
    """Dynamically import models/<model_name>/predict.py and return its
    predict_fake_probability function."""
    p = REPO_ROOT / "models" / model_name / "predict.py"
    if not p.exists():
        raise SystemExit(f"no predict.py at {p}")
    spec = importlib.util.spec_from_file_location(
        f"predict_{model_name}", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "predict_fake_probability"):
        raise SystemExit(
            f"{p} must define predict_fake_probability(paths) -> np.ndarray")
    return mod.predict_fake_probability


def list_ood():
    paths, labels = [], []
    for label, cls in ((0, "REAL"), (1, "FAKE")):
        fs = sorted(glob.glob(str(OOD_ROOT / cls / "*.jpg")))
        paths += fs; labels += [label] * len(fs)
    return paths, np.asarray(labels, np.int64)


def run_robust(model_name, predict, te_p, yt, thr, cfg):
    """Robustness battery — re-encode each test image under the perturbation
    via a temp dir of JPEGs, then call predict()."""
    for pname, (levels, fac) in BATTERY.items():
        for lvl in levels:
            t0 = time.time()
            f = fac(lvl)
            with tempfile.TemporaryDirectory() as tdir:
                perturbed = []
                for i, p in enumerate(te_p):
                    img = load_image(p); img = f(img)
                    out = os.path.join(tdir, f"{i:06d}.jpg")
                    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)) \
                         .save(out, "JPEG", quality=95)
                    perturbed.append(out)
                s = predict(perturbed)
            rec = evaluate(yt, s, model_name, f"robust:{pname}@{lvl}",
                           threshold=thr, threshold_policy="best_val_youden",
                           record_key=f"robust_{pname}_{lvl}",
                           config={**cfg, "perturb": pname, "level": lvl},
                           save_scores=True)
            print(f"  {pname:7s}@{lvl:<5} AUROC={rec['metrics']['auroc']:.4f}  "
                  f"acc={rec['metrics']['accuracy']:.4f}  ({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="model name (must match a folder under models/)")
    ap.add_argument("--splits", nargs="+",
                    default=["val", "test", "ood"],
                    choices=["val", "test", "ood"],
                    help="splits to evaluate (default: all three)")
    ap.add_argument("--skip-robust", action="store_true",
                    help="skip the 19-perturbation robustness battery")
    args = ap.parse_args()

    predict = load_predict(args.model)
    cfg = {"model": args.model, "evaluator": "scripts/evaluate.py"}
    sp = make_splits()

    # 1. val (needed first to derive Youden threshold)
    if "val" in args.splits or "test" in args.splits or "ood" in args.splits or not args.skip_robust:
        print(f"\n[val] inference + threshold")
        t0 = time.time()
        va_p, ya = sp["val"]
        p_val = predict(va_p)
        thr = best_threshold(ya, p_val)
        rec = evaluate(ya, p_val, args.model, "val",
                       threshold=thr, threshold_policy="best_val_youden",
                       record_key="val", config=cfg, save_scores=True)
        print(f"  {summarize(rec)}  ({time.time()-t0:.0f}s)  thr={thr:.3f}")

    # 2. test
    if "test" in args.splits:
        print(f"\n[test] sealed test")
        t0 = time.time()
        te_p, yt = sp["test"]
        p_test = predict(te_p)
        rec = evaluate(yt, p_test, args.model, "test",
                       threshold=thr, threshold_policy="best_val_youden",
                       record_key="test", config=cfg, save_scores=True)
        print(f"  {summarize(rec)}  ({time.time()-t0:.0f}s)")

    # 3. OOD
    if "ood" in args.splits:
        print(f"\n[ood:sdturbo] cross-generator")
        t0 = time.time()
        op, yo = list_ood()
        if not op:
            print(f"  no OOD images found at {OOD_ROOT}; run scripts/generate_ood.py first")
        else:
            p_ood = predict(op)
            rec = evaluate(yo, p_ood, args.model, "ood:sdturbo",
                           threshold=thr, threshold_policy="best_val_youden",
                           record_key="ood_sdturbo", config=cfg, save_scores=True)
            print(f"  {summarize(rec)}  ({time.time()-t0:.0f}s)")

    # 4. Robustness battery
    if not args.skip_robust and "test" in args.splits:
        print(f"\n[robust] 19-perturbation battery on sealed test")
        te_p, yt = sp["test"]
        run_robust(args.model, predict, te_p, yt, thr, cfg)


if __name__ == "__main__":
    main()
