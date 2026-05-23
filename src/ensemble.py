"""5-model team ensemble runner.

Discovers every model directory under project/results/ that contains
test_scores.npz, builds probability-averaged ensembles, and reports the best
subset + leave-one-out contributions for test, OOD, and each robust split.

Pull-style: no hardcoded model list. As soon as a teammate drops their
*_scores.npz into results/<model>/, it'll be included automatically.

Run:  python team_ensemble_runner.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
from itertools import combinations
from sklearn.metrics import roc_auc_score, accuracy_score

import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parent.parent
RES = os.environ.get("CIFAKE_RESULTS", str(_REPO_ROOT / "results" / "per_model"))

# Models we explicitly want in the team-level comparison. (Frequency variants
# are deliberately excluded because in earlier runs they hurt the ensemble.)
TEAM_MODELS = [
    "cnn_baseline_yin",       # Yin's from-scratch CNN
    "resnet18_nathan",        # Nathan's ResNet-18 ImageNet pretrained
    "vit_small_alex",         # Alex's ViT-Small
    "clip_mlp_vit_b32_laion", # Leyi's final CLIP probe (Model 5)
]


def _load(model_dir, key_suffix):
    """Load (y, score) from results/<model_dir>/<stem>_scores.npz.
    Returns None if missing."""
    p = f"{RES}/{model_dir}/{key_suffix}_scores.npz"
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d["y_true"], d["y_score"]


def ensemble_run(model_keys, key_suffix, split_label):
    """Compute pairwise + full ensembles for the given suffix across the
    supplied model list. Returns a dict of metrics or None if any missing."""
    y = None
    scores = {}
    for name in model_keys:
        sc = _load(name, key_suffix)
        if sc is None:
            return None, f"{name} missing {key_suffix}_scores.npz"
        y_i, s_i = sc
        if y is None:
            y = y_i
        elif not np.array_equal(y, y_i):
            return None, f"{name} label mismatch on {key_suffix}"
        scores[name] = s_i

    per_model = {n: float(roc_auc_score(y, s)) for n, s in scores.items()}
    full = float(roc_auc_score(y, np.stack(list(scores.values())).mean(0)))

    pairs = {}
    for a, b in combinations(scores.keys(), 2):
        S = np.stack([scores[a], scores[b]])
        pairs[f"{a}__+__{b}"] = float(roc_auc_score(y, S.mean(0)))

    triples = {}
    if len(scores) >= 3:
        for trio in combinations(scores.keys(), 3):
            S = np.stack([scores[k] for k in trio])
            triples["__+__".join(trio)] = float(roc_auc_score(y, S.mean(0)))

    loo = {}
    if len(scores) >= 3:
        for held in scores:
            sub = {k: v for k, v in scores.items() if k != held}
            S = np.stack(list(sub.values()))
            sub_auc = float(roc_auc_score(y, S.mean(0)))
            loo[held] = {"auroc_without": sub_auc,
                          "contribution_pp": (full - sub_auc) * 100}

    return {
        "split": split_label, "n": int(len(y)),
        "per_model_auroc": per_model,
        "pair_ensembles": pairs,
        "triple_ensembles": triples,
        "all_ensemble_auroc": full,
        "leave_one_out": loo,
        "best_pair": max(pairs.items(), key=lambda x: x[1]),
        "best_triple": max(triples.items(), key=lambda x: x[1]) if triples else None,
    }, None


def main():
    # Validate which TEAM_MODELS have any record at all
    available = []
    for m in TEAM_MODELS:
        if os.path.exists(f"{RES}/{m}/test_scores.npz"):
            available.append(m)
        else:
            print(f"  [skip] {m} (no test_scores.npz)")
    print(f"available models: {available}")
    if len(available) < 2:
        print("Need at least 2 models for an ensemble. Bailing.")
        sys.exit(1)

    print()
    out = {"models": available, "results": {}}

    # In-distribution test
    res, err = ensemble_run(available, "test", "test")
    if err:
        print(f"test: SKIP ({err})")
    else:
        print(f"=== TEST (n={res['n']}) ===")
        for m, a in res["per_model_auroc"].items():
            print(f"  {m:30s}  {a:.4f}")
        print(f"  best pair:  {res['best_pair'][0]:60s} {res['best_pair'][1]:.4f}")
        if res["best_triple"]:
            print(f"  best 3:     {res['best_triple'][0]:60s} {res['best_triple'][1]:.4f}")
        print(f"  all ensemble: {res['all_ensemble_auroc']:.4f}")
        if res["leave_one_out"]:
            print("  leave-one-out (positive = contribution):")
            for m, c in res["leave_one_out"].items():
                print(f"    drop {m:30s}: AUROC -> {c['auroc_without']:.4f}  ({c['contribution_pp']:+.2f} pp)")
        out["results"]["test"] = res

    # OOD
    print()
    res, err = ensemble_run(available, "ood_sdturbo", "ood:sdturbo")
    if err:
        print(f"OOD: SKIP ({err})")
    else:
        print(f"=== OOD sd-turbo (n={res['n']}) ===")
        for m, a in res["per_model_auroc"].items():
            print(f"  {m:30s}  {a:.4f}")
        print(f"  best pair:  {res['best_pair'][0]:60s} {res['best_pair'][1]:.4f}")
        if res["best_triple"]:
            print(f"  best 3:     {res['best_triple'][0]:60s} {res['best_triple'][1]:.4f}")
        print(f"  all ensemble: {res['all_ensemble_auroc']:.4f}")
        if res["leave_one_out"]:
            print("  leave-one-out:")
            for m, c in res["leave_one_out"].items():
                print(f"    drop {m:30s}: AUROC -> {c['auroc_without']:.4f}  ({c['contribution_pp']:+.2f} pp)")
        out["results"]["ood_sdturbo"] = res

    # Robustness summary
    print()
    print("=== ROBUSTNESS (per-perturb best subset) ===")
    PERTURBS = [
        ("jpeg",    [90, 75, 60, 40, 25, 10]),
        ("blur",    [0.3, 0.5, 0.8, 1.0, 1.5]),
        ("noise",   [2, 4, 8, 16, 32]),
        ("rescale", [24, 16, 12]),
    ]
    for pname, levels in PERTURBS:
        for lvl in levels:
            suffix = f"robust_{pname}_{lvl}"
            res, err = ensemble_run(available, suffix, f"robust:{pname}@{lvl}")
            if err:
                continue
            best_name, best_auc = res["best_pair"]
            print(f"  {pname:7s}@{lvl:<5}  ensemble_all={res['all_ensemble_auroc']:.4f}  best_pair_auc={best_auc:.4f}")
            out["results"][f"robust_{pname}_{lvl}"] = res

    # Save full report
    out_path = f"{RES}/team_ensemble_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nfull report -> {out_path}")


if __name__ == "__main__":
    main()
