"""Regenerate the canonical comparison figures into results/figures/.

Reads only from `results/per_model/*` and `results/team_ensemble_report.json`,
so it works as long as those are populated by `scripts/evaluate.py` and
`scripts/run_team_ensemble.py`.

Produces (overwrites the curated headline figures in place):
    01_clean_vs_ood.png         all models side-by-side on test + OOD AUROC
    02_robustness_curves.png    per-perturbation AUROC curves (4 panels: jpeg, blur, noise, rescale)
    03_team_ensemble.png        best pair / best triple / all-ensemble AUROC + leave-one-out
"""
from __future__ import annotations
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = REPO_ROOT / "results"
PER = RES / "per_model"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Display order + colours. Edit here if you re-arrange the lineup.
MODELS = [
    ("Yin small CNN (from scratch)", "cnn_baseline_yin",       "#2ca02c"),
    ("Nathan ResNet-18 (ImageNet)",  "resnet18_nathan",        "#d62728"),
    ("Alex ViT-Small",               "vit_small_alex",         "#9467bd"),
    ("Leyi freq detector (best)",    "freq_detector",          "#1f77b4"),
    ("Leyi CLIP probe (LAION+MLP)",  "clip_mlp_vit_b32_laion", "#7c3aed"),
]

PERTURBS = {
    "jpeg":    [90, 75, 60, 40, 25, 10],
    "blur":    [0.3, 0.5, 0.8, 1.0, 1.5],
    "noise":   [2, 4, 8, 16, 32],
    "rescale": [24, 16, 12],
}


def _load_json(p):
    if not p.exists(): return None
    try: return json.load(open(p))
    except Exception: return None


def _auc(model_dir: str, split_key: str):
    """split_key in {'test', 'val', 'ood_sdturbo', 'robust_<p>_<l>'}"""
    rec = _load_json(PER / model_dir / f"{split_key}.json")
    return rec["metrics"]["auroc"] if rec else None


# --------------------------------------------------------------------------- #
def fig_clean_vs_ood():
    have = []
    for name, mdir, color in MODELS:
        t = _auc(mdir, "test"); o = _auc(mdir, "ood_sdturbo")
        if t is not None and o is not None:
            have.append((name, color, t, o))
        else:
            print(f"  [skip {name}] test={t} ood={o}")
    if not have:
        print("  no models with both test and ood; skipping fig_clean_vs_ood")
        return
    names, colors, ts, os_ = zip(*[(n, c, t, o) for (n, c, t, o) in have])
    x = np.arange(len(names)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - w/2, ts, w, color=colors, edgecolor="black", lw=0.5,
           label="in-dist test (CIFAKE)")
    ax.bar(x + w/2, os_, w, color=colors, edgecolor="black", lw=0.5,
           alpha=0.55, hatch="//", label="cross-gen OOD (sd-turbo)")
    for i, (t, o) in enumerate(zip(ts, os_)):
        ax.text(i - w/2, t + 0.005, f"{t:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, o + 0.005, f"{o:.3f}", ha="center", fontsize=8)
        ax.text(i, 0.46, f"Δ={(t-o)*100:+.1f}", ha="center",
                fontsize=8, color="firebrick")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("AUROC"); ax.set_ylim(0.45, 1.02); ax.grid(alpha=0.3, axis="y")
    ax.set_title("All 5 models — clean test vs cross-generator OOD AUROC")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "01_clean_vs_ood.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {FIG/'01_clean_vs_ood.png'}")


def fig_robust_curves():
    fig, ax = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for k, (pname, levels) in enumerate(PERTURBS.items()):
        for name, mdir, color in MODELS:
            ys = [_auc(mdir, f"robust_{pname}_{lvl}") for lvl in levels]
            if all(y is None for y in ys): continue
            ax[k].plot(levels, ys, "-o", color=color, label=name, lw=1.7)
        ax[k].set_title(pname); ax[k].set_xlabel("perturbation strength")
        ax[k].grid(alpha=0.3); ax[k].set_ylim(0.4, 1.0)
        if k == 0: ax[k].set_ylabel("test AUROC")
    ax[-1].legend(loc="lower left", fontsize=7)
    fig.suptitle("Robustness curves — inference-time perturbations on sealed test")
    fig.tight_layout(); fig.savefig(FIG / "02_robustness_curves.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {FIG/'02_robustness_curves.png'}")


def fig_ensemble():
    rep = _load_json(RES / "team_ensemble_report.json")
    if rep is None:
        print("  no team_ensemble_report.json — run scripts/run_team_ensemble.py first")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, kind in zip(axes, ["test", "ood_sdturbo"]):
        info = rep["results"].get(kind)
        if not info:
            ax.set_title(f"{kind}: no data"); continue
        per = info["per_model_auroc"]
        pairs = info.get("pair_ensembles", {})
        triples = info.get("triple_ensembles", {})
        full = info["all_ensemble_auroc"]
        labels = list(per) + list(pairs) + (list(triples) if triples else []) + ["ALL"]
        values = list(per.values()) + list(pairs.values()) + \
                 (list(triples.values()) if triples else []) + [full]
        # colour by group
        n_p = len(per); n_pair = len(pairs); n_tri = len(triples) if triples else 0
        colors = ["#9ecae1"] * n_p + ["#74c476"] * n_pair + \
                 ["#a1d99b"] * n_tri + ["#31a354"]
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors, edgecolor="black", lw=0.5)
        for i, v in enumerate(values):
            ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"ensemble · {kind}  (best single → pairs → triples → all)")
        ax.set_ylabel("AUROC"); ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(min(values) - 0.02, 1.0)
    fig.suptitle("Team ensemble across diverse inductive biases")
    fig.tight_layout(); fig.savefig(FIG / "03_team_ensemble.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {FIG/'03_team_ensemble.png'}")


def main():
    print("=== fig_clean_vs_ood ==="); fig_clean_vs_ood()
    print("=== fig_robust_curves ==="); fig_robust_curves()
    print("=== fig_ensemble ==="); fig_ensemble()
    print(f"\nall figures -> {FIG}")


if __name__ == "__main__":
    main()
