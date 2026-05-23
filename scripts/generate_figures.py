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
def _render_clean_vs_ood(have, mode, out_name, title):
    """Render the headline figure in one of three modes.

    mode:
        "full"          → both in-dist + OOD bars, with Δ overlay (writeup)
        "in_dist_only"  → only the left bars rendered, same canvas/labels
                          (used on slides 4-7 to avoid spoiling the slide-8 reveal)
        "ood_reveal"    → both bars rendered + Δ overlay, but OOD bars in full
                          colour to make the reveal punchy (used on slide 8)
    """
    names, colors, ts, os_ = zip(*[(n, c, t, o) for (n, c, t, o) in have])
    x = np.arange(len(names)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.8))

    if mode == "in_dist_only":
        # Center the test bars on each x tick; keep the same y-axis as
        # the full version so the next slide's reveal is visually congruent.
        ax.bar(x, ts, w, color=colors, edgecolor="black", lw=0.5,
               label="in-dist test (CIFAKE)")
        for i, t in enumerate(ts):
            ax.text(i, t + 0.005, f"{t:.3f}", ha="center", fontsize=8)
        ax.set_title(f"{title} — clean in-distribution test only")
        ax.legend(loc="lower right")
    elif mode == "ood_reveal":
        # Plot test bars in muted form to anchor the comparison, then the
        # OOD bars in full colour with Δ annotation — this is the slide-8
        # transition where Leyi says "...and here's what happens cross-gen".
        ax.bar(x - w/2, ts, w, color=colors, edgecolor="black", lw=0.5,
               alpha=0.35, label="in-dist test (CIFAKE)")
        ax.bar(x + w/2, os_, w, color=colors, edgecolor="black", lw=0.5,
               hatch="//", label="cross-gen OOD (sd-turbo) — the reveal")
        for i, (t, o) in enumerate(zip(ts, os_)):
            ax.text(i - w/2, t + 0.005, f"{t:.3f}", ha="center", fontsize=8,
                    color="gray")
            ax.text(i + w/2, o + 0.005, f"{o:.3f}", ha="center", fontsize=9,
                    color="black", fontweight="bold")
            ax.text(i, 0.46, f"Δ={(t-o)*100:+.1f}", ha="center",
                    fontsize=9, color="firebrick", fontweight="bold")
        ax.set_title(f"{title} — cross-generator OOD reveal")
        ax.legend(loc="lower right")
    else:  # "full"
        ax.bar(x - w/2, ts, w, color=colors, edgecolor="black", lw=0.5,
               label="in-dist test (CIFAKE)")
        ax.bar(x + w/2, os_, w, color=colors, edgecolor="black", lw=0.5,
               alpha=0.55, hatch="//", label="cross-gen OOD (sd-turbo)")
        for i, (t, o) in enumerate(zip(ts, os_)):
            ax.text(i - w/2, t + 0.005, f"{t:.3f}", ha="center", fontsize=8)
            ax.text(i + w/2, o + 0.005, f"{o:.3f}", ha="center", fontsize=8)
            ax.text(i, 0.46, f"Δ={(t-o)*100:+.1f}", ha="center",
                    fontsize=8, color="firebrick")
        ax.set_title(title)
        ax.legend(loc="lower right")

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("AUROC"); ax.set_ylim(0.45, 1.02); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / out_name, dpi=130)
    plt.close(fig)
    print(f"  wrote {FIG/out_name}")


def fig_clean_vs_ood(presentation_mode: bool = False):
    """Headline figure: every model's in-dist + cross-generator OOD AUROC.

    Default: emits a single composite `01_clean_vs_ood.png` (used in the
    writeup and as the slide-8 fallback).

    `presentation_mode=True` additionally emits `01a_in_dist_only.png` (for
    slides 4-7 — does not spoil the slide-8 OOD reveal) and
    `01b_ood_reveal.png` (for the slide-8 transition).
    """
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
    title = "All 5 models — clean test vs cross-generator OOD AUROC"
    _render_clean_vs_ood(have, "full", "01_clean_vs_ood.png", title)
    if presentation_mode:
        _render_clean_vs_ood(have, "in_dist_only", "01a_in_dist_only.png", title)
        _render_clean_vs_ood(have, "ood_reveal",   "01b_ood_reveal.png",   title)


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
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--presentation-mode", action="store_true",
                    help="Additionally emit the split-reveal variants of "
                         "figure 01 used in the slide deck "
                         "(01a_in_dist_only.png, 01b_ood_reveal.png).")
    args = ap.parse_args()

    print("=== fig_clean_vs_ood ===")
    fig_clean_vs_ood(presentation_mode=args.presentation_mode)
    print("=== fig_robust_curves ===")
    fig_robust_curves()
    print("=== fig_ensemble ===")
    fig_ensemble()
    print(f"\nall figures -> {FIG}")


if __name__ == "__main__":
    main()
