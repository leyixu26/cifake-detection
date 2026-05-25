"""Compute PR-AUC and GFLOPs for the frequency detector and the CLIP probe,
then write the values back into the corresponding per-model JSONs.

These two metrics were not part of the original shared eval harness, but the
team comparison table on the deck reports them for every model, so they are
needed here for consistency.

PR-AUC is computed from the persisted per-sample scores in
    results/per_model/<model>/{test,ood_sdturbo}_scores.npz
which are emitted by `scripts/evaluate.py` (via `eval_harness.evaluate(...,
save_scores=True)`). The values are exact and reproducible from the same
predictions used for ROC-AUC.

GFLOPs is computed via fvcore.nn.FlopCountAnalysis on a forward pass with the
canonical input shape per model:
    frequency detector  -- 3 x 32 x 32   (SpectrumCNN, Variant B headline)
    CLIP probe          -- 3 x 224 x 224 (frozen ViT-B/32 + MLP head)
fvcore is the same tool the teammates used (per the `flop_counter:
fvcore-0.1.5.post20221221` field in their results_CNN_from_scratch.json).

Run:
    pip install fvcore
    PYTHONPATH=. python scripts/compute_pr_auc_and_gflops.py
"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import average_precision_score

REPO = pathlib.Path(__file__).resolve().parent.parent
PER = REPO / "results" / "per_model"

# Architecture notes per model — used to instantiate forwards for GFLOPs.
FREQ_INPUT = (1, 3, 32, 32)
CLIP_INPUT = (1, 3, 224, 224)


@contextlib.contextmanager
def _quiet():
    """Silence fvcore's noisy unsupported-op warnings. The unsupported ops
    (element-wise add/mul, gelu, scaled_dot_product_attention) are not the
    FLOP-dominant ones for the architectures here; linear and convolutional
    ops dominate, and fvcore counts those correctly."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        yield


def pr_auc(model_dir: str, split: str) -> float:
    f = PER / model_dir / f"{split}_scores.npz"
    d = np.load(f)
    return float(average_precision_score(d["y_true"], d["y_score"]))


def gflops_freq() -> tuple[float, int]:
    """Variant B SpectrumCNN. Returns (gflops, trainable_params)."""
    sys.path.insert(0, str(REPO))
    from src.freq_detector.models import SpectrumCNN

    m = SpectrumCNN(in_ch=3).eval()
    with _quiet():
        flops = FlopCountAnalysis(m, torch.randn(*FREQ_INPUT)).total()
    params = sum(p.numel() for p in m.parameters())
    return flops / 1e9, params


def gflops_clip() -> tuple[float, int, int]:
    """Frozen ViT-B/32 LAION encoder + trainable 132 k-param MLP head.
    Returns (full_forward_gflops, encoder_params, head_params)."""
    import open_clip

    encoder, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    encoder.visual.eval()

    class MLPHead(nn.Module):
        def __init__(self, dim=512, hidden=256, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                nn.Dropout(dropout),    nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    head = MLPHead().eval()
    with _quiet():
        enc_flops = FlopCountAnalysis(
            encoder.visual, torch.randn(*CLIP_INPUT)).total()
        head_flops = FlopCountAnalysis(head, torch.randn(1, 512)).total()

    enc_params = sum(p.numel() for p in encoder.visual.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    total_gflops = (enc_flops + head_flops) / 1e9
    return total_gflops, enc_params, head_params


def _update_json(path: pathlib.Path, pr_auc_val: float, gflops: float | None,
                 gflops_note: str | None):
    rec = json.load(open(path))
    rec.setdefault("metrics", {})["pr_auc"] = round(pr_auc_val, 6)
    if gflops is not None:
        rec["gflops"] = round(gflops, 4)
    if gflops_note is not None:
        rec["gflops_note"] = gflops_note
    rec["_added_by"] = "scripts/compute_pr_auc_and_gflops.py"
    json.dump(rec, open(path, "w"), indent=2)


def main():
    print("=== PR-AUC (computed from persisted *_scores.npz) ===")
    freq_test_pr   = pr_auc("freq_detector",          "test")
    freq_ood_pr    = pr_auc("freq_detector",          "ood_sdturbo")
    clip_test_pr   = pr_auc("clip_mlp_vit_b32_laion", "test")
    clip_ood_pr    = pr_auc("clip_mlp_vit_b32_laion", "ood_sdturbo")
    print(f"  freq_detector          test {freq_test_pr:.4f}  ood {freq_ood_pr:.4f}")
    print(f"  clip_mlp_vit_b32_laion test {clip_test_pr:.4f}  ood {clip_ood_pr:.4f}")

    print()
    print("=== GFLOPs (via fvcore) ===")
    freq_gf, freq_pars = gflops_freq()
    clip_gf, clip_enc_pars, clip_head_pars = gflops_clip()
    print(f"  freq_detector (Variant B, 3x32x32)            {freq_gf:.4f} GFLOPs  "
          f"params {freq_pars:,}")
    print(f"  clip_probe   (frozen encoder + MLP, 3x224x224) {clip_gf:.4f} GFLOPs  "
          f"params {clip_enc_pars:,} frozen + {clip_head_pars:,} trained")

    print()
    print("=== writing back to per-model JSONs ===")
    # Freq detector: gflops applies to the headline test split; we also
    # record the value on ood_sdturbo for symmetry (same model, same FLOPs).
    _update_json(PER / "freq_detector" / "test.json",
                 freq_test_pr, freq_gf,
                 "Variant B SpectrumCNN forward at 3x32x32 (fvcore)")
    _update_json(PER / "freq_detector" / "ood_sdturbo.json",
                 freq_ood_pr, freq_gf,
                 "Variant B SpectrumCNN forward at 3x32x32 (fvcore)")
    _update_json(PER / "clip_mlp_vit_b32_laion" / "test.json",
                 clip_test_pr, clip_gf,
                 "Frozen ViT-B/32 (LAION) + MLP head at 3x224x224 (fvcore). "
                 "Encoder is frozen at training time but still counted at "
                 "inference for apples-to-apples comparison with the other "
                 "rows in the comparison table.")
    _update_json(PER / "clip_mlp_vit_b32_laion" / "ood_sdturbo.json",
                 clip_ood_pr, clip_gf,
                 "Frozen ViT-B/32 (LAION) + MLP head at 3x224x224 (fvcore).")
    for p in [PER / "freq_detector" / "test.json",
              PER / "freq_detector" / "ood_sdturbo.json",
              PER / "clip_mlp_vit_b32_laion" / "test.json",
              PER / "clip_mlp_vit_b32_laion" / "ood_sdturbo.json"]:
        print(f"  wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
