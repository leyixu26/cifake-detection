"""CLIP-probe end-to-end pipeline (Model 5).

Consolidated from the original `clip_v2.py` (encoder + linear/MLP probes + TTA)
and `clip_robust_final.py` (robustness battery).  The pipeline is:

   1. extract_embeddings()       : encode all CIFAKE splits via frozen CLIP
   2. fit_linear()  / fit_mlp()  : train a small probe on the cached embeddings
   3. tta_linear()               : flip-averaging on the linear probe
   4. run_robust_battery()       : re-encode test under each perturbation, predict
                                    with the saved MLP, emit shared-harness records

All metrics go through `src.eval_harness.evaluate(..., save_scores=True)`.

Encoder choices kept here:
   * `vit_b32_laion`  : OpenCLIP ViT-B-32 with LAION-2B weights (the FINAL choice)
   * `vit_b32_openai` : original OpenAI CLIP (kept as baseline reference)

Embeddings are cached under `<embed_root>/<tag>/{split}_{X,y}.npy`. Default
`embed_root` resolves to `<repo>/src/clip_probe/embeds/` but can be overridden
via the `CIFAKE_CLIP_EMBEDS` env var.
"""
from __future__ import annotations
import glob
import os
import pathlib
import time
from typing import Optional

import joblib
import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Allow this module to be imported either as a package (preferred) or directly.
try:
    from ..eval_harness import best_threshold, evaluate, summarize  # type: ignore
    from ..perturbations import BATTERY  # type: ignore
    from ..freq_detector.datasets import make_splits, load_image  # type: ignore
except (ImportError, ValueError):
    import sys
    _REPO = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_REPO))
    from eval_harness import best_threshold, evaluate, summarize  # type: ignore
    from perturbations import BATTERY  # type: ignore
    from freq_detector.datasets import make_splits, load_image  # type: ignore


# --------------------------------------------------------------------------- #
# Defaults: encoder choice + cache locations
# --------------------------------------------------------------------------- #
ENCODERS = {
    "vit_b32_laion":  ("ViT-B-32",           "laion2b_s34b_b79k"),  # final
    "vit_b32_openai": ("ViT-B-32-quickgelu", "openai"),             # baseline reference
}
DEFAULT_TAG = "vit_b32_laion"
BATCH = 128

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EMBED_ROOT = os.environ.get(
    "CIFAKE_CLIP_EMBEDS",
    str(_REPO_ROOT / "src" / "clip_probe" / "embeds"),
)
OOD_ROOT = os.environ.get(
    "CIFAKE_OOD",
    str(_REPO_ROOT / "data" / "ood_sdturbo"),
)


def _device() -> str:
    if torch.cuda.is_available():     return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_encoder(tag: str = DEFAULT_TAG):
    """Return (model, preprocess, device) for the chosen encoder tag."""
    if tag not in ENCODERS:
        raise ValueError(f"unknown encoder tag {tag!r}; choose one of {list(ENCODERS)}")
    name, pretrained = ENCODERS[tag]
    dev = _device()
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model = model.to(dev).eval()
    return model, preprocess, dev


def _list_ood():
    paths, labels = [], []
    for label, cls in ((0, "REAL"), (1, "FAKE")):
        fs = sorted(glob.glob(f"{OOD_ROOT}/{cls}/*.jpg"))
        paths += fs; labels += [label] * len(fs)
    return paths, np.asarray(labels, np.int64)


# --------------------------------------------------------------------------- #
# Embedding extraction (cached per encoder)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _encode_paths(paths, model, preprocess, dev, flip: bool = False) -> np.ndarray:
    D = model.visual.output_dim
    out = np.empty((len(paths), D), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        imgs = []
        for p in chunk:
            img = Image.open(p).convert("RGB")
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            imgs.append(preprocess(img))
        x = torch.stack(imgs).to(dev)
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)   # L2-normalise
        out[i:i + len(chunk)] = feats.float().cpu().numpy()
        if (i // BATCH) % 50 == 0:
            rate = (i + len(chunk)) / max(time.time() - t0, 1e-6)
            print(f"    {i + len(chunk)}/{len(paths)}  ({rate:.0f} img/s)")
    return out


def extract_embeddings(tag: str = DEFAULT_TAG, also_flip: bool = True):
    """Extract and cache embeddings for train / val / test / OOD (and optional flip)."""
    model, preprocess, dev = load_encoder(tag)
    edir = pathlib.Path(EMBED_ROOT) / tag
    edir.mkdir(parents=True, exist_ok=True)
    print(f"== extract {tag} -> {edir}  (D={model.visual.output_dim})")

    sp = make_splits()
    op, yo = _list_ood()
    jobs = [
        ("train", sp["train"][0], sp["train"][1]),
        ("val",   sp["val"][0],   sp["val"][1]),
        ("test",  sp["test"][0],  sp["test"][1]),
        ("ood",   op, yo),
    ]
    for split, paths, labels in jobs:
        xp = edir / f"{split}_X.npy"; yp = edir / f"{split}_y.npy"
        if xp.exists() and yp.exists():
            print(f"   {split}: cached"); continue
        print(f"   {split}: encoding {len(paths)} images...")
        X = _encode_paths(paths, model, preprocess, dev)
        np.save(xp, X); np.save(yp, np.asarray(labels, np.int64))
        print(f"   {split}: saved {X.shape}  (~{X.nbytes/1e6:.1f} MB)")

    if also_flip:
        for split, paths, _ in jobs[1:]:   # skip train; only val/test/ood for TTA
            xp = edir / f"{split}_X_flip.npy"
            if xp.exists():
                print(f"   {split} flip: cached"); continue
            print(f"   {split} flip: encoding...")
            np.save(xp, _encode_paths(paths, model, preprocess, dev, flip=True))


def _load(tag: str, split: str):
    edir = pathlib.Path(EMBED_ROOT) / tag
    return np.load(edir / f"{split}_X.npy"), np.load(edir / f"{split}_y.npy")


# --------------------------------------------------------------------------- #
# Linear probe (Ojha 2023 recipe)
# --------------------------------------------------------------------------- #
def fit_linear(tag: str = DEFAULT_TAG, save_records: bool = True) -> dict:
    Xtr, ytr = _load(tag, "train")
    Xva, yva = _load(tag, "val")
    Xte, yte = _load(tag, "test")
    Xoo, yoo = _load(tag, "ood")
    best = None
    for C in (0.1, 1.0, 10.0, 100.0):
        lr = LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr)
        s = lr.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yva, s)
        print(f"    C={C:>6}  val AUROC={auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, C, lr, s)
    auc_v, C, lr, sv = best
    thr = best_threshold(yva, sv)
    cfg = {"variant": "linear_probe", "encoder": tag, "C": C}
    model_id = f"clip_probe_{tag}"
    if save_records:
        evaluate(yva, sv, model_id, "val", threshold=thr,
                 threshold_policy="best_val_youden", record_key="val",
                 config=cfg, save_scores=True)
        st = lr.predict_proba(Xte)[:, 1]
        evaluate(yte, st, model_id, "test", threshold=thr,
                 threshold_policy="best_val_youden", record_key="test",
                 config=cfg, save_scores=True)
        so = lr.predict_proba(Xoo)[:, 1]
        evaluate(yoo, so, model_id, "ood:sdturbo", threshold=thr,
                 threshold_policy="best_val_youden", record_key="ood_sdturbo",
                 config=cfg, save_scores=True)
    # persist alongside cached embeddings
    bundle_path = pathlib.Path(EMBED_ROOT) / tag / "best_linear.joblib"
    joblib.dump({"model": lr, "C": C}, bundle_path)
    return {"tag": tag, "C": C, "val_auroc": auc_v}


# --------------------------------------------------------------------------- #
# MLP head (the FINAL probe — 132k params, 256 hidden, dropout 0.3)
# --------------------------------------------------------------------------- #
class MLPHead(nn.Module):
    def __init__(self, d: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(tag: str = DEFAULT_TAG, save_records: bool = True,
            ckpt_path: Optional[str] = None) -> dict:
    dev = _device()
    Xtr, ytr = _load(tag, "train")
    Xva, yva = _load(tag, "val")
    Xte, yte = _load(tag, "test")
    Xoo, yoo = _load(tag, "ood")

    torch.manual_seed(42); np.random.seed(42)
    net = MLPHead(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    Xva_t = torch.from_numpy(Xva).to(dev)

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(40):
        net.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        for i in range(0, len(Xtr_t), 1024):
            idx = perm[i:i + 1024]
            opt.zero_grad()
            lossf(net(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            sva = torch.sigmoid(net(Xva_t)).cpu().numpy()
        auc = float(roc_auc_score(yva, sva))
        if auc > best_auc + 1e-5:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break

    net.load_state_dict(best_state); net.eval()

    # Persist the trained head for robustness + downstream use
    if ckpt_path is None:
        ckpt_path = str(pathlib.Path(EMBED_ROOT) / tag / "best_mlp.pt")
    torch.save({"state_dict": best_state, "in_dim": int(Xtr.shape[1]),
                "hidden": 256, "dropout": 0.3}, ckpt_path)

    def _sc(X):
        with torch.no_grad():
            return torch.sigmoid(net(torch.from_numpy(X).to(dev))).cpu().numpy()

    sv = _sc(Xva); st = _sc(Xte); so = _sc(Xoo)
    thr = best_threshold(yva, sv)
    cfg = {"variant": "mlp_probe", "encoder": tag, "hidden": 256, "dropout": 0.3}
    model_id = f"clip_mlp_{tag}"
    if save_records:
        evaluate(yva, sv, model_id, "val",  threshold=thr,
                 threshold_policy="best_val_youden", record_key="val",
                 config=cfg, save_scores=True)
        evaluate(yte, st, model_id, "test", threshold=thr,
                 threshold_policy="best_val_youden", record_key="test",
                 config=cfg, save_scores=True)
        evaluate(yoo, so, model_id, "ood:sdturbo", threshold=thr,
                 threshold_policy="best_val_youden", record_key="ood_sdturbo",
                 config=cfg, save_scores=True)
    return {"tag": tag, "val_auroc": best_auc,
            "test_auroc": float(roc_auc_score(yte, st)),
            "ood_auroc":  float(roc_auc_score(yoo, so)),
            "ckpt_path":  ckpt_path}


def load_mlp_from_ckpt(ckpt_path: str):
    """Load a saved MLPHead from disk. Returns (model, device)."""
    dev = _device()
    ckpt = torch.load(ckpt_path, map_location=dev)
    net = MLPHead(ckpt["in_dim"], hidden=ckpt["hidden"], dropout=ckpt["dropout"]).to(dev)
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    return net, dev


# --------------------------------------------------------------------------- #
# TTA (horizontal-flip averaging on linear probe)
# --------------------------------------------------------------------------- #
def tta_linear(tag: str = DEFAULT_TAG, save_records: bool = True) -> dict:
    bundle_path = pathlib.Path(EMBED_ROOT) / tag / "best_linear.joblib"
    bundle = joblib.load(bundle_path)
    lr = bundle["model"]

    def _scored(split):
        X  = np.load(pathlib.Path(EMBED_ROOT) / tag / f"{split}_X.npy")
        Xf = np.load(pathlib.Path(EMBED_ROOT) / tag / f"{split}_X_flip.npy")
        avg = (X + Xf) / 2
        avg /= np.linalg.norm(avg, axis=1, keepdims=True) + 1e-9
        return lr.predict_proba(avg)[:, 1]

    yva = np.load(pathlib.Path(EMBED_ROOT) / tag / "val_y.npy")
    yte = np.load(pathlib.Path(EMBED_ROOT) / tag / "test_y.npy")
    yoo = np.load(pathlib.Path(EMBED_ROOT) / tag / "ood_y.npy")
    sv = _scored("val"); st = _scored("test"); so = _scored("ood")
    thr = best_threshold(yva, sv)
    cfg = {"variant": "linear_probe_TTA(flip)", "encoder": tag, "C": bundle["C"]}
    model_id = f"clip_tta_{tag}"
    if save_records:
        for y, s, split, key in [(yva, sv, "val", "val"),
                                  (yte, st, "test", "test"),
                                  (yoo, so, "ood:sdturbo", "ood_sdturbo")]:
            evaluate(y, s, model_id, split, threshold=thr,
                     threshold_policy="best_val_youden", record_key=key,
                     config=cfg, save_scores=True)
    return {"tag": tag, "val_auroc": float(roc_auc_score(yva, sv)),
            "test_auroc": float(roc_auc_score(yte, st)),
            "ood_auroc":  float(roc_auc_score(yoo, so))}


# --------------------------------------------------------------------------- #
# Robustness battery (re-encode test under each perturbation, then predict)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _encode_perturbed(paths, model, preprocess, dev, perturb):
    D = model.visual.output_dim
    out = np.empty((len(paths), D), np.float32)
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        imgs = []
        for p in chunk:
            img = load_image(p)             # float (H, W, 3) in [0, 1]
            img = perturb(img)
            pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            imgs.append(preprocess(pil))
        x = torch.stack(imgs).to(dev)
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out[i:i + len(chunk)] = feats.float().cpu().numpy()
    return out


def run_robust_battery(tag: str = DEFAULT_TAG,
                       ckpt_path: Optional[str] = None,
                       save_records: bool = True):
    """Re-encode every test image under each perturbation and predict via MLP head."""
    model, preprocess, dev = load_encoder(tag)
    if ckpt_path is None:
        ckpt_path = str(pathlib.Path(EMBED_ROOT) / tag / "best_mlp.pt")
    mlp, _ = load_mlp_from_ckpt(ckpt_path)

    # Reuse val-best Youden threshold from cached val embeddings
    val_X = np.load(pathlib.Path(EMBED_ROOT) / tag / "val_X.npy")
    val_y = np.load(pathlib.Path(EMBED_ROOT) / tag / "val_y.npy")
    with torch.no_grad():
        sv = torch.sigmoid(mlp(torch.from_numpy(val_X).to(dev))).cpu().numpy()
    thr = best_threshold(val_y, sv)
    print(f"val-Youden threshold = {thr:.3f}")

    sp = make_splits()
    te_p, yt = sp["test"]
    model_id = f"clip_mlp_{tag}"

    for pname, (levels, fac) in BATTERY.items():
        for lvl in levels:
            t0 = time.time()
            f = fac(lvl)
            X = _encode_perturbed(te_p, model, preprocess, dev, f)
            with torch.no_grad():
                s = torch.sigmoid(mlp(torch.from_numpy(X).to(dev))).cpu().numpy()
            rec = evaluate(yt, s, model_id, f"robust:{pname}@{lvl}",
                           threshold=thr, threshold_policy="best_val_youden",
                           record_key=f"robust_{pname}_{lvl}",
                           config={"variant": "mlp_probe", "encoder": tag,
                                   "perturb": pname, "level": lvl},
                           save_scores=save_records)
            print(f"  {pname:7s}@{lvl:<5} AUROC={rec['metrics']['auroc']:.4f} "
                  f"acc={rec['metrics']['accuracy']:.4f}  ({time.time()-t0:.0f}s)")
