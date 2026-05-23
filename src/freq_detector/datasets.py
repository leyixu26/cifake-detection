"""CIFAKE indexing, frozen split, and feature extraction.

Label convention:  REAL = 0,  FAKE = 1  (positive class = AI-generated).

* ``test/`` is SEALED: never used for fitting, tuning, or early stopping.
* The 100k ``train/`` pool is split 90/10 (stratified, seed=42) into the
  working train / validation sets used for everything pre-final.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .fft_utils import SpectralNormalizer, compute_logmag
from .radial_psd import spectral_features

# Resolve relative to repo root so the module works regardless of where the
# repo is cloned. Override with CIFAKE_DATA env var.
import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parent.parent.parent
DATA_ROOT = os.environ.get("CIFAKE_DATA", str(_REPO_ROOT / "data" / "cifake"))
SEED = 42


def _list_split(split: str) -> tuple[list[str], np.ndarray]:
    """All (path, label) for 'train' or 'test', deterministically ordered."""
    paths, labels = [], []
    for label, cls in ((0, "REAL"), (1, "FAKE")):
        files = sorted(glob.glob(os.path.join(DATA_ROOT, split, cls, "*.jpg")))
        paths += files
        labels += [label] * len(files)
    return paths, np.asarray(labels, dtype=np.int64)


def make_splits(val_frac: float = 0.10, seed: int = SEED):
    """Return dict with 'train', 'val', 'test' -> (paths list, labels array).

    Stratified, fixed-seed; the same call always yields the same partition.
    """
    paths, labels = _list_split("train")
    paths = np.asarray(paths)
    rng = np.random.RandomState(seed)
    tr_idx, va_idx = [], []
    for c in (0, 1):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        va_idx.append(idx[:n_val])
        tr_idx.append(idx[n_val:])
    tr = np.sort(np.concatenate(tr_idx))
    va = np.sort(np.concatenate(va_idx))
    test_paths, test_labels = _list_split("test")
    return {
        "train": (paths[tr].tolist(), labels[tr]),
        "val": (paths[va].tolist(), labels[va]),
        "test": (test_paths, test_labels),
    }


def load_image(path: str) -> np.ndarray:
    """JPEG -> float32 (H, W, 3) in [0, 1]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Variant B: spectrum tensors (computed on the fly so ablations are free)
# --------------------------------------------------------------------------- #
class SpectrumDataset(Dataset):
    def __init__(
        self,
        paths,
        labels,
        color: str = "rgb",
        transform: str = "fft",
        window: bool = True,
        remove_dc: int = 1,
        normalizer=None,
        perturb=None,
    ):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.color, self.transform = color, transform
        self.window, self.remove_dc = window, remove_dc
        self.normalizer = normalizer  # SpectralNormalizer or None
        self.perturb = perturb  # callable(img_uint8_like float[0,1]) -> img, for robustness

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = load_image(self.paths[i])
        if self.perturb is not None:
            img = self.perturb(img)
        spec = compute_logmag(
            img,
            color=self.color,
            transform=self.transform,
            window=self.window,
            remove_dc=self.remove_dc,
        )
        if spec.ndim == 2:  # luma -> add channel dim
            spec = spec[None]
        if self.normalizer is not None:
            flat = self.normalizer.transform(spec.reshape(1, -1))
            spec = flat.reshape(spec.shape)
        x = torch.from_numpy(np.ascontiguousarray(spec)).float()
        y = torch.tensor(self.labels[i])
        return x, y


# --------------------------------------------------------------------------- #
# Variant A: bulk handcrafted feature extraction (cached)
# --------------------------------------------------------------------------- #
def fit_normalizer(
    train_paths,
    color: str = "rgb",
    transform: str = "fft",
    window: bool = True,
    remove_dc: int = 1,
    cache: str | None = None,
) -> SpectralNormalizer:
    """Streaming per-bin mean/std over TRAIN ONLY. Leakage-safe by construction.

    Streams sum / sum-of-squares so we never hold all spectra in RAM, then
    builds a frozen ``SpectralNormalizer``. Persisted to ``cache`` (.npz).
    """
    if cache and os.path.exists(cache):
        return SpectralNormalizer.load(cache)

    s = ssq = None
    n = 0
    for p in train_paths:
        spec = compute_logmag(load_image(p), color, transform, window, remove_dc)
        v = spec.reshape(-1).astype(np.float64)
        if s is None:
            s = np.zeros_like(v)
            ssq = np.zeros_like(v)
        s += v
        ssq += v * v
        n += 1
    mu = s / n
    var = np.maximum(ssq / n - mu * mu, 0.0)
    norm = SpectralNormalizer.from_stats(mu.astype(np.float32),
                                         np.sqrt(var).astype(np.float32))
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        norm.save(cache)
    return norm


def extract_features(
    paths,
    color: str = "luma",
    transform: str = "fft",
    window: bool = True,
    remove_dc: int = 1,
    cache: str | None = None,
    perturb=None,
    **feat_kwargs,
) -> np.ndarray:
    """(N, D) handcrafted spectral feature matrix. Cached to ``cache`` (.npy).

    ``perturb``  optional callable(img) -> img applied before the spectrum
    transform (used for the robustness battery / OOD; disables caching).
    """
    from .fft_utils import (  # local import to avoid cycle at module load
        dct_magnitude,
        fft_magnitude,
        to_planes,
        zero_dc,
    )

    if cache and os.path.exists(cache) and perturb is None:
        return np.load(cache)

    feats = []
    for p in paths:
        img = load_image(p)
        if perturb is not None:
            img = perturb(img)
        planes = to_planes(img, color)
        if transform == "fft":
            mag = zero_dc(fft_magnitude(planes, window), k=remove_dc, shifted=True)
        else:
            mag = zero_dc(dct_magnitude(planes, window), k=remove_dc, shifted=False)
        feats.append(spectral_features(mag, **feat_kwargs))
    feats = np.asarray(feats, dtype=np.float32)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, feats)
    return feats
