"""Robustness perturbation battery (inference-time only).

Each factory returns ``f(img) -> img`` where img is float32 (H, W, 3) in [0, 1].
Used via ``SpectrumDataset(..., perturb=f)`` on the SEALED test set to produce
accuracy/AUROC-vs-strength curves. ``jpeg`` also doubles as the
compression-confound probe (a pure-compression detector collapses under it).

BATTERY maps name -> (levels, factory(level)).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

_RNG = np.random.RandomState(1234)  # fixed so robustness eval is reproducible


def _to_u8(img):
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def jpeg_recompress(quality: int):
    def f(img):
        buf = io.BytesIO()
        Image.fromarray(_to_u8(img)).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"), np.float32) / 255.0
    return f


def gaussian_blur(sigma: float):
    def f(img):
        out = np.empty_like(img)
        for c in range(3):
            out[..., c] = gaussian_filter(img[..., c], sigma=float(sigma))
        return np.clip(out, 0, 1).astype(np.float32)
    return f


def additive_noise(sigma_255: float):
    s = float(sigma_255) / 255.0

    def f(img):
        n = _RNG.normal(0, s, img.shape).astype(np.float32)
        return np.clip(img + n, 0, 1).astype(np.float32)
    return f


def rescale(small: int):
    def f(img):
        im = Image.fromarray(_to_u8(img))
        im = im.resize((small, small), Image.BICUBIC).resize((32, 32), Image.BICUBIC)
        return np.asarray(im.convert("RGB"), np.float32) / 255.0
    return f


BATTERY = {
    "jpeg": ([90, 75, 60, 40, 25, 10], jpeg_recompress),
    "blur": ([0.3, 0.5, 0.8, 1.0, 1.5], gaussian_blur),
    "noise": ([2, 4, 8, 16, 32], additive_noise),
    "rescale": ([24, 16, 12], rescale),
}
