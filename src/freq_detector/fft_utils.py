"""Spectral preprocessing for the frequency-domain AI-image detector.

All functions operate on a single image given as a float32 array in [0, 1] with
shape (H, W, 3) (the layout returned by ``np.asarray(PIL.Image)`` / 255).

The pipeline is deliberately split so ablations can toggle each stage:

    image --(color)--> plane(s) --(window)--> FFT/DCT --> magnitude
        |-> to_logmag(remove_dc)  -> CNN input  (Variant B)
        |-> power = magnitude**2  -> radial PSD (Variant A, see radial_psd.py)

Conventions
-----------
* "luma"  : single BT.601 luminance plane, shape (H, W).
* "rgb"   : per-channel, shape (3, H, W)  (channels-first for torch).
* magnitude / log-magnitude are always ``fftshift``-ed (DC centred).
"""
from __future__ import annotations

import hashlib

import numpy as np
import scipy.fft as sfft

EPS = 1e-8
# BT.601 luma weights (matches JPEG / CIFAR convention).
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Color
# --------------------------------------------------------------------------- #
def to_planes(img: np.ndarray, color: str) -> np.ndarray:
    """Return planes to transform.

    color="luma" -> (H, W)        single luminance plane
    color="rgb"  -> (3, H, W)     channels-first stack
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got {img.shape}")
    if color == "luma":
        return img @ _LUMA  # (H, W)
    if color == "rgb":
        return np.transpose(img, (2, 0, 1))  # (3, H, W)
    raise ValueError(f"unknown color mode {color!r}")


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def hann2d(h: int, w: int) -> np.ndarray:
    """Separable 2-D Hann window, shape (h, w), peak 1.0."""
    wy = np.hanning(h + 2)[1:-1]  # drop the zero endpoints (all-zero rows hurt)
    wx = np.hanning(w + 2)[1:-1]
    return np.outer(wy, wx).astype(np.float32)


def _apply_window(plane: np.ndarray, window: bool) -> np.ndarray:
    if not window:
        return plane
    h, w = plane.shape[-2:]
    return plane * hann2d(h, w)


# --------------------------------------------------------------------------- #
# FFT / DCT magnitude
# --------------------------------------------------------------------------- #
def fft_magnitude(planes: np.ndarray, window: bool = True) -> np.ndarray:
    """fftshift-ed magnitude spectrum. Same leading shape as ``planes``."""
    p = _apply_window(planes, window)
    spec = sfft.fft2(p, axes=(-2, -1))
    spec = sfft.fftshift(spec, axes=(-2, -1))
    return np.abs(spec).astype(np.float32)


def dct_magnitude(planes: np.ndarray, window: bool = False) -> np.ndarray:
    """|2-D type-II orthonormal DCT| (Frank et al. style).

    Not fftshift-ed: DCT energy is naturally compacted at the top-left, which is
    fine for a CNN and for band statistics. Windowing defaults off for DCT.
    """
    p = _apply_window(planes, window)
    coeffs = sfft.dctn(p, type=2, norm="ortho", axes=(-2, -1))
    return np.abs(coeffs).astype(np.float32)


def _center(shape) -> tuple[int, int]:
    h, w = shape[-2:]
    return h // 2, w // 2


def zero_dc(mag: np.ndarray, k: int = 1, shifted: bool = True) -> np.ndarray:
    """Zero the DC term (and ``k``-1 rings of low-freq neighbours).

    k=0 keeps DC; k=1 zeros only DC; k=2 zeros the central 3x3, etc.
    ``shifted`` True  -> DC at centre  (fftshift-ed FFT magnitude).
    ``shifted`` False -> DC at (0, 0)  (DCT or un-shifted FFT).
    """
    if k <= 0:
        return mag
    out = mag.copy()
    h, w = out.shape[-2:]
    if shifted:
        cy, cx = _center(out.shape)
        y0, y1 = max(cy - k + 1, 0), min(cy + k, h)
        x0, x1 = max(cx - k + 1, 0), min(cx + k, w)
        out[..., y0:y1, x0:x1] = 0.0
    else:
        out[..., :k, :k] = 0.0
    return out


def to_logmag(mag: np.ndarray, remove_dc: int = 1, shifted: bool = True) -> np.ndarray:
    """log(magnitude + eps) with optional DC removal. CNN input (Variant B)."""
    m = zero_dc(mag, k=remove_dc, shifted=shifted)
    return np.log(m + EPS).astype(np.float32)


def compute_complex(
    img: np.ndarray, window: bool = True, remove_dc: int = 1
) -> np.ndarray:
    """End-to-end image -> 6-channel [Re, Im] FFT (per RGB channel).

    Returns (6, H, W).  Used by the complex-spectrum CNN sibling ablation:
    the complex FFT preserves all the input information (it's a bijection),
    so this is the strongest within-frequency-domain comparison to magnitude.
    """
    planes = to_planes(img, "rgb")              # (3, H, W)
    p = _apply_window(planes, window)
    spec = sfft.fft2(p, axes=(-2, -1))
    spec = sfft.fftshift(spec, axes=(-2, -1))   # (3, H, W) complex
    re = spec.real.astype(np.float32)
    im = spec.imag.astype(np.float32)
    # signed-log on each part to compress dynamic range while keeping sign info
    re = np.sign(re) * np.log1p(np.abs(re))
    im = np.sign(im) * np.log1p(np.abs(im))
    out = np.concatenate([re, im], axis=0)      # (6, H, W)
    # zero DC bin on every channel (Re of DC = sum of pixels, very large)
    if remove_dc > 0:
        out = zero_dc(out, k=remove_dc, shifted=True)
    return out.astype(np.float32)


def compute_logmag(
    img: np.ndarray,
    color: str = "rgb",
    transform: str = "fft",
    window: bool = True,
    remove_dc: int = 1,
) -> np.ndarray:
    """End-to-end image -> log-magnitude spectrum (Variant B convenience).

    Returns (H, W) for color="luma", (3, H, W) for color="rgb".
    """
    planes = to_planes(img, color)
    if transform == "fft":
        mag = fft_magnitude(planes, window=window)
        return to_logmag(mag, remove_dc=remove_dc, shifted=True)
    if transform == "dct":
        mag = dct_magnitude(planes, window=window)
        return to_logmag(mag, remove_dc=remove_dc, shifted=False)
    raise ValueError(f"unknown transform {transform!r}")


# --------------------------------------------------------------------------- #
# Leakage-safe per-bin normalizer
# --------------------------------------------------------------------------- #
class SpectralNormalizer:
    """Per-bin standardization with TRAIN-ONLY statistics.

    Guard rails: ``transform`` raises if called before ``fit``; ``fit`` raises
    if called twice (prevents accidental refit on val/test). Persisted to .npz
    and restored without ever recomputing.
    """

    def __init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self._fitted = False

    def fit(self, feats: np.ndarray) -> "SpectralNormalizer":
        if self._fitted:
            raise RuntimeError(
                "SpectralNormalizer already fitted - refusing to refit "
                "(would leak val/test statistics)."
            )
        self.mu = feats.mean(axis=0).astype(np.float32)
        self.sigma = feats.std(axis=0).astype(np.float32)
        self._fitted = True
        return self

    def transform(self, feats: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SpectralNormalizer.transform called before fit().")
        return ((feats - self.mu) / (self.sigma + EPS)).astype(np.float32)

    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("nothing to save: normalizer not fitted.")
        np.savez(path, mu=self.mu, sigma=self.sigma)

    @classmethod
    def from_stats(cls, mu: np.ndarray, sigma: np.ndarray) -> "SpectralNormalizer":
        """Build from pre-computed (train-only) streaming statistics."""
        obj = cls()
        obj.mu = np.asarray(mu, np.float32)
        obj.sigma = np.asarray(sigma, np.float32)
        obj._fitted = True
        return obj

    @classmethod
    def load(cls, path: str) -> "SpectralNormalizer":
        d = np.load(path)
        obj = cls()
        obj.mu, obj.sigma = d["mu"], d["sigma"]
        obj._fitted = True
        return obj

    def hash(self) -> str:
        """Short content hash of (mu, sigma) for run provenance logging."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.mu).tobytes())
        h.update(np.ascontiguousarray(self.sigma).tobytes())
        return h.hexdigest()[:12]
