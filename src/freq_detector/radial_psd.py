"""Handcrafted spectral features for Variant A (Durall / Dzanic style).

Input is an fftshift-ed magnitude spectrum (DC centred), shape (H, W) for luma
or (C, H, W) for per-channel.  Power = magnitude ** 2.

Feature vector (luma, default, 32x32):
    radial PSD            : 16  (log power, integer radii 1..16, DC dropped)
    azimuthal PSD         :  8  (log power, 8 angular sectors)
    scalar summaries      :  5  (HF slope, HF energy frac, centroid, flatness, rolloff)
    --------------------------------------------------------------------------
    total                 : 29

The radius/angle index maps depend only on (H, W) so they are cached.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

EPS = 1e-8


@lru_cache(maxsize=8)
def _index_maps(h: int, w: int, n_radial: int, n_sectors: int):
    """Return (radial_bin, sector_bin) integer maps of shape (h, w).

    radial_bin in 0..n_radial (0 == DC ring, dropped later).
    sector_bin in 0..n_sectors-1.
    """
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    dy, dx = yy - cy, xx - cx
    r = np.sqrt(dy * dy + dx * dx)
    # Normalise so that Nyquist (min(cy, cx)) maps to n_radial.
    r_max = float(min(cy, cx))
    rb = np.clip(np.round(r / r_max * n_radial), 0, n_radial).astype(np.int64)
    theta = np.arctan2(dy, dx)  # [-pi, pi]
    sb = np.floor((theta + np.pi) / (2 * np.pi) * n_sectors).astype(np.int64)
    sb = np.clip(sb, 0, n_sectors - 1)
    return rb, sb


def _radial_psd(power: np.ndarray, n_radial: int) -> np.ndarray:
    """Azimuthally-averaged power per integer radius, radii 1..n_radial."""
    h, w = power.shape[-2:]
    rb, _ = _index_maps(h, w, n_radial, 1)
    flat_r = rb.ravel()
    out = []
    for plane in np.atleast_3d(power.reshape(-1, h, w)):
        p = plane.ravel()
        sums = np.bincount(flat_r, weights=p, minlength=n_radial + 1)
        cnts = np.bincount(flat_r, minlength=n_radial + 1)
        prof = sums / np.maximum(cnts, 1)
        out.append(prof[1:])  # drop bin 0 == DC ring
    return np.asarray(out)  # (n_planes, n_radial)


def _azimuthal_psd(power: np.ndarray, n_sectors: int) -> np.ndarray:
    """Mean power per angular sector (captures anisotropic grid artefacts)."""
    h, w = power.shape[-2:]
    _, sb = _index_maps(h, w, 1, n_sectors)
    flat_s = sb.ravel()
    out = []
    for plane in np.atleast_3d(power.reshape(-1, h, w)):
        p = plane.ravel()
        sums = np.bincount(flat_s, weights=p, minlength=n_sectors)
        cnts = np.bincount(flat_s, minlength=n_sectors)
        out.append(sums / np.maximum(cnts, 1))
    return np.asarray(out)  # (n_planes, n_sectors)


def _scalars(rpsd: np.ndarray) -> np.ndarray:
    """Five interpretable scalars from one radial profile (length R).

    1. HF slope        : slope of log(rpsd) vs log(radius) over the top half.
    2. HF energy frac   : energy in the top half / total.
    3. spectral centroid: sum(k * p) / sum(p), normalised to [0, 1].
    4. spectral flatness: geo-mean / arith-mean of the profile.
    5. roll-off radius  : normalised radius holding 85% of cumulative energy.
    """
    r = len(rpsd)
    k = np.arange(1, r + 1, dtype=np.float64)
    p = np.maximum(rpsd.astype(np.float64), EPS)
    half = r // 2

    logk, logp = np.log(k[half:]), np.log(p[half:])
    slope = np.polyfit(logk, logp, 1)[0] if len(logk) > 1 else 0.0

    total = p.sum()
    hf_frac = p[half:].sum() / total
    centroid = float((k * p).sum() / total) / r
    flatness = float(np.exp(np.log(p).mean()) / p.mean())
    cum = np.cumsum(p) / total
    rolloff = float(np.searchsorted(cum, 0.85) + 1) / r
    return np.array([slope, hf_frac, centroid, flatness, rolloff], dtype=np.float32)


def spectral_features(
    mag: np.ndarray,
    n_radial: int = 16,
    n_sectors: int = 8,
    log: bool = True,
    use_azimuthal: bool = True,
    use_scalars: bool = True,
) -> np.ndarray:
    """Magnitude spectrum -> 1-D handcrafted feature vector.

    Per-channel inputs are averaged into a luma-like profile after pooling so
    the vector length is independent of color mode (color is an ablation knob
    handled upstream by choosing the magnitude's plane count).
    """
    power = (mag.astype(np.float64)) ** 2
    rp = _radial_psd(power, n_radial).mean(axis=0)  # (n_radial,)
    parts = [np.log(rp + EPS) if log else rp]
    if use_azimuthal:
        ap = _azimuthal_psd(power, n_sectors).mean(axis=0)
        parts.append(np.log(ap + EPS) if log else ap)
    if use_scalars:
        parts.append(_scalars(rp))
    return np.concatenate(parts).astype(np.float32)


def feature_names(
    n_radial: int = 16,
    n_sectors: int = 8,
    use_azimuthal: bool = True,
    use_scalars: bool = True,
) -> list[str]:
    """Human-readable names aligned with ``spectral_features`` (for LR coefs)."""
    names = [f"radial_{i}" for i in range(1, n_radial + 1)]
    if use_azimuthal:
        names += [f"azim_{i}" for i in range(n_sectors)]
    if use_scalars:
        names += ["hf_slope", "hf_energy_frac", "centroid", "flatness", "rolloff85"]
    return names
