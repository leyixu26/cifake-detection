"""Yin's from-scratch SimpleCNN inference wrapper.

Architecture and preprocessing copied verbatim from his cnn_baseline.ipynb so the
predict function is fully self-contained (no notebook scope dependencies).

* Input: 32x32 native resolution, no resize
* Normalize: CIFAR-10 stats
* FAKE=0 (ImageFolder alphabetical) -> P(FAKE) = softmax(logits)[:, 0]

Public API:
    predict_fake_probability(paths: list[str]) -> np.ndarray
"""
from __future__ import annotations
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import pathlib as _pl, os
CKPT = str(_pl.Path(__file__).resolve().parent / "best_cnn.pt")
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)
BATCH = 256


def _device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


# --- Architecture (copied from Yin's cnn_baseline.ipynb cell 11) ---
class ConvBlock(nn.Module):
    """2x (Conv 3x3 -> BN -> ReLU) then MaxPool 2x2."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return self.pool(x)


class SimpleCNN(nn.Module):
    """3 conv blocks (32 -> 64 -> 128) + GAP + Linear -> 2 logits."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.block1 = ConvBlock(3, 32)      # 32x32 -> 16x16
        self.block2 = ConvBlock(32, 64)     # 16x16 -> 8x8
        self.block3 = ConvBlock(64, 128)    # 8x8 -> 4x4
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


_EVAL_TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class _PathsDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return _EVAL_TFM(Image.open(self.paths[i]).convert("RGB"))


_MODEL = None
_DEV   = None


def _load_model():
    global _MODEL, _DEV
    if _MODEL is not None:
        return _MODEL, _DEV
    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"Yin's CNN checkpoint not found at {CKPT}. "
            f"See models/cnn_baseline_yin/README.md for how to obtain it."
        )
    _DEV = _device()
    model = SimpleCNN(num_classes=2)
    ckpt = torch.load(CKPT, map_location=_DEV)
    # Yin's notebook saves: {"state_dict": ..., "epoch": ..., "val_loss": ...}
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(_DEV).eval()
    _MODEL = model
    return _MODEL, _DEV


@torch.no_grad()
def predict_fake_probability(paths: List[str]) -> np.ndarray:
    """Return P(FAKE) ∈ [0, 1] for the given image paths, shape (N,) float32.

    See `models/README.md` for the harness-wide contract this satisfies.
    The model is loaded lazily on the first call and cached in a module-level
    global; subsequent calls reuse the loaded weights.
    """
    model, dev = _load_model()
    ds = _PathsDataset(paths)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    out = np.empty(len(paths), dtype=np.float32)
    i = 0
    for batch in dl:
        x = batch.to(dev, non_blocking=True)
        probs = F.softmax(model(x), dim=1)
        p_fake = probs[:, 0].cpu().numpy()       # FAKE=0 alphabetical
        out[i:i + len(p_fake)] = p_fake
        i += len(p_fake)
    return out


if __name__ == "__main__":
    import glob
    DATA = os.environ.get("CIFAKE_DATA",
        str(_pl.Path(__file__).resolve().parent.parent.parent / "data" / "cifake"))
    paths = sorted(glob.glob(f"{DATA}/test/REAL/*.jpg"))[:32] + \
            sorted(glob.glob(f"{DATA}/test/FAKE/*.jpg"))[:32]
    p = predict_fake_probability(paths)
    print(f"smoke: P(FAKE|REAL)={p[:32].mean():.3f}  P(FAKE|FAKE)={p[32:].mean():.3f}  "
          f"range=[{p.min():.3f}, {p.max():.3f}]")
