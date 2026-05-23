"""Alex's ViT-Small (vit_small_patch16_224, timm) inference wrapper.

Mirrors the preprocessing in train_vit.py:
* 32x32 -> 224x224 via BICUBIC resize
* Per-channel normalize with the stats Alex used (note: differs slightly from Nathan's)
* FAKE=0 in the model's softmax -> P(FAKE) = softmax[:, 0]

Public API:
    predict_fake_probability(paths: list[str]) -> np.ndarray  (shape (N,), in [0,1])
"""
from __future__ import annotations
from typing import List
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import timm

import pathlib as _pl
CKPT = str(_pl.Path(__file__).resolve().parent / "best_vit.pt")
# IMPORTANT: Alex's exact CIFAR std (NOT the more common [0.247, 0.2435, 0.2616]).
MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]
BATCH = 64   # ViT-Small @ 224 is heavier than ResNet-18; smaller batch is safer on MPS


def _device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


_EVAL_TFM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class _PathsDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return _EVAL_TFM(img)


_MODEL = None
_DEV   = None


def _load_model():
    global _MODEL, _DEV
    if _MODEL is not None:
        return _MODEL, _DEV
    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"Alex's ViT-Small checkpoint not found at {CKPT}. "
            f"Pull it via `git lfs pull` (it is tracked by Git LFS)."
        )
    _DEV = _device()
    # Same architecture Alex trained: timm vit_small_patch16_224 with 2-class head.
    model = timm.create_model("vit_small_patch16_224", pretrained=False)
    model.head = nn.Linear(in_features=model.head.in_features, out_features=2)
    state = torch.load(CKPT, map_location=_DEV)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(_DEV).eval()
    _MODEL = model
    return _MODEL, _DEV


@torch.no_grad()
def predict_fake_probability(paths: List[str]) -> np.ndarray:
    model, dev = _load_model()
    ds = _PathsDataset(paths)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    out = np.empty(len(paths), dtype=np.float32)
    i = 0
    for batch in dl:
        x = batch.to(dev, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        p_fake = probs[:, 0].cpu().numpy()
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
    real_mean = p[:32].mean(); fake_mean = p[32:].mean()
    print(f"smoke: P(FAKE|REAL)={real_mean:.3f}  P(FAKE|FAKE)={fake_mean:.3f}  "
          f"(want low/high)  range=[{p.min():.3f}, {p.max():.3f}]")
