"""Models for the frequency-domain detector.

* ShallowMLP   - Variant A learned classifier on ~29-d handcrafted features.
* SpectrumCNN  - Variant B, ~270k params, matches the teammate small-CNN scale.

Both output a single logit (use BCEWithLogitsLoss).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ShallowMLP(nn.Module):
    def __init__(self, in_dim: int = 29, hidden=(64, 32), dropout: float = 0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _conv_block(cin, cout, pool: bool):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
              nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SpectrumCNN(nn.Module):
    """log-magnitude spectrum -> logit.

    in_ch=1 (luma) or 3 (rgb). 32x32 -> 32 -> 16 -> 8 -> GAP.
    ~0.22M params at the default widths (in the 150-400k target band,
    comparable to the teammate from-scratch small-CNN baseline).
    """

    def __init__(self, in_ch: int = 3, widths=(48, 96, 192), dropout: float = 0.3):
        super().__init__()
        w1, w2, w3 = widths
        self.features = nn.Sequential(
            _conv_block(in_ch, w1, pool=False),
            _conv_block(w1, w2, pool=True),   # 32 -> 16
            _conv_block(w2, w3, pool=True),   # 16 -> 8
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(w3, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":  # quick shape / param sanity
    for ch in (1, 3):
        m = SpectrumCNN(in_ch=ch)
        o = m(torch.randn(2, ch, 32, 32))
        print(f"SpectrumCNN in_ch={ch}: out {tuple(o.shape)}  params {count_params(m):,}")
    mlp = ShallowMLP(29)
    print(f"ShallowMLP: out {tuple(mlp(torch.randn(2, 29)).shape)}  params {count_params(mlp):,}")
