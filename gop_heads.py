"""CTC head architectures shared by training / eval / scorer pipelines.

'linear' is the released default (41K params, 162 KB).
'cnn' adds a depthwise-separable temporal trunk (frame-level denoising)
before the phone softmax; no normalization layers, so padded frames cannot
leak across batches through BN stats.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DWCNNHead(nn.Module):
    """[T,B,D] -> pointwise/depthwise(k=7)x2 -> per-frame logits [T,B,V+1]."""

    def __init__(self, dim: int, nout: int, ch: int = 128):
        super().__init__()
        self.stem = nn.Conv1d(dim, ch, 1)
        self.pw1 = nn.Conv1d(ch, ch, 1)
        self.dw1 = nn.Conv1d(ch, ch, 7, padding=3, groups=ch)
        self.pw2 = nn.Conv1d(ch, ch, 1)
        self.dw2 = nn.Conv1d(ch, ch, 7, padding=3, groups=ch)
        self.out = nn.Linear(ch, nout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.permute(1, 2, 0)                      # [T, B, D] -> [B, D, T]
        h = self.gelu(self.dw1(self.pw1(self.stem(h))))
        h = self.gelu(self.dw2(self.pw2(h)))        # [B, ch, T]
        h = h.permute(2, 0, 1)                      # [T, B, ch]
        return self.out(h)                          # [T, B, nout]


def build_head(kind: str, dim: int, nout: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(dim, nout)
    if kind == "cnn":
        return DWCNNHead(dim, nout)
    raise ValueError(f"unknown head kind: {kind}")


def head_kind_from_state(sd: dict) -> str:
    return "cnn" if "stem.weight" in sd else "linear"


def load_head(path, V: int, device: str) -> nn.Module:
    """Rebuild + load a saved head; architecture and input dim auto-detected."""
    sd = torch.load(path, map_location=device)
    kind = head_kind_from_state(sd)
    key = "stem.weight" if kind == "cnn" else "weight"
    model = build_head(kind, sd[key].shape[1], V + 1)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model
