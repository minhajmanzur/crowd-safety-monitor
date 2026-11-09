"""Crowd density estimation: image in, people-per-cell density map out.

The network regresses a density map whose sum is the head count, so it keeps
working when people overlap and a detector would lose them. It never produces
boxes, crops, embeddings or anything else that could identify a person.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

STRIDE = 8  # density map is 1/8 of the input resolution
TARGET_SCALE = 100.0  # regression targets are scaled up for a healthier loss
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class DensityNet(nn.Module):
    """ResNet-18 front end (to stride 8) plus a dilated-conv density head."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.front = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)

        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=2, dilation=2), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

        self.back = nn.Sequential(*block(128, 128), *block(128, 64), *block(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.back(self.front(x))


def to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - _MEAN) / _STD).transpose(2, 0, 1))


def gt_density(points: np.ndarray, h: int, w: int) -> np.ndarray:
    """Ground-truth density at 1/STRIDE resolution. Sums to len(points).

    Each head becomes a Gaussian whose width grows towards the bottom of the
    frame (a cheap perspective prior: nearer heads are bigger). Each Gaussian
    is normalised inside the image so border heads still count as one person.
    """
    full = np.zeros((h, w), np.float32)
    for x, y in points:
        x = float(np.clip(x, 0, w - 1))
        y = float(np.clip(y, 0, h - 1))
        sigma = 3.0 + 9.0 * (y / h)
        r = int(3 * sigma)
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        gx = np.exp(-((np.arange(x0, x1) - x) ** 2) / (2 * sigma**2))
        gy = np.exp(-((np.arange(y0, y1) - y) ** 2) / (2 * sigma**2))
        g = np.outer(gy, gx)
        full[y0:y1, x0:x1] += (g / g.sum()).astype(np.float32)
    return full.reshape(h // STRIDE, STRIDE, w // STRIDE, STRIDE).sum(axis=(1, 3))


class DensityEstimator:
    """Inference wrapper: BGR frame -> non-negative density map (numpy)."""

    def __init__(self, weights: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DensityNet(pretrained=False)
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        frame_bgr = frame_bgr[: h - h % STRIDE, : w - w % STRIDE]
        x = to_tensor(frame_bgr).unsqueeze(0).to(self.device)
        out = self.model(x)
        return (out.float().clamp(min=0) / TARGET_SCALE)[0, 0].cpu().numpy()
