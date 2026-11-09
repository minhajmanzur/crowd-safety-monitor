"""Loader for the Mall crowd-counting dataset (Chen et al., BMVC 2012).

2,000 frames (640x480) from one fixed camera, with every head annotated as a
point. The standard protocol trains on frames 0-799 and tests on 800-1999.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

TRAIN = range(0, 800)
TEST = range(800, 2000)


@dataclass
class MallDataset:
    root: Path
    points: list[np.ndarray]  # per frame, (N, 2) array of head (x, y) in pixels
    counts: np.ndarray  # per frame total head count

    def __len__(self) -> int:
        return len(self.points)

    def frame_path(self, i: int) -> Path:
        return self.root / "frames" / f"seq_{i + 1:06d}.jpg"

    def frame(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.frame_path(i)))
        if img is None:
            raise FileNotFoundError(self.frame_path(i))
        return img


def load_mall(root: str | Path) -> MallDataset:
    root = Path(root)
    if not (root / "mall_gt.mat").exists():
        raise FileNotFoundError(
            f"Mall dataset not found at {root}. Run: python scripts/download_mall.py"
        )
    gt = loadmat(str(root / "mall_gt.mat"))
    points = [
        np.asarray(f["loc"][0, 0], dtype=np.float32).reshape(-1, 2)
        for f in gt["frame"][0]
    ]
    counts = np.asarray(gt["count"], dtype=np.float32).ravel()
    return MallDataset(root=root, points=points, counts=counts)
