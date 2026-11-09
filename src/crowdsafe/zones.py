"""Camera zones: polygons in normalised image coordinates.

A zone turns a density map (or a set of head points) into one number: how many
people are in it. No identities are involved at any stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from matplotlib.path import Path as MplPath


@dataclass
class Zone:
    name: str
    polygon: list[list[float]]  # [[x, y], ...] in 0..1 image coordinates
    threshold: float | None = None  # people; the safe-occupancy limit
    area_m2: float | None = None  # floor area, if the camera is calibrated
    scale: float = 1.0  # count calibration: corrects the model's systematic over/undercount here
    _mask_cache: dict = field(default_factory=dict, repr=False)

    def mask(self, h: int, w: int) -> np.ndarray:
        """Fractional coverage mask at (h, w): 1 inside, 0 outside, soft edges."""
        key = (h, w)
        if key not in self._mask_cache:
            # Supersample so coarse density cells get fractional weights. Testing
            # sub-pixel centres (rather than filling the polygon) means zones
            # that share an edge never both claim the same sliver of floor.
            s = 4
            ys, xs = np.mgrid[0 : h * s, 0 : w * s]
            centres = np.column_stack([(xs.ravel() + 0.5) / (w * s), (ys.ravel() + 0.5) / (h * s)])
            inside = MplPath(np.asarray(self.polygon, float)).contains_points(centres)
            self._mask_cache[key] = (
                inside.reshape(h, s, w, s).mean(axis=(1, 3)).astype(np.float32)
            )
        return self._mask_cache[key]

    def count_density(self, density: np.ndarray) -> float:
        """People in the zone according to a density map (any resolution)."""
        return self.scale * float((density * self.mask(*density.shape)).sum())

    def count_points(self, points: np.ndarray, h: int, w: int) -> int:
        """People in the zone according to ground-truth head points."""
        if len(points) == 0:
            return 0
        poly = np.array([[x * w, y * h] for x, y in self.polygon], np.float32)
        return int(
            sum(
                cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                for x, y in points
            )
        )

    def density_per_m2(self, count: float) -> float | None:
        return count / self.area_m2 if self.area_m2 else None


def zones_from_config(cfg: dict) -> list[Zone]:
    return [
        Zone(
            name=z["name"],
            polygon=z["polygon"],
            threshold=z.get("threshold"),
            area_m2=z.get("area_m2"),
            scale=float(z.get("scale", 1.0)),
        )
        for z in cfg["zones"]
    ]


def zone_counts_from_points(zones: list[Zone], points: list[np.ndarray], h: int, w: int) -> np.ndarray:
    """(n_frames, n_zones) ground-truth occupancy."""
    return np.array(
        [[z.count_points(p, h, w) for z in zones] for p in points], dtype=np.float32
    )


def derive_thresholds(zones: list[Zone], gt_counts: np.ndarray, percentile: float) -> None:
    """Fill in missing thresholds from a percentile of ground-truth occupancy.

    Benchmark-only convenience: a real site sets thresholds from floor area and
    a safe density (for example 2 people per square metre for moving crowds).
    """
    for j, z in enumerate(zones):
        if z.threshold is None:
            z.threshold = float(np.ceil(np.percentile(gt_counts[:, j], percentile)))
