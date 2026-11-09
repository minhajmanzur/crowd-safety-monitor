"""Crowd flow direction per zone from dense optical flow.

Motion is measured on pixels, weighted by where the density map says people
are. Nobody is tracked from frame to frame, so there are no trajectories or
identities to store.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .zones import Zone

FLOW_SIZE = (320, 240)  # (w, h) working resolution
DIRECTIONS = ["right", "down", "left", "up"]  # image coordinates, y grows downward


@dataclass
class ZoneFlow:
    dx: float  # mean motion, pixels/frame at FLOW_SIZE
    dy: float
    speed: float  # magnitude of the mean motion vector
    coherence: float  # 0 = people moving every which way, 1 = one shared direction
    direction: str  # "right" | "down" | "left" | "up" | "still"


def direction_label(dx: float, dy: float, min_speed: float) -> str:
    if float(np.hypot(dx, dy)) < min_speed:
        return "still"
    angle = np.degrees(np.arctan2(dy, dx)) % 360
    return DIRECTIONS[int(((angle + 45) % 360) // 90)]


class FlowEstimator:
    def __init__(self, min_speed: float = 0.4, min_pixel_motion: float = 0.5):
        self.prev: np.ndarray | None = None
        self.min_speed = min_speed
        self.min_pixel_motion = min_pixel_motion

    def update(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Returns flow (h, w, 2) versus the previous frame, or None on frame one."""
        grey = cv2.cvtColor(cv2.resize(frame_bgr, FLOW_SIZE), cv2.COLOR_BGR2GRAY)
        flow = None
        if self.prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev, grey, None, pyr_scale=0.5, levels=4, winsize=21,
                iterations=3, poly_n=7, poly_sigma=1.5, flags=0,
            )
        self.prev = grey
        return flow

    def zone_flow(self, flow: np.ndarray | None, density: np.ndarray, zone: Zone) -> ZoneFlow:
        if flow is None:
            return ZoneFlow(0.0, 0.0, 0.0, 0.0, "still")
        h, w = flow.shape[:2]
        mag = np.linalg.norm(flow, axis=2)
        weight = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR).clip(min=0)
        weight = weight * zone.mask(h, w) * (mag > self.min_pixel_motion)
        total = float(weight.sum())
        if total < 1e-6:
            return ZoneFlow(0.0, 0.0, 0.0, 0.0, "still")
        dx = float((flow[..., 0] * weight).sum() / total)
        dy = float((flow[..., 1] * weight).sum() / total)
        speed = float(np.hypot(dx, dy))
        mean_mag = float((mag * weight).sum() / total)
        coherence = speed / mean_mag if mean_mag > 1e-6 else 0.0
        return ZoneFlow(dx, dy, speed, min(coherence, 1.0), direction_label(dx, dy, self.min_speed))
