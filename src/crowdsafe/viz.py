"""Operator view: density heatmap, zone status, flow arrows, recommended actions."""
from __future__ import annotations

import cv2
import numpy as np

from .agent import Recommendation, ZoneSnapshot
from .alerts import Level
from .flow import ZoneFlow
from .zones import Zone

COLORS = {Level.NORMAL: (90, 200, 90), Level.WARNING: (0, 190, 255), Level.CRITICAL: (60, 60, 240)}  # BGR
PANEL_W = 330
FONT = cv2.FONT_HERSHEY_SIMPLEX


def pixelate(frame: np.ndarray, block: int = 10) -> np.ndarray:
    """Coarsen the frame so no face or outfit detail survives in saved output."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // block, h // block), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def render(
    frame: np.ndarray,
    density: np.ndarray,
    zones: list[Zone],
    snaps: list[ZoneSnapshot],
    flows: list[ZoneFlow],
    active: list[Recommendation],
    frame_idx: int,
    anonymize: bool = True,
) -> np.ndarray:
    h, w = frame.shape[:2]
    base = pixelate(frame) if anonymize else frame.copy()

    heat = cv2.resize(density, (w, h), interpolation=cv2.INTER_CUBIC).clip(min=0)
    heat = cv2.GaussianBlur(heat, (0, 0), 6)
    norm = np.clip(heat / (heat.max() + 1e-9), 0, 1)
    colour = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (0.65 * norm**0.6)[..., None]
    img = (base * (1 - alpha) + colour * alpha).astype(np.uint8)

    for z, s, f in zip(zones, snaps, flows):
        col = COLORS[s.level]
        pts = np.array([[x * w, y * h] for x, y in z.polygon], np.int32)
        cv2.polylines(img, [pts], True, col, 3 if s.level else 1, cv2.LINE_AA)
        x0, y0 = pts[:, 0].min() + 8, pts[:, 1].min() + 20
        label = f"{z.name}  {s.count:.0f}/{s.threshold:.0f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.rectangle(img, (x0 - 5, y0 - th - 5), (x0 + tw + 5, y0 + 5), (28, 28, 28), -1)
        cv2.rectangle(img, (x0 - 5, y0 - th - 5), (x0 - 2, y0 + 5), col, -1)
        cv2.putText(img, label, (x0 + 2, y0), FONT, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        if f.direction != "still":
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            n = max(f.speed, 1e-6)
            tip = (int(cx + 45 * f.dx / n), int(cy + 45 * f.dy / n))
            cv2.arrowedLine(img, (cx, cy), tip, (0, 0, 0), 6, cv2.LINE_AA, tipLength=0.35)
            cv2.arrowedLine(img, (cx, cy), tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)

    panel = np.full((h, PANEL_W, 3), 28, np.uint8)
    y = 26
    cv2.putText(panel, "CROWD SAFETY MONITOR", (12, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(panel, f"frame {frame_idx}", (12, y), FONT, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    y += 14
    for s, f in zip(snaps, flows):
        col = COLORS[s.level]
        y += 22
        cv2.putText(panel, s.name, (12, y), FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(panel, s.level.name, (PANEL_W - 92, y), FONT, 0.45, col, 1, cv2.LINE_AA)
        y += 10
        bw = PANEL_W - 24
        cv2.rectangle(panel, (12, y), (12 + bw, y + 8), (70, 70, 70), -1)
        cv2.rectangle(panel, (12, y), (12 + int(bw * min(s.load / 1.3, 1.0)), y + 8), col, -1)
        lim = 12 + int(bw / 1.3)
        cv2.line(panel, (lim, y - 3), (lim, y + 11), (255, 255, 255), 1)
        y += 24
        trend = "rising" if s.slope > 0.05 else "falling" if s.slope < -0.05 else "steady"
        eta = f", limit in ~{s.eta:.0f}f" if s.eta is not None and s.eta < 60 else ""
        cv2.putText(panel, f"{s.count:.0f} of {s.threshold:.0f}  {trend}{eta}  flow {f.direction}",
                    (12, y), FONT, 0.36, (160, 160, 160), 1, cv2.LINE_AA)
        y += 6

    y += 26
    cv2.putText(panel, "RECOMMENDED ACTIONS", (12, y), FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    if not active:
        y += 22
        cv2.putText(panel, "none - all zones normal", (12, y), FONT, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
    for r in sorted(active, key=lambda r: r.priority)[:9]:
        y += 22
        if y > h - 30:
            break
        cv2.putText(panel, r.text(), (12, y), FONT, 0.4, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(panel, "aggregate counts only - no face recognition", (12, h - 10), FONT, 0.34, (120, 120, 120), 1, cv2.LINE_AA)
    return np.hstack([img, panel])
