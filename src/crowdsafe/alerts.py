"""Per-zone alerting with an early-warning stage.

The monitor smooths the zone count and fits a short trend. It raises WARNING
when the count, projected `horizon` frames along any upward trend, reaches
`warn_ratio` of the limit. That happens before the count itself crosses, and
the gap is the lead time staff get to act. CRITICAL means the smoothed count is
already over the limit.

Lead time is bought with false alerts: a lower `warn_ratio` or longer `horizon`
warns earlier and cries wolf more often. The benchmark measures that trade.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Level(IntEnum):
    NORMAL = 0
    WARNING = 1  # approaching the limit
    CRITICAL = 2  # over the threshold now


@dataclass
class AlertConfig:
    ema_alpha: float = 0.5  # smoothing of the raw count (1 = none)
    trend_window: int = 6  # frames used to fit the trend
    horizon: int = 5  # frames ahead to project an upward trend
    warn_ratio: float = 0.9  # WARNING once the projection reaches this share of the limit
    confirm: int = 2  # consecutive frames a condition must hold before alerting
    clear_ratio: float = 0.85  # alert clears once the count falls below this share of the limit
    predictive: bool = True  # False = reactive baseline (alert only once over)


@dataclass
class ZoneState:
    level: Level
    count: float  # smoothed occupancy
    slope: float  # people per frame
    forecast: float  # predicted occupancy at now + horizon
    eta: float | None  # frames until the trend reaches the threshold, if rising
    new_alert: bool  # True on the frame an alert episode starts
    escalated: bool  # True on the frame WARNING becomes CRITICAL


class ZoneMonitor:
    def __init__(self, threshold: float, cfg: AlertConfig | None = None):
        self.threshold = float(threshold)
        self.cfg = cfg or AlertConfig()
        self.level = Level.NORMAL
        self._ema: float | None = None
        self._hist: deque[float] = deque(maxlen=max(2, self.cfg.trend_window))
        self._warn_run = 0
        self._crit_run = 0

    def update(self, raw_count: float) -> ZoneState:
        c, T = self.cfg, self.threshold
        self._ema = raw_count if self._ema is None else c.ema_alpha * raw_count + (1 - c.ema_alpha) * self._ema
        self._hist.append(self._ema)

        slope = 0.0
        n = len(self._hist)
        if n >= 3:  # least-squares slope, closed form
            mid = (n - 1) / 2
            slope = sum((i - mid) * v for i, v in enumerate(self._hist)) / (n * (n * n - 1) / 12)
        forecast = self._ema + slope * c.horizon
        eta = (T - self._ema) / slope if slope > 1e-6 and self._ema < T else None

        over_now = self._ema >= T
        heading_over = c.predictive and self._ema + max(slope, 0.0) * c.horizon >= c.warn_ratio * T
        clear_at = (min(c.clear_ratio, c.warn_ratio - 0.05) if c.predictive else c.clear_ratio) * T
        self._crit_run = self._crit_run + 1 if over_now else 0
        self._warn_run = self._warn_run + 1 if (heading_over or over_now) else 0

        prev = self.level
        if self._crit_run >= c.confirm:
            self.level = Level.CRITICAL
        elif self.level == Level.NORMAL:
            if c.predictive and self._warn_run >= c.confirm:
                self.level = Level.WARNING
        elif self._ema < clear_at:
            # Hysteresis: an alert only stands down once the zone has clearly
            # emptied, so a count hovering at the limit does not flap.
            self.level = Level.WARNING if heading_over else Level.NORMAL

        return ZoneState(
            level=self.level,
            count=self._ema,
            slope=slope,
            forecast=forecast,
            eta=eta,
            new_alert=prev == Level.NORMAL and self.level != Level.NORMAL,
            escalated=prev == Level.WARNING and self.level == Level.CRITICAL,
        )


def run_monitor(counts: np.ndarray, threshold: float, cfg: AlertConfig) -> np.ndarray:
    """Alert level for every frame of a count series (offline convenience)."""
    m = ZoneMonitor(threshold, cfg)
    return np.array([int(m.update(float(c)).level) for c in counts])
