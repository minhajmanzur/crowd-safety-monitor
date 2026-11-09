"""Alert metrics: lead time before a threshold breach, and false-alert rate.

Definitions (all in frames; divide by the camera's frame rate for seconds):

* Breach: a run of frames where the ground-truth zone count is at or over the
  threshold, lasting at least `min_len` frames. Runs separated by `merge_gap`
  frames or fewer count as one breach, so a count wobbling around the limit is
  one event rather than many.
* Alert: a run of frames where the monitor's level is above NORMAL.
* Lead time of a breach: if an alert is active on the frame the breach starts,
  lead = breach start - that alert's start (capped at `window`). The alert has
  to be continuously on up to the breach, so an unrelated earlier blip earns
  nothing. If no alert is active at the start, lead is negative: minus the
  delay until the first alert during the breach. No alert at all = missed.
* False alert: an alert that never overlaps a breach.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def episodes(active: np.ndarray, min_len: int = 1, merge_gap: int = 0) -> list[tuple[int, int]]:
    """Maximal runs of True as (start, end_exclusive)."""
    runs: list[list[int]] = []
    start = None
    for i, a in enumerate(list(active) + [False]):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if runs and start - runs[-1][1] <= merge_gap:
                runs[-1][1] = i
            else:
                runs.append([start, i])
            start = None
    return [(s, e) for s, e in runs if e - s >= min_len]


@dataclass
class AlertMetrics:
    breaches: int
    detected: int
    detected_early: int
    missed: int
    alerts: int
    false_alerts: int
    recall: float
    early_rate: float
    false_alert_rate: float
    false_alerts_per_1k_frames: float
    mean_lead: float
    median_lead: float
    alert_time_fraction: float  # share of zone-frames spent in alert (guards against "always on")
    breach_time_fraction: float  # share of zone-frames actually over the threshold
    frames: int


def score(
    gt_counts: np.ndarray,
    levels: np.ndarray,
    threshold: float,
    window: int = 20,
    min_len: int = 2,
    merge_gap: int = 3,
) -> tuple[list[float], dict]:
    """Returns (lead time per detected breach, raw tallies) for one zone."""
    breaches = episodes(gt_counts >= threshold, min_len=min_len, merge_gap=merge_gap)
    alerts = episodes(levels > 0)
    leads: list[float] = []
    alert_true = [False] * len(alerts)
    missed = 0
    for bs, be in breaches:
        at_onset = [a_s for a_s, a_e in alerts if a_s <= bs < a_e]
        during = [a_s for a_s, a_e in alerts if bs < a_s < be]
        if at_onset:
            leads.append(float(min(bs - at_onset[0], window)))
        elif during:
            leads.append(float(bs - during[0]))
        else:
            missed += 1
    for k, (a_s, a_e) in enumerate(alerts):
        alert_true[k] = any(a_s < be and a_e > bs for bs, be in breaches)
    return leads, {
        "breaches": len(breaches),
        "missed": missed,
        "alerts": len(alerts),
        "false_alerts": alert_true.count(False),
        "alert_frames": int((levels > 0).sum()),
        "breach_frames": int((gt_counts >= threshold).sum()),
        "frames": len(gt_counts),
    }


def aggregate(per_zone: list[tuple[list[float], dict]]) -> AlertMetrics:
    leads = [x for l, _ in per_zone for x in l]
    t = {k: sum(d[k] for _, d in per_zone) for k in per_zone[0][1]}
    detected = t["breaches"] - t["missed"]
    return AlertMetrics(
        breaches=t["breaches"],
        detected=detected,
        detected_early=sum(1 for x in leads if x > 0),
        missed=t["missed"],
        alerts=t["alerts"],
        false_alerts=t["false_alerts"],
        recall=detected / t["breaches"] if t["breaches"] else float("nan"),
        early_rate=sum(1 for x in leads if x > 0) / t["breaches"] if t["breaches"] else float("nan"),
        false_alert_rate=t["false_alerts"] / t["alerts"] if t["alerts"] else 0.0,
        false_alerts_per_1k_frames=1000 * t["false_alerts"] / t["frames"],
        mean_lead=float(np.mean(leads)) if leads else float("nan"),
        median_lead=float(np.median(leads)) if leads else float("nan"),
        alert_time_fraction=t["alert_frames"] / t["frames"],
        breach_time_fraction=t["breach_frames"] / t["frames"],
        frames=t["frames"],
    )


def evaluate(gt: np.ndarray, levels: np.ndarray, thresholds: list[float], **kw) -> AlertMetrics:
    """gt and levels are (n_frames, n_zones)."""
    return aggregate([score(gt[:, j], levels[:, j], thresholds[j], **kw) for j in range(gt.shape[1])])


def as_dict(m: AlertMetrics) -> dict:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(m).items()}
