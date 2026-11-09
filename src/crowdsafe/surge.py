"""Surge replay: how much warning would the monitor give if a zone really filled up?

The Mall footage is an ordinary shopping day. Occupancy drifts, but nothing
surges, so it cannot show how the alerting behaves when it matters. This module
replays simulated surges through the same monitor and the same metrics, with
the density model's *real* counting errors laid on top:

* True occupancy = a baseline, plus either a surge (a ramp past the limit at a
  set fill rate) or a near miss (a ramp that stops short of the limit), plus
  the frame-to-frame fluctuation measured on the Mall ground truth.
* Estimated occupancy = true occupancy + counting errors resampled in blocks
  from the model's actual errors on Mall frames (bias included). The errors
  are used at the size measured on a zone holding about 17 people; in a denser
  zone they would probably be larger, so read the results as optimistic.

Simulated crowd dynamics, real perception error. Results are labelled as such.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alerts import AlertConfig, run_monitor
from .evaluate import episodes


@dataclass
class SurgeSpec:
    limit: float = 40.0  # people; e.g. a 20 m2 zone at 2 people/m2
    base_frac: float = 0.55  # normal occupancy as a share of the limit
    frames: int = 600  # 5 minutes at 2 fps
    rates: tuple = (0.1, 0.2, 0.4, 0.8)  # fill rates, people per frame
    peak: float = 1.25  # surges top out at this share of the limit
    hold: int = 40  # frames at the peak
    block: int = 50  # block length for resampling counting errors


@dataclass
class Scenario:
    true: np.ndarray
    est: np.ndarray
    rate: float | None  # fill rate for a surge, None for a near miss


def fluctuation_params(gt_series: np.ndarray, smooth: int = 40) -> tuple[float, float]:
    """AR(1) coefficient and std of occupancy around its slow drift."""
    kernel = np.ones(smooth) / smooth
    resid = (gt_series - np.convolve(gt_series, kernel, mode="same"))[smooth:-smooth]
    phi = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    return phi, float(resid.std())


def make_scenarios(
    n: int, rng: np.random.Generator, errors: np.ndarray,
    phi: float, sigma: float, spec: SurgeSpec | None = None,
) -> list[Scenario]:
    spec = spec or SurgeSpec()
    T, out = spec.limit, []
    for k in range(n):
        base = spec.base_frac * T * rng.uniform(0.9, 1.1)
        surge = k % 2 == 0
        rate = float(spec.rates[(k // 2) % len(spec.rates)])
        top = spec.peak * T if surge else rng.uniform(0.7, 0.9) * T
        t0 = int(rng.integers(150, 300))
        up = int(np.ceil((top - base) / rate))
        curve = np.full(spec.frames, base)
        ramp = base + rate * np.arange(up)
        down = top - (rate / 2) * np.arange(int(np.ceil((top - base) / (rate / 2))))
        shape = np.concatenate([ramp, np.full(spec.hold, top), down])[: spec.frames - t0]
        curve[t0 : t0 + len(shape)] = shape

        noise = np.zeros(spec.frames)
        scale = sigma * np.sqrt(1 - phi**2)
        for t in range(1, spec.frames):
            noise[t] = phi * noise[t - 1] + rng.normal(0, scale)
        true = np.round(np.clip(curve + noise, 0, None))

        err = np.concatenate([
            errors[s : s + spec.block]
            for s in rng.integers(0, len(errors) - spec.block, spec.frames // spec.block + 1)
        ])[: spec.frames]
        est = np.clip(true + err, 0, None)
        out.append(Scenario(true=true, est=est, rate=rate if surge else None))
    return out


def evaluate_surges(scenarios: list[Scenario], cfg: AlertConfig, limit: float,
                    min_len: int = 4, merge_gap: int = 6, window: int = 120) -> dict:
    """Lead time is scored on the first breach of each scenario (the surge arriving).

    Scenarios are labelled by what truly happened, not by what was intended: a
    near miss whose random fluctuation does carry it over the limit is a breach.
    False-alert rate keeps the Mall definition: alerts that never overlap a breach.
    """
    leads, missed, alerts_n, false_n, quiet, quiet_alarmed = [], 0, 0, 0, 0, 0
    for s in scenarios:
        alerts = episodes(run_monitor(s.est, limit, cfg) > 0)
        breaches = episodes(s.true >= limit, min_len, merge_gap)
        alerts_n += len(alerts)
        false_n += sum(not any(a < be and e > bs for bs, be in breaches) for a, e in alerts)
        if not breaches:
            quiet += 1
            quiet_alarmed += bool(alerts)
            continue
        bs, be = breaches[0]
        at_onset = [a for a, e in alerts if a <= bs < e]
        during = [a for a, e in alerts if bs < a < be]
        if at_onset:
            leads.append(min(bs - at_onset[0], window))
        elif during:
            leads.append(bs - during[0])
        else:
            missed += 1
    n = len(leads) + missed
    return {
        "surges": n,
        "missed": missed,
        "warned_early": round(sum(l > 0 for l in leads) / n, 3) if n else None,
        "median_lead_frames": float(np.median(leads)) if leads else None,
        "p10_lead_frames": float(np.percentile(leads, 10)) if leads else None,
        "alerts": alerts_n,
        "false_alert_rate": round(false_n / alerts_n, 3) if alerts_n else 0.0,
        "quiet_scenarios": quiet,
        "quiet_scenarios_alarmed": round(quiet_alarmed / quiet, 3) if quiet else None,
    }
