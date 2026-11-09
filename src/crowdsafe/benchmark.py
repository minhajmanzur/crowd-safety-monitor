"""Benchmark on the Mall dataset: counting accuracy, alert lead time, false alerts.

    python -m crowdsafe.benchmark --config configs/mall.yaml

Protocol
--------
* Density model: trained on frames 0-799, tested on 800-1999 (standard split).
* Thresholds: per zone, a percentile of ground-truth occupancy on frames 0-799.
* Alert settings: tuned on frames 0-799 using *out-of-fold* predictions (a model
  trained on 0-399 predicts 400-799 and vice versa), so tuning sees the same
  kind of counting error the final model makes on unseen frames.
* Everything reported is measured once on frames 800-1999.
* Surge replay (see surge.py): simulated surges with the model's real counting
  errors on top. Settings are tuned on one set of scenarios using out-of-fold
  errors and reported on a separate set using held-out test errors.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .alerts import AlertConfig, run_monitor
from .density import DensityEstimator
from .evaluate import AlertMetrics, as_dict, evaluate
from .mall import TEST, TRAIN, load_mall
from .surge import SurgeSpec, evaluate_surges, fluctuation_params, make_scenarios
from .zones import derive_thresholds, zone_counts_from_points, zones_from_config

GRID = {
    "ema_alpha": [0.3, 0.5, 0.8, 1.0],
    "warn_ratio": [0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    "horizon": [0, 3, 6, 10],
    "confirm": [1, 2, 3],
}


def predict(ds, zones, weights: str, frames) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (zone counts, total counts, frames per second)."""
    est = DensityEstimator(weights)
    zc, tot = [], []
    est(ds.frame(frames[0]))  # warm-up
    t = 0.0
    for i in frames:
        f = ds.frame(i)
        t0 = time.perf_counter()
        d = est(f)
        t += time.perf_counter() - t0
        zc.append([z.count_density(d) for z in zones])
        tot.append(d.sum())
    return np.array(zc, np.float32), np.array(tot, np.float32), len(frames) / t


def levels_for(counts: np.ndarray, thresholds: list[float], cfg: AlertConfig) -> np.ndarray:
    return np.stack([run_monitor(counts[:, j], thresholds[j], cfg) for j in range(counts.shape[1])], axis=1)


def selection_score(m: AlertMetrics, max_far: float) -> tuple[float, float, float]:
    """Earliest warning within the false-alert budget (the way a site would set it)."""
    within = m.false_alert_rate <= max_far
    return (float(within), m.mean_lead if within else -m.false_alert_rate, m.early_rate)


def surge_benchmark(gt_train, gt_test, pred_train, pred_test, ev: dict, max_far: float, fps: float) -> dict:
    """Zone 0 (the busiest) supplies the fluctuation statistics and the counting errors."""
    spec = SurgeSpec()
    kw = {"min_len": ev["min_len"], "merge_gap": ev["merge_gap"], "window": 120}
    phi, sigma = fluctuation_params(gt_train[:, 0])
    tune = make_scenarios(160, np.random.default_rng(1), pred_train[:, 0] - gt_train[:, 0], phi, sigma, spec)
    test = make_scenarios(400, np.random.default_rng(2), pred_test[:, 0] - gt_test[:, 0], phi, sigma, spec)

    def pick(a: AlertConfig):
        m = evaluate_surges(tune, a, spec.limit, **kw)
        ok = m["false_alert_rate"] <= max_far and m["missed"] <= 0.02 * m["surges"]
        return (ok, m["median_lead_frames"] if ok else -m["false_alert_rate"] - m["missed"])

    grid = [AlertConfig(**dict(zip(GRID, v))) for v in itertools.product(*GRID.values())]
    best = max(grid, key=pick)
    reactive = AlertConfig(ema_alpha=1.0, confirm=1, clear_ratio=1.0, predictive=False)
    systems = (("reactive", reactive), ("early_warning", best))
    out = {
        "spec": {"limit_people": spec.limit, "normal_occupancy": spec.base_frac, "surge_peak": spec.peak,
                 "near_miss_peak": "0.70-0.90 of limit", "scenarios_test": len(test), "frames_each": spec.frames,
                 "lead_cap_frames": kw["window"], "fluctuation_ar1": round(phi, 2),
                 "fluctuation_std_people": round(sigma, 2)},
        "tuned_alert_config": {k: getattr(best, k) for k in GRID},
        "overall": {name: evaluate_surges(test, a, spec.limit, **kw) for name, a in systems},
        "by_fill_rate": [],
    }
    for r in spec.rates:
        row = {"people_per_second": round(r * fps, 2), "people_per_frame": r}
        for name, a in systems:
            m = evaluate_surges([s for s in test if s.rate == r], a, spec.limit, **kw)
            row[name] = {k: m[k] for k in ("surges", "missed", "warned_early", "median_lead_frames", "p10_lead_frames")}
            row[name]["median_lead_seconds"] = None if m["median_lead_frames"] is None else round(m["median_lead_frames"] / fps, 1)
        out["by_fill_rate"].append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mall.yaml")
    ap.add_argument("--fold-a", default="runs/fold_a.pt", help="model trained on frames 0-399")
    ap.add_argument("--fold-b", default="runs/fold_b.pt", help="model trained on frames 400-799")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = load_mall(cfg["dataset"]["root"])
    zones = zones_from_config(cfg)
    for z in zones:
        z.scale = 1.0  # calibration is re-derived below from training data only
    names = [z.name for z in zones]
    ev = dict(cfg["evaluation"])
    max_far = ev.pop("max_false_alert_rate")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt = zone_counts_from_points(zones, ds.points, 480, 640)
    gt_train, gt_test = gt[list(TRAIN)], gt[list(TEST)]
    derive_thresholds(zones, gt_train, cfg["threshold_percentile"])
    thr = [z.threshold for z in zones]

    print("predicting test frames with the final model...")
    pred_test, tot_test, fps = predict(ds, zones, cfg["weights"], list(TEST))
    print("predicting training frames out-of-fold...")
    oof_lo, _, _ = predict(ds, zones, args.fold_b, list(range(0, 400)))
    oof_hi, _, _ = predict(ds, zones, args.fold_a, list(range(400, 800)))
    pred_train = np.concatenate([oof_lo, oof_hi])

    # Per-zone count calibration, fitted on out-of-fold training predictions:
    # the density model undercounts a little, and a late count means a late alert.
    scale = gt_train.sum(0) / pred_train.sum(0)
    raw_zone_mae = np.abs(pred_test - gt_test).mean(0)
    pred_train, pred_test = pred_train * scale, pred_test * scale

    # ---- counting accuracy -------------------------------------------------
    err = tot_test - ds.counts[list(TEST)]
    mean_baseline = np.abs(ds.counts[list(TRAIN)].mean() - ds.counts[list(TEST)]).mean()
    counting = {
        "test_frames": len(err),
        "mae": round(float(np.abs(err).mean()), 2),
        "rmse": round(float(np.sqrt((err**2).mean())), 2),
        "mean_gt_count": round(float(ds.counts[list(TEST)].mean()), 2),
        "mae_predict_train_mean": round(float(mean_baseline), 2),
        "zone_scale": {n: round(float(scale[j]), 3) for j, n in enumerate(names)},
        "zone_mae_uncalibrated": {n: round(float(raw_zone_mae[j]), 2) for j, n in enumerate(names)},
        "zone_bias": {n: round(float((pred_test[:, j] - gt_test[:, j]).mean()), 2) for j, n in enumerate(names)},
        "zone_mae": {n: round(float(np.abs(pred_test[:, j] - gt_test[:, j]).mean()), 2) for j, n in enumerate(names)},
        "zone_mae_out_of_fold_train": {n: round(float(np.abs(pred_train[:, j] - gt_train[:, j]).mean()), 2) for j, n in enumerate(names)},
        "inference_fps": round(fps, 1),
    }

    # ---- tune alert settings on out-of-fold training predictions -----------
    print("tuning alert settings on the training split...")
    sweep = []
    for vals in itertools.product(*GRID.values()):
        a = AlertConfig(**dict(zip(GRID, vals)))
        m_tr = evaluate(gt_train, levels_for(pred_train, thr, a), thr, **ev)
        m_te = evaluate(gt_test, levels_for(pred_test, thr, a), thr, **ev)
        sweep.append((a, m_tr, m_te))
    best_cfg, best_train, _ = max(sweep, key=lambda s: selection_score(s[1], max_far))

    # ---- final numbers on the test split -----------------------------------
    default_cfg = AlertConfig(ema_alpha=0.5, horizon=5, warn_ratio=0.9, confirm=2)  # an "earlier, noisier" operating point
    reactive = AlertConfig(ema_alpha=1.0, confirm=1, clear_ratio=1.0, predictive=False)
    systems = {
        "reactive": (pred_test, reactive),
        "early warning (eager: smoothed, 5-frame trend)": (pred_test, default_cfg),
        "early warning (tuned on train)": (pred_test, best_cfg),
        "early warning (tuned), perfect counts": (gt_test, best_cfg),
    }
    alert_results, level_traces = {}, {}
    for name, (counts, a) in systems.items():
        lv = levels_for(counts, thr, a)
        level_traces[name] = lv
        alert_results[name] = as_dict(evaluate(gt_test, lv, thr, **ev))

    result = {
        "protocol": {
            "train_frames": "0-799", "test_frames": "800-1999",
            "threshold_percentile": cfg["threshold_percentile"],
            "thresholds": dict(zip(names, thr)),
            "evaluation": ev, "max_false_alert_rate_for_tuning": max_far, "nominal_fps": cfg["dataset"]["fps"],
        },
        "counting": counting,
        "tuned_alert_config": {k: getattr(best_cfg, k) for k in GRID},
        "tuned_alert_config_train_metrics": as_dict(best_train),
        "alerts_test": alert_results,
        "surge_replay": surge_benchmark(gt_train, gt_test, pred_train, pred_test, ev, max_far, cfg["dataset"]["fps"]),
        "sweep_test": [
            {**{k: getattr(a, k) for k in GRID}, "train": as_dict(mt), "test": as_dict(me)} for a, mt, me in sweep
        ],
    }
    (out / "benchmark.json").write_text(json.dumps(result, indent=2))
    np.savez(
        out / "traces.npz", gt_test=gt_test, pred_test=pred_test, thresholds=np.array(thr),
        levels_tuned=level_traces["early warning (tuned on train)"],
        levels_reactive=level_traces["reactive"],
    )

    print(json.dumps({k: v for k, v in result.items() if k != "sweep_test"}, indent=2))


if __name__ == "__main__":
    main()
