"""Figures for the README, drawn from results/benchmark.json and results/traces.npz.

    python -m crowdsafe.plots
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import episodes

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WARNING, CRITICAL = "#fab219", "#d03b3b"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def timeline(res: dict, tr, out: Path, span: int = 420) -> None:
    gt, pred, thr, lv = tr["gt_test"], tr["pred_test"], tr["thresholds"], tr["levels_tuned"]
    names = list(res["protocol"]["thresholds"])
    ev = res["protocol"]["evaluation"]
    # Pick the zone and window with the most breaches so the figure shows the mechanism.
    best = (-1, 0, 0)
    for j in range(gt.shape[1]):
        br = episodes(gt[:, j] >= thr[j], ev["min_len"], ev["merge_gap"])
        for s0 in range(0, gt.shape[0] - span, 20):
            n = sum(1 for b, _ in br if s0 + 20 <= b < s0 + span - 5)
            if n > best[0]:
                best = (n, j, s0)
    _, j, s0 = best
    x = np.arange(s0, s0 + span) + 800
    g, p, l = gt[s0 : s0 + span, j], pred[s0 : s0 + span, j], lv[s0 : s0 + span, j]

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=160, facecolor=SURFACE)
    style(ax)
    top = max(g.max(), p.max()) + 2.5
    for a, b in episodes(l > 0):
        ax.axvspan(x[a] - 0.5, x[b - 1] + 0.5, color=WARNING, alpha=0.28, linewidth=0)
    for a, b in episodes(g >= thr[j], ev["min_len"], ev["merge_gap"]):
        ax.plot([x[a], x[b - 1]], [top - 0.6] * 2, color=CRITICAL, linewidth=4, solid_capstyle="butt")
    ax.plot(x, g, color=INK2, linewidth=1.2, label="Ground truth (annotated heads)")
    ax.plot(x, p, color=BLUE, linewidth=2, label="Estimated from density map")
    ax.axhline(thr[j], color=INK, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(x[0] + 2, thr[j] + 0.3, f"limit = {thr[j]:.0f} people", color=INK, fontsize=9, va="bottom")
    ax.plot([], [], color=WARNING, alpha=0.5, linewidth=8, label="Alert active")
    ax.plot([], [], color=CRITICAL, linewidth=4, label="Actual breach")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, top)
    ax.set_xlabel("frame (about 2 per second)", color=INK2, fontsize=9)
    ax.set_ylabel("people in zone", color=INK2, fontsize=9)
    ax.set_title(f"Zone '{names[j]}', held-out frames: estimated count, alerts and actual breaches",
                 color=INK, fontsize=11, loc="left", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def tradeoff(res: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160, facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    xs = [100 * s["test"]["false_alert_rate"] for s in res["sweep_test"]]
    ys = [s["test"]["mean_lead"] for s in res["sweep_test"]]
    ax.scatter(xs, ys, s=22, color=MUTED, alpha=0.45, linewidths=0, label="Other settings in the sweep")
    picks = [
        ("early warning (tuned on train)", BLUE, "Early warning (tuned on train)"),
        ("reactive", ORANGE, "Reactive baseline"),
    ]
    for key, col, label in picks:
        m = res["alerts_test"][key]
        ax.scatter([100 * m["false_alert_rate"]], [m["mean_lead"]], s=90, color=col,
                   edgecolors=SURFACE, linewidths=2, zorder=3, label=label)
        ax.annotate(label, (100 * m["false_alert_rate"], m["mean_lead"]), xytext=(9, 6),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("false alerts (% of all alerts)", color=INK2, fontsize=9)
    ax.set_ylabel("mean lead time before breach (frames)", color=INK2, fontsize=9)
    ax.set_title("Earlier warning costs false alerts (each dot is one alert setting, held-out frames)",
                 color=INK, fontsize=10.5, loc="left", pad=12)
    ax.text(0.99, 0.03, "above 0: warned before the breach.  below 0: alerted after it began.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    out = Path("results")
    res = json.loads((out / "benchmark.json").read_text())
    tr = np.load(out / "traces.npz")
    timeline(res, tr, out / "timeline.png")
    tradeoff(res, out / "tradeoff.png")
    print("wrote results/timeline.png and results/tradeoff.png")


if __name__ == "__main__":
    main()
