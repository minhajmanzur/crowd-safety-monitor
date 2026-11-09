"""Run the monitor over a frame sequence or video.

    python -m crowdsafe.run --config configs/mall.yaml --start 800 --frames 300 --out runs/demo.mp4
    python -m crowdsafe.run --config configs/mall.yaml --video my_camera.mp4 --out runs/out.mp4

Writes an annotated video (pixelated by default) and a JSONL event log that
contains only aggregate numbers and recommended actions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

from .agent import SafetyAgent, Site, snapshot, to_json
from .alerts import AlertConfig, ZoneMonitor
from .density import DensityEstimator
from .flow import FlowEstimator
from .mall import TRAIN, load_mall
from .viz import render
from .zones import derive_thresholds, zone_counts_from_points, zones_from_config


def frame_source(cfg: dict, video: str | None, start: int, n: int):
    if video:
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(n):
            ok, f = cap.read()
            if not ok:
                break
            yield start + i, f
    else:
        ds = load_mall(cfg["dataset"]["root"])
        for i in range(start, min(start + n, len(ds))):
            yield i, ds.frame(i)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mall.yaml")
    ap.add_argument("--video", help="video file instead of the Mall frames")
    ap.add_argument("--start", type=int, default=800)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default="runs/demo.mp4")
    ap.add_argument("--gif", help="also write an animated GIF here")
    ap.add_argument("--llm", action="store_true", help="refine recommendations with Claude")
    ap.add_argument("--no-anonymize", action="store_true", help="do not pixelate the output")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    zones = zones_from_config(cfg)
    if any(z.threshold is None for z in zones):
        ds = load_mall(cfg["dataset"]["root"])
        gt = zone_counts_from_points(zones, [ds.points[i] for i in TRAIN], 480, 640)
        derive_thresholds(zones, gt, cfg["threshold_percentile"])

    estimator = DensityEstimator(cfg["weights"])
    flow_est = FlowEstimator()
    acfg = AlertConfig(**cfg.get("alerts", {}))
    monitors = [ZoneMonitor(z.threshold, acfg) for z in zones]
    advisor = None
    if args.llm:
        from .llm_agent import LLMAdvisor

        advisor = LLMAdvisor()
    agent = SafetyAgent(Site.from_config(cfg), advisor)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = open(out.with_suffix(".jsonl"), "w")
    writer, gif_frames = None, []
    fps = float(cfg["dataset"].get("fps", 2.0))

    for idx, frame in frame_source(cfg, args.video, args.start, args.frames):
        density = estimator(frame)
        flow = flow_est.update(frame)
        states = [m.update(z.count_density(density)) for m, z in zip(monitors, zones)]
        flows = [flow_est.zone_flow(flow, density, z) for z in zones]
        snaps = [snapshot(z.name, z.threshold, s, f) for z, s, f in zip(zones, states, flows)]
        new = agent.step(snaps)

        log.write(json.dumps({
            "frame": idx,
            "zones": [{"name": s.name, "level": s.level.name, "count": round(s.count, 1),
                       "threshold": s.threshold, "flow": s.flow_direction} for s in snaps],
            "new_recommendations": [to_json(r) for r in new],
        }) + "\n")
        for r in new:
            print(f"[frame {idx}] {r.zone}: {r.text()} - {r.reason}")

        view = render(frame, density, zones, snaps, flows, agent.active, idx, anonymize=not args.no_anonymize)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps * 4, view.shape[1::-1])
        writer.write(view)
        if args.gif:
            gif_frames.append(cv2.cvtColor(cv2.resize(view, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))

    if writer:
        writer.release()
    log.close()
    if args.gif and gif_frames:
        import imageio.v2 as imageio

        imageio.mimsave(args.gif, gif_frames, duration=1 / (fps * 4), loop=0)
    print(f"wrote {out} and {out.with_suffix('.jsonl')}")


if __name__ == "__main__":
    main()
