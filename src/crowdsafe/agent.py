"""The action-recommending agent.

It reads the zone snapshot (occupancy, trend, flow, alert level) plus the site
layout (gates, staff, neighbouring zones) and proposes concrete actions for a
human operator. It recommends; it never actuates anything.

Two layers:

* `RulePolicy` is deterministic and always runs. It is the safety floor.
* `LLMAdvisor` (llm_agent.py, optional) asks Claude to re-prioritise and
  explain. Its output is validated against the site layout, and it can add to
  the rule-based actions but cannot remove them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .alerts import Level, ZoneState
from .flow import ZoneFlow

ACTIONS = (
    "open_gate",  # target: gate id
    "dispatch_staff",  # target: zone, amount: number of staff
    "hold_inflow",  # target: upstream zone to meter or pause entry from
    "redirect_flow",  # target: zone to steer people towards
    "pa_announcement",  # target: zone
    "escalate",  # target: "control_room"
)

OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


@dataclass
class ZoneSnapshot:
    name: str
    level: Level
    count: float
    threshold: float
    slope: float
    eta: float | None
    flow_direction: str
    flow_coherence: float

    @property
    def load(self) -> float:
        return self.count / self.threshold if self.threshold else 0.0


@dataclass
class Recommendation:
    action: str
    zone: str  # the zone the action is meant to relieve
    target: str
    priority: int  # 1 = do first
    reason: str
    amount: int = 0
    source: str = "rules"

    def text(self) -> str:
        verb = {
            "open_gate": f"Open gate {self.target}",
            "dispatch_staff": f"Send {self.amount} staff to {self.target}",
            "hold_inflow": f"Hold inflow from {self.target}",
            "redirect_flow": f"Redirect people towards {self.target}",
            "pa_announcement": f"PA announcement in {self.target}",
            "escalate": "Escalate to control room supervisor",
        }[self.action]
        return f"P{self.priority} {verb}"


@dataclass
class Site:
    staff_available: int
    zones: dict  # name -> {"gates": [...], "neighbors": {direction: zone}}
    staff_committed: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict) -> "Site":
        s = cfg["site"]
        return cls(staff_available=int(s["staff_available"]), zones=s["zones"])

    def closed_gates(self, zone: str) -> list[dict]:
        return [g for g in self.zones.get(zone, {}).get("gates", []) if g["state"] == "closed"]

    def gate_ids(self) -> set[str]:
        return {g["id"] for z in self.zones.values() for g in z.get("gates", [])}

    def neighbors(self, zone: str) -> dict:
        return self.zones.get(zone, {}).get("neighbors", {})

    def staff_free(self) -> int:
        return self.staff_available - sum(self.staff_committed.values())


def snapshot(name: str, threshold: float, state: ZoneState, flow: ZoneFlow) -> ZoneSnapshot:
    return ZoneSnapshot(
        name=name, level=state.level, count=state.count, threshold=threshold,
        slope=state.slope, eta=state.eta,
        flow_direction=flow.direction, flow_coherence=flow.coherence,
    )


def feeders(zone: ZoneSnapshot, all_zones: dict[str, ZoneSnapshot], site: Site) -> list[str]:
    """Neighbouring zones whose crowd is currently moving into `zone`."""
    out = []
    for direction, nb in site.neighbors(zone.name).items():
        nb_snap = all_zones.get(nb)
        if nb_snap and nb_snap.flow_direction == OPPOSITE[direction]:
            out.append(nb)
    return out


class RulePolicy:
    """Deterministic playbook. One set of actions per alert stage per zone."""

    def recommend(self, zones: list[ZoneSnapshot], site: Site) -> list[Recommendation]:
        by_name = {z.name: z for z in zones}
        recs: list[Recommendation] = []
        for z in sorted(zones, key=lambda z: (-int(z.level), -z.load)):
            if z.level == Level.NORMAL:
                continue
            critical = z.level == Level.CRITICAL
            pct = f"{100 * z.load:.0f}% of limit"
            eta = f", about {z.eta:.0f} frames to limit" if z.eta is not None else ""
            p = 1

            for g in site.closed_gates(z.name)[: None if critical else 1]:
                recs.append(Recommendation(
                    "open_gate", z.name, g["id"], p,
                    f"{z.name} at {pct}{eta}; {g['name']} is closed and adds exit capacity."))
            p += 1

            for src in feeders(z, by_name, site):
                recs.append(Recommendation(
                    "hold_inflow", z.name, src, p,
                    f"Crowd in {src} is moving towards {z.name}; meter entry until it clears."))

            want = (2 if critical else 1) - site.staff_committed.get(z.name, 0)
            n = max(0, min(want, site.staff_free()))
            if n:
                recs.append(Recommendation(
                    "dispatch_staff", z.name, z.name, p,
                    f"{z.name} is {'over its limit' if critical else 'approaching its limit'}; "
                    "staff on the ground to guide people and confirm the camera reading.",
                    amount=n))
            p += 1

            quiet = [
                nb for nb in site.neighbors(z.name).values()
                if nb in by_name and by_name[nb].level == Level.NORMAL and by_name[nb].load < 0.7
            ]
            if quiet and (critical or z.flow_coherence < 0.3):
                recs.append(Recommendation(
                    "redirect_flow", z.name, quiet[0], p,
                    f"{quiet[0]} has spare room ({100 * by_name[quiet[0]].load:.0f}% of limit)."))

            if critical:
                recs.append(Recommendation(
                    "pa_announcement", z.name, z.name, p,
                    "Ask people to keep moving and use the other exits."))
                if z.load >= 1.2 or site.staff_free() - n <= 0:
                    recs.append(Recommendation(
                        "escalate", z.name, "control_room", 1,
                        f"{z.name} at {pct}" + ("" if z.load >= 1.2 else " with no staff left to send") + "."))
        return recs


def validate(rec: Recommendation, site: Site, zone_names: set[str]) -> bool:
    """Reject anything that does not exist on this site. Applied to LLM output."""
    if rec.action not in ACTIONS or rec.zone not in zone_names:
        return False
    if rec.action == "open_gate":
        return rec.target in site.gate_ids()
    if rec.action == "escalate":
        return rec.target == "control_room"
    if rec.action == "dispatch_staff":
        return rec.target in zone_names and 1 <= rec.amount <= site.staff_available
    return rec.target in zone_names


class SafetyAgent:
    """Calls the policy when a zone's alert stage changes, not on every frame."""

    def __init__(self, site: Site, advisor=None):
        self.site = site
        self.rules = RulePolicy()
        self.advisor = advisor  # optional LLMAdvisor
        self._stage: dict[str, Level] = {}
        self.active: list[Recommendation] = []

    def step(self, zones: list[ZoneSnapshot]) -> list[Recommendation]:
        """Returns newly issued recommendations (empty on most frames)."""
        changed = [z for z in zones if z.level > self._stage.get(z.name, Level.NORMAL)]
        cleared = [z for z in zones if z.level == Level.NORMAL and self._stage.get(z.name, Level.NORMAL) != Level.NORMAL]
        for z in zones:
            self._stage[z.name] = z.level
        for z in cleared:
            self.site.staff_committed.pop(z.name, None)
            self.active = [r for r in self.active if r.zone != z.name]
        if not changed:
            return []

        names = {z.name for z in changed}
        recs = [r for r in self.rules.recommend(zones, self.site) if r.zone in names]
        if self.advisor is not None:
            recs = self.advisor.refine(zones, self.site, recs)

        # Staff requests are always for the shortfall, so they are always new.
        seen = {(r.action, r.zone, r.target) for r in self.active if r.action != "dispatch_staff"}
        new = [r for r in recs if (r.action, r.zone, r.target) not in seen]
        for r in new:
            if r.action == "dispatch_staff":
                self.site.staff_committed[r.zone] = self.site.staff_committed.get(r.zone, 0) + r.amount
        self.active += new
        return new


def to_json(rec: Recommendation) -> dict:
    return asdict(rec)
