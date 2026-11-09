"""Optional Claude-backed advisor.

Claude sees only aggregate numbers (zone occupancy, trend, flow direction) and
the site layout. No images or per-person data are ever sent. It may re-order,
re-word and add actions, but:

* every action is validated against the site layout before it is shown;
* rule-based actions are never dropped (they are the safety floor);
* any API error, refusal or invalid output falls back to the rules unchanged.

Needs `pip install anthropic` and an ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from .agent import Recommendation, Site, ZoneSnapshot, to_json, validate

MODEL = "claude-opus-5-5"

SYSTEM = """You advise the control-room operator of a venue on crowd safety.

You receive a JSON snapshot: per-zone occupancy against its safe limit, the trend, \
the crowd's flow direction, the alert level, the site layout (gates, neighbouring \
zones, staff still free) and a baseline list of actions from a fixed rule playbook.

Return the action list the operator should see. You may re-prioritise the baseline \
actions, sharpen their reasons, and add actions the playbook missed, such as relieving \
a zone through a quieter neighbour. Use only gates, zones and staff numbers that appear \
in the snapshot. Keep each reason to one sentence that states the evidence, because the \
operator reads it under time pressure and decides whether to act. A human makes every \
decision; you only recommend. Counts come from a camera model and can be wrong, so when \
a reading is borderline prefer actions that are cheap to undo, like sending one staff \
member to check, over ones that disrupt the venue."""


class _Action(BaseModel):
    action: Literal["open_gate", "dispatch_staff", "hold_inflow", "redirect_flow", "pa_announcement", "escalate"]
    zone: str
    target: str
    priority: int
    amount: int
    reason: str


class _Plan(BaseModel):
    actions: list[_Action]


class LLMAdvisor:
    def __init__(self, client=None, model: str = MODEL):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.last_error: str | None = None

    def refine(self, zones: list[ZoneSnapshot], site: Site, baseline: list[Recommendation]) -> list[Recommendation]:
        payload = {
            "zones": [
                {
                    "name": z.name, "alert_level": z.level.name,
                    "occupancy": round(z.count, 1), "safe_limit": z.threshold,
                    "trend_people_per_frame": round(z.slope, 2),
                    "frames_until_limit": None if z.eta is None else round(z.eta),
                    "flow_direction": z.flow_direction,
                    "flow_coherence": round(z.flow_coherence, 2),
                }
                for z in zones
            ],
            "site": {"staff_free": site.staff_free(), "zones": site.zones},
            "baseline_actions": [to_json(r) for r in baseline],
        }
        self.last_error = None
        try:
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=4000,
                system=SYSTEM,
                messages=[{"role": "user", "content": json.dumps(payload)}],
                output_config={"effort": "low"},  # latency matters more than depth here
                output_format=_Plan,
            )
            if resp.stop_reason != "end_turn" or resp.parsed_output is None:
                raise RuntimeError(f"stop_reason={resp.stop_reason}")
            plan = resp.parsed_output
        except Exception as e:  # noqa: BLE001 - the monitor must never die on the advisor
            self.last_error = f"{type(e).__name__}: {e}"
            return baseline

        names = {z.name for z in zones}
        out: list[Recommendation] = []
        for a in plan.actions:
            rec = Recommendation(a.action, a.zone, a.target, max(1, a.priority), a.reason, a.amount, source="llm")
            if validate(rec, site, names):
                out.append(rec)
        kept = {(r.action, r.zone, r.target) for r in out}
        out += [r for r in baseline if (r.action, r.zone, r.target) not in kept]
        return sorted(out, key=lambda r: r.priority)
