"""Unit tests that need no dataset, weights or GPU."""
import numpy as np

from crowdsafe.agent import Recommendation, RulePolicy, SafetyAgent, Site, ZoneSnapshot, validate
from crowdsafe.alerts import AlertConfig, Level, ZoneMonitor, run_monitor
from crowdsafe.density import gt_density
from crowdsafe.evaluate import episodes, evaluate
from crowdsafe.flow import FlowEstimator, direction_label
from crowdsafe.zones import Zone

SITE = {
    "site": {
        "staff_available": 3,
        "zones": {
            "a": {"gates": [{"id": "G1", "name": "Side exit", "state": "closed"}], "neighbors": {"down": "b"}},
            "b": {"gates": [], "neighbors": {"up": "a"}},
        },
    }
}


def snap(name, level, count, thr=10, direction="still", coherence=0.8):
    return ZoneSnapshot(name, level, count, thr, 0.5, 4.0, direction, coherence)


# ---- density and zones -----------------------------------------------------
def test_gt_density_preserves_count_even_at_borders():
    pts = np.array([[0, 0], [639, 479], [320, 240], [5, 470]], np.float32)
    assert abs(gt_density(pts, 480, 640).sum() - 4) < 1e-3


def test_zone_counts_split_a_density_map_without_loss():
    top = Zone("top", [[0, 0], [1, 0], [1, 0.5], [0, 0.5]])
    bottom = Zone("bottom", [[0, 0.5], [1, 0.5], [1, 1], [0, 1]])
    d = np.random.default_rng(0).random((60, 80)).astype(np.float32)
    assert abs(top.count_density(d) + bottom.count_density(d) - d.sum()) < 1e-2
    pts = np.array([[10, 10], [10, 400], [600, 100]], np.float32)
    assert (top.count_points(pts, 480, 640), bottom.count_points(pts, 480, 640)) == (2, 1)


# ---- flow ------------------------------------------------------------------
def test_direction_labels():
    assert direction_label(2, 0, 0.4) == "right"
    assert direction_label(0, 2, 0.4) == "down"
    assert direction_label(-2, 0.3, 0.4) == "left"
    assert direction_label(0, -2, 0.4) == "up"
    assert direction_label(0.1, 0.1, 0.4) == "still"


def test_flow_recovers_direction_of_moving_texture():
    rng = np.random.default_rng(1)
    tex = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    tex = np.kron(tex[::8, ::8], np.ones((8, 8, 1), np.uint8))  # blocky so it is trackable
    shifted = np.roll(tex, 12, axis=1)  # everything moves right
    fe = FlowEstimator()
    assert fe.update(tex) is None
    flow = fe.update(shifted)
    zone = Zone("all", [[0, 0], [1, 0], [1, 1], [0, 1]])
    zf = fe.zone_flow(flow, np.ones((60, 80), np.float32), zone)
    assert zf.direction == "right" and zf.coherence > 0.8


# ---- alerts ----------------------------------------------------------------
def test_predictive_alert_fires_before_a_steady_ramp_crosses():
    counts = np.concatenate([np.full(20, 5.0), np.linspace(5, 15, 40), np.full(20, 15.0)])
    levels = run_monitor(counts, 12, AlertConfig(ema_alpha=0.6, trend_window=6, horizon=5, warn_ratio=0.9, confirm=2))
    first_alert = int(np.argmax(levels > 0))
    first_breach = int(np.argmax(counts >= 12))
    assert 0 < first_alert < first_breach
    assert levels[first_alert] == Level.WARNING and levels.max() == Level.CRITICAL


def test_flat_series_below_threshold_never_alerts():
    noise = np.random.default_rng(0).normal(0, 0.5, 300)
    assert run_monitor(6 + noise, 12, AlertConfig()).max() == Level.NORMAL


def test_hysteresis_keeps_alert_until_zone_clearly_empties():
    m = ZoneMonitor(10, AlertConfig(ema_alpha=1.0, confirm=1, clear_ratio=0.8))
    assert [m.update(c).level for c in [11, 9.5, 9, 7.9]] == [
        Level.CRITICAL, Level.CRITICAL, Level.CRITICAL, Level.NORMAL]


# ---- evaluation ------------------------------------------------------------
def test_episodes_merge_and_min_length():
    a = np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0], bool)
    assert episodes(a) == [(1, 3), (4, 5), (9, 10)]
    assert episodes(a, min_len=2, merge_gap=1) == [(1, 5)]


def test_lead_time_and_false_alert_accounting():
    gt = np.zeros((100, 1))
    gt[50:60] = 20  # one breach at frame 50
    levels = np.zeros((100, 1), int)
    levels[44:60] = 1  # true alert, 6 frames early
    levels[10:13] = 1  # false alert
    m = evaluate(gt, levels, [10.0], window=20, min_len=2, merge_gap=3)
    assert (m.breaches, m.detected, m.alerts, m.false_alerts) == (1, 1, 2, 1)
    assert m.mean_lead == 6 and m.false_alert_rate == 0.5 and m.early_rate == 1.0


def test_earlier_blip_that_cleared_earns_no_lead():
    gt = np.zeros((100, 1))
    gt[50:60] = 20
    levels = np.zeros((100, 1), int)
    levels[40:43] = 1  # cleared 7 frames before the breach: a false alert, not a warning
    levels[52:60] = 2
    m = evaluate(gt, levels, [10.0], window=20)
    assert m.mean_lead == -2 and m.false_alerts == 1 and m.early_rate == 0.0


def test_surge_scenarios_breach_only_when_they_should():
    from crowdsafe.surge import SurgeSpec, make_scenarios

    spec = SurgeSpec()
    sc = make_scenarios(16, np.random.default_rng(0), np.zeros(500), 0.5, 0.0, spec)
    for s in sc:
        assert (s.true.max() >= spec.limit) == (s.rate is not None)
        assert np.array_equal(s.true, s.est)  # zero counting error in, zero out


def test_late_alert_gives_negative_lead_and_missed_breach_counts():
    gt = np.zeros((100, 1))
    gt[20:30] = 20
    gt[70:80] = 20
    levels = np.zeros((100, 1), int)
    levels[23:30] = 2  # 3 frames late; second breach never alerted
    m = evaluate(gt, levels, [10.0], window=20)
    assert m.mean_lead == -3 and m.missed == 1 and m.recall == 0.5 and m.early_rate == 0.0


# ---- agent -----------------------------------------------------------------
def test_warning_opens_gate_and_sends_one_staff():
    site = Site.from_config(SITE)
    recs = RulePolicy().recommend([snap("a", Level.WARNING, 9), snap("b", Level.NORMAL, 3)], site)
    acts = {(r.action, r.target, r.amount) for r in recs}
    assert ("open_gate", "G1", 0) in acts and ("dispatch_staff", "a", 1) in acts
    assert not any(r.action == "escalate" for r in recs)


def test_inflow_from_neighbour_is_held():
    site = Site.from_config(SITE)
    zones = [snap("a", Level.NORMAL, 4, direction="down"), snap("b", Level.WARNING, 9)]
    recs = RulePolicy().recommend(zones, site)
    assert any(r.action == "hold_inflow" and r.target == "a" for r in recs)


def test_agent_only_speaks_on_stage_changes_and_tracks_staff():
    agent = SafetyAgent(Site.from_config(SITE))
    quiet = [snap("a", Level.NORMAL, 3), snap("b", Level.NORMAL, 3)]
    warn = [snap("a", Level.WARNING, 9), snap("b", Level.NORMAL, 3)]
    crit = [snap("a", Level.CRITICAL, 13), snap("b", Level.NORMAL, 3)]
    assert agent.step(quiet) == []
    first = agent.step(warn)
    assert first and agent.step(warn) == []  # no repeats while nothing changes
    second = agent.step(crit)
    assert any(r.action == "pa_announcement" for r in second)
    assert not any(r.action == "open_gate" for r in second)  # already recommended
    assert agent.site.staff_committed["a"] == 2
    agent.step(quiet)
    assert agent.active == [] and agent.site.staff_free() == 3


def test_validation_rejects_things_that_do_not_exist_on_site():
    site = Site.from_config(SITE)
    ok = Recommendation("open_gate", "a", "G1", 1, "r")
    assert validate(ok, site, {"a", "b"})
    assert not validate(Recommendation("open_gate", "a", "G99", 1, "r"), site, {"a", "b"})
    assert not validate(Recommendation("dispatch_staff", "a", "a", 1, "r", amount=50), site, {"a", "b"})
    assert not validate(Recommendation("launch_drone", "a", "a", 1, "r"), site, {"a", "b"})


class _FakeClient:
    """Stands in for anthropic.Anthropic so the advisor is testable offline."""

    def __init__(self, plan=None, fail=False):
        self.plan, self.fail = plan, fail
        self.messages = self

    def parse(self, **kwargs):
        if self.fail:
            raise ConnectionError("offline")
        return type("R", (), {"stop_reason": "end_turn", "parsed_output": self.plan})()


def test_llm_advisor_keeps_rule_floor_and_drops_invalid_actions():
    from crowdsafe.llm_agent import LLMAdvisor, _Action, _Plan

    site = Site.from_config(SITE)
    zones = [snap("a", Level.CRITICAL, 13), snap("b", Level.NORMAL, 3)]
    baseline = RulePolicy().recommend(zones, site)
    plan = _Plan(actions=[
        _Action(action="open_gate", zone="a", target="G99", priority=1, amount=0, reason="invented gate"),
        _Action(action="redirect_flow", zone="a", target="b", priority=1, amount=0, reason="b is quiet"),
    ])
    out = LLMAdvisor(client=_FakeClient(plan)).refine(zones, site, baseline)
    keys = {(r.action, r.target) for r in out}
    assert ("open_gate", "G99") not in keys
    assert {(r.action, r.target) for r in baseline} <= keys


def test_llm_advisor_falls_back_to_rules_on_error():
    from crowdsafe.llm_agent import LLMAdvisor

    site = Site.from_config(SITE)
    zones = [snap("a", Level.WARNING, 9), snap("b", Level.NORMAL, 3)]
    baseline = RulePolicy().recommend(zones, site)
    adv = LLMAdvisor(client=_FakeClient(fail=True))
    assert adv.refine(zones, site, baseline) == baseline and "offline" in adv.last_error
