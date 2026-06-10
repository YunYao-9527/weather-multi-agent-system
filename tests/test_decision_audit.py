import datetime as dt

from weather_agent.base import ForecastAgent
from weather_agent.models import AgentOutput, EvidenceCard, Observation
from weather_agent.orchestrator import ForecastOrchestrator, OrchestratorConfig


class _FixedAgent(ForecastAgent):
    def __init__(self, name: str, score: float, proxy: bool = False):
        self.name = name
        self._score = score
        self._proxy = proxy

    def run(self, obs: Observation) -> AgentOutput:
        card = EvidenceCard(
            agent=self.name,
            claim="fixed",
            confidence=0.9,
            hazard_scores={"short_rain": self._score, "wind": self._score, "hail": self._score, "tornado": self._score},
            supporting_features={"source": self.name, "proxy_source": self._proxy},
            proxy_source=self._proxy,
            upstream_sources=[self.name],
        )
        return AgentOutput(agent=self.name, evidence=card)


def _obs(**meta) -> Observation:
    return Observation(
        timestamp=dt.datetime(2025, 3, 1, 12, 0),
        city="Tianjin",
        vertical_velocity=-1.0,
        low_level_convergence=0.7,
        cape=1500,
        dcape=900,
        shear_0_6km=16,
        t850_500=25,
        wbz_km=4.0,
        humidity_low=0.7,
        radar_dbz_max=50,
        radar_bow_echo=False,
        storm_motion_ms=10,
        prob_guidance={"short_rain": 0.6, "wind": 0.6, "hail": 0.3, "tornado": 0.1},
        source_meta={
            "data_quality_score": "0.9",
            "model_spread_score": "0.1",
            "signal_persist_minutes": "30",
            "model_count": "3",
            "model_coverage_score": "1.0",
            "radar_age_minutes": "10",
            "radar_freshness_score": "0.9",
            **{k: str(v) for k, v in meta.items()},
        },
    )


def test_degraded_mode_triggers_manual_review():
    cfg = OrchestratorConfig(issue_threshold=0.5, clear_threshold=0.3, min_readiness_score=0.2)
    agents = [_FixedAgent("a", 0.9), _FixedAgent("b", 0.85)]
    orch = ForecastOrchestrator(config=cfg, agents=agents)

    obs = _obs(data_quality_score=0.3, radar_age_minutes=180)
    cycle = orch.run_cycle(obs)

    assert cycle.decision.degraded_mode is True
    assert cycle.decision.requires_human_review is True
    assert cycle.decision.action == "manual_review"
    assert cycle.decision.issue is False


def test_decision_hysteresis_issue_then_clear():
    cfg = OrchestratorConfig(issue_threshold=0.55, clear_threshold=0.35, min_readiness_score=0.2)
    orch = ForecastOrchestrator(config=cfg, agents=[_FixedAgent("a", 0.9), _FixedAgent("b", 0.9)])

    c1 = orch.run_cycle(_obs())
    assert c1.decision.issue is True

    # still above clear threshold; should stay issued due hysteresis
    orch.agents = [_FixedAgent("a", 0.4), _FixedAgent("b", 0.4)]
    c2 = orch.run_cycle(_obs())
    assert c2.decision.issue is True

    # below clear threshold; should clear
    orch.agents = [_FixedAgent("a", 0.1), _FixedAgent("b", 0.1)]
    c3 = orch.run_cycle(_obs())
    assert c3.decision.issue is False


def test_audit_record_contains_provenance_and_trace():
    cfg = OrchestratorConfig(issue_threshold=0.5, clear_threshold=0.3, min_readiness_score=0.2)
    orch = ForecastOrchestrator(config=cfg, agents=[_FixedAgent("a", 0.7, proxy=True), _FixedAgent("b", 0.7)])
    cycle = orch.run_cycle(_obs())

    assert cycle.audit is not None
    assert cycle.audit.run_id
    assert "a" in cycle.audit.provenance
    assert "decision_trace" in cycle.audit.decision_snapshot
    assert "per_agent_contribution" in cycle.audit.fusion_snapshot
