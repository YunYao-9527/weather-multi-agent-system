import datetime as dt

from weather_agent.fusion import fuse_evidence
from weather_agent.models import EvidenceCard, FusionResult, HazardObject, Observation, ObservationEnvelope
from weather_agent.orchestrator import ForecastOrchestrator, OrchestratorConfig
from weather_agent.policy_engine import PolicySnapshot, decide_packet
from weather_agent.providers import ProxyObjectiveGuidanceProvider
from weather_agent.replay import ReplayStore
from weather_agent.storage import RegistryStore
from weather_agent.truth_factory import TruthFactory


def _obs(meta: dict | None = None) -> Observation:
    return Observation(
        timestamp=dt.datetime(2025, 4, 1, 12, 0),
        city="Tianjin",
        vertical_velocity=-1.0,
        low_level_convergence=0.7,
        cape=1500.0,
        dcape=900.0,
        shear_0_6km=18.0,
        t850_500=25.0,
        wbz_km=4.0,
        humidity_low=0.75,
        radar_dbz_max=52.0,
        radar_bow_echo=True,
        storm_motion_ms=12.0,
        prob_guidance={"short_rain": 0.7, "wind": 0.65, "hail": 0.3, "tornado": 0.1},
        source_meta={
            "mode": "manual",
            "analysis_time": "2025-04-01T12:00:00",
            "valid_time": "2025-04-01T12:00:00",
            "lead_time": "0",
            "signal_persist_minutes": "40",
            "area_coverage_ratio": "0.3",
            "data_quality_score": "0.9",
            "model_spread_score": "0.1",
            "model_count": "3",
            "model_coverage_score": "1.0",
            "radar_age_minutes": "10",
            "radar_freshness_score": "0.95",
            **(meta or {}),
        },
    )


def test_provider_normalization_marks_proxy_tier():
    provider = ProxyObjectiveGuidanceProvider()
    payload = provider.normalize(provider.fetch(guidance={"wind": 0.6}, now=dt.datetime(2025, 4, 1, 12, 0)), domain_id="city:tianjin")
    record = provider.to_record(payload)
    qc = provider.validate(payload)
    assert record.source_tier == "proxy"
    assert "proxy_source" in qc.flags


def test_family_aware_fusion_limits_same_family_reinforcement():
    cards = [
        EvidenceCard(agent="radar", family="radar", source_tier="experimental", claim="a", confidence=0.9, hazard_scores={"wind": 0.9}, supporting_features={"source": "r1"}),
        EvidenceCard(agent="nowcast", family="radar", source_tier="experimental", claim="b", confidence=0.9, hazard_scores={"wind": 0.9}, supporting_features={"source": "r2"}),
        EvidenceCard(agent="environment", family="environment", source_tier="experimental", claim="c", confidence=0.7, hazard_scores={"wind": 0.5}, supporting_features={"source": "env"}),
    ]
    fused = fuse_evidence(cards, data_quality=0.9, model_spread_score=0.1)
    radar_family = fused.family_contribution["radar"]["wind"]
    assert radar_family <= 0.9
    assert fused.hazard_prob_raw["wind"] < 0.9


def test_policy_engine_vetoes_proxy_over_limit():
    fusion = FusionResult(
        p_release={"wind": 0.8},
        lower_confidence_bound={"wind": 0.7},
        proxy_dependency_ratio=0.8,
        independent_family_count=3,
        effective_source_tier="proxy",
        family_contribution={"radar": {"wind": 0.8}},
    )
    obj = HazardObject(
        object_id="obj1",
        hazard_type="wind",
        area_km2=200.0,
        evidence_persistence=3,
        min_area_met=True,
        min_duration_met=True,
        lifecycle_state="recommend_issue",
    )
    packet = decide_packet(obj, fusion, PolicySnapshot(max_proxy_share=0.35))
    assert packet.requires_human_review is True
    assert "proxy_share_exceeded" in packet.veto_reasons


def test_truth_versions_are_append_only(tmp_path):
    factory = TruthFactory(root=tmp_path / "truth", registry=RegistryStore(tmp_path / "registry"))
    reconciled = {"point_truth": [], "grid_truth": [], "object_truth": [], "event_truth": [], "summary": {"point_truth": 0}}
    v1 = factory.version_truth(city="Tianjin", period={"start": "2025-04-01", "end": "2025-04-02"}, reconciled=reconciled)
    v2 = factory.version_truth(city="Tianjin", period={"start": "2025-04-01", "end": "2025-04-03"}, reconciled=reconciled)
    assert v1.truth_version != v2.truth_version
    assert len(factory.list_versions()) == 2


def test_replay_bundle_is_deterministic(tmp_path):
    orch = ForecastOrchestrator(config=OrchestratorConfig(issue_threshold=0.5, clear_threshold=0.3))
    cycle = orch.run_cycle(ObservationEnvelope.from_observation(_obs()), request_id="req-1", trace_id="trace-1")
    store = ReplayStore(root=tmp_path / "runs")
    store.save(cycle)
    check = store.compare_bundle(cycle.replay_bundle.bundle_id, store.replay_bundle(cycle.replay_bundle.bundle_id)["cycle"])
    assert check["deterministic"] is True


def test_full_chain_persists_object_and_trace(tmp_path):
    orch = ForecastOrchestrator(config=OrchestratorConfig(issue_threshold=0.5, clear_threshold=0.3))
    cycle = orch.run_cycle(ObservationEnvelope.from_observation(_obs()), request_id="req-chain", trace_id="trace-chain")
    store = ReplayStore(root=tmp_path / "runs")
    store.save(cycle)
    assert cycle.hazard_object.object_id
    assert cycle.audit.request_id == "req-chain"
    assert cycle.decision.family_contribution
