import datetime as dt

from weather_agent.agents.environment import EnvironmentAgent
from weather_agent.agents.llm_situation import LLMSituationAgent
from weather_agent.fusion import fuse_evidence
from weather_agent.llm_openai import LLMSituationEvidencePayload, OpenAIResponsesClient, _build_llm_prompts
from weather_agent.models import Observation, ObservationEnvelope
from weather_agent.orchestrator import ForecastOrchestrator, OrchestratorConfig


def _obs() -> Observation:
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
        source_meta={"mode": "manual", "signal_persist_minutes": "40", "area_coverage_ratio": "0.3"},
    )


class FakeLLMClient:
    model = "gpt-4o-mini"
    ready = True

    def generate_situation_evidence(self, envelope: ObservationEnvelope) -> LLMSituationEvidencePayload:
        return LLMSituationEvidencePayload.model_validate(
            {
                "claim": "Cross-source pattern supports organized severe convection with manual review fallback if conflict grows.",
                "confidence": 0.58,
                "hazard_scores": {"short_rain": 0.66, "wind": 0.61, "hail": 0.34, "tornado": 0.12},
                "key_factors": ["High CAPE", "Radar core present", "Guidance supports wind"],
                "caution_flags": ["Experimental reasoning layer"],
                "action_bias": "manual_review",
            }
        )


def test_llm_shadow_agent_emits_diagnostic_only_evidence():
    envelope = ObservationEnvelope.from_observation(_obs())
    agent = LLMSituationAgent(client=FakeLLMClient(), mode="shadow")
    card = agent.run_envelope(envelope).evidence
    assert card.agent == "llm_situation"
    assert card.family == "llm_reasoning"
    assert card.supporting_features["diagnostic_only"] is True
    assert card.model_version == "openai.responses:gpt-4o-mini"


def test_llm_reasoning_family_does_not_increase_independent_family_count():
    envelope = ObservationEnvelope.from_observation(_obs())
    llm_card = LLMSituationAgent(client=FakeLLMClient(), mode="assist").run_envelope(envelope).evidence
    env_card = EnvironmentAgent().run(_obs()).evidence
    fused = fuse_evidence([env_card, llm_card], data_quality=0.95, model_spread_score=0.1)
    assert "llm_reasoning" in fused.family_contribution
    assert fused.independent_family_count == 0


def test_orchestrator_keeps_llm_shadow_evidence_out_of_weighted_family_fusion():
    envelope = ObservationEnvelope.from_observation(_obs())
    orch = ForecastOrchestrator(
        config=OrchestratorConfig(issue_threshold=0.5, clear_threshold=0.3),
        agents=[EnvironmentAgent(), LLMSituationAgent(client=FakeLLMClient(), mode="shadow")],
    )
    cycle = orch.run_cycle(envelope, request_id="llm-shadow", trace_id="llm-shadow")
    agents = [item.agent for item in cycle.decision.rationale]
    assert "llm_situation" in agents
    assert "llm_reasoning" not in cycle.fusion_result.family_contribution


def test_openai_client_parses_structured_output(monkeypatch):
    class DummyResponse:
        status_code = 200

        def json(self):
            return {
                "output_text": (
                    '{"claim":"Conservative structured analysis supports monitoring with human review if needed.",'
                    '"confidence":0.42,'
                    '"hazard_scores":{"short_rain":0.5,"wind":0.4,"hail":0.2,"tornado":0.1},'
                    '"key_factors":["Radar reflectivity"],'
                    '"caution_flags":["Experimental source"],'
                    '"action_bias":"monitor"}'
                )
            }

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("weather_agent.llm_openai.requests.post", fake_post)
    client = OpenAIResponsesClient(api_key="test-key", model="gpt-4o-mini", timeout_sec=9.0)
    payload = client.generate_situation_evidence(ObservationEnvelope.from_observation(_obs()))
    assert payload.action_bias == "monitor"
    assert captured["json"]["text"]["format"]["type"] == "json_schema"
    assert captured["url"].endswith("/responses")


def test_deepseek_client_uses_chat_completions(monkeypatch):
    class DummyResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"claim":"DeepSeek conservative analysis supports monitoring.",'
                                '"confidence":0.41,'
                                '"hazard_scores":{"short_rain":0.49,"wind":0.38,"hail":0.16,"tornado":0.05},'
                                '"key_factors":["Radar reflectivity"],'
                                '"caution_flags":["Experimental reasoning layer"],'
                                '"action_bias":"monitor"}'
                            )
                        }
                    }
                ]
            }

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("weather_agent.llm_openai.requests.post", fake_post)
    client = OpenAIResponsesClient(api_key="test-key", provider="deepseek", model="deepseek-chat", timeout_sec=9.0)
    payload = client.generate_situation_evidence(ObservationEnvelope.from_observation(_obs()))
    assert payload.action_bias == "monitor"
    assert captured["json"]["response_format"]["type"] == "json_object"
    assert captured["url"].endswith("/chat/completions")


def test_deepseek_prompt_explicitly_forbids_echoing_envelope():
    system_prompt, user_prompt = _build_llm_prompts({"analysis_time": "x", "environment": {"cape": 1}}, "deepseek")
    assert "exactly these top-level keys" in system_prompt
    assert "Do not include analysis_time" in user_prompt
