from __future__ import annotations

from weather_agent.base import ForecastAgent
from weather_agent.models import AgentOutput, EvidenceCard, Observation, clamp01, hazard_template


class CirculationAgent(ForecastAgent):
    name = "circulation"

    def run(self, obs: Observation) -> AgentOutput:
        # Upward motion (negative omega) + convergence imply convective trigger support
        trigger = clamp01((-obs.vertical_velocity / 2.0) + obs.low_level_convergence * 0.7)
        scores = hazard_template()
        scores["short_rain"] = clamp01(trigger * 0.9)
        scores["wind"] = clamp01(trigger * 0.6)
        scores["hail"] = clamp01(trigger * 0.4)

        claim = "存在明显动力触发背景，短时内本地对流触发概率升高" if trigger >= 0.5 else "动力触发背景一般，需等待更多证据"
        card = EvidenceCard(
            agent=self.name,
            claim=claim,
            confidence=clamp01(0.45 + trigger * 0.5),
            hazard_scores=scores,
            supporting_features={
                "vertical_velocity": obs.vertical_velocity,
                "low_level_convergence": obs.low_level_convergence,
                "source": "dynamic_fields",
                "proxy_source": False,
            },
            proxy_source=False,
            upstream_sources=["dynamic_fields"],
            rule_version="circulation.v2",
            model_version="rule_based.v2",
        )
        return AgentOutput(agent=self.name, evidence=card)
