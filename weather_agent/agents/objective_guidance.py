from __future__ import annotations

from weather_agent.base import ForecastAgent
from weather_agent.models import AgentOutput, EvidenceCard, Observation, clamp01, hazard_template


class ObjectiveGuidanceAgent(ForecastAgent):
    name = "objective_guidance"

    def run(self, obs: Observation) -> AgentOutput:
        scores = hazard_template()
        for k in scores:
            scores[k] = clamp01(float(obs.prob_guidance.get(k, 0.0)))

        card = EvidenceCard(
            agent=self.name,
            claim="客观概率产品已融合，作为主观研判校验参考",
            confidence=0.75,
            hazard_scores=scores,
            supporting_features={"source": "prob_guidance_proxy", "city": obs.city, "proxy_source": True},
            proxy_source=True,
            upstream_sources=["objective_guidance_proxy"],
            rule_version="objective_guidance.v2",
            model_version="heuristic_proxy.v2",
        )
        return AgentOutput(agent=self.name, evidence=card)
