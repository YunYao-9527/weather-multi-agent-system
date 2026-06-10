from __future__ import annotations

from weather_agent.base import ForecastAgent
from weather_agent.models import AgentOutput, EvidenceCard, Observation, clamp01, hazard_template


class NowcastAgent(ForecastAgent):
    name = "nowcast"

    def run(self, obs: Observation) -> AgentOutput:
        # 0-2h tendency proxy
        growth = clamp01(obs.low_level_convergence * 0.45 + obs.humidity_low * 0.3 + (-obs.vertical_velocity / 2.0) * 0.25)

        scores = hazard_template()
        scores["short_rain"] = clamp01(growth * 0.85)
        scores["wind"] = clamp01(growth * 0.55)
        scores["hail"] = clamp01(growth * 0.35)

        card = EvidenceCard(
            agent=self.name,
            claim="未来0-2小时对流维持或增强概率较高" if growth >= 0.5 else "未来0-2小时对流维持信号一般",
            confidence=clamp01(0.45 + growth * 0.45),
            hazard_scores=scores,
            supporting_features={
                "growth_signal": round(growth, 3),
                "window": "0-2h",
                "source": "nowcast_proxy",
                "proxy_source": True,
            },
            proxy_source=True,
            upstream_sources=["nowcast_proxy"],
            rule_version="nowcast.v2",
            model_version="heuristic_proxy.v2",
        )
        return AgentOutput(agent=self.name, evidence=card)
