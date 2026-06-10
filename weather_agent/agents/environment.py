from __future__ import annotations

from weather_agent.base import ForecastAgent
from weather_agent.models import AgentOutput, EvidenceCard, Observation, clamp01, hazard_template
from weather_agent.rules import thresholds_for


class EnvironmentAgent(ForecastAgent):
    name = "environment"

    def run(self, obs: Observation) -> AgentOutput:
        th = thresholds_for(obs.timestamp)
        low_humidity = clamp01(1.0 - obs.humidity_low)

        hail_factor = clamp01(
            clamp01((th.hail_wbz_km_max - obs.wbz_km + 1.0) / 2.0) * 0.25
            + clamp01(obs.cape / max(th.hail_cape_min, 1.0)) * 0.4
            + clamp01(obs.shear_0_6km / max(th.hail_shear_min, 1.0)) * 0.35
        )

        wind_factor = clamp01(
            clamp01(obs.dcape / max(th.wind_dcape_min, 1.0)) * 0.45
            + clamp01(obs.t850_500 / max(th.wind_t850_500_min, 1.0)) * 0.25
            + clamp01(obs.shear_0_6km / max(th.wind_shear_min, 1.0)) * 0.2
            + low_humidity * 0.1
        )

        rain_factor = clamp01(
            clamp01(obs.humidity_low / max(th.rain_humidity_min, 0.01)) * 0.5
            + clamp01(obs.cape / max(th.rain_cape_min, 1.0)) * 0.4
            + clamp01(obs.low_level_convergence) * 0.1
        )

        scores = hazard_template()
        scores["short_rain"] = rain_factor
        scores["wind"] = wind_factor
        scores["hail"] = hail_factor

        if max(rain_factor, wind_factor, hail_factor) > 0.65:
            claim = "环境条件支持强对流发生，需重点关注灾种分型"
        else:
            claim = "环境场支持度中等，需结合雷达与客观概率进一步确认"

        card = EvidenceCard(
            agent=self.name,
            claim=claim,
            confidence=clamp01(0.5 + max(scores.values()) * 0.4),
            hazard_scores=scores,
            supporting_features={
                "cape": obs.cape,
                "dcape": obs.dcape,
                "shear_0_6km": obs.shear_0_6km,
                "t850_500": obs.t850_500,
                "wbz_km": obs.wbz_km,
                "humidity_low": obs.humidity_low,
                "month": obs.timestamp.month,
                "source": "environment_fields",
                "proxy_source": False,
            },
            proxy_source=False,
            upstream_sources=["environment_fields"],
            rule_version="environment.v2",
            model_version="rule_based.v2",
        )
        return AgentOutput(agent=self.name, evidence=card)
