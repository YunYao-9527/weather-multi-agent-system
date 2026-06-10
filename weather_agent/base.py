from __future__ import annotations

from abc import ABC, abstractmethod
from weather_agent.models import AgentOutput, Observation, ObservationEnvelope


class ForecastAgent(ABC):
    name: str

    def run_envelope(self, envelope: ObservationEnvelope) -> AgentOutput:
        return self.run(envelope.to_observation())

    @abstractmethod
    def run(self, obs: Observation) -> AgentOutput:
        raise NotImplementedError
