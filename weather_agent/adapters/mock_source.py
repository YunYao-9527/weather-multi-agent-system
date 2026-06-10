from __future__ import annotations

from abc import ABC, abstractmethod

from weather_agent.demo_data import demo_observation
from weather_agent.models import Observation


class ObservationSource(ABC):
    @abstractmethod
    def read(self) -> Observation:
        raise NotImplementedError


class DemoObservationSource(ObservationSource):
    def __init__(self, city: str = "天津"):
        self.city = city

    def read(self) -> Observation:
        return demo_observation(self.city)
