from __future__ import annotations

from datetime import datetime

from weather_agent.models import Observation


def demo_observation(city: str = "天津") -> Observation:
    return Observation(
        timestamp=datetime.now(),
        city=city,
        vertical_velocity=-1.2,
        low_level_convergence=0.72,
        cape=1850.0,
        dcape=920.0,
        shear_0_6km=18.0,
        t850_500=25.2,
        wbz_km=3.9,
        humidity_low=0.78,
        radar_dbz_max=56.0,
        radar_bow_echo=True,
        storm_motion_ms=13.0,
        prob_guidance={
            "short_rain": 0.71,
            "wind": 0.63,
            "hail": 0.36,
            "tornado": 0.08,
        },
        source_meta={"mode": "demo"},
    )
