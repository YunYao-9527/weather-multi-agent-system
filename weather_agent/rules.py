from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class HazardThresholds:
    # Hail
    hail_wbz_km_max: float
    hail_dbz_trigger: float
    hail_cape_min: float
    hail_shear_min: float

    # Wind
    wind_dcape_min: float
    wind_t850_500_min: float
    wind_shear_min: float

    # Short rain
    rain_humidity_min: float
    rain_cape_min: float


def month_profile(ts: datetime) -> str:
    m = ts.month
    if m in (4, 5):
        return "spring_hail_shear"
    if m in (7, 8):
        return "summer_high_cape"
    return "default"


def thresholds_for(ts: datetime) -> HazardThresholds:
    p = month_profile(ts)

    # Values are anchored by your doc guidance:
    # - WBZ for hail discrimination (<=4.2km small hail favorable)
    # - T850-500 >=24C supports wind potential
    # - Seasonal difference: Apr-May often lower CAPE with stronger shear, Jul-Aug high CAPE can still work with weaker shear
    if p == "spring_hail_shear":
        return HazardThresholds(
            hail_wbz_km_max=4.2,
            hail_dbz_trigger=50.0,
            hail_cape_min=800.0,
            hail_shear_min=18.0,
            wind_dcape_min=700.0,
            wind_t850_500_min=24.0,
            wind_shear_min=16.0,
            rain_humidity_min=0.62,
            rain_cape_min=900.0,
        )

    if p == "summer_high_cape":
        return HazardThresholds(
            hail_wbz_km_max=4.2,
            hail_dbz_trigger=50.0,
            hail_cape_min=1800.0,
            hail_shear_min=12.0,
            wind_dcape_min=900.0,
            wind_t850_500_min=24.0,
            wind_shear_min=12.0,
            rain_humidity_min=0.70,
            rain_cape_min=1400.0,
        )

    return HazardThresholds(
        hail_wbz_km_max=4.2,
        hail_dbz_trigger=50.0,
        hail_cape_min=1200.0,
        hail_shear_min=15.0,
        wind_dcape_min=800.0,
        wind_t850_500_min=24.0,
        wind_shear_min=14.0,
        rain_humidity_min=0.65,
        rain_cape_min=1100.0,
    )
