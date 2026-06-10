from __future__ import annotations

from weather_agent.adapters.utils import clamp


def hazard_probs_from_features(
    cape: float,
    lifted_index: float,
    humidity: float,
    gust: float,
    shear: float,
    wbz_km: float,
    precip_prob_pct: float,
) -> dict:
    rain_cape = clamp(cape / 2200.0, 0.0, 1.0)
    rain_humidity = clamp(humidity, 0.0, 1.0)
    rain_precip = clamp(precip_prob_pct / 100.0, 0.0, 1.0)
    short_rain = clamp(0.40 * rain_cape + 0.35 * rain_humidity + 0.25 * rain_precip, 0.0, 1.0)

    # Conservative mapping to avoid over-amplifying single noisy features.
    wind_gust = clamp((gust - 8.0) / 20.0, 0.0, 1.0)
    wind_shear = clamp((shear - 10.0) / 25.0, 0.0, 1.0)
    unstable_li = clamp((-lifted_index - 1.0) / 5.0, 0.0, 1.0)
    wind = clamp(0.45 * wind_gust + 0.25 * wind_shear + 0.15 * unstable_li + 0.15 * short_rain, 0.0, 1.0)

    hail_cape = clamp((cape - 500.0) / 2200.0, 0.0, 1.0)
    hail_shear = clamp((shear - 12.0) / 22.0, 0.0, 1.0)
    low_wbz = clamp((3.8 - wbz_km) / 1.8, 0.0, 1.0)
    hail = clamp(0.40 * hail_cape + 0.25 * hail_shear + 0.25 * low_wbz + 0.10 * unstable_li, 0.0, 1.0)

    tornadic_shear = clamp((shear - 15.0) / 20.0, 0.0, 1.0)
    near_surface_moisture = clamp((humidity - 0.65) / 0.2, 0.0, 1.0)
    tornado = clamp(0.20 * wind + 0.15 * hail + 0.20 * tornadic_shear + 0.10 * near_surface_moisture, 0.0, 1.0)
    return {
        "short_rain": short_rain,
        "wind": wind,
        "hail": hail,
        "tornado": tornado,
    }
