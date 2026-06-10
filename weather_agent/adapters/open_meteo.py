from __future__ import annotations

import datetime as dt
from typing import Any, Dict

import requests

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_FALLBACK_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def geocode_city(city: str) -> dict:
    r = requests.get(
        GEO_URL,
        params={"name": city, "count": 1, "language": "zh", "format": "json"},
        timeout=20,
    )
    r.raise_for_status()
    js = r.json()
    results = js.get("results") or []
    if not results:
        raise ValueError(f"city not found: {city}")
    return results[0]


def fetch_model_hourly(lat: float, lon: float, model: str, timezone: str = "Asia/Shanghai") -> dict:
    hourly_vars = ",".join(
        [
            "cape",
            "convective_inhibition",
            "lifted_index",
            "relative_humidity_2m",
            "wind_gusts_10m",
            "wind_speed_10m",
            "vertical_velocity_700hPa",
            "temperature_850hPa",
            "temperature_500hPa",
            "freezing_level_height",
            "precipitation_probability",
            "precipitation",
            "showers",
            "wind_speed_850hPa",
            "wind_speed_500hPa",
            "wind_direction_850hPa",
            "wind_direction_500hPa",
        ]
    )

    r = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": hourly_vars,
            "forecast_days": 2,
            "timezone": timezone,
            "models": model,
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def nearest_hour_index(hourly_time: list[str], target: dt.datetime) -> int:
    target_naive = target.replace(tzinfo=None)
    parsed = [dt.datetime.fromisoformat(t) for t in hourly_time]
    best_idx = 0
    best_delta = abs((parsed[0] - target_naive).total_seconds())
    for i, t in enumerate(parsed[1:], start=1):
        d = abs((t - target_naive).total_seconds())
        if d < best_delta:
            best_idx = i
            best_delta = d
    return best_idx


def fetch_archive_hourly(lat: float, lon: float, start_date: str, end_date: str, timezone: str = "Asia/Shanghai") -> dict:
    hourly_vars = ",".join(
        [
            "cape",
            "convective_inhibition",
            "lifted_index",
            "relative_humidity_2m",
            "wind_gusts_10m",
            "wind_speed_10m",
            "vertical_velocity_700hPa",
            "temperature_850hPa",
            "temperature_500hPa",
            "freezing_level_height",
            "precipitation",
            "showers",
            "wind_speed_850hPa",
            "wind_speed_500hPa",
            "wind_direction_850hPa",
            "wind_direction_500hPa",
        ]
    )
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_vars,
        "timezone": timezone,
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    errors = []
    for url in (ARCHIVE_URL, ARCHIVE_FALLBACK_URL):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            js = r.json()
            js["source_endpoint"] = url
            return js
        except Exception as e:
            errors.append(f"{url}: {e}")
            continue
    raise RuntimeError(f"failed to fetch archive hourly from open-meteo: {' | '.join(errors)}")
