from __future__ import annotations

import io
import time
from typing import Any

import requests
from PIL import Image

from weather_agent.adapters.utils import clamp, latlon_to_tile

MAPS_URL = "https://api.rainviewer.com/public/weather-maps.json"
_MANIFEST_CACHE: dict[str, Any] = {"fetched_at": 0.0, "payload": None}


def fetch_weather_maps_manifest(timeout: int = 20, max_age_sec: int = 120) -> dict[str, Any]:
    now = time.time()
    cached = _MANIFEST_CACHE.get("payload")
    fetched_at = float(_MANIFEST_CACHE.get("fetched_at", 0.0) or 0.0)
    if cached and now - fetched_at <= max_age_sec:
        return dict(cached)

    r = requests.get(MAPS_URL, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    _MANIFEST_CACHE["payload"] = dict(js)
    _MANIFEST_CACHE["fetched_at"] = now
    return js


def latest_radar_frame(manifest: dict[str, Any]) -> dict[str, Any]:
    radar = manifest.get("radar", {})
    frames = radar.get("past", [])
    if not frames:
        return {}
    return dict(frames[-1])


def latest_radar_tile_descriptor(timeout: int = 20) -> dict[str, Any]:
    manifest = fetch_weather_maps_manifest(timeout=timeout)
    host = str(manifest.get("host", "") or "")
    frame = latest_radar_frame(manifest)
    path = str(frame.get("path", "") or "")
    frame_time = frame.get("time")
    if not host or not path:
        return {}
    return {
        "host": host,
        "frame_path": path,
        "frame_time": frame_time,
        "tile_url_template": f"{host}{path}/256/{{z}}/{{x}}/{{y}}/2/1_1.png",
    }


def fetch_radar_dbz(lat: float, lon: float, z: int = 7) -> dict:
    js = fetch_weather_maps_manifest(timeout=20)

    host = js.get("host", "")
    frame = latest_radar_frame(js)
    frames = [frame] if frame else []
    if not host or not frames:
        return {"dbz": 0.0, "bow_echo": False, "frame_time": None, "source": "rainviewer"}

    frame = frames[-1]
    path = frame.get("path")
    if not path:
        return {"dbz": 0.0, "bow_echo": False, "frame_time": None, "source": "rainviewer"}
    frame_time = frame.get("time")

    tile_x, tile_y, x_float, y_float = latlon_to_tile(lat, lon, z)
    url = f"{host}{path}/256/{z}/{tile_x}/{tile_y}/2/1_1.png"

    img_resp = requests.get(url, timeout=20)
    img_resp.raise_for_status()
    img = Image.open(io.BytesIO(img_resp.content)).convert("RGBA")
    px = img.load()

    px_x = int((x_float - tile_x) * 256)
    px_y = int((y_float - tile_y) * 256)
    r_, g_, b_, a_ = px[px_x, px_y]

    if a_ == 0:
        dbz = 0.0
    else:
        intensity = max(r_, g_, b_) / 255.0
        dbz = clamp(20.0 + intensity * 45.0, 20.0, 70.0)

    # Very light placeholder: bow echo signal unavailable from single-pixel extraction
    return {
        "dbz": float(dbz),
        "bow_echo": False,
        "frame_time": frame_time,
        "source": "rainviewer",
        "proxy_source": True,
        "feature_kind": "proxy_tile",
        "tile": {"z": z, "x": tile_x, "y": tile_y},
        "tile_url_template": f"{host}{path}/256/{{z}}/{{x}}/{{y}}/2/1_1.png",
        "frame_path": path,
    }
