from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Iterable


@dataclass
class LocalGridConfig:
    path: str = "data/radar_grids/latest.json"
    max_distance_km: float = 180.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    return 2.0 * r * asin(sqrt(max(a, 0.0)))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _parse_iso_time(v: str | None) -> float | None:
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _load_cells_json(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cells = raw.get("cells")
    if not isinstance(cells, list):
        return [], raw if isinstance(raw, dict) else {}
    return cells, raw if isinstance(raw, dict) else {}


def _load_cells_csv(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "lat": r.get("lat"),
                    "lon": r.get("lon"),
                    "dbz": r.get("dbz") or r.get("reflectivity") or r.get("reflectivity_dbz"),
                    "echo_top_km": r.get("echo_top_km") or r.get("echo_top"),
                    "vil": r.get("vil"),
                }
            )
    return rows, {"source": "local_csv_grid"}


def _nearest_cells(
    cells: Iterable[dict[str, Any]],
    lat: float,
    lon: float,
    max_distance_km: float,
    keep: int = 12,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for c in cells:
        c_lat = _safe_float(c.get("lat"), default=9999.0)
        c_lon = _safe_float(c.get("lon"), default=9999.0)
        if abs(c_lat) > 90 or abs(c_lon) > 180:
            continue
        d = _haversine_km(lat, lon, c_lat, c_lon)
        if d > max_distance_km:
            continue
        ranked.append(
            {
                "lat": c_lat,
                "lon": c_lon,
                "dbz": _safe_float(c.get("dbz"), 0.0),
                "echo_top_km": _safe_float(c.get("echo_top_km"), 0.0),
                "vil": _safe_float(c.get("vil"), 0.0),
                "distance_km": d,
            }
        )
    ranked.sort(key=lambda x: x["distance_km"])
    return ranked[:keep]


def fetch_local_radar_grid(lat: float, lon: float, cfg: LocalGridConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LocalGridConfig()
    fp = Path(cfg.path)
    if not fp.exists():
        return {
            "dbz": 0.0,
            "bow_echo": False,
            "frame_time": None,
            "source": "local_radar_grid",
            "status": "not_found",
            "proxy_source": False,
            "feature_kind": "grid",
            "grid_path": str(fp),
        }

    try:
        if fp.suffix.lower() == ".csv":
            cells, meta = _load_cells_csv(fp)
        else:
            cells, meta = _load_cells_json(fp)
    except Exception as e:
        return {
            "dbz": 0.0,
            "bow_echo": False,
            "frame_time": None,
            "source": "local_radar_grid",
            "status": f"parse_error:{e}",
            "proxy_source": False,
            "feature_kind": "grid",
            "grid_path": str(fp),
        }

    near = _nearest_cells(cells, lat=lat, lon=lon, max_distance_km=float(cfg.max_distance_km))
    if not near:
        return {
            "dbz": 0.0,
            "bow_echo": False,
            "frame_time": _parse_iso_time(str(meta.get("generated_at", ""))),
            "source": "local_radar_grid",
            "status": "no_nearby_cells",
            "proxy_source": False,
            "feature_kind": "grid",
            "grid_path": str(fp),
        }

    nearest = near[0]
    dbz_vals = [float(c["dbz"]) for c in near]
    dbz_max = max(dbz_vals) if dbz_vals else 0.0
    dbz_mean = sum(dbz_vals) / max(len(dbz_vals), 1)
    echo_top_max = max(float(c.get("echo_top_km", 0.0)) for c in near)
    vil_max = max(float(c.get("vil", 0.0)) for c in near)
    bow_echo = bool(dbz_max >= 45.0 and dbz_mean >= 38.0 and vil_max >= 20.0)
    frame_time = _parse_iso_time(str(meta.get("generated_at", "")))

    return {
        "dbz": round(float(nearest["dbz"]), 2),
        "dbz_max_nearby": round(float(dbz_max), 2),
        "dbz_mean_nearby": round(float(dbz_mean), 2),
        "echo_top_km_max_nearby": round(float(echo_top_max), 2),
        "vil_max_nearby": round(float(vil_max), 2),
        "bow_echo": bow_echo,
        "frame_time": frame_time,
        "source": str(meta.get("source", "local_radar_grid")),
        "status": "ok",
        "proxy_source": False,
        "feature_kind": "grid",
        "grid_path": str(fp),
        "nearest_distance_km": round(float(nearest["distance_km"]), 2),
        "cell_count_used": len(near),
    }
