from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from weather_agent.adapters.area_resolver import _reverse_admin


def _point_offset_km(lat: float, lon: float, dist_km: float, bearing_deg: float) -> Tuple[float, float]:
    rad = math.radians(bearing_deg)
    dlat = (dist_km / 111.0) * math.cos(rad)
    dlon = (dist_km / (111.0 * max(math.cos(math.radians(lat)), 1e-6))) * math.sin(rad)
    return lat + dlat, lon + dlon


def _sample_points(lat: float, lon: float, radius_km: float) -> List[Tuple[float, float]]:
    pts = [(lat, lon)]
    for b in (0, 45, 90, 135, 180, 225, 270, 315):
        pts.append(_point_offset_km(lat, lon, radius_km, b))
    for b in (0, 90, 180, 270):
        pts.append(_point_offset_km(lat, lon, radius_km * 0.5, b))
    return pts


def _risk_radius_km(prob: float) -> float:
    p = float(prob)
    if p >= 0.8:
        return 80.0
    if p >= 0.65:
        return 60.0
    if p >= 0.5:
        return 40.0
    return 25.0


class AdminResolverCache:
    def __init__(self, path: str | Path = "data/cache/admin_reverse_cache.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def _key(self, lat: float, lon: float) -> str:
        return f"{lat:.4f},{lon:.4f}"

    def get_label(self, lat: float, lon: float) -> str:
        k = self._key(lat, lon)
        if k in self.data:
            return str(self.data[k])
        label = ""
        try:
            adm = _reverse_admin(lat, lon)
            label = str(adm.get("label", "") or "")
        except Exception:
            label = ""
        self.data[k] = label
        return label

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")


def predicted_admin_labels(city_lat: float, city_lon: float, hazard_prob: float, cache: AdminResolverCache) -> List[str]:
    radius = _risk_radius_km(hazard_prob)
    labels: List[str] = []
    for p_lat, p_lon in _sample_points(city_lat, city_lon, radius):
        lbl = cache.get_label(p_lat, p_lon)
        if lbl:
            labels.append(lbl)
    return sorted(set(labels))


def truth_admin_labels(label_row: dict, hazard: str, station_catalog: dict, cache: AdminResolverCache) -> List[str]:
    flags = label_row.get("station_event_flags") or {}
    out = []
    for sid, vals in flags.items():
        if int((vals or {}).get(hazard, 0)) != 1:
            continue
        st = station_catalog.get(sid) or {}
        lat = st.get("lat")
        lon = st.get("lon")
        if lat is None or lon is None:
            continue
        lbl = cache.get_label(float(lat), float(lon))
        if lbl:
            out.append(lbl)
    return sorted(set(out))


def admin_hit_bias(pred_labels: Iterable[str], truth_labels_set: Iterable[str]) -> Dict[str, float | int]:
    p = set(pred_labels)
    t = set(truth_labels_set)
    if not t:
        return {
            "truth_admin_count": 0,
            "pred_admin_count": len(p),
            "admin_hit_rate": 0.0,
            "admin_coverage_bias": 0.0,
            "admin_metrics_applicable": False,
        }
    inter = p & t
    hit = len(inter) / max(len(t), 1)
    bias = len(p) / max(len(t), 1)
    return {
        "truth_admin_count": len(t),
        "pred_admin_count": len(p),
        "admin_hit_rate": round(hit, 4),
        "admin_coverage_bias": round(bias, 4),
        "admin_metrics_applicable": True,
    }


def aggregate_admin_metrics(
    samples: List[dict],
    hazard: str,
    city_lat: float,
    city_lon: float,
    station_catalog: dict,
    cache: AdminResolverCache,
    prob_key: str = "enhanced",
) -> Dict[str, float]:
    rows = []
    for s in samples:
        label_row = s.get("truth_row") or {}
        if not label_row:
            continue
        pred = predicted_admin_labels(city_lat, city_lon, float(s.get(prob_key, {}).get(hazard, 0.0)), cache)
        truth = truth_admin_labels(label_row, hazard, station_catalog, cache)
        rows.append(admin_hit_bias(pred, truth))

    if not rows:
        return {
            "admin_hit_rate": 0.0,
            "admin_coverage_bias": 0.0,
            "admin_metrics_applicable_hours": 0,
        }

    hit_vals = [float(r["admin_hit_rate"]) for r in rows if bool(r["admin_metrics_applicable"])]
    bias_vals = [float(r["admin_coverage_bias"]) for r in rows if bool(r["admin_metrics_applicable"])]
    return {
        "admin_hit_rate": round(sum(hit_vals) / max(len(hit_vals), 1), 4) if hit_vals else 0.0,
        "admin_coverage_bias": round(sum(bias_vals) / max(len(bias_vals), 1), 4) if bias_vals else 0.0,
        "admin_metrics_applicable_hours": len(hit_vals),
    }


def _build_grid_points(lat: float, lon: float, radius_km: float, step_km: float = 15.0) -> List[Tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    span = int(max(1, radius_km // step_km))
    for iy in range(-span, span + 1):
        for ix in range(-span, span + 1):
            d = math.sqrt(float(ix * ix + iy * iy)) * step_km
            if d <= radius_km:
                # translate local xy(km) to lat/lon
                dlat = (iy * step_km) / 111.0
                dlon = (ix * step_km) / (111.0 * max(math.cos(math.radians(lat)), 1e-6))
                pts.append((lat + dlat, lon + dlon))
    return pts


def _truth_station_coords(label_row: dict, hazard: str, station_catalog: dict) -> List[Tuple[float, float]]:
    out: list[tuple[float, float]] = []
    flags = label_row.get("station_event_flags") or {}
    for sid, vals in flags.items():
        if int((vals or {}).get(hazard, 0)) != 1:
            continue
        st = station_catalog.get(sid) or {}
        lat = st.get("lat")
        lon = st.get("lon")
        if lat is None or lon is None:
            continue
        out.append((float(lat), float(lon)))
    return out


def _point_distance_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    # local approximation sufficient for evaluation grid sizing
    dlat = (a_lat - b_lat) * 111.0
    dlon = (a_lon - b_lon) * 111.0 * max(math.cos(math.radians((a_lat + b_lat) / 2.0)), 1e-6)
    return math.sqrt(dlat * dlat + dlon * dlon)


def _pred_truth_grid_sets(
    city_lat: float,
    city_lon: float,
    hazard_prob: float,
    truth_stations: List[Tuple[float, float]],
    pred_radius_km: float | None = None,
    truth_radius_km: float = 22.0,
    grid_step_km: float = 15.0,
    domain_radius_km: float = 120.0,
) -> Tuple[set[int], set[int], int]:
    pred_r = float(pred_radius_km if pred_radius_km is not None else _risk_radius_km(hazard_prob))
    grid = _build_grid_points(city_lat, city_lon, radius_km=domain_radius_km, step_km=grid_step_km)
    pred_set: set[int] = set()
    truth_set: set[int] = set()
    for i, (g_lat, g_lon) in enumerate(grid):
        d_city = _point_distance_km(g_lat, g_lon, city_lat, city_lon)
        if d_city <= pred_r:
            pred_set.add(i)
        for s_lat, s_lon in truth_stations:
            if _point_distance_km(g_lat, g_lon, s_lat, s_lon) <= truth_radius_km:
                truth_set.add(i)
                break
    return pred_set, truth_set, len(grid)


def grid_hit_bias(
    pred_cells: set[int],
    truth_cells: set[int],
    total_cells: int,
) -> Dict[str, float | int | bool]:
    if total_cells <= 0:
        return {
            "grid_hit_rate": 0.0,
            "grid_coverage_bias": 0.0,
            "grid_false_alarm_rate": 0.0,
            "grid_csi": 0.0,
            "grid_metrics_applicable": False,
            "truth_grid_count": 0,
            "pred_grid_count": 0,
            "grid_total_cells": 0,
        }
    if not truth_cells:
        return {
            "grid_hit_rate": 0.0,
            "grid_coverage_bias": 0.0,
            "grid_false_alarm_rate": 0.0,
            "grid_csi": 0.0,
            "grid_metrics_applicable": False,
            "truth_grid_count": 0,
            "pred_grid_count": len(pred_cells),
            "grid_total_cells": total_cells,
        }

    tp = len(pred_cells & truth_cells)
    fn = len(truth_cells - pred_cells)
    fp = len(pred_cells - truth_cells)
    hit = tp / max(len(truth_cells), 1)
    bias = len(pred_cells) / max(len(truth_cells), 1)
    far = fp / max(tp + fp, 1)
    csi = tp / max(tp + fp + fn, 1)
    return {
        "grid_hit_rate": round(hit, 4),
        "grid_coverage_bias": round(bias, 4),
        "grid_false_alarm_rate": round(far, 4),
        "grid_csi": round(csi, 4),
        "grid_metrics_applicable": True,
        "truth_grid_count": len(truth_cells),
        "pred_grid_count": len(pred_cells),
        "grid_total_cells": total_cells,
    }


def aggregate_grid_metrics(
    samples: List[dict],
    hazard: str,
    city_lat: float,
    city_lon: float,
    station_catalog: dict,
    prob_key: str = "enhanced",
) -> Dict[str, float | int]:
    rows = []
    for s in samples:
        label_row = s.get("truth_row") or {}
        if not label_row:
            continue
        truth_pts = _truth_station_coords(label_row, hazard, station_catalog)
        pred_prob = float(s.get(prob_key, {}).get(hazard, 0.0))
        pred_cells, truth_cells, total_cells = _pred_truth_grid_sets(
            city_lat=city_lat,
            city_lon=city_lon,
            hazard_prob=pred_prob,
            truth_stations=truth_pts,
        )
        rows.append(grid_hit_bias(pred_cells, truth_cells, total_cells))

    if not rows:
        return {
            "grid_hit_rate": 0.0,
            "grid_coverage_bias": 0.0,
            "grid_false_alarm_rate": 0.0,
            "grid_csi": 0.0,
            "grid_metrics_applicable_hours": 0,
        }

    app = [r for r in rows if bool(r.get("grid_metrics_applicable"))]
    if not app:
        return {
            "grid_hit_rate": 0.0,
            "grid_coverage_bias": 0.0,
            "grid_false_alarm_rate": 0.0,
            "grid_csi": 0.0,
            "grid_metrics_applicable_hours": 0,
        }
    return {
        "grid_hit_rate": round(sum(float(r.get("grid_hit_rate", 0.0)) for r in app) / len(app), 4),
        "grid_coverage_bias": round(sum(float(r.get("grid_coverage_bias", 0.0)) for r in app) / len(app), 4),
        "grid_false_alarm_rate": round(sum(float(r.get("grid_false_alarm_rate", 0.0)) for r in app) / len(app), 4),
        "grid_csi": round(sum(float(r.get("grid_csi", 0.0)) for r in app) / len(app), 4),
        "grid_metrics_applicable_hours": len(app),
    }
