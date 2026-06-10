import datetime as dt

from weather_agent.adapters.live_snapshot import _build_radar_visualization_template, _fetch_radar
from weather_agent.agents.radar import RadarAgent
from weather_agent.models import Observation


def _obs(meta: dict) -> Observation:
    return Observation(
        timestamp=dt.datetime(2025, 3, 1, 12, 0),
        city="Tianjin",
        vertical_velocity=-1.0,
        low_level_convergence=0.6,
        cape=1200,
        dcape=800,
        shear_0_6km=16,
        t850_500=24,
        wbz_km=3.8,
        humidity_low=0.7,
        radar_dbz_max=48.0,
        radar_bow_echo=False,
        storm_motion_ms=10.0,
        prob_guidance={"short_rain": 0.5, "wind": 0.4, "hail": 0.3, "tornado": 0.1},
        source_meta=meta,
    )


def test_radar_agent_marks_proxy_from_source_meta():
    agent = RadarAgent()
    out = agent.run(
        _obs(
            {
                "radar_source": "rainviewer",
                "radar_feature_kind": "proxy_tile",
                "radar_proxy_source": "1",
                "radar_provider_used": "rainviewer",
            }
        )
    )
    assert out.evidence.proxy_source is True
    assert out.evidence.supporting_features["source"] == "rainviewer"


def test_radar_agent_marks_real_source_from_source_meta():
    agent = RadarAgent()
    out = agent.run(
        _obs(
            {
                "radar_source": "noaa_nowcoast_nexrad",
                "radar_feature_kind": "grid",
                "radar_proxy_source": "0",
                "radar_provider_used": "nowcoast",
            }
        )
    )
    assert out.evidence.proxy_source is False
    assert out.evidence.supporting_features["provider"] == "nowcoast"


def test_fetch_radar_fallback_chain_returns_attempts(monkeypatch):
    monkeypatch.setenv("AGENT_RADAR_PROVIDER", "auto")
    monkeypatch.setenv("AGENT_RADAR_PROVIDER_PRIORITY", "local_grid,rainviewer")
    monkeypatch.setenv("AGENT_RADAR_GRID_FILE", "not_exists.json")
    monkeypatch.setattr(
        "weather_agent.adapters.live_snapshot.fetch_radar_dbz",
        lambda lat, lon: {"dbz": 35.0, "bow_echo": False, "frame_time": None, "source": "rainviewer"},
    )
    out = _fetch_radar(39.1, 117.2)
    assert "attempts" in out
    assert out.get("selected_provider") in {"rainviewer", "local_grid", ""}


def test_build_radar_visualization_template_uses_same_origin_proxy(monkeypatch):
    monkeypatch.setattr(
        "weather_agent.adapters.live_snapshot.latest_radar_tile_descriptor",
        lambda timeout=12: {"frame_path": "/v2/radar/demo_frame", "frame_time": 1234567890},
    )
    out = _build_radar_visualization_template({"data": {"source": "local_radar_grid"}})
    assert out == "/api/v1/tiles/radar/{z}/{x}/{y}.png?frame_path=%2Fv2%2Fradar%2Fdemo_frame&frame_time=1234567890"
