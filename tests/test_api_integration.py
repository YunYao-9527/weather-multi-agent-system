from fastapi.testclient import TestClient

import weather_agent.api as api_mod
from weather_agent.api import app


client = TestClient(app)


def _auth():
    return {"Authorization": "Bearer agent-dev-token"}


def test_manual_forecast_then_audit_lookup():
    body = {
        "city": "Tianjin",
        "area": "测试区域",
        "auto_area": False,
        "min_issue_prob": 0.55,
        "window_minutes": 120,
        "save_run": True,
        "observation": {
            "vertical_velocity": -1.2,
            "low_level_convergence": 0.72,
            "cape": 1850.0,
            "dcape": 920.0,
            "shear_0_6km": 18.0,
            "t850_500": 25.2,
            "wbz_km": 3.9,
            "humidity_low": 0.78,
            "radar_dbz_max": 56.0,
            "radar_bow_echo": True,
            "storm_motion_ms": 13.0,
            "prob_guidance": {
                "short_rain": 0.71,
                "wind": 0.63,
                "hail": 0.36,
                "tornado": 0.08,
            },
        },
    }

    r = client.post('/api/v1/forecast/manual', json=body, headers=_auth())
    assert r.status_code == 200
    payload = r.json()["data"]
    run_id = payload.get("run_id")
    assert run_id

    r2 = client.get(f'/api/v1/audit/{run_id}', headers=_auth())
    assert r2.status_code == 200
    audit = r2.json()["data"]
    assert audit.get("run_id") == run_id

    r3 = client.get('/api/v1/registry/predict/recent?limit=3&city=Tianjin', headers=_auth())
    assert r3.status_code == 200
    items = r3.json()["data"]["items"]
    assert isinstance(items, list)
    assert len(items) >= 1
    assert "readiness" in items[0]


def test_window_scan_api_with_mock(monkeypatch):
    monkeypatch.setattr(
        api_mod,
        "run_window_scan",
        lambda cfg: {
            "reports": {"json": "x.json", "markdown": "x.md"},
            "result": {"summary": {"window_count_scanned": 10, "window_count_passed": 2}},
        },
    )
    body = {
        "city": "Tianjin",
        "search_start": "2025-03-01",
        "search_end": "2025-04-01",
        "window_days": 3,
        "step_days": 1,
        "min_truth_coverage": 0.6,
        "min_total_positive_labels": 5,
        "min_train_positive_labels": 1,
        "min_calibration_positive_labels": 1,
        "headline_tiers": ["gold", "silver"],
        "top_k": 10,
    }
    r = client.post('/api/v1/windows/scan', json=body, headers=_auth())
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["result"]["summary"]["window_count_scanned"] == 10


def test_radar_tile_proxy_returns_png(monkeypatch):
    monkeypatch.setattr(api_mod, "fetch_weather_maps_manifest", lambda timeout=12: {"host": "https://tile.example"})
    monkeypatch.setattr(api_mod, "latest_radar_frame", lambda manifest: {"path": "/v2/radar/demo"})

    class _Resp:
        content = b"png-bytes"
        headers = {"content-type": "image/png"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(api_mod.requests, "get", lambda url, timeout=20: _Resp())

    r = client.get("/api/v1/tiles/radar/7/106/49.png")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/png")
    assert r.headers["x-radar-upstream"].endswith("/v2/radar/demo/256/7/106/49/2/1_1.png")


def test_llm_runtime_endpoint_updates_mode_and_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(api_mod.SETTINGS, "llm_agent_enabled", False)
    monkeypatch.setattr(api_mod.SETTINGS, "llm_agent_mode", "shadow")
    monkeypatch.setattr(api_mod.SETTINGS, "llm_provider", "openai")
    monkeypatch.setattr(api_mod.SETTINGS, "llm_model", "gpt-4o-mini")

    r0 = client.get("/api/v1/runtime/llm", headers=_auth())
    assert r0.status_code == 200
    assert r0.json()["data"]["effective_mode"] == "off"

    r1 = client.post(
        "/api/v1/runtime/llm",
        json={"mode": "shadow", "model": "gpt-4o-mini", "api_key": "test-key"},
        headers=_auth(),
    )
    assert r1.status_code == 200
    data = r1.json()["data"]
    assert data["effective_mode"] == "shadow"
    assert data["ready"] is True
    assert data["key_present"] is True

    r2 = client.post("/api/v1/runtime/llm", json={"mode": "off"}, headers=_auth())
    assert r2.status_code == 200
    assert r2.json()["data"]["effective_mode"] == "off"


def test_llm_runtime_endpoint_supports_deepseek_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(api_mod.SETTINGS, "llm_agent_enabled", False)
    monkeypatch.setattr(api_mod.SETTINGS, "llm_agent_mode", "shadow")
    monkeypatch.setattr(api_mod.SETTINGS, "llm_provider", "openai")
    monkeypatch.setattr(api_mod.SETTINGS, "llm_model", "gpt-4o-mini")

    r = client.post(
        "/api/v1/runtime/llm",
        json={"provider": "deepseek", "mode": "shadow", "model": "deepseek-chat", "api_key": "deepseek-test-key"},
        headers=_auth(),
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["provider"] == "deepseek"
    assert data["effective_mode"] == "shadow"
    assert data["key_present"] is True
    assert data["ready"] is True
