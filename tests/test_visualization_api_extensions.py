from fastapi.testclient import TestClient

import weather_agent.api as api_mod
from weather_agent.api import app


client = TestClient(app)


def _auth():
    return {"Authorization": "Bearer agent-dev-token"}


def test_truth_compare_endpoints(monkeypatch):
    class _Factory:
        def list_versions(self):
            return [{"name": "truth.a"}, {"name": "truth.b"}]

        def get_version(self, truth_version):
            return {"manifest": {"truth_version": truth_version, "record_counts": {"point_truth": 3}}, "snapshot": {"event_truth": []}}

        def compare_versions(self, left_version, right_version):
            return {
                "left": {"truth_version": left_version, "record_counts": {"point_truth": 3}},
                "right": {"truth_version": right_version, "record_counts": {"point_truth": 5}},
                "delta": {"point_truth": {"left": 3, "right": 5, "delta": 2}},
                "same_headline_tier": True,
            }

    monkeypatch.setattr(api_mod, "TruthFactory", lambda: _Factory())

    r1 = client.get("/api/v1/truth/versions", headers=_auth())
    assert r1.status_code == 200
    assert r1.json()["data"]["count"] == 2

    r2 = client.get("/api/v1/truth/versions/truth.a", headers=_auth())
    assert r2.status_code == 200
    assert r2.json()["data"]["manifest"]["truth_version"] == "truth.a"

    r3 = client.post("/api/v1/truth/compare", json={"left_truth_version": "truth.a", "right_truth_version": "truth.b"}, headers=_auth())
    assert r3.status_code == 200
    assert r3.json()["data"]["delta"]["point_truth"]["delta"] == 2


def test_replay_bundle_listing_and_compare(monkeypatch):
    class _Replay:
        def list_bundles(self, limit=20):
            return [{"bundle_id": "bundle.a", "run_id": "run.a"}, {"bundle_id": "bundle.b", "run_id": "run.b"}]

        def replay_bundle(self, bundle_id):
            return {
                "bundle": {"bundle_id": bundle_id},
                "cycle": {"decision": {"hazard_prob": {"wind": 0.7 if bundle_id == "bundle.a" else 0.5}}},
            }

        def compare_bundle(self, bundle_id, replayed_cycle, tolerance=1e-6):
            return {
                "bundle_id": bundle_id,
                "deterministic": False,
                "tolerance": tolerance,
                "hazard_prob": {"wind": {"left": 0.7, "right": 0.5, "delta": 0.2}},
            }

    monkeypatch.setattr(api_mod, "ReplayStore", lambda: _Replay())

    r1 = client.get("/api/v1/replay/bundles?limit=10", headers=_auth())
    assert r1.status_code == 200
    assert r1.json()["data"]["count"] == 2

    r2 = client.post(
        "/api/v1/replay/compare-bundles",
        json={"left_bundle_id": "bundle.a", "right_bundle_id": "bundle.b"},
        headers=_auth(),
    )
    assert r2.status_code == 200
    assert r2.json()["data"]["comparison"]["hazard_prob"]["wind"]["delta"] == 0.2
