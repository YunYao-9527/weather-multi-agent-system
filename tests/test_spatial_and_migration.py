import json
from pathlib import Path

import sqlite3

from weather_agent.migrations import apply_registry_migrations, migrate_memory_profiles
from weather_agent.spatial_eval import admin_hit_bias, grid_hit_bias


def test_admin_hit_bias_basic():
    r = admin_hit_bias(["A", "B"], ["B", "C"])
    assert r["admin_metrics_applicable"] is True
    assert r["admin_hit_rate"] == 0.5
    assert r["admin_coverage_bias"] == 1.0


def test_admin_hit_bias_no_truth():
    r = admin_hit_bias(["A"], [])
    assert r["admin_metrics_applicable"] is False
    assert r["admin_hit_rate"] == 0.0


def test_grid_hit_bias_basic():
    r = grid_hit_bias({1, 2, 3}, {2, 3, 4}, total_cells=10)
    assert r["grid_metrics_applicable"] is True
    assert r["grid_hit_rate"] == 0.6667
    assert r["grid_coverage_bias"] == 1.0
    assert r["grid_false_alarm_rate"] == 0.3333
    assert r["grid_csi"] == 0.5


def test_memory_migration_adds_required_fields(tmp_path: Path):
    fp = tmp_path / "memory_profiles.json"
    fp.write_text(json.dumps({"x:1": {"agent_weights": {"a": 1.0}}}, ensure_ascii=False), encoding="utf-8")
    out = migrate_memory_profiles(fp)
    assert out["exists"] is True
    payload = json.loads(fp.read_text(encoding="utf-8"))
    p = payload["x:1"]
    assert "profile_version" in p
    assert "sample_count" in p
    assert "coverage_ratio" in p


def test_registry_migration_dry_run_keeps_schema_unchanged(tmp_path: Path):
    db = tmp_path / "registry.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE predict_runs(run_id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

    out = apply_registry_migrations(db_path=db, dry_run=True, create_backup=False)
    assert out["dry_run"] is True
    assert out["count"] == 0

    conn = sqlite3.connect(str(db))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(predict_runs)").fetchall()}
    conn.close()
    assert "report_path" not in cols
