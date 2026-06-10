from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


MigrationFn = Callable[[sqlite3.Connection], list[str]]


@dataclass
class MigrationSpec:
    version: str
    description: str
    fn: MigrationFn


def _ensure_migration_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now')),
            description TEXT
        )
        """
    )


def _has_migration(conn: sqlite3.Connection, version: str) -> bool:
    row = conn.execute("SELECT 1 FROM schema_migrations WHERE version = ?", (version,)).fetchone()
    return bool(row)


def _mark_migration(conn: sqlite3.Connection, version: str, description: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO schema_migrations(version, description) VALUES (?, ?)",
        (version, description),
    )


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _m001_predict_metadata(conn: sqlite3.Connection) -> list[str]:
    ops = []
    cols = _table_columns(conn, "predict_runs")
    if "report_path" not in cols:
        conn.execute("ALTER TABLE predict_runs ADD COLUMN report_path TEXT")
        ops.append("predict_runs.report_path")
    if "metadata_json" not in cols:
        conn.execute("ALTER TABLE predict_runs ADD COLUMN metadata_json TEXT")
        ops.append("predict_runs.metadata_json")
    return ops


def _m002_eval_nullable_target(conn: sqlite3.Connection) -> list[str]:
    # schema-compatible noop for SQLite; target_met can already be null.
    conn.execute("SELECT 1")
    return ["eval_runs.target_met_nullable_verified"]


def _m003_registry_indexes(conn: sqlite3.Connection) -> list[str]:
    ops: list[str] = []
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "predict_runs" in tables:
        cols = _table_columns(conn, "predict_runs")
        if {"city", "created_at"}.issubset(cols):
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predict_city_time ON predict_runs(city, created_at DESC)")
            ops.append("idx_predict_city_time")
    if "eval_runs" in tables:
        cols = _table_columns(conn, "eval_runs")
        if {"city", "created_at"}.issubset(cols):
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_city_time ON eval_runs(city, created_at DESC)")
            ops.append("idx_eval_city_time")
    if "evolve_runs" in tables:
        cols = _table_columns(conn, "evolve_runs")
        if {"city", "created_at"}.issubset(cols):
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evolve_city_time ON evolve_runs(city, created_at DESC)")
            ops.append("idx_evolve_city_time")
    return ops


def _migration_specs() -> list[MigrationSpec]:
    return [
        MigrationSpec("001_predict_metadata", "add predict report/metadata columns", _m001_predict_metadata),
        MigrationSpec("002_eval_nullable_target", "allow nullable target gate", _m002_eval_nullable_target),
        MigrationSpec("003_registry_indexes", "add registry query indexes", _m003_registry_indexes),
    ]


def _backup_file(path: Path, suffix: str = "bak") -> str | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".{stamp}.{suffix}")
    shutil.copy2(path, backup)
    return str(backup)


def apply_registry_migrations(
    db_path: str | Path = "runs/registry.db",
    *,
    dry_run: bool = False,
    create_backup: bool = True,
) -> dict:
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    backup_path = _backup_file(db) if create_backup and not dry_run else None

    work_db = db
    if dry_run:
        work_db = db.with_suffix(db.suffix + ".dryrun.tmp")
        if db.exists():
            shutil.copy2(db, work_db)
        elif work_db.exists():
            work_db.unlink(missing_ok=True)

    conn = sqlite3.connect(str(work_db))
    try:
        _ensure_migration_table(conn)
        applied = []
        skipped = []
        operations: list[str] = []
        for spec in _migration_specs():
            if _has_migration(conn, spec.version):
                skipped.append(spec.version)
                continue
            operations.extend(spec.fn(conn))
            if not dry_run:
                _mark_migration(conn, spec.version, spec.description)
                applied.append(spec.version)

        if dry_run:
            conn.rollback()
        else:
            conn.commit()
        result = {
            "db_path": str(db),
            "work_db_path": str(work_db),
            "dry_run": dry_run,
            "backup_path": backup_path,
            "applied": applied,
            "skipped": skipped,
            "operations": operations,
            "count": len(applied),
        }
        return result
    finally:
        conn.close()
        if dry_run and work_db != db:
            work_db.unlink(missing_ok=True)


def migrate_memory_profiles(
    path: str | Path = "memory/memory_profiles.json",
    *,
    dry_run: bool = False,
    create_backup: bool = True,
) -> dict:
    fp = Path(path)
    if not fp.exists():
        return {"path": str(fp), "updated": 0, "exists": False, "dry_run": dry_run, "backup_path": None}

    try:
        raw = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {"path": str(fp), "updated": 0, "exists": True, "error": "invalid_json", "dry_run": dry_run}

    updated = 0
    touched_keys: list[str] = []
    for key, p in (raw or {}).items():
        if not isinstance(p, dict):
            continue
        changed = False
        if "profile_version" not in p:
            p["profile_version"] = "memory.v2"
            changed = True
        if "generated_at" not in p:
            p["generated_at"] = ""
            changed = True
        if "coverage_ratio" not in p:
            p["coverage_ratio"] = 0.0
            changed = True
        if "sample_count" not in p:
            p["sample_count"] = 0
            changed = True
        if "valid_window" not in p:
            p["valid_window"] = {}
            changed = True
        if "key" not in p:
            p["key"] = key
            changed = True
        if "schema_version" not in p:
            p["schema_version"] = "memory_profile.v2"
            changed = True
        if "code_version" not in p:
            p["code_version"] = "unknown"
            changed = True
        if changed:
            updated += 1
            touched_keys.append(key)

    backup_path = _backup_file(fp) if create_backup and updated > 0 and not dry_run else None
    if updated > 0 and not dry_run:
        fp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "path": str(fp),
        "updated": updated,
        "exists": True,
        "dry_run": dry_run,
        "backup_path": backup_path,
        "touched_keys": touched_keys,
    }


def run_all_migrations(
    *,
    db_path: str | Path = "runs/registry.db",
    memory_path: str | Path = "memory/memory_profiles.json",
    dry_run: bool = False,
    create_backup: bool = True,
) -> dict:
    reg = apply_registry_migrations(db_path=db_path, dry_run=dry_run, create_backup=create_backup)
    mem = migrate_memory_profiles(path=memory_path, dry_run=dry_run, create_backup=create_backup)
    return {
        "generated_at": datetime.now().isoformat(),
        "dry_run": dry_run,
        "registry": reg,
        "memory": mem,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run registry/memory schema migrations")
    parser.add_argument("--db-path", default="runs/registry.db")
    parser.add_argument("--memory-path", default="memory/memory_profiles.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    out = run_all_migrations(
        db_path=args.db_path,
        memory_path=args.memory_path,
        dry_run=bool(args.dry_run),
        create_backup=not bool(args.no_backup),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
