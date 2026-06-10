from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict

from weather_agent.storage import RegistryStore, SQLiteMetadataStore


@dataclass
class RunMeta:
    run_id: str
    created_at: str
    config_hash: str
    code_hash: str


def _hash_payload(payload: Dict[str, Any]) -> str:
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _code_hash() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if proc.returncode == 0:
            h = (proc.stdout or "").strip()
            if h:
                return h
    except Exception:
        pass
    core = ["weather_agent/api.py", "weather_agent/orchestrator.py", "weather_agent/fusion.py", "weather_agent/evaluator.py"]
    payload = ""
    for p in core:
        fp = Path(p)
        if fp.exists():
            try:
                payload += fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
    return "local_" + sha256(payload.encode("utf-8")).hexdigest()[:8] if payload else "nogit"


class ExperimentRegistry:
    def __init__(self, db_path: str | Path = "runs/registry.db"):
        self.db_path = Path(db_path)
        self.db = SQLiteMetadataStore(self.db_path)
        self.registry = RegistryStore()
        self._init_db()
        self._bootstrap_registries()

    def _init_db(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS predict_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                city TEXT NOT NULL,
                mode TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                data_window_start TEXT,
                data_window_end TEXT,
                level TEXT,
                action TEXT,
                issue INTEGER,
                degraded_mode INTEGER,
                report_path TEXT,
                metadata_json TEXT
            );
            CREATE TABLE IF NOT EXISTS eval_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                city TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                period_start TEXT,
                period_end TEXT,
                samples INTEGER,
                qualified_coverage REAL,
                target_met INTEGER,
                report_json TEXT,
                metadata_json TEXT
            );
            CREATE TABLE IF NOT EXISTS evolve_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                city TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                period_start TEXT,
                period_end TEXT,
                train_samples INTEGER,
                calibration_samples INTEGER,
                test_samples INTEGER,
                fallback_reason TEXT,
                metadata_json TEXT
            );
            CREATE TABLE IF NOT EXISTS memory_profiles (
                profile_key TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                city TEXT,
                month INTEGER,
                sample_count INTEGER,
                coverage_ratio REAL,
                profile_version TEXT,
                code_hash TEXT,
                payload_json TEXT
            );
            """
        )

    def _bootstrap_registries(self) -> None:
        self.registry.register("models", "multi_agent_rules.v3", {"model_version": "multi_agent_rules.v3", "kind": "heuristic_multi_agent"})
        self.registry.register("features", "feature_schema.v1", {"feature_version": "feature_schema.v1", "schema": "ObservationEnvelope"})
        self.registry.register("providers", "provider_schema.v1", {"provider_version": "provider_schema.v1", "tiers": ["official", "experimental", "proxy"]})

    def build_run_meta(self, payload: Dict[str, Any], prefix: str) -> RunMeta:
        now = datetime.now().isoformat()
        cfg_hash = _hash_payload(payload)
        run_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg_hash[:6]}"
        return RunMeta(run_id=run_id, created_at=now, config_hash=cfg_hash, code_hash=_code_hash())

    def record_predict(self, meta: RunMeta, city: str, mode: str, decision: Dict[str, Any], data_window: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        self.db.execute(
            """
            INSERT OR REPLACE INTO predict_runs (
                run_id, created_at, city, mode, config_hash, code_hash,
                data_window_start, data_window_end, level, action, issue,
                degraded_mode, report_path, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta.run_id,
                meta.created_at,
                city,
                mode,
                meta.config_hash,
                meta.code_hash,
                data_window.get("start"),
                data_window.get("end"),
                decision.get("level"),
                decision.get("action"),
                1 if decision.get("issue") else 0,
                1 if decision.get("degraded_mode") else 0,
                metadata.get("saved_path"),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )

    def record_eval(self, meta: RunMeta, city: str, result: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        self.db.execute(
            """
            INSERT OR REPLACE INTO eval_runs (
                run_id, created_at, city, config_hash, code_hash,
                period_start, period_end, samples, qualified_coverage,
                target_met, report_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta.run_id,
                meta.created_at,
                city,
                meta.config_hash,
                meta.code_hash,
                result.get("period", {}).get("start"),
                result.get("period", {}).get("end"),
                int(result.get("samples", 0)),
                float(result.get("truth_labels", {}).get("qualified_coverage_ratio", 0.0)),
                None,
                json.dumps(result.get("reports", {}), ensure_ascii=False),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )

    def record_evolve(self, meta: RunMeta, city: str, result: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        qc = result.get("qualified_counts", {})
        self.db.execute(
            """
            INSERT OR REPLACE INTO evolve_runs (
                run_id, created_at, city, config_hash, code_hash,
                period_start, period_end, train_samples, calibration_samples,
                test_samples, fallback_reason, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta.run_id,
                meta.created_at,
                city,
                meta.config_hash,
                meta.code_hash,
                result.get("trained_period", {}).get("start"),
                result.get("trained_period", {}).get("end"),
                int(qc.get("train", 0)),
                int(qc.get("calibration", 0)),
                int(qc.get("test", 0)),
                str(result.get("fallback_reason", "")),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )

    def record_memory_profile(self, profile_key: str, payload: Dict[str, Any]) -> None:
        self.db.execute(
            """
            INSERT OR REPLACE INTO memory_profiles (
                profile_key, created_at, city, month, sample_count, coverage_ratio,
                profile_version, code_hash, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_key,
                datetime.now().isoformat(),
                payload.get("city"),
                int(payload.get("month", 0)),
                int(payload.get("sample_count", 0)),
                float(payload.get("coverage_ratio", 0.0)),
                str(payload.get("profile_version", "v1")),
                _code_hash(),
                json.dumps(payload, ensure_ascii=False),
            ),
        )

    def _fetch_one(self, table: str, run_id: str) -> dict | None:
        rows = self.db.query(f"SELECT * FROM {table} WHERE run_id = ?", (run_id,))
        return rows[0] if rows else None

    def fetch_audit_run(self, run_id: str) -> dict | None:
        return self._fetch_one("predict_runs", run_id)

    def fetch_eval_run(self, run_id: str) -> dict | None:
        return self._fetch_one("eval_runs", run_id)

    def fetch_evolve_run(self, run_id: str) -> dict | None:
        return self._fetch_one("evolve_runs", run_id)

    def fetch_recent_predict_runs(self, limit: int = 3, city: str | None = None) -> list[dict]:
        lim = max(1, min(int(limit), 20))
        sql = "SELECT * FROM predict_runs"
        params: tuple[Any, ...]
        if city:
            sql += " WHERE city = ? ORDER BY created_at DESC LIMIT ?"
            params = (city, lim)
        else:
            sql += " ORDER BY created_at DESC LIMIT ?"
            params = (lim,)
        rows = self.db.query(sql, params)
        out: list[dict] = []
        for rec in rows:
            meta = {}
            try:
                meta = json.loads(rec.get("metadata_json") or "{}")
            except Exception:
                meta = {}
            rec["metadata"] = meta
            out.append(rec)
        return out

    def list_models(self) -> list[dict]:
        return self.registry.list("models")

    def list_policies(self) -> list[dict]:
        return self.registry.list("policies")

    def list_features(self) -> list[dict]:
        return self.registry.list("features")

    def list_truth(self) -> list[dict]:
        return self.registry.list("truth")

    def list_providers(self) -> list[dict]:
        return self.registry.list("providers")
