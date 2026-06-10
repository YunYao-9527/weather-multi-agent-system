from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from weather_agent.models import CycleResult
from weather_agent.serialize import cycle_to_dict
from weather_agent.storage import AuditIndex, dump_json, load_json


def _compare_numeric_dict(a: Dict[str, Any], b: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    keys = sorted(set(a.keys()) | set(b.keys()))
    diffs = {}
    ok = True
    for key in keys:
        av = float(a.get(key, 0.0))
        bv = float(b.get(key, 0.0))
        delta = abs(av - bv)
        if delta > tolerance:
            ok = False
        diffs[key] = {"left": av, "right": bv, "delta": round(delta, 8)}
    return {"ok": ok, "diff": diffs}


class ReplayStore:
    def __init__(self, root: str | Path = "runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cycle_dir = self.root / "cycles"
        self.audit_dir = self.root / "audit"
        self.bundle_dir = self.root / "replay_bundles"
        self.cycle_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.audit_index = AuditIndex(self.audit_dir)

    def save(self, result: CycleResult) -> Path:
        run_id = (result.audit.run_id if result.audit else "").strip() or result.generated_at.strftime("%Y%m%d_%H%M%S")
        path = self.cycle_dir / f"cycle_{run_id}.json"
        payload = cycle_to_dict(result)
        dump_json(path, payload)

        if result.audit:
            ap = self.audit_dir / f"audit_{run_id}.json"
            dump_json(ap, asdict(result.audit))
            self.audit_index.record(
                result.audit.request_id,
                {
                    "run_id": run_id,
                    "bundle_id": result.audit.bundle_id,
                    "trace_id": result.audit.trace_id,
                    "policy_version": result.audit.policy_version,
                    "generated_at": result.audit.generated_at,
                },
            )

        if result.replay_bundle:
            bundle_id = result.replay_bundle.bundle_id
            folder = self.bundle_dir / bundle_id
            folder.mkdir(parents=True, exist_ok=True)
            dump_json(folder / "bundle.json", asdict(result.replay_bundle))
            dump_json(folder / "cycle.json", payload)
        return path

    def load_cycle(self, run_id: str) -> Dict[str, Any] | None:
        p = self.cycle_dir / f"cycle_{run_id}.json"
        return load_json(p)

    def load_audit(self, run_id: str) -> Dict[str, Any] | None:
        p = self.audit_dir / f"audit_{run_id}.json"
        return load_json(p)

    def load_bundle(self, bundle_id: str) -> Dict[str, Any] | None:
        return load_json(self.bundle_dir / bundle_id / "bundle.json")

    def latest_run_id(self) -> str | None:
        files = sorted(self.cycle_dir.glob("cycle_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not files:
            return None
        return files[0].stem.replace("cycle_", "", 1)

    def trace(self, request_id: str) -> Dict[str, Any] | None:
        rec = self.audit_index.get(request_id)
        if not rec:
            return None
        out = dict(rec)
        run_id = rec.get("run_id")
        if run_id:
            out["audit"] = self.load_audit(run_id)
            out["cycle"] = self.load_cycle(run_id)
        bundle_id = rec.get("bundle_id")
        if bundle_id:
            out["bundle"] = self.load_bundle(bundle_id)
        return out

    def compare_runs(self, baseline_run_id: str, enhanced_run_id: str) -> Dict[str, Any]:
        b = self.load_cycle(baseline_run_id)
        e = self.load_cycle(enhanced_run_id)
        if not b or not e:
            raise FileNotFoundError("baseline or enhanced run not found")

        b_prob = b.get("decision", {}).get("hazard_prob", {})
        e_prob = e.get("decision", {}).get("hazard_prob", {})
        hazards = sorted(set(b_prob.keys()) | set(e_prob.keys()))
        diff = {
            h: {
                "baseline": float(b_prob.get(h, 0.0)),
                "enhanced": float(e_prob.get(h, 0.0)),
                "delta": round(float(e_prob.get(h, 0.0)) - float(b_prob.get(h, 0.0)), 4),
            }
            for h in hazards
        }
        return {
            "baseline_run_id": baseline_run_id,
            "enhanced_run_id": enhanced_run_id,
            "hazard_prob_diff": diff,
            "issue_changed": bool(b.get("decision", {}).get("issue") != e.get("decision", {}).get("issue")),
            "level_changed": bool(b.get("decision", {}).get("level") != e.get("decision", {}).get("level")),
            "readiness_delta": round(
                float(e.get("decision", {}).get("evidence_readiness_score", 0.0))
                - float(b.get("decision", {}).get("evidence_readiness_score", 0.0)),
                4,
            ),
        }

    def replay_bundle(self, bundle_id: str) -> Dict[str, Any]:
        bundle = self.load_bundle(bundle_id)
        if not bundle:
            raise FileNotFoundError(bundle_id)
        cycle = load_json(self.bundle_dir / bundle_id / "cycle.json", default={})
        return {"bundle": bundle, "cycle": cycle}

    def list_bundles(self, limit: int = 20) -> list[dict]:
        folders = sorted([p for p in self.bundle_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for folder in folders[: max(1, limit)]:
            bundle = load_json(folder / "bundle.json", default={})
            cycle = load_json(folder / "cycle.json", default={})
            out.append(
                {
                    "bundle_id": folder.name,
                    "run_id": bundle.get("run_id") or cycle.get("audit", {}).get("run_id"),
                    "request_id": bundle.get("request_id"),
                    "trace_id": bundle.get("trace_id"),
                    "generated_at": cycle.get("generated_at"),
                    "action": cycle.get("decision", {}).get("action"),
                    "level": cycle.get("decision", {}).get("level"),
                }
            )
        return out

    def compare_bundle(self, bundle_id: str, replayed_cycle: Dict[str, Any], tolerance: float = 1e-6) -> Dict[str, Any]:
        stored = load_json(self.bundle_dir / bundle_id / "cycle.json", default={})
        result = _compare_numeric_dict(
            stored.get("decision", {}).get("hazard_prob", {}),
            replayed_cycle.get("decision", {}).get("hazard_prob", {}),
            tolerance,
        )
        return {
            "bundle_id": bundle_id,
            "deterministic": result["ok"],
            "tolerance": tolerance,
            "hazard_prob": result["diff"],
        }
