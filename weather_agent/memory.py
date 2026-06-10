from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class MemoryManager:
    def __init__(self, root: str | Path = "memory", min_sample_threshold: int = 24):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.weights_file = self.root / "agent_weights.json"
        self.calibrators_file = self.root / "prob_calibrators.json"
        self.profiles_file = self.root / "memory_profiles.json"
        self.min_sample_threshold = int(min_sample_threshold)

    def _read_json(self, fp: Path) -> dict:
        if not fp.exists():
            return {}
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_json(self, fp: Path, payload: dict) -> None:
        fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _profile_key(self, city: str, month: int) -> str:
        return f"{city.lower()}:{int(month)}"

    def _global_key(self, month: int) -> str:
        return f"global:{int(month)}"

    def _resolve_read_key(self, city: str, month: int) -> str:
        profiles = self._read_json(self.profiles_file)
        city_key = self._profile_key(city, month)
        city_profile = profiles.get(city_key, {})
        if int(city_profile.get("sample_count", 0)) >= self.min_sample_threshold:
            return city_key
        global_key = self._global_key(month)
        if global_key in profiles:
            return global_key
        return city_key

    def load_profile(self, city: str, month: int) -> dict | None:
        profiles = self._read_json(self.profiles_file)
        key = self._resolve_read_key(city, month)
        return profiles.get(key)

    def save_profile(
        self,
        city: str,
        month: int,
        *,
        agent_weights: Dict[str, float],
        calibrators: Dict[str, object],
        sample_count: int,
        coverage_ratio: float,
        valid_window: Dict[str, str] | None = None,
        profile_version: str = "v2",
        code_version: str = "nogit",
        shrinkage: float = 1.0,
    ) -> None:
        key = self._profile_key(city, month)
        profiles = self._read_json(self.profiles_file)
        profiles[key] = {
            "key": key,
            "city": city,
            "month": int(month),
            "sample_count": int(sample_count),
            "coverage_ratio": float(coverage_ratio),
            "valid_window": valid_window or {},
            "profile_version": profile_version,
            "code_version": code_version,
            "shrinkage": float(shrinkage),
            "generated_at": datetime.now().isoformat(),
            "agent_weights": agent_weights,
            "calibrators": calibrators,
        }

        # update monthly global profile by sample-weighted smoothing
        gk = self._global_key(month)
        g = profiles.get(gk)
        if g:
            old_n = max(1, int(g.get("sample_count", 0)))
            new_n = max(1, int(sample_count))
            merged_w: Dict[str, float] = {}
            all_agents = set(agent_weights) | set(g.get("agent_weights", {}).keys())
            for a in all_agents:
                old_v = float(g.get("agent_weights", {}).get(a, 1.0))
                new_v = float(agent_weights.get(a, 1.0))
                merged_w[a] = round((old_v * old_n + new_v * new_n) / (old_n + new_n), 6)
            g["agent_weights"] = merged_w
            g["sample_count"] = old_n + new_n
            g["coverage_ratio"] = round((float(g.get("coverage_ratio", 0.0)) * old_n + float(coverage_ratio) * new_n) / (old_n + new_n), 6)
            g["generated_at"] = datetime.now().isoformat()
            # keep calibrators from larger sample profile
            if new_n >= old_n:
                g["calibrators"] = calibrators
            profiles[gk] = g
        else:
            profiles[gk] = {
                "key": gk,
                "city": "global",
                "month": int(month),
                "sample_count": int(sample_count),
                "coverage_ratio": float(coverage_ratio),
                "valid_window": valid_window or {},
                "profile_version": profile_version,
                "code_version": code_version,
                "shrinkage": 1.0,
                "generated_at": datetime.now().isoformat(),
                "agent_weights": agent_weights,
                "calibrators": calibrators,
            }

        self._write_json(self.profiles_file, profiles)

        # compatibility exports
        self.save_weights(city, month, agent_weights)
        self.save_calibrators(city, month, calibrators)

    def load_weights(self, city: str, month: int) -> Dict[str, float] | None:
        profiles = self._read_json(self.profiles_file)
        key = self._resolve_read_key(city, month)
        p = profiles.get(key)
        if p and isinstance(p.get("agent_weights"), dict):
            return p.get("agent_weights")

        data = self._read_json(self.weights_file)
        return data.get(key)

    def save_weights(self, city: str, month: int, weights: Dict[str, float]) -> None:
        data = self._read_json(self.weights_file)
        key = self._profile_key(city, month)
        data[key] = weights
        self._write_json(self.weights_file, data)

    def load_calibrators(self, city: str, month: int) -> Dict[str, object] | None:
        profiles = self._read_json(self.profiles_file)
        key = self._resolve_read_key(city, month)
        p = profiles.get(key)
        if p and isinstance(p.get("calibrators"), dict):
            return p.get("calibrators")

        data = self._read_json(self.calibrators_file)
        return data.get(key)

    def save_calibrators(self, city: str, month: int, calibrators: Dict[str, object]) -> None:
        data = self._read_json(self.calibrators_file)
        key = self._profile_key(city, month)
        data[key] = calibrators
        self._write_json(self.calibrators_file, data)
