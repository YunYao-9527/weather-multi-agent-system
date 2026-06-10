from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

from weather_agent.models import GridSpec, TruthRecord, TruthVersion
from weather_agent.storage import RegistryStore, dump_json, load_json
from weather_agent.truth_labels import TruthConfig, build_truth_label_artifact


class TruthFactory:
    def __init__(self, root: str | Path = "runs/truth_factory", registry: RegistryStore | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or RegistryStore()

    def ingest_truth(
        self,
        *,
        city: str,
        start_date: date,
        end_date: date,
        cfg: TruthConfig | None = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        return build_truth_label_artifact(
            city=city,
            start_date=start_date,
            end_date=end_date,
            cfg=cfg or TruthConfig(),
            force_rebuild=force_rebuild,
        )

    def normalize_truth(self, artifact: Dict[str, Any]) -> Dict[str, List[TruthRecord]]:
        meta = artifact.get("meta", {})
        labels = artifact.get("labels_by_hour", {})
        city = str(meta.get("city") or meta.get("city_resolved") or "unknown")
        grid_spec = GridSpec(grid_id=f"grid:{city.lower().replace(' ', '_')}", nx=8, ny=8, dx_km=10.0, dy_km=10.0)
        point_records: list[TruthRecord] = []
        grid_records: list[TruthRecord] = []
        object_records: list[TruthRecord] = []
        event_records: list[TruthRecord] = []

        event_counter = 0
        active_event_by_hazard: dict[str, dict] = {}
        previous_ts = None
        for ts, row in sorted(labels.items()):
            for hazard in ("short_rain", "wind", "hail", "tornado"):
                label = int(row.get(hazard, 0))
                if label <= 0:
                    if hazard in active_event_by_hazard:
                        evt = active_event_by_hazard.pop(hazard)
                        event_records.append(
                            TruthRecord(
                                truth_id=evt["event_id"],
                                truth_layer="event",
                                timestamp=evt["start_time"],
                                city=city,
                                event_id=evt["event_id"],
                                labels={hazard: 1},
                                label_tier=str(evt["label_tier"]),
                                source_meta={"start_time": evt["start_time"], "end_time": previous_ts or ts, "hazard": hazard},
                            )
                        )
                    continue

                truth_id = sha256(f"{city}|{ts}|{hazard}|point".encode("utf-8")).hexdigest()[:16]
                point_records.append(
                    TruthRecord(
                        truth_id=truth_id,
                        truth_layer="point",
                        timestamp=ts,
                        city=city,
                        labels={hazard: 1},
                        label_tier=str(row.get("label_tier", "proxy")),
                        source_meta={"station_ids": row.get("station_ids", []), "sources": row.get("sources", {})},
                    )
                )
                grid_id = f"{grid_spec.grid_id}:c03r03"
                grid_records.append(
                    TruthRecord(
                        truth_id=sha256(f"{city}|{ts}|{hazard}|{grid_id}".encode("utf-8")).hexdigest()[:16],
                        truth_layer="grid",
                        timestamp=ts,
                        city=city,
                        grid_id=grid_id,
                        labels={hazard: 1},
                        label_tier=str(row.get("label_tier", "proxy")),
                        source_meta={"grid_spec": asdict(grid_spec)},
                    )
                )
                object_id = f"truth_obj_{sha256(f'{city}|{hazard}|{ts}'.encode('utf-8')).hexdigest()[:12]}"
                object_records.append(
                    TruthRecord(
                        truth_id=object_id,
                        truth_layer="object",
                        timestamp=ts,
                        city=city,
                        object_id=object_id,
                        labels={hazard: 1},
                        label_tier=str(row.get("label_tier", "proxy")),
                        source_meta={"geometry": "city_center_proxy", "hazard": hazard},
                    )
                )
                active = active_event_by_hazard.get(hazard)
                if not active:
                    event_counter += 1
                    active_event_by_hazard[hazard] = {
                        "event_id": f"event_{city.lower().replace(' ', '_')}_{hazard}_{event_counter:04d}",
                        "start_time": ts,
                        "label_tier": str(row.get("label_tier", "proxy")),
                    }
            previous_ts = ts

        for hazard, evt in list(active_event_by_hazard.items()):
            event_records.append(
                TruthRecord(
                    truth_id=evt["event_id"],
                    truth_layer="event",
                    timestamp=evt["start_time"],
                    city=city,
                    event_id=evt["event_id"],
                    labels={hazard: 1},
                    label_tier=str(evt["label_tier"]),
                    source_meta={"start_time": evt["start_time"], "end_time": previous_ts or evt["start_time"], "hazard": hazard},
                )
            )
        return {
            "point_truth": point_records,
            "grid_truth": grid_records,
            "object_truth": object_records,
            "event_truth": event_records,
        }

    def reconcile_truth(self, normalized: Dict[str, List[TruthRecord]]) -> Dict[str, Any]:
        out: dict[str, Any] = {}
        for layer, records in normalized.items():
            out[layer] = [asdict(r) for r in records]
        out["summary"] = {layer: len(records) for layer, records in normalized.items()}
        return out

    def version_truth(self, *, city: str, period: Dict[str, str], reconciled: Dict[str, Any], headline_tier: str = "gold") -> TruthVersion:
        payload = {
            "city": city,
            "period": period,
            "headline_tier": headline_tier,
            "summary": reconciled.get("summary", {}),
            "sha": sha256(repr(reconciled).encode("utf-8")).hexdigest(),
        }
        truth_version = f"truth.{city.lower().replace(' ', '_')}.{period['start']}.{period['end']}.{payload['sha'][:8]}"
        folder = self.root / truth_version
        folder.mkdir(parents=True, exist_ok=True)
        snapshot_path = dump_json(folder / "snapshot.json", reconciled)
        manifest = TruthVersion(
            truth_version=truth_version,
            created_at=datetime.now().isoformat(),
            city=city,
            period=period,
            headline_tier=headline_tier,
            manifest_path=str(folder / "manifest.json"),
            snapshot_path=str(snapshot_path),
            record_counts=dict(reconciled.get("summary", {})),
            notes=[
                "headline metrics must use gold only",
                "silver may be used for supplemental analysis",
                "proxy truth cannot enter headline evaluation or policy activation",
            ],
        )
        dump_json(folder / "manifest.json", asdict(manifest))
        self.registry.register("truth", truth_version, asdict(manifest))
        return manifest

    def publish_truth_snapshot(
        self,
        *,
        city: str,
        start_date: date,
        end_date: date,
        cfg: TruthConfig | None = None,
        force_rebuild: bool = False,
        headline_tier: str = "gold",
    ) -> Dict[str, Any]:
        artifact = self.ingest_truth(city=city, start_date=start_date, end_date=end_date, cfg=cfg, force_rebuild=force_rebuild)
        normalized = self.normalize_truth(artifact)
        reconciled = self.reconcile_truth(normalized)
        version = self.version_truth(
            city=city,
            period={"start": start_date.isoformat(), "end": end_date.isoformat()},
            reconciled=reconciled,
            headline_tier=headline_tier,
        )
        return {
            "artifact": artifact,
            "normalized_layers": {k: len(v) for k, v in normalized.items()},
            "truth_version": asdict(version),
        }

    def list_versions(self) -> List[Dict[str, Any]]:
        return self.registry.list("truth")

    def get_version(self, truth_version: str) -> Dict[str, Any] | None:
        record = self.registry.get("truth", truth_version)
        if not record:
            return None
        snapshot = load_json(record.get("snapshot_path"), default={})
        manifest = load_json(record.get("manifest_path"), default=record)
        return {"manifest": manifest, "snapshot": snapshot}

    def compare_versions(self, left_version: str, right_version: str) -> Dict[str, Any]:
        left = self.get_version(left_version)
        right = self.get_version(right_version)
        if not left or not right:
            missing = []
            if not left:
                missing.append(left_version)
            if not right:
                missing.append(right_version)
            raise FileNotFoundError(",".join(missing))

        def _summary(entry: Dict[str, Any]) -> Dict[str, Any]:
            manifest = entry.get("manifest", {})
            snapshot = entry.get("snapshot", {})
            return {
                "truth_version": manifest.get("truth_version"),
                "headline_tier": manifest.get("headline_tier"),
                "record_counts": manifest.get("record_counts", {}),
                "event_count": len(snapshot.get("event_truth", [])),
                "object_count": len(snapshot.get("object_truth", [])),
                "grid_count": len(snapshot.get("grid_truth", [])),
                "point_count": len(snapshot.get("point_truth", [])),
            }

        lsum = _summary(left)
        rsum = _summary(right)
        keys = sorted(set(lsum.get("record_counts", {}).keys()) | set(rsum.get("record_counts", {}).keys()))
        deltas = {
            k: {
                "left": int(lsum.get("record_counts", {}).get(k, 0)),
                "right": int(rsum.get("record_counts", {}).get(k, 0)),
                "delta": int(rsum.get("record_counts", {}).get(k, 0)) - int(lsum.get("record_counts", {}).get(k, 0)),
            }
            for k in keys
        }
        return {
            "left": lsum,
            "right": rsum,
            "delta": deltas,
            "same_headline_tier": lsum.get("headline_tier") == rsum.get("headline_tier"),
        }

    def validate_truth_version(self, truth_version: str, headline_tier: str = "gold") -> Dict[str, Any]:
        record = self.registry.get("truth", truth_version)
        if not record:
            return {"ok": False, "issues": ["truth_version_not_found"]}
        issues: list[str] = []
        if headline_tier == "gold" and str(record.get("headline_tier", "gold")) != "gold":
            issues.append("headline_tier_not_gold")
        snap = load_json(record.get("snapshot_path"))
        if not snap:
            issues.append("missing_snapshot")
        return {"ok": not issues, "issues": issues, "truth_version": truth_version}
