from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Unsupported json type: {type(obj)!r}")


def dump_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return target


def load_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    text = target.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if default is not None:
            return default
        raise


class MetadataStore(ABC):
    @abstractmethod
    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict]:
        raise NotImplementedError


class SQLiteMetadataStore(MetadataStore):
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._connect() as conn:
            conn.execute(sql, params)

    def executescript(self, script: str) -> None:
        with self._connect() as conn:
            conn.executescript(script)

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


class ManifestObjectStore:
    def __init__(self, root: str | Path = "runs/object_store"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, category: str, object_id: str, payload: Any) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.root / category / object_id / f"{ts}.json"
        return dump_json(path, payload)

    def put_named(self, category: str, name: str, payload: Any) -> Path:
        path = self.root / category / name
        return dump_json(path, payload)

    def latest(self, category: str, object_id: str) -> dict | None:
        folder = self.root / category / object_id
        files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return None
        return load_json(files[0])

    def list_category(self, category: str) -> list[Path]:
        folder = self.root / category
        if not folder.exists():
            return []
        return sorted(folder.rglob("*.json"))


class RegistryStore:
    def __init__(self, root: str | Path = "runs/registry"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _index_path(self, kind: str) -> Path:
        return self.root / kind / "index.json"

    def register(self, kind: str, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        folder = self.root / kind
        folder.mkdir(parents=True, exist_ok=True)
        item_path = folder / f"{name}.json"
        dump_json(item_path, payload)
        index = load_json(self._index_path(kind), default={"items": []})
        items = [x for x in index.get("items", []) if x.get("name") != name]
        items.append(
            {
                "name": name,
                "path": str(item_path),
                "updated_at": datetime.now().isoformat(),
                "version": payload.get("policy_version")
                or payload.get("truth_version")
                or payload.get("feature_version")
                or payload.get("model_version")
                or payload.get("provider_version")
                or payload.get("version")
                or name,
            }
        )
        index["items"] = sorted(items, key=lambda x: x["updated_at"], reverse=True)
        dump_json(self._index_path(kind), index)
        return payload

    def list(self, kind: str) -> List[Dict[str, Any]]:
        return list((load_json(self._index_path(kind), default={"items": []}) or {}).get("items", []))

    def get(self, kind: str, name: str) -> Dict[str, Any] | None:
        path = self.root / kind / f"{name}.json"
        return load_json(path)

    def set_active(self, kind: str, name: str) -> None:
        dump_json(self.root / kind / "active.json", {"name": name, "updated_at": datetime.now().isoformat()})

    def get_active(self, kind: str) -> Dict[str, Any] | None:
        return load_json(self.root / kind / "active.json")


class ObjectRepository:
    def __init__(self, root: str | Path = "runs/objects"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, payload: Dict[str, Any]) -> Path:
        object_id = str(payload.get("object_id") or "object.unknown")
        folder = self.root / object_id
        folder.mkdir(parents=True, exist_ok=True)
        version = int(payload.get("object_version", 1) or 1)
        path = folder / f"v{version:04d}.json"
        return dump_json(path, payload)

    def latest(self, object_id: str) -> Dict[str, Any] | None:
        folder = self.root / object_id
        files = sorted(folder.glob("v*.json"), reverse=True)
        if not files:
            return None
        return load_json(files[0])

    def history(self, object_id: str) -> List[Dict[str, Any]]:
        folder = self.root / object_id
        files = sorted(folder.glob("v*.json"))
        return [load_json(p) for p in files]

    def active(self) -> List[Dict[str, Any]]:
        items: list[dict] = []
        for folder in self.root.iterdir() if self.root.exists() else []:
            if not folder.is_dir():
                continue
            latest = self.latest(folder.name)
            if latest and latest.get("lifecycle_state") not in {"cleared"}:
                items.append(latest)
        return sorted(items, key=lambda x: x.get("last_update_time", ""), reverse=True)


class AuditIndex:
    def __init__(self, root: str | Path = "runs/audit"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "trace_index.json"

    def record(self, request_id: str, payload: Dict[str, Any]) -> None:
        if not request_id:
            return
        index = load_json(self.path, default={"items": {}})
        index.setdefault("items", {})[request_id] = payload
        dump_json(self.path, index)

    def get(self, request_id: str) -> Dict[str, Any] | None:
        data = load_json(self.path, default={"items": {}})
        return data.get("items", {}).get(request_id)
