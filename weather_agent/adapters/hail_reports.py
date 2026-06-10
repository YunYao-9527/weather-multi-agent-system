from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable

from zoneinfo import ZoneInfo


def _safe_parse_dt(v: str, tz_name: str) -> datetime | None:
    if not v:
        return None
    raw = v.strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        fmts = ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M")
        dt = None
        for f in fmts:
            try:
                dt = datetime.strptime(raw, f)
                break
            except Exception:
                continue
        if dt is None:
            return None

    tz = ZoneInfo(tz_name)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return dt


def _hour_key(dt: datetime) -> str:
    return dt.replace(tzinfo=None, minute=0, second=0, microsecond=0).isoformat()


def _candidate_files(root: Path, city: str) -> Iterable[Path]:
    city_l = city.lower()
    for p in root.glob("*.csv"):
        name = p.stem.lower()
        if city_l in name or "hail" in name:
            yield p


def load_hail_reports(
    city: str,
    start_date: date,
    end_date: date,
    timezone: str = "Asia/Shanghai",
    source_dir: str | Path = "data/hail_reports",
) -> dict:
    root = Path(source_dir)
    if not root.exists():
        return {"by_hour": {}, "files": [], "records": 0}

    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=ZoneInfo(timezone))
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=ZoneInfo(timezone))
    by_hour: Dict[str, int] = {}
    files = []
    total = 0

    for fp in _candidate_files(root, city):
        files.append(str(fp))
        try:
            with fp.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get("time") or row.get("timestamp") or row.get("datetime") or ""
                    dt = _safe_parse_dt(ts, timezone)
                    if not dt:
                        continue
                    if dt < start_dt or dt >= end_dt:
                        continue

                    city_col = (row.get("city") or row.get("location") or "").strip()
                    if city_col and city_col.lower() not in city.lower():
                        # If file contains mixed cities, keep loose matching.
                        if city.lower() not in city_col.lower():
                            continue

                    has_hail = row.get("hail")
                    if has_hail is None or has_hail == "":
                        has_hail = "1"
                    if str(has_hail).strip().lower() in {"0", "false", "no", "n"}:
                        continue

                    key = _hour_key(dt)
                    by_hour[key] = 1
                    total += 1
        except Exception:
            continue

    return {"by_hour": by_hour, "files": files, "records": total}
