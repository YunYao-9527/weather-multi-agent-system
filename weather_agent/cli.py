from __future__ import annotations

import argparse
import json
from datetime import datetime

from weather_agent.demo_data import demo_observation
from weather_agent.orchestrator import ForecastOrchestrator, OrchestratorConfig
from weather_agent.replay import ReplayStore
from weather_agent.serialize import cycle_to_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run severe-convection multi-agent forecast cycle")
    p.add_argument("--city", default="天津")
    p.add_argument("--area", default="全市")
    p.add_argument("--min-issue-prob", type=float, default=0.45)
    p.add_argument("--save", action="store_true", help="save cycle output to runs/*.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    obs = demo_observation(args.city)

    orchestrator = ForecastOrchestrator(
        OrchestratorConfig(
            min_issue_prob=args.min_issue_prob,
            region_name=args.area,
        )
    )

    cycle = orchestrator.run_cycle(obs)
    d = cycle.decision

    print("=== Multi-Agent Warning Decision ===")
    print(f"time: {datetime.now().isoformat(timespec='seconds')}")
    print(f"city: {obs.city}")
    print(f"issue warning: {d.issue}")
    print(f"level: {d.level}")
    print(f"window: {d.start_time.strftime('%H:%M')} - {d.end_time.strftime('%H:%M')}")
    print(f"area: {d.affected_area}")
    print(f"system confidence: {d.confidence:.2f}")
    print("hazard probability:")
    for k, v in d.hazard_prob.items():
        print(f"  - {k}: {v:.2f}")

    if d.conflicts:
        print("conflicts:")
        for c in d.conflicts:
            print(f"  - {c}")

    print("evidence chain:")
    for e in d.rationale:
        print(f"  - [{e.agent}] conf={e.confidence:.2f} | {e.claim}")

    if args.save:
        out = ReplayStore().save(cycle)
        print(f"saved: {out}")

    payload = cycle_to_dict(cycle)
    print("json:")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
