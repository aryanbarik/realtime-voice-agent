"""End-of-turn sweep: python -m eval.sweep_eot --profile v3-hosted --scenarios 'long_pause-*,simple-*'."""

from __future__ import annotations

import argparse
import asyncio

from eval.run_eval import run_label, select_scenarios


def sweep_points() -> list[tuple[str, dict]]:
    pts = [
        (f"eot-silence-{ms}", {"turn": {"detector": "silence", "silence_ms": ms}})
        for ms in (300, 500, 700, 1000)
    ]
    pts += [
        (f"eot-smart-{th}", {"turn": {"detector": "smart_turn", "smart_turn_threshold": th}})
        for th in (0.3, 0.5, 0.7)
    ]
    return pts


async def run_sweep(profile: str, patterns: str) -> None:
    scenarios = select_scenarios(patterns)
    for label, overrides in sweep_points():
        res = await run_label(profile=profile, scenarios=scenarios, label=label, overrides=overrides)
        s = res["summary"]
        print(f"{label}: success={s['task_success_rate']} wait_p50={s['endpoint_wait_ms_p50']} "
              f"false_cut_in={s['false_cut_in_rate']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="v3-hosted")
    ap.add_argument("--scenarios", default="long_pause-*,simple-*")
    a = ap.parse_args()
    asyncio.run(run_sweep(a.profile, a.scenarios))


if __name__ == "__main__":
    main()
