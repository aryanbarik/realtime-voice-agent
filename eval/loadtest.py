"""Concurrency load test: python -m eval.loadtest --profile v4-hosted --levels 1,5,10,20 --scenarios 'simple-*'."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json

from eval.run_eval import RESULTS, run_label, select_scenarios


async def run_levels(profile: str, levels: list[int], patterns: str, overrides: dict) -> dict:
    base = select_scenarios(patterns)
    out: dict[str, dict] = {}
    for level in levels:
        n = max(level, 6)
        chosen = list(itertools.islice(itertools.cycle(base), n))
        res = await run_label(
            profile=profile, scenarios=chosen, label=f"loadtest-{profile}-L{level}",
            overrides=overrides, concurrency=level,
        )
        calls = res["scenarios"]
        out[str(level)] = {
            "client_v2v_p50": res["summary"]["client_v2v_p50"],
            "client_v2v_p95": res["summary"]["client_v2v_p95"],
            "success_rate": res["summary"]["task_success_rate"],
            "errors": sum(1 for c in calls if c.get("error")),
        }
        print(f"level {level}: {out[str(level)]}", flush=True)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"loadtest-{profile}.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--levels", default="1,5,10,20")
    ap.add_argument("--scenarios", default="simple-*")
    ap.add_argument("--overrides", default="{}")
    a = ap.parse_args()
    asyncio.run(run_levels(a.profile, [int(x) for x in a.levels.split(",")], a.scenarios, json.loads(a.overrides)))


if __name__ == "__main__":
    main()
