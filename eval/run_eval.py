"""Eval runner: spawns the agent server, runs simulated callers, grades from DB state, summarizes."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import fnmatch
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import yaml
from openai import AsyncOpenAI

from eval.grade import call_wer, entity_counts, grade
from eval.scenario import Scenario, load_scenarios
from eval.simulate import SimulatedCaller, load_voices
from server.cost import cost_from_totals, scale_to_3min, selfhosted_cost_usd
from server.config import Settings, load_pipeline_config

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PRICES_PATH = Path(__file__).parent / "prices.yaml"
EVAL_TODAY = "2026-11-23"
CALL_TIMEOUT_S = 330

SERVER_STAGES = ("endpoint_wait_ms", "asr_finalize_ms", "llm_ttft_ms", "tool_ms", "tts_ttfa_ms", "send_ms")


def pct(values: list[float], q: float) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(float(np.percentile(vals, q, method="linear")), 1)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def make_tenant(label: str, scenario_id: str) -> str:
    base = re.sub(r"[^a-z0-9-]", "-", f"{label}-{scenario_id}".lower())[:57]
    return f"{base}-{uuid.uuid4().hex[:6]}"


def select_scenarios(patterns: str | None) -> list[Scenario]:
    scenarios = load_scenarios()
    if not patterns:
        return scenarios
    pats = [p.strip() for p in patterns.split(",") if p.strip()]
    chosen = [s for s in scenarios if any(fnmatch.fnmatch(s.id, p) for p in pats)]
    if not chosen:
        raise SystemExit(f"no scenarios match {patterns!r}")
    return chosen


class AgentServer:
    """Our server in a subprocess (always started: it owns the backend and admin API)."""

    def __init__(self, profile: str, overrides: dict, out_dir: Path):
        self.port = free_port()
        self.admin_token = secrets.token_hex(16)
        self.out_dir = out_dir
        self.env = {
            **os.environ,
            "VA_PROFILE": profile,
            "VA_OVERRIDES": json.dumps(overrides),
            "VA_TODAY": EVAL_TODAY,
            "VA_DB_PATH": str((out_dir / "backend.db").resolve()),
            "VA_TRACE_DIR": str((out_dir / "traces").resolve()),
            "VA_ADMIN_TOKEN": self.admin_token,
            "VA_PUBLIC_DEMO": "0",
        }
        self.proc: subprocess.Popen | None = None

    @property
    def http_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws"

    async def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.out_dir / "server.log", "wb")
        self.proc = subprocess.Popen(
            ["uv", "run", "uvicorn", "server.app:app", "--port", str(self.port), "--log-level", "warning"],
            cwd=ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        deadline = time.monotonic() + 60
        async with httpx.AsyncClient() as http:
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"server exited early; see {self.out_dir / 'server.log'}")
                try:
                    if (await http.get(f"{self.http_url}/healthz")).status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                await asyncio.sleep(0.5)
        self.stop()
        raise RuntimeError("server /healthz did not come up within 60 s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _failed_call(scenario: Scenario, err: str) -> dict:
    return {
        "scenario_id": scenario.id, "end_reason": "error", "duration_s": 0.0, "caller_lines": [],
        "user_finals": [], "agent_finals": [], "turns": [], "turn_metrics": [], "tools": [],
        "usage": None, "error": err,
    }


async def run_one(scenario, label, server, agent_url, caller, admin_http) -> dict:
    tenant = make_tenant(label, scenario.id)
    admin = f"/backend/admin/tenants/{tenant}"
    try:
        r = await admin_http.put(admin, json=scenario.seed.model_dump())
        r.raise_for_status()
        ws_url = agent_url or server.ws_url
        extra = {"backend": f"{server.http_url}/backend"} if agent_url else None
        try:
            res = await asyncio.wait_for(caller.run(ws_url, scenario, tenant, extra), CALL_TIMEOUT_S)
            out = dataclasses.asdict(res)
        except Exception as e:  # a failed call is a failed scenario, not a failed run
            out = _failed_call(scenario, f"{type(e).__name__}: {e}")
        rows = (await admin_http.get(f"{admin}/reservations")).json()
        out["grade"] = grade(scenario, rows)
        out["reservations"] = rows
    except Exception as e:
        out = _failed_call(scenario, f"{type(e).__name__}: {e}")
        out["grade"] = {"success": False, "reasons": [out["error"]]}
        out["reservations"] = []
    out["scenario_id"] = scenario.id
    out["category"] = scenario.category
    out["tenant"] = tenant
    return out


def load_prices() -> dict:
    return yaml.safe_load(PRICES_PATH.read_text())


def selfhost_capacity() -> int:
    """Highest load-test level of v4-selfhosted with client p95 < 1200 ms; 1 if there is no file."""
    path = RESULTS / "loadtest-v4-selfhosted.json"
    if not path.exists():
        return 1
    data = json.loads(path.read_text())
    ok = [int(k) for k, v in data.items() if v.get("client_v2v_p95") is not None and v["client_v2v_p95"] < 1200]
    return max(ok) if ok else 1


def call_cost_usd(profile: str, overrides: dict, call: dict, prices: dict, n_calls_per_gpu: int) -> float | None:
    """Cost of one call scaled to 180 s (simulated-caller spend excluded)."""
    dur = call.get("duration_s") or 0
    if dur <= 0:
        return None
    cfg = load_pipeline_config(Settings(va_profile=profile, va_overrides=json.dumps(overrides)))
    if cfg.llm.provider == "vllm":
        return scale_to_3min(selfhosted_cost_usd(dur, n_calls_per_gpu, prices=prices), dur)
    usage = call.get("usage")
    if not usage or usage.get("asr_audio_s") is None:
        return None
    usd = cost_from_totals(
        usage["asr_audio_s"], usage, llm_model=cfg.llm.model,
        batch_asr=cfg.mode == "batch", prices=prices,
    )
    return scale_to_3min(usd, dur)


def summarize(profile: str, overrides: dict, calls: list[dict], scenarios: dict[str, Scenario]) -> dict:
    turns = [t for c in calls for t in c.get("turns", [])]
    client_v2v = [t["client_v2v_ms"] for t in turns if t.get("client_v2v_ms") is not None]
    metrics = [m for c in calls for m in c.get("turn_metrics", [])]
    derived = [m.get("derived", {}) for m in metrics]
    server_v2v = [d["v2v_ms"] for d in derived if d.get("v2v_ms") is not None]

    by_turn = {}
    for c in calls:
        for m in c.get("turn_metrics", []):
            by_turn[(c["scenario_id"], m.get("turn_id"))] = m.get("derived", {}).get("v2v_ms")
    network = []
    for c in calls:
        for t in c.get("turns", []):
            s = by_turn.get((c["scenario_id"], t.get("agent_turn_id")))
            if s is not None and t.get("client_v2v_ms") is not None:
                network.append(t["client_v2v_ms"] - s)

    non_interrupt = [t for t in turns if not t.get("is_interrupt")]
    false_cuts = sum(1 for t in non_interrupt if t.get("false_cut_in"))
    barge_client = [t["client_barge_in_ms"] for t in turns if t.get("client_barge_in_ms") is not None]
    barge_server = [d["barge_in_ms"] for d in derived if d.get("barge_in_ms") is not None]

    wers = []
    missing = total = 0
    for c in calls:
        sc = scenarios[c["scenario_id"]]
        w = call_wer(c.get("caller_lines", []), c.get("user_finals", []))
        if w is not None:
            wers.append(w)
        m, t = entity_counts(sc, c.get("user_finals", []))
        missing += m
        total += t

    prices = load_prices()
    n_gpu = selfhost_capacity()
    costs = [x for x in (call_cost_usd(profile, overrides, c, prices, n_gpu) for c in calls) if x is not None]
    durs = [c["duration_s"] for c in calls if c.get("duration_s")]

    summary = {
        "task_success_rate": round(sum(c["grade"]["success"] for c in calls) / len(calls), 4) if calls else None,
        "client_v2v_p50": pct(client_v2v, 50),
        "client_v2v_p95": pct(client_v2v, 95),
        "server_v2v_p50": pct(server_v2v, 50),
        "server_v2v_p95": pct(server_v2v, 95),
    }
    for k in SERVER_STAGES:
        summary[f"{k}_p50"] = pct([d[k] for d in derived if d.get(k) is not None], 50)
    summary.update({
        "network_ms_p50": pct(network, 50),
        "barge_in_client_p50": pct(barge_client, 50),
        "barge_in_client_p95": pct(barge_client, 95),
        "barge_in_server_p50": pct(barge_server, 50),
        "false_cut_in_rate": round(false_cuts / len(non_interrupt), 4) if non_interrupt else None,
        "wer": round(float(np.mean(wers)), 4) if wers else None,
        "name_number_error_rate": round(missing / total, 4) if total else None,
        "cost_per_3min_call_usd": round(float(np.mean(costs)), 4) if costs else None,
        "mean_call_s": round(float(np.mean(durs)), 1) if durs else None,
    })
    return summary


def summary_markdown(results: dict) -> str:
    lines = [
        f"# {results['label']}",
        "",
        f"- profile: `{results['profile']}`  overrides: `{json.dumps(results['overrides'])}`",
        f"- git: `{results['git_sha']}`  started: {results['started_at']}  scenarios: {len(results['scenarios'])}",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for k, v in results["summary"].items():
        lines.append(f"| {k} | {'n/a' if v is None else v} |")
    fails = [c for c in results["scenarios"] if not c["grade"]["success"]]
    if fails:
        lines += ["", "## Failed scenarios", ""]
        for c in fails:
            lines.append(f"- `{c['scenario_id']}` ({c.get('end_reason')}): {'; '.join(c['grade']['reasons'])}")
    return "\n".join(lines) + "\n"


async def run_label(
    *,
    profile: str,
    scenarios: list[Scenario],
    label: str,
    overrides: dict | None = None,
    concurrency: int = 1,
    agent_url: str | None = None,
) -> dict:
    overrides = overrides or {}
    out_dir = RESULTS / label
    started = datetime.now().isoformat(timespec="seconds")
    settings = Settings()
    server = AgentServer(profile, overrides, out_dir)
    await server.start()
    try:
        voices = load_voices()
        async with (
            httpx.AsyncClient(timeout=60) as provider_http,
            httpx.AsyncClient(
                base_url=server.http_url, headers={"X-Admin-Token": server.admin_token}, timeout=30
            ) as admin_http,
        ):
            openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            caller = SimulatedCaller(openai_client, provider_http, voices)
            sem = asyncio.Semaphore(concurrency)

            async def guarded(sc: Scenario) -> dict:
                async with sem:
                    res = await run_one(sc, label, server, agent_url, caller, admin_http)
                    ok = "ok" if res["grade"]["success"] else "FAIL"
                    print(f"[{label}] {sc.id}: {ok}", flush=True)
                    return res

            calls = await asyncio.gather(*(guarded(sc) for sc in scenarios))
    finally:
        server.stop()
    by_id = {s.id: s for s in scenarios}
    results = {
        "label": label,
        "profile": profile,
        "overrides": overrides,
        "git_sha": git_sha(),
        "started_at": started,
        "scenarios": list(calls),
        "summary": summarize(profile, overrides, list(calls), by_id),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out_dir / "summary.md").write_text(summary_markdown(results))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--scenarios", default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--label", default=None)
    ap.add_argument("--agent-url", default=None)
    a = ap.parse_args()
    label = a.label or f"{a.profile}-{datetime.now():%Y%m%d-%H%M}"
    results = asyncio.run(run_label(
        profile=a.profile, scenarios=select_scenarios(a.scenarios), label=label,
        overrides=json.loads(a.overrides), concurrency=a.concurrency, agent_url=a.agent_url,
    ))
    print((RESULTS / label / "summary.md").read_text())
    print(f"wrote {RESULTS / label / 'results.json'}")


if __name__ == "__main__":
    main()
