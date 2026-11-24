"""Public-demo guardrails: per-IP, global concurrency, and a daily spend cap.

Wiring (server/app.py, only when settings.va_public_demo):
    limiter = DemoLimiter(backend.state.db, settings.va_daily_budget_usd)   # once
    ip = client_ip(ws)
    try:
        lease = limiter.acquire(ip)          # before ws.accept() or after; see refuse()
    except Refused as r:
        await ws.accept(); await transport.send_json({"type":"error","message":r.reason})
        await transport.close(code=1013); return
    session_id = ...; tenant = demo_tenant(session_id)
    meter = SpendMeter(limiter, lambda: (session.asr_audio_s, session.usage_totals))
    try: run session with meter.run() as an extra task; max_call_s = DEMO_MAX_CALL_S
    finally: await meter.final(); lease.release()
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import date
from typing import Any

from server.cost import cost_from_totals

DEMO_MAX_CALL_S = 180
MAX_CONCURRENT_PER_IP = 1
MAX_CALLS_PER_IP_HOUR = 3
MAX_CONCURRENT_GLOBAL = 4
SPEND_FLUSH_S = 30.0

MSG_BUDGET = "Demo budget for today is used up."
MSG_IP = "Too many calls from your address; try again later."
MSG_BUSY = "Demo is busy; try again in a minute."
REFUSE_CLOSE_CODE = 1013


class Refused(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def client_ip(ws: Any) -> str:
    """`Fly-Client-IP` header, else the socket peer."""
    ip = ws.headers.get("fly-client-ip")
    if ip:
        return ip.strip()
    return ws.client.host if ws.client else "unknown"


def demo_tenant(session_id: str) -> str:
    return f"demo-{session_id[:8]}"


def apply_demo_config(cfg):
    """Return a copy of PipelineConfig with max_call_s forced to the demo cap."""
    return cfg.model_copy(update={"max_call_s": DEMO_MAX_CALL_S})


class Lease:
    def __init__(self, limiter: "DemoLimiter", ip: str):
        self._limiter, self.ip, self._released = limiter, ip, False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._limiter._release(self.ip)


class DemoLimiter:
    """db: backend Database (needs add_spend(day, usd) and spend(day))."""

    def __init__(self, db, daily_budget_usd: float, today: Callable[[], date] = date.today,
                 clock: Callable[[], float] = time.monotonic):
        self.db, self.budget, self._today, self._clock = db, daily_budget_usd, today, clock
        self._active: dict[str, int] = defaultdict(int)
        self._total = 0
        self._starts: dict[str, deque[float]] = defaultdict(deque)

    def day(self) -> str:
        return self._today().isoformat()

    def over_budget(self) -> bool:
        return self.db.spend(self.day()) >= self.budget

    def acquire(self, ip: str) -> Lease:
        """Raises Refused. No awaits, so check+increment is atomic on the loop."""
        if self.over_budget():
            raise Refused(MSG_BUDGET)
        now = self._clock()
        starts = self._starts[ip]
        while starts and now - starts[0] > 3600:
            starts.popleft()
        if self._active[ip] >= MAX_CONCURRENT_PER_IP or len(starts) >= MAX_CALLS_PER_IP_HOUR:
            raise Refused(MSG_IP)
        if self._total >= MAX_CONCURRENT_GLOBAL:
            raise Refused(MSG_BUSY)
        starts.append(now)
        self._active[ip] += 1
        self._total += 1
        return Lease(self, ip)

    def _release(self, ip: str) -> None:
        self._active[ip] = max(self._active[ip] - 1, 0)
        if not self._active[ip]:
            del self._active[ip]
        self._total = max(self._total - 1, 0)

    def add_spend(self, usd: float) -> float:
        return self.db.add_spend(self.day(), usd)


class SpendMeter:
    """Charges the cost delta of one session to the daily budget.

    `snapshot()` returns (asr_audio_s, usage_totals dict with TurnTrace.usage keys),
    cumulative for the session. `run()` is a task that flushes every 30 s and
    returns (letting the caller end the call) once the budget is exhausted;
    `final()` flushes the remainder at session end.
    """

    def __init__(self, limiter: DemoLimiter, snapshot: Callable[[], tuple[float, dict]],
                 llm_model: str = "gpt-6-luna", interval_s: float = SPEND_FLUSH_S):
        self.limiter, self.snapshot, self.llm_model, self.interval_s = limiter, snapshot, llm_model, interval_s
        self._charged = 0.0

    def flush(self) -> float:
        asr_s, usage = self.snapshot()
        total = cost_from_totals(asr_s, usage, llm_model=self.llm_model)
        delta = total - self._charged
        if delta > 0:
            self._charged = total
            self.limiter.add_spend(delta)
        return delta

    async def run(self) -> str:
        """Returns "budget" when the cap is hit; otherwise runs until cancelled."""
        while True:
            await asyncio.sleep(self.interval_s)
            self.flush()
            if self.limiter.over_budget():
                return "budget"

    async def final(self) -> None:
        self.flush()
