"""Deterministically generate the 62 eval scenarios into eval/scenarios/*.yaml.

Run: uv run python -m eval.gen_scenarios
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import yaml

from backend.db import SLOTS, order_alternatives
from eval.scenario import SCENARIO_DIR, Scenario

FIRST_DAY = date(2026, 11, 24)
DAYS = 14  # 2026-11-24 .. 2026-12-07

NAMES = [
    "Priya Raman", "Alex Johnson", "Maria Garcia", "James Carter", "Emily Chen",
    "David Okafor", "Sophie Martin", "Daniel Kim", "Laura Rossi", "Michael Brown",
    "Hannah Schmidt", "Omar Haddad", "Grace Turner", "Lucas Silva", "Anna Petrova",
    "Ethan Walker", "Isabella Costa", "Noah Patel", "Chloe Dubois", "Ryan Murphy",
    "Zoe Anderson", "Mateo Fernandez", "Olivia Bennett", "Samuel Adler", "Nora Larsen",
    "Victor Ramos", "Tara Singh", "Henry Foster", "Leah Cohen", "Marcus Webb",
]
SPELL_NAMES = [
    "Siobhan O'Donoghue", "Nguyen Thi Thao", "Krzysztof Wojciechowski",
    "Aoife Ni Bhriain", "Oluwaseun Adeyemi", "Xiaoqing Zhou",
]
PERSONAS = [
    "Friendly and concise.",
    "Polite but in a hurry.",
    "Chatty and warm.",
    "Calm and direct.",
    "A little unsure, but cooperative.",
]
# Requested times leave room for alternatives on both sides.
TIMES = ["18:00", "18:30", "19:00", "19:30", "20:00", "20:30"]

# Pause, in ms, between the two halves of the first caller line (long_pause).
PAUSES = [1200, 1500, 1800, 1200, 1500, 1800]
SNRS = [20, 10, 5]

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def human_date(d: str) -> str:
    dt = date.fromisoformat(d)
    return f"{dt.strftime('%A')}, {dt.strftime('%B')} {dt.day}"


def human_time(t: str) -> str:
    h, m = int(t[:2]), int(t[3:])
    h12 = h - 12 if h > 12 else h
    return f"{h12} PM" if m == 0 else f"{h12}:{m:02d} PM"


class Gen:
    def __init__(self) -> None:
        self.rng = random.Random(7)
        self.names = list(NAMES)
        self.rng.shuffle(self.names)
        self.name_i = 0

    def name(self) -> str:
        n = self.names[self.name_i % len(self.names)]
        self.name_i += 1
        return n

    def day(self) -> str:
        return (FIRST_DAY + timedelta(days=self.rng.randrange(DAYS))).isoformat()

    def persona(self) -> str:
        return self.rng.choice(PERSONAS)

    def party(self, lo: int = 2, hi: int = 6) -> int:
        return self.rng.randint(lo, hi)

    def time(self) -> str:
        return self.rng.choice(TIMES)

    def other_time(self, *avoid: str) -> str:
        return self.rng.choice([t for t in TIMES if t not in avoid])

    # -- builders: each returns a plain dict matching eval.scenario.Scenario --

    def simple(self, sid: str, category: str = "simple") -> dict:
        name, d, t, n = self.name(), self.day(), self.time(), self.party()
        goal = (
            f"Book a table for {WORDS[n]} on {human_date(d)} at {human_time(t)} "
            f"under the name {name}."
        )
        return _scn(
            sid, category, name, goal, self.persona(),
            expected_res=[(name, n, d, t)],
        )

    def counteroffer(self, sid: str, category: str = "counteroffer") -> dict:
        name, d, t, n = self.name(), self.day(), self.time(), self.party()
        available = [s for s in SLOTS if s != t]
        first_alt = order_alternatives(t, available)[0]
        goal = (
            f"Book a table for {WORDS[n]} on {human_date(d)} at {human_time(t)} "
            f"under the name {name}. If that time is not available, accept the "
            f"first alternative the agent offers that is within one hour of your "
            f"requested time."
        )
        return _scn(
            sid, category, name, goal, self.persona(),
            blocked=[(d, t)], expected_res=[(name, n, d, first_alt)],
        )

    def cancel(self, sid: str) -> dict:
        name, other = self.name(), self.name()
        d, t, n = self.day(), self.time(), self.party()
        od, ot, on = self.day(), self.time(), self.party()
        mine = (name, n, d, t)
        goal = (
            f"Cancel your reservation on {human_date(d)} under the name {name}. "
            f"Do not book anything new."
        )
        s = _scn(
            sid, "cancel", name, goal, self.persona(),
            seed_res=[mine, (other, on, od, ot)], expected_res=[], expected_cancelled=[mine],
        )
        return s

    def spelling(self, sid: str, name: str) -> dict:
        d, t, n = self.day(), self.time(), self.party()
        goal = (
            f"Book a table for {WORDS[n]} on {human_date(d)} at {human_time(t)} "
            f"under the name {name}. The name is hard to hear, so spell it out "
            f"letter by letter after saying it."
        )
        return _scn(
            sid, "spelling", name, goal, self.persona(),
            spell=True, expected_res=[(name, n, d, t)],
        )

    def change_mind(self, sid: str) -> dict:
        name, d, n = self.name(), self.day(), self.party()
        a = self.time()
        b = self.other_time(a)
        goal = (
            f"Start by asking for a table for {WORDS[n]} on {human_date(d)} at "
            f"{human_time(a)} under the name {name}. After the agent reads the "
            f"booking details back, but before you say yes, change your mind and "
            f"ask for {human_time(b)} instead. Confirm only the {human_time(b)} booking."
        )
        return _scn(
            sid, "change_mind", name, goal, self.persona(),
            expected_res=[(name, n, d, b)],
        )

    def interrupt(self, sid: str) -> dict:
        name, d, t = self.name(), self.day(), self.time()
        n = self.party(2, 4)
        n2 = self.rng.choice([x for x in range(2, 8) if x != n])
        goal = (
            f"Book a table on {human_date(d)} at {human_time(t)} under the name "
            f"{name}. Your party size is {WORDS[n2]}. Initially you may mention "
            f"{WORDS[n]}, but you will correct it to {WORDS[n2]} when you interrupt "
            f"the agent. The final booking must be for {WORDS[n2]}."
        )
        s = _scn(
            sid, "interrupt", name, goal, self.persona(),
            expected_res=[(name, n2, d, t)],
        )
        s["behaviors"]["interrupt"] = {
            "agent_turn": 2,
            "after_ms": 1200,
            "instruction": f"Interrupt to change the party size to {n2}",
        }
        return s

    def long_pause(self, sid: str, pause_ms: int) -> dict:
        s = self.simple(sid, "long_pause")
        s["behaviors"]["pause_ms"] = pause_ms
        return s


def _scn(
    sid: str,
    category: str,
    name: str,
    goal: str,
    persona: str,
    *,
    spell: bool = False,
    blocked: list[tuple[str, str]] | None = None,
    seed_res: list[tuple[str, int, str, str]] | None = None,
    expected_res: list[tuple[str, int, str, str]] | None = None,
    expected_cancelled: list[tuple[str, int, str, str]] | None = None,
) -> dict:
    def res(rows):
        return [
            {"name": a, "party_size": b, "date": c, "time": d} for a, b, c, d in rows or []
        ]

    return {
        "id": sid,
        "category": category,
        "caller": {"name": name, "goal": goal, "persona": persona, "spell_name": spell},
        "behaviors": {"interrupt": None, "pause_ms": 0},
        "noise": None,
        "seed": {
            "capacity_covers": 24,
            "blocked": [{"date": d, "time": t} for d, t in blocked or []],
            "reservations": res(seed_res),
        },
        "expected": {
            "reservations": res(expected_res),
            "cancelled": res(expected_cancelled),
        },
        "max_caller_turns": 10,
    }


def generate() -> list[dict]:
    g = Gen()
    out: list[dict] = []
    out += [g.simple(f"simple-{i:02d}") for i in range(1, 13)]
    out += [g.counteroffer(f"counteroffer-{i:02d}") for i in range(1, 9)]
    out += [g.cancel(f"cancel-{i:02d}") for i in range(1, 7)]
    out += [g.spelling(f"spelling-{i:02d}", n) for i, n in enumerate(SPELL_NAMES, 1)]
    out += [g.change_mind(f"change_mind-{i:02d}") for i in range(1, 7)]
    out += [g.interrupt(f"interrupt-{i:02d}") for i in range(1, 7)]
    out += [g.long_pause(f"long_pause-{i:02d}", p) for i, p in enumerate(PAUSES, 1)]
    # noise: 4 bases (2 simple, 2 counteroffer) x 3 SNRs
    idx = 0
    for kind in ("simple", "simple", "counteroffer", "counteroffer"):
        base = None
        for snr in SNRS:
            idx += 1
            sid = f"noise-{idx:02d}"
            if base is None:
                base = getattr(g, kind)(sid, "noise")
                s = base
            else:
                s = {**base, "id": sid}
            s = {**s, "noise": {"snr_db": snr}}
            out.append(s)
    return out


def main() -> None:
    scenarios = generate()
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    for old in SCENARIO_DIR.glob("*.yaml"):
        old.unlink()
    for s in scenarios:
        Scenario.model_validate(s)  # fail fast on schema drift
        with open(SCENARIO_DIR / f"{s['id']}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(s, f, sort_keys=False, allow_unicode=True, width=100)
    print(f"wrote {len(scenarios)} scenarios to {SCENARIO_DIR}")


if __name__ == "__main__":
    main()
