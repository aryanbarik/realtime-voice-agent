"""Scenario schema for the synthetic-caller eval (loaded from YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

SCENARIO_DIR = Path(__file__).parent / "scenarios"

Category = Literal[
    "simple",
    "counteroffer",
    "cancel",
    "spelling",
    "change_mind",
    "interrupt",
    "long_pause",
    "noise",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Caller(_Model):
    name: str
    goal: str
    persona: str
    spell_name: bool = False


class Interrupt(_Model):
    agent_turn: int  # the greeting is turn 0
    after_ms: int
    instruction: str


class Behaviors(_Model):
    interrupt: Interrupt | None = None
    pause_ms: int = 0


class Noise(_Model):
    snr_db: float


class BlockedSlot(_Model):
    date: str
    time: str


class SeedReservation(_Model):
    name: str
    party_size: int
    date: str
    time: str


class Seed(_Model):
    capacity_covers: int = 24
    blocked: list[BlockedSlot] = []
    reservations: list[SeedReservation] = []


class Expected(_Model):
    reservations: list[SeedReservation] = []
    cancelled: list[SeedReservation] = []


class Scenario(_Model):
    id: str
    category: Category
    caller: Caller
    behaviors: Behaviors = Behaviors()
    noise: Noise | None = None
    seed: Seed = Seed()
    expected: Expected = Expected()
    max_caller_turns: int = 10


def load_scenario(path: str | Path) -> Scenario:
    with open(path, encoding="utf-8") as f:
        return Scenario.model_validate(yaml.safe_load(f))


def load_scenarios(directory: str | Path = SCENARIO_DIR) -> list[Scenario]:
    return [load_scenario(p) for p in sorted(Path(directory).glob("*.yaml"))]
