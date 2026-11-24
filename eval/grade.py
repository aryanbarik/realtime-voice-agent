"""Grading and per-call quality metrics. Success is decided from backend database state only."""

from __future__ import annotations

import re
import unicodedata

import jiwer

from backend.db import normalize_name
from eval.scenario import Scenario

_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
}


def _key(row: dict) -> tuple:
    return (normalize_name(row["name"]), int(row["party_size"]), row["date"], row["time"])


def grade(scenario: Scenario, rows: list[dict]) -> dict:
    """`rows` is the admin reservations list (all statuses) for the scenario's tenant."""
    caller = normalize_name(scenario.caller.name)
    confirmed = {
        (r["party_size"], r["date"], r["time"])
        for r in rows
        if r["status"] == "confirmed" and normalize_name(r["name"]) == caller
    }
    wanted = {(e.party_size, e.date, e.time) for e in scenario.expected.reservations}
    reasons: list[str] = []
    if confirmed != wanted:
        reasons.append(f"confirmed reservations {confirmed} != expected {wanted}")

    cancelled_keys = {_key(r) for r in rows if r["status"] == "cancelled"}
    for e in scenario.expected.cancelled:
        if _key(e.model_dump()) not in cancelled_keys:
            reasons.append(f"expected cancellation missing: {e.name} {e.date} {e.time}")

    for s in scenario.seed.reservations:
        k = _key(s.model_dump())
        if k[0] == caller:
            continue
        if k in cancelled_keys:
            reasons.append(f"seed reservation of another guest was cancelled: {s.name}")

    return {"success": not reasons, "reasons": reasons}


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def call_wer(caller_lines: list[str], user_finals: list[str]) -> float | None:
    ref = _norm_text(" ".join(caller_lines))
    hyp = _norm_text(" ".join(user_finals))
    if not ref:
        return None
    return float(jiwer.wer(ref, hyp))


def entity_counts(scenario: Scenario, user_finals: list[str]) -> tuple[int, int]:
    """Returns (missing, total) over caller name tokens and the party size."""
    text = _norm_text(" ".join(user_finals))
    words = set(text.split())
    squashed = text.replace(" ", "")
    total = missing = 0
    for tok in scenario.caller.name.split():
        t = _norm_text(tok).replace(" ", "")
        if not t:
            continue
        total += 1
        if t not in words and not (len(t) >= 5 and t in squashed):
            missing += 1
    if scenario.expected.reservations:
        n = scenario.expected.reservations[0].party_size
        total += 1
        candidates = {str(n)}
        if n in _NUMBER_WORDS:
            candidates.add(_NUMBER_WORDS[n])
        if not candidates & words:
            missing += 1
    return missing, total
