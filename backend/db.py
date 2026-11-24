"""SQLite-backed reservation store for the mock restaurant backend."""

from __future__ import annotations

import re
import sqlite3
import threading
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SLOTS: list[str] = [f"{17 + m // 60:02d}:{m % 60:02d}" for m in range(0, 271, 30)]
DEFAULT_CAPACITY = 24
MAX_PARTY = 8
HORIZON_DAYS = 60
TENANT_RE = re.compile(r"^[a-z0-9-]{1,64}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants(id TEXT PRIMARY KEY, capacity_covers INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS blocked_slots(tenant TEXT, date TEXT, time TEXT, PRIMARY KEY(tenant,date,time));
CREATE TABLE IF NOT EXISTS reservations(id INTEGER PRIMARY KEY AUTOINCREMENT, tenant TEXT NOT NULL, name TEXT NOT NULL,
  party_size INTEGER NOT NULL, date TEXT NOT NULL, time TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('confirmed','cancelled')), created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS daily_spend(date TEXT PRIMARY KEY, usd REAL NOT NULL);
"""


class ValidationError(ValueError):
    """Raised for invalid request values; message is human-readable."""


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().split())


def _minutes(t: str) -> int:
    return int(t[:2]) * 60 + int(t[3:])


def order_alternatives(
    requested_time: str, available: list[str], limit: int = 3
) -> list[str]:
    """Up to `limit` slots ordered by distance from requested_time; ties go earlier."""
    req = _minutes(requested_time)
    ranked = sorted(
        (s for s in available if s != requested_time),
        key=lambda s: (abs(_minutes(s) - req), _minutes(s)),
    )
    return ranked[:limit]


def first_weekday_on_or_after(today: date, weekday: int) -> date:
    return today + timedelta(days=(weekday - today.weekday()) % 7)


def validate_party_size(n: int) -> None:
    if not 1 <= n <= MAX_PARTY:
        raise ValidationError(f"party_size must be between 1 and {MAX_PARTY}")


def validate_time(t: str) -> None:
    if t not in SLOTS:
        raise ValidationError(
            "time must be a 30-minute seating between 17:00 and 21:30"
        )


def parse_date(d: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d or ""):
        raise ValidationError("date must be formatted YYYY-MM-DD")
    try:
        return date.fromisoformat(d)
    except ValueError:
        raise ValidationError("date must be a valid calendar date") from None


class Database:
    def __init__(self, path: str, today: date):
        self.today = today
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # -- helpers (callers hold the lock) --------------------------------
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def validate_date(self, d: str) -> date:
        parsed = parse_date(d)
        last = self.today + timedelta(days=HORIZON_DAYS)
        if not self.today <= parsed <= last:
            raise ValidationError(
                f"date must be between {self.today.isoformat()} and {last.isoformat()}"
            )
        return parsed

    def _ensure_tenant(self, tenant: str) -> None:
        c = self.conn
        if c.execute("SELECT 1 FROM tenants WHERE id=?", (tenant,)).fetchone():
            return
        now = self._now()
        c.execute(
            "INSERT INTO tenants VALUES(?,?,?)", (tenant, DEFAULT_CAPACITY, now)
        )
        friday = first_weekday_on_or_after(self.today, 4).isoformat()
        saturday = first_weekday_on_or_after(self.today, 5).isoformat()
        c.execute("INSERT INTO blocked_slots VALUES(?,?,?)", (tenant, friday, "19:00"))
        c.execute(
            "INSERT INTO reservations(tenant,name,party_size,date,time,status,created_at)"
            " VALUES(?,?,?,?,?,'confirmed',?)",
            (tenant, "John Smith", 2, saturday, "20:00", now),
        )
        c.commit()

    def _slot_available(
        self, tenant: str, d: str, t: str, party: int, capacity: int
    ) -> bool:
        c = self.conn
        if c.execute(
            "SELECT 1 FROM blocked_slots WHERE tenant=? AND date=? AND time=?",
            (tenant, d, t),
        ).fetchone():
            return False
        used = c.execute(
            "SELECT COALESCE(SUM(party_size),0) FROM reservations"
            " WHERE tenant=? AND date=? AND time=? AND status='confirmed'",
            (tenant, d, t),
        ).fetchone()[0]
        return used + party <= capacity

    def _capacity(self, tenant: str) -> int:
        return self.conn.execute(
            "SELECT capacity_covers FROM tenants WHERE id=?", (tenant,)
        ).fetchone()[0]

    def _alternatives(self, tenant: str, d: str, t: str, party: int) -> list[str]:
        cap = self._capacity(tenant)
        free = [s for s in SLOTS if self._slot_available(tenant, d, s, party, cap)]
        return order_alternatives(t, free)

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        return {
            "reservation_id": r["id"],
            "name": r["name"],
            "party_size": r["party_size"],
            "date": r["date"],
            "time": r["time"],
            "status": r["status"],
        }

    # -- public API ------------------------------------------------------
    def availability(self, tenant: str, d: str, t: str, party: int) -> dict:
        validate_party_size(party)
        validate_time(t)
        self.validate_date(d)
        with self.lock:
            self._ensure_tenant(tenant)
            ok = self._slot_available(tenant, d, t, party, self._capacity(tenant))
            alts = [] if ok else self._alternatives(tenant, d, t, party)
        return {
            "date": d,
            "time": t,
            "party_size": party,
            "available": ok,
            "alternatives": alts,
        }

    def book(self, tenant: str, name: str, party: int, d: str, t: str) -> dict:
        """Returns the reservation dict, or {"error": "slot_unavailable", "alternatives": [...]}."""
        if not name.strip():
            raise ValidationError("name must not be empty")
        validate_party_size(party)
        validate_time(t)
        self.validate_date(d)
        with self.lock:
            self._ensure_tenant(tenant)
            if not self._slot_available(tenant, d, t, party, self._capacity(tenant)):
                return {
                    "error": "slot_unavailable",
                    "alternatives": self._alternatives(tenant, d, t, party),
                }
            cur = self.conn.execute(
                "INSERT INTO reservations(tenant,name,party_size,date,time,status,created_at)"
                " VALUES(?,?,?,?,?,'confirmed',?)",
                (tenant, name.strip(), party, d, t, self._now()),
            )
            self.conn.commit()
        return {
            "reservation_id": cur.lastrowid,
            "name": name.strip(),
            "party_size": party,
            "date": d,
            "time": t,
            "status": "confirmed",
        }

    def cancel(self, tenant: str, name: str, d: str) -> list[dict]:
        parse_date(d)
        target = normalize_name(name)
        with self.lock:
            self._ensure_tenant(tenant)
            rows = self.conn.execute(
                "SELECT * FROM reservations WHERE tenant=? AND date=? AND status='confirmed'"
                " ORDER BY id",
                (tenant, d),
            ).fetchall()
            hits = [r for r in rows if normalize_name(r["name"]) == target]
            for r in hits:
                self.conn.execute(
                    "UPDATE reservations SET status='cancelled' WHERE id=?", (r["id"],)
                )
            self.conn.commit()
        return [{**self._row(r), "status": "cancelled"} for r in hits]

    def seed(
        self,
        tenant: str,
        capacity: int,
        blocked: list[tuple[str, str]],
        reservations: list[tuple[str, int, str, str]],
    ) -> None:
        if capacity < 1:
            raise ValidationError("capacity_covers must be at least 1")
        for d, t in blocked:
            parse_date(d)
            validate_time(t)
        for name, party, d, t in reservations:
            if not name.strip():
                raise ValidationError("name must not be empty")
            validate_party_size(party)
            parse_date(d)
            validate_time(t)
        with self.lock:
            c = self.conn
            now = self._now()
            c.execute("DELETE FROM blocked_slots WHERE tenant=?", (tenant,))
            c.execute("DELETE FROM reservations WHERE tenant=?", (tenant,))
            c.execute("DELETE FROM tenants WHERE id=?", (tenant,))
            c.execute("INSERT INTO tenants VALUES(?,?,?)", (tenant, capacity, now))
            c.executemany(
                "INSERT OR IGNORE INTO blocked_slots VALUES(?,?,?)",
                [(tenant, d, t) for d, t in blocked],
            )
            c.executemany(
                "INSERT INTO reservations(tenant,name,party_size,date,time,status,created_at)"
                " VALUES(?,?,?,?,?,'confirmed',?)",
                [(tenant, n.strip(), p, d, t, now) for n, p, d, t in reservations],
            )
            c.commit()

    def list_reservations(self, tenant: str) -> list[dict]:
        with self.lock:
            self._ensure_tenant(tenant)
            rows = self.conn.execute(
                "SELECT * FROM reservations WHERE tenant=? ORDER BY id", (tenant,)
            ).fetchall()
        return [self._row(r) for r in rows]

    def add_spend(self, day: str, usd: float) -> float:
        with self.lock:
            self.conn.execute(
                "INSERT INTO daily_spend VALUES(?,?)"
                " ON CONFLICT(date) DO UPDATE SET usd=usd+excluded.usd",
                (day, usd),
            )
            self.conn.commit()
            return self.conn.execute(
                "SELECT usd FROM daily_spend WHERE date=?", (day,)
            ).fetchone()[0]

    def spend(self, day: str) -> float:
        with self.lock:
            r = self.conn.execute(
                "SELECT usd FROM daily_spend WHERE date=?", (day,)
            ).fetchone()
        return r[0] if r else 0.0
