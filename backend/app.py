"""Mock restaurant backend (FastAPI)."""

from __future__ import annotations

import hmac
from datetime import date

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .db import TENANT_RE, Database, ValidationError


class BookBody(BaseModel):
    name: str
    party_size: int
    date: str
    time: str


class CancelBody(BaseModel):
    name: str
    date: str


class SlotRef(BaseModel):
    date: str
    time: str


class SeedReservation(BaseModel):
    name: str
    party_size: int
    date: str
    time: str


class SeedBody(BaseModel):
    capacity_covers: int = 24
    blocked: list[SlotRef] = []
    reservations: list[SeedReservation] = []


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def create_backend_app(
    db_path: str, today: date, admin_token: str | None, public_demo: bool
) -> FastAPI:
    db = Database(db_path, today)
    app = FastAPI(title="Bella Vista backend")
    app.state.db = db

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        errs = exc.errors()
        if errs:
            e = errs[0]
            loc = ".".join(str(p) for p in e["loc"] if p not in ("body", "query"))
            msg = f"{loc}: {e['msg']}" if loc else str(e["msg"])
        else:
            msg = "invalid request"
        return _error(422, msg)

    @app.exception_handler(ValidationError)
    async def _domain(_: Request, exc: ValidationError):
        return _error(422, str(exc))

    def tenant_id(tenant: str) -> str:
        if not TENANT_RE.fullmatch(tenant):
            raise ValidationError("tenant must match ^[a-z0-9-]{1,64}$")
        return tenant

    @app.get("/t/{tenant}/availability")
    def availability(tenant: str, date: str, time: str, party_size: int):
        return db.availability(tenant_id(tenant), date, time, party_size)

    @app.post("/t/{tenant}/reservations", status_code=201)
    def book(tenant: str, body: BookBody):
        res = db.book(
            tenant_id(tenant), body.name, body.party_size, body.date, body.time
        )
        if "error" in res:
            return JSONResponse(res, status_code=409)
        return res

    @app.post("/t/{tenant}/reservations/cancel")
    def cancel(tenant: str, body: CancelBody):
        cancelled = db.cancel(tenant_id(tenant), body.name, body.date)
        if not cancelled:
            return _error(404, "not_found")
        return {"cancelled": cancelled}

    if public_demo and not admin_token:
        return app  # admin routes are not registered at all

    def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
        if admin_token and not (
            x_admin_token
            and hmac.compare_digest(x_admin_token.encode(), admin_token.encode())
        ):
            raise HTTPException(status_code=401, detail="invalid admin token")

    admin = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])

    @admin.put("/tenants/{tenant}")
    def seed(tenant: str, body: SeedBody):
        db.seed(
            tenant_id(tenant),
            body.capacity_covers,
            [(b.date, b.time) for b in body.blocked],
            [(r.name, r.party_size, r.date, r.time) for r in body.reservations],
        )
        return {"ok": True}

    @admin.get("/tenants/{tenant}/reservations")
    def reservations(tenant: str):
        return db.list_reservations(tenant_id(tenant))

    app.include_router(admin)
    return app
