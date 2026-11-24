from __future__ import annotations

import json

import httpx

TIMEOUT_S = 2.0


def _err(code: str) -> str:
    return json.dumps({"error": code})


class ReservationClient:
    def __init__(self, http: httpx.AsyncClient, tenant: str):
        self.http = http
        self.tenant = tenant

    async def execute(self, name: str, arguments: str) -> str:
        """Run a tool against the backend. Always returns a JSON string; never raises."""
        try:
            args = json.loads(arguments)
            if not isinstance(args, dict):
                return _err("invalid_arguments")
        except (json.JSONDecodeError, TypeError):
            return _err("invalid_arguments")

        base = f"/t/{self.tenant}"
        try:
            if name == "check_availability":
                params = {k: args[k] for k in ("date", "time", "party_size")}
                resp = await self.http.get(f"{base}/availability", params=params, timeout=TIMEOUT_S)
            elif name == "book_reservation":
                body = {k: args[k] for k in ("name", "party_size", "date", "time")}
                resp = await self.http.post(f"{base}/reservations", json=body, timeout=TIMEOUT_S)
            elif name == "cancel_reservation":
                body = {k: args[k] for k in ("name", "date")}
                resp = await self.http.post(f"{base}/reservations/cancel", json=body, timeout=TIMEOUT_S)
            else:
                return _err("unknown_tool")
        except KeyError:
            return _err("invalid_arguments")
        except (httpx.TimeoutException, httpx.TransportError):
            return _err("backend_unavailable")
        except Exception:
            return _err("backend_unavailable")

        if resp.status_code >= 500:
            return _err("backend_unavailable")
        return resp.text
