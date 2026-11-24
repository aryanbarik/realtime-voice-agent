from datetime import date

import pytest
from fastapi.testclient import TestClient

from backend.app import create_backend_app

TODAY = date(2026, 11, 23)
FRI = "2026-11-27"


@pytest.fixture
def client(tmp_path):
    app = create_backend_app(str(tmp_path / "b.db"), TODAY, None, False)
    return TestClient(app)


def avail(c, tenant="a", d=FRI, t="19:00", n=2):
    return c.get(f"/t/{tenant}/availability", params={"date": d, "time": t, "party_size": n})


def test_blocked_slot_alternatives(client):
    r = avail(client)
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["alternatives"] == ["18:30", "19:30", "18:00"]


def test_over_capacity_409(client):
    for _ in range(3):
        r = client.post("/t/a/reservations", json={"name": "A", "party_size": 8, "date": "2026-11-25", "time": "18:00"})
        assert r.status_code == 201
    r = client.post("/t/a/reservations", json={"name": "B", "party_size": 1, "date": "2026-11-25", "time": "18:00"})
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "slot_unavailable"
    assert body["alternatives"]


def test_cancel_normalized_name(client):
    client.post("/t/a/reservations", json={"name": "Priya Raman", "party_size": 2, "date": "2026-11-25", "time": "18:00"})
    r = client.post("/t/a/reservations/cancel", json={"name": "  priya RAMAN ", "date": "2026-11-25"})
    assert r.status_code == 200
    assert r.json()["cancelled"][0]["name"] == "Priya Raman"
    r = client.post("/t/a/reservations/cancel", json={"name": "Priya Raman", "date": "2026-11-25"})
    assert r.status_code == 404
    assert r.json() == {"error": "not_found"}


def test_seed_isolated_per_tenant(client):
    avail(client, "b")  # lazily create b with default seed
    before = client.get("/admin/tenants/b/reservations").json()
    r = client.put("/admin/tenants/a", json={"capacity_covers": 24, "blocked": [], "reservations": [{"name": "Zed", "party_size": 2, "date": FRI, "time": "18:00"}]})
    assert r.status_code == 200
    assert avail(client, "a").json()["available"] is True
    assert [x["name"] for x in client.get("/admin/tenants/a/reservations").json()] == ["Zed"]
    assert client.get("/admin/tenants/b/reservations").json() == before
    assert avail(client, "b").json()["available"] is False


def test_party_size_422(client):
    r = avail(client, n=9)
    assert r.status_code == 422
    assert r.json()["error"] == "party_size must be between 1 and 8"
    r = client.post("/t/a/reservations", json={"name": "A", "party_size": 9, "date": FRI, "time": "18:00"})
    assert r.status_code == 422 and "error" in r.json()


def test_default_seed_and_validation(client):
    rows = client.get("/admin/tenants/z/reservations").json()
    assert [(x["name"], x["party_size"]) for x in rows] == [("John Smith", 2)]
    assert avail(client, t="19:15").status_code == 422
    assert avail(client, d="2026-11-22").status_code == 422


def test_admin_auth_and_public_demo(tmp_path):
    app = create_backend_app(str(tmp_path / "c.db"), TODAY, "secret", False)
    c = TestClient(app)
    assert c.get("/admin/tenants/a/reservations").status_code == 401
    assert c.get("/admin/tenants/a/reservations", headers={"X-Admin-Token": "secret"}).status_code == 200
    app = create_backend_app(str(tmp_path / "d.db"), TODAY, None, True)
    assert TestClient(app).get("/admin/tenants/a/reservations").status_code == 404
