"""API-key auth, signed-URL helpers, UUID parsing, rate limiter, CORS defaults."""

import uuid

import pytest
from fastapi import HTTPException

from api import security

EVENT = str(uuid.uuid4())
PHOTO = str(uuid.uuid4())


# ── Upload / admin keys ───────────────────────────────────────────────────────

def test_upload_requires_key(client, jpeg_bytes):
    r = client.post("/upload", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")}, data={"event_id": EVENT})
    assert r.status_code == 401


def test_upload_rejects_wrong_key(client, jpeg_bytes):
    r = client.post("/upload", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")}, data={"event_id": EVENT},
                    headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_bulk_upload_requires_key(client, jpeg_bytes):
    r = client.post("/upload/bulk", files=[("files", ("a.jpg", jpeg_bytes, "image/jpeg"))], data={"event_id": EVENT})
    assert r.status_code == 401


def test_admin_routes_reject_upload_key(client, upload_headers):
    assert client.post(f"/admin/batch/{EVENT}", headers=upload_headers).status_code == 401
    assert client.get(f"/admin/events/{EVENT}/stats", headers=upload_headers).status_code == 401


def test_admin_routes_reject_missing_key(client):
    assert client.post(f"/admin/batch/{EVENT}").status_code == 401
    assert client.get(f"/admin/events/{EVENT}/stats").status_code == 401


def test_admin_key_is_accepted_on_upload_dependency():
    # Dependency-level check (no DB needed): the admin key passes the upload guard.
    security.require_upload_key(x_api_key="test-admin-key")
    security.require_upload_key(x_api_key="test-upload-key")
    with pytest.raises(HTTPException) as e:
        security.require_upload_key(x_api_key="wrong")
    assert e.value.status_code == 401


def test_admin_dependency_rejects_upload_key():
    with pytest.raises(HTTPException) as e:
        security.require_admin_key(x_api_key="test-upload-key")
    assert e.value.status_code == 401


@pytest.mark.parametrize("var", ["UPLOAD_API_KEY", "ADMIN_API_KEY"])
@pytest.mark.parametrize("value", ["", "   "])
def test_unset_or_empty_key_returns_503(client, monkeypatch, var, value, jpeg_bytes):
    monkeypatch.setenv(var, value)
    if var == "UPLOAD_API_KEY":
        r = client.post("/upload", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")}, data={"event_id": EVENT},
                        headers={"X-API-Key": value or "x"})
    else:
        r = client.get(f"/admin/events/{EVENT}/stats", headers={"X-API-Key": value or "x"})
    assert r.status_code == 503


def test_empty_header_never_matches_empty_key(client, monkeypatch):
    # An empty configured key must not turn "send an empty header" into a bypass.
    monkeypatch.setenv("ADMIN_API_KEY", "")
    r = client.get(f"/admin/events/{EVENT}/stats", headers={"X-API-Key": ""})
    assert r.status_code == 503
    assert security.is_admin_key("") is False
    assert security.is_admin_key(None) is False


def test_startup_warns_about_unset_secrets(monkeypatch, caplog):
    for name in ("UPLOAD_API_KEY", "ADMIN_API_KEY", "PHOTO_URL_SECRET"):
        monkeypatch.setenv(name, "")
    with caplog.at_level("WARNING", logger="media_ai.security"):
        security.log_startup_warnings()
    text = caplog.text
    assert "UPLOAD_API_KEY" in text and "ADMIN_API_KEY" in text and "PHOTO_URL_SECRET" in text


def test_keys_are_compared_in_constant_time(monkeypatch):
    calls = []
    real = security.hmac.compare_digest
    monkeypatch.setattr(security.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    security.is_admin_key("test-admin-key")
    assert calls, "hmac.compare_digest was not used"


# ── Signed URLs ───────────────────────────────────────────────────────────────

def test_signature_roundtrip_and_expiry():
    url = security.signed_photo_url(EVENT, PHOTO, "full", now=1_000_000)
    qs = dict(p.split("=") for p in url.split("?", 1)[1].split("&"))
    exp, sig = int(qs["exp"]), qs["sig"]
    assert exp == 1_000_000 + 3600
    assert security.verify_photo_signature(EVENT, PHOTO, "full", exp, sig, now=1_000_100)
    assert not security.verify_photo_signature(EVENT, PHOTO, "full", exp, sig, now=exp + 1)  # expired


@pytest.mark.parametrize("change", ["event", "photo", "size", "exp"])
def test_signature_covers_every_field(change):
    url = security.signed_photo_url(EVENT, PHOTO, "full", now=1_000_000)
    qs = dict(p.split("=") for p in url.split("?", 1)[1].split("&"))
    exp, sig = int(qs["exp"]), qs["sig"]
    args = {"event": EVENT, "photo": PHOTO, "size": "full", "exp": exp}
    args[change] = {"event": str(uuid.uuid4()), "photo": str(uuid.uuid4()), "size": "thumb", "exp": exp + 1}[change]
    assert not security.verify_photo_signature(args["event"], args["photo"], args["size"], args["exp"], sig, now=1_000_100)


def test_signing_fails_closed_without_secret(monkeypatch):
    monkeypatch.setenv("PHOTO_URL_SECRET", "")
    with pytest.raises(HTTPException) as e:
        security.signed_photo_url(EVENT, PHOTO, "full")
    assert e.value.status_code == 503
    with pytest.raises(HTTPException) as e:
        security.verify_photo_signature(EVENT, PHOTO, "full", 9999999999, "0" * 64)
    assert e.value.status_code == 503


def test_ttl_is_configurable(monkeypatch):
    monkeypatch.setenv("PHOTO_URL_TTL_SECONDS", "60")
    url = security.signed_photo_url(EVENT, PHOTO, "thumb", now=1000)
    assert "exp=1060" in url


# ── UUID parsing ──────────────────────────────────────────────────────────────

def test_parse_uuid_canonicalises_and_rejects():
    u = uuid.uuid4()
    assert security.parse_uuid(str(u).upper()) == str(u)
    for bad in ["", "abc", "../../etc/passwd", "1' OR '1'='1", str(u) + "/x"]:
        with pytest.raises(HTTPException) as e:
            security.parse_uuid(bad)
        assert e.value.status_code == 422


def test_admin_routes_reject_non_uuid_event(client, admin_headers):
    assert client.get("/admin/events/not-a-uuid/stats", headers=admin_headers).status_code == 422
    assert client.post("/admin/batch/not-a-uuid", headers=admin_headers).status_code == 422


# ── Rate limiter ──────────────────────────────────────────────────────────────

def test_limiter_window_and_independent_keys():
    lim = security.SlidingWindowLimiter()
    assert all(lim.allow("a", 3, now=0) for _ in range(3))
    assert not lim.allow("a", 3, now=1)
    assert lim.allow("b", 3, now=1)          # other client unaffected
    assert lim.allow("a", 3, now=61)         # window slid


def test_search_returns_429_after_limit(client, monkeypatch, jpeg_bytes):
    monkeypatch.setenv("SEARCH_RATE_LIMIT_PER_MINUTE", "2")
    data = {"event_id": EVENT, "consent": "false"}
    files = {"selfie": ("s.jpg", jpeg_bytes, "image/jpeg")}
    codes = [client.post("/search/face", files=files, data=data).status_code for _ in range(4)]
    assert codes[:2] == [400, 400] and codes[2:] == [429, 429]


# ── CORS defaults ─────────────────────────────────────────────────────────────

def test_cors_is_same_origin_by_default(client):
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_storage_is_not_served(client):
    assert client.get("/storage/anything.jpg").status_code == 404
