"""
Security helpers: API-key auth, signed photo URLs, upload validation, path
safety and a small per-IP rate limiter.

All settings are read from the environment at call time (not import time) so
tests can change them with monkeypatch.setenv.

Environment variables
    UPLOAD_API_KEY                  X-API-Key accepted on /upload*
    ADMIN_API_KEY                   X-API-Key accepted on /admin/* (also accepted on /upload*
                                    and when fetching photos directly)
    PHOTO_URL_SECRET                HMAC secret for signed photo URLs
    PHOTO_URL_TTL_SECONDS           lifetime of a signed URL (default 3600)
    MAX_UPLOAD_MB                   per-photo size cap (default 25)
    MAX_SELFIE_MB                   per-selfie size cap (default 5)
    MAX_IMAGE_PIXELS                decoded-size cap, protects against decompression bombs
                                    (default 64,000,000)
    SEARCH_RATE_LIMIT_PER_MINUTE    per-client-IP limit on /search/face (default 10; 0 disables)
    CONSENT_VERSION                 stored with each consent record (default "v1")
    STORAGE_PATH                    root for stored photos
"""

import hashlib
import hmac
import io
import logging
import os
import threading
import time
import uuid
import warnings
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Header, HTTPException, Request, UploadFile
from PIL import Image

log = logging.getLogger("media_ai.security")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# Pillow only warns between 1x and 2x of this value and errors above 2x; we turn the
# warning into an error in validate_image() and also check the size ourselves.
Image.MAX_IMAGE_PIXELS = int(_env("MAX_IMAGE_PIXELS", "64000000"))


# ── API-key auth ──────────────────────────────────────────────────────────────

def _key_matches(provided: str | None, expected: str) -> bool:
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _unauthorized() -> HTTPException:
    return HTTPException(status_code=401, detail="Invalid or missing API key", headers={"WWW-Authenticate": "ApiKey"})


def require_upload_key(x_api_key: str | None = Header(default=None)) -> None:
    """Dependency for /upload*. 503 if UPLOAD_API_KEY is unset; the admin key is also accepted."""
    upload_key = _env("UPLOAD_API_KEY")
    if not upload_key:
        raise HTTPException(status_code=503, detail="Uploads are not configured (UPLOAD_API_KEY is not set).")
    if _key_matches(x_api_key, upload_key) or _key_matches(x_api_key, _env("ADMIN_API_KEY")):
        return
    raise _unauthorized()


def require_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    """Dependency for /admin/*. 503 if ADMIN_API_KEY is unset."""
    admin_key = _env("ADMIN_API_KEY")
    if not admin_key:
        raise HTTPException(status_code=503, detail="Admin routes are not configured (ADMIN_API_KEY is not set).")
    if _key_matches(x_api_key, admin_key):
        return
    raise _unauthorized()


def is_admin_key(provided: str | None) -> bool:
    return _key_matches(provided, _env("ADMIN_API_KEY"))


def log_startup_warnings() -> None:
    for name, what in (
        ("UPLOAD_API_KEY", "/upload* routes will return 503"),
        ("ADMIN_API_KEY", "/admin/* routes will return 503"),
        ("PHOTO_URL_SECRET", "photo URLs cannot be signed, so face search and photo serving will return 503"),
    ):
        if not _env(name):
            log.warning("%s is not set: %s", name, what)


# ── Signed photo URLs ─────────────────────────────────────────────────────────

PHOTO_SIZES = ("thumb", "full")


def require_photo_secret() -> str:
    secret = _env("PHOTO_URL_SECRET")
    if not secret:
        raise HTTPException(status_code=503, detail="Photo links are not configured (PHOTO_URL_SECRET is not set).")
    return secret


def photo_url_ttl() -> int:
    return int(_env("PHOTO_URL_TTL_SECONDS", "3600"))


def _photo_sig(event_id: str, photo_id: str, size: str, exp: int) -> str:
    secret = require_photo_secret()
    msg = f"{event_id}|{photo_id}|{size}|{exp}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def photo_path(event_id: str, photo_id: str, size: str) -> str:
    return f"/events/{event_id}/photos/{photo_id}?size={size}"


def signed_photo_url(event_id: str, photo_id: str, size: str, now: float | None = None) -> str:
    """Relative URL that serves one photo until it expires. Raises 503 if PHOTO_URL_SECRET is unset."""
    exp = int((time.time() if now is None else now) + photo_url_ttl())
    sig = _photo_sig(event_id, photo_id, size, exp)
    return f"{photo_path(event_id, photo_id, size)}&exp={exp}&sig={sig}"


def verify_photo_signature(event_id: str, photo_id: str, size: str, exp: int, sig: str, now: float | None = None) -> bool:
    """True only if the signature matches and the link has not expired. Raises 503 if the secret is unset."""
    expected = _photo_sig(event_id, photo_id, size, exp)
    if not hmac.compare_digest(expected.encode("utf-8"), sig.encode("utf-8")):
        return False
    return exp >= (time.time() if now is None else now)


# ── IDs and storage paths ─────────────────────────────────────────────────────

def parse_uuid(value: str, field: str = "id") -> str:
    """Return the canonical lowercase form of a UUID, or raise 422."""
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=422, detail=f"{field} must be a UUID")


def storage_root() -> Path:
    return Path(_env("STORAGE_PATH", "./storage")).resolve()


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except (ValueError, OSError):
        return False


def resolve_storage_path(stored: str | None) -> Path | None:
    """
    Turn a stored photo path into a real file path that is guaranteed to be inside
    STORAGE_PATH, or None.

    New rows store paths relative to STORAGE_PATH ("<event>/original/<uuid>.jpg").
    Rows written by earlier versions hold absolute paths (or paths relative to the
    working directory) and are still read; an absolute path from a different mount is
    re-rooted by taking whatever follows its "storage" directory.
    """
    if not stored:
        return None
    root = storage_root()
    raw = Path(stored)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(root / raw)
    parts = raw.parts
    if "storage" in parts:
        candidates.append(root.joinpath(*parts[parts.index("storage") + 1:]))
    first_valid = None
    for cand in candidates:
        if not _under(cand, root):
            continue
        if cand.exists():
            return cand.resolve()
        first_valid = first_valid or cand.resolve()
    return first_valid


def relative_storage_path(path: Path) -> str:
    return path.resolve().relative_to(storage_root()).as_posix()


# ── Upload validation ─────────────────────────────────────────────────────────

ALLOWED_FORMATS = {"JPEG": "jpg", "PNG": "png"}
_CHUNK = 1024 * 1024


def max_upload_bytes() -> int:
    return int(float(_env("MAX_UPLOAD_MB", "25")) * 1024 * 1024)


def max_selfie_bytes() -> int:
    return int(float(_env("MAX_SELFIE_MB", "5")) * 1024 * 1024)


def reject_oversized_request(request: Request, max_bytes: int, slack: int = 256 * 1024) -> None:
    """Cheap early check on Content-Length so an obviously huge body is refused up front."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + slack:
        raise HTTPException(status_code=413, detail=f"File too large (limit {max_bytes // (1024 * 1024)} MB)")


async def read_capped(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in chunks and stop as soon as it exceeds max_bytes."""
    buf = io.BytesIO()
    total = 0
    while True:
        chunk = await upload.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large (limit {max_bytes // (1024 * 1024)} MB)")
        buf.write(chunk)
    return buf.getvalue()


def validate_image(data: bytes) -> tuple[str, int, int]:
    """
    Check the bytes really are a JPEG or PNG (by content, not by client header) and
    are not a decompression bomb. Returns (extension, width, height).
    """
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            fmt = img.format
            width, height = img.size
            if fmt not in ALLOWED_FORMATS:
                raise HTTPException(status_code=415, detail="Only JPEG and PNG images are accepted")
            if width * height > Image.MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image has too many pixels")
            img.verify()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(status_code=413, detail="Image has too many pixels")
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid image")
    return ALLOWED_FORMATS[fmt], width, height


# ── Rate limiting (in-process) ────────────────────────────────────────────────

class SlidingWindowLimiter:
    """Allow at most `limit` hits per key in any `window` seconds. Not shared across workers."""

    def __init__(self):
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = 60.0, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            if len(self._hits) > 10000:  # drop idle clients so the dict cannot grow without bound
                for k in [k for k, v in self._hits.items() if not v or v[-1] <= now - window]:
                    del self._hits[k]
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


search_limiter = SlidingWindowLimiter()


def enforce_search_rate_limit(request: Request) -> None:
    """429 when a client IP exceeds SEARCH_RATE_LIMIT_PER_MINUTE. Uses the socket address."""
    limit = int(_env("SEARCH_RATE_LIMIT_PER_MINUTE", "10"))
    if limit <= 0:
        return
    client = request.client.host if request.client else "unknown"
    if not search_limiter.allow(client, limit):
        raise HTTPException(status_code=429, detail="Too many searches. Please wait a minute and try again.", headers={"Retry-After": "60"})
