"""
Shared fixtures.

Safety rules baked in here:
  * Tests never touch the real `media_ai` database. DB_NAME is forced to `media_ai_test`
    before anything can open a connection, and tests that need real SQL use a throwaway
    database created and dropped by the `test_db` fixture. If Postgres is not reachable those
    tests are skipped with a clear reason.
  * CompreFace is never called: tests monkeypatch `extract_face_embedding` / `requests.post`.
  * Test secrets are fake and set before the app is imported.
"""

import io
import math
import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Real values (from the environment / .env) are read ONLY by the integration test.
from dotenv import dotenv_values  # noqa: E402

REAL_ENV = {**dotenv_values(ROOT / ".env"), **{k: v for k, v in os.environ.items()}}

# Fake config, set before `api.*` / `processing.*` are imported.
TEST_UPLOAD_KEY = "test-upload-key"
TEST_ADMIN_KEY = "test-admin-key"
TEST_PHOTO_SECRET = "test-photo-secret"
os.environ.update(
    {
        "UPLOAD_API_KEY": TEST_UPLOAD_KEY,
        "ADMIN_API_KEY": TEST_ADMIN_KEY,
        "PHOTO_URL_SECRET": TEST_PHOTO_SECRET,
        "PHOTO_URL_TTL_SECONDS": "3600",
        "COMPREFACE_API_KEY": "test-compreface-key",
        "COMPREFACE_URL": "http://compreface.invalid",
        "SEARCH_RATE_LIMIT_PER_MINUTE": "0",
        "MIN_FACE_CONFIDENCE": "0.70",
        "MAX_UPLOAD_MB": "25",
        "MAX_SELFIE_MB": "5",
        "CORS_ORIGINS": "",
        "DB_NAME": "media_ai_test",
    }
)
os.environ.setdefault("DB_HOST", REAL_ENV.get("DB_HOST") or "localhost")
os.environ.setdefault("DB_PORT", REAL_ENV.get("DB_PORT") or "5432")
os.environ.setdefault("DB_USER", REAL_ENV.get("DB_USER") or "media_user")
os.environ.setdefault("DB_PASS", REAL_ENV.get("DB_PASS") or "media_pass")

from PIL import Image  # noqa: E402

TEST_DB = "media_ai_test"


# ── Database ──────────────────────────────────────────────────────────────────

def _admin_conn():
    import psycopg2

    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASS"],
        dbname="postgres",
        connect_timeout=3,
    )
    conn.autocommit = True
    return conn


@pytest.fixture(scope="session")
def test_db():
    """Create a throwaway `media_ai_test` database with the real schema; drop it afterwards."""
    import psycopg2
    import db.connection as dbc

    try:
        admin = _admin_conn()
    except psycopg2.OperationalError as e:
        pytest.skip(f"Postgres not reachable at {os.environ['DB_HOST']}:{os.environ['DB_PORT']} ({e.__class__.__name__}); start it with `docker compose up -d db`")

    assert os.environ["DB_NAME"] == TEST_DB, "tests must never use the real database"
    cur = admin.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    cur.execute(f"CREATE DATABASE {TEST_DB}")

    if dbc._pool is not None:
        dbc._pool.closeall()
    dbc._pool = None
    schema = (ROOT / "db" / "schema.sql").read_text(encoding="utf-8")
    conn = dbc.get_conn()
    try:
        with conn.cursor() as c:
            c.execute(schema)
        conn.commit()
    finally:
        dbc.release_conn(conn)

    yield TEST_DB

    if dbc._pool is not None:
        dbc._pool.closeall()
    dbc._pool = None
    cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    admin.close()


@pytest.fixture
def clean_db(test_db):
    """Empty every table before the test (the schema stays)."""
    from db.connection import execute_query

    execute_query(
        "TRUNCATE media_ai.search_results, media_ai.search_requests, media_ai.processing_errors, "
        "media_ai.face_embeddings, media_ai.bib_detections, media_ai.batch_runs, media_ai.event_photos CASCADE"
    )
    return test_db


# ── App / storage ─────────────────────────────────────────────────────────────

@pytest.fixture
def storage_dir(tmp_path, monkeypatch):
    d = tmp_path / "storage"
    d.mkdir()
    monkeypatch.setenv("STORAGE_PATH", str(d))
    return d


@pytest.fixture
def client(storage_dir):
    """TestClient without running startup events (no scheduler)."""
    from fastapi.testclient import TestClient
    from api.main import app
    from api.security import search_limiter

    search_limiter.reset()
    return TestClient(app)


@pytest.fixture
def upload_headers():
    return {"X-API-Key": TEST_UPLOAD_KEY}


@pytest.fixture
def admin_headers():
    return {"X-API-Key": TEST_ADMIN_KEY}


# ── Image / vector helpers ────────────────────────────────────────────────────

def make_image_bytes(fmt="JPEG", size=(64, 48), color=(200, 80, 80)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def jpeg_bytes():
    return make_image_bytes("JPEG")


@pytest.fixture
def png_bytes():
    return make_image_bytes("PNG")


def vec(similarity_to_axis: float) -> list[float]:
    """512-d unit vector whose cosine similarity to vec(1.0) is `similarity_to_axis`."""
    s = max(-1.0, min(1.0, similarity_to_axis))
    v = [0.0] * 512
    v[0] = s
    v[1] = math.sqrt(max(0.0, 1.0 - s * s))
    return v


def face(similarity: float, w: int = 100, h: int = 100) -> dict:
    return {"embedding": vec(similarity), "confidence": 0.99, "bounding_box": {"x": 0, "y": 0, "w": w, "h": h}}


def insert_photo(event_id: str, image_rel: str | None = None, thumb_rel: str | None = None, **cols) -> str:
    """Insert an event_photos row and return its photo_id."""
    from db.connection import execute_query

    photo_id = str(uuid.uuid4())
    execute_query(
        """
        INSERT INTO media_ai.event_photos (photo_id, event_id, image_url, thumbnail_url, uploaded_at, face_attempted_at)
        VALUES (%s, %s, %s, %s, COALESCE(%s, NOW()), %s)
        """,
        (photo_id, event_id, image_rel or f"{event_id}/original/{photo_id}.jpg", thumb_rel,
         cols.get("uploaded_at"), cols.get("face_attempted_at")),
    )
    return photo_id


def insert_face(photo_id: str, embedding: list[float]) -> None:
    from db.connection import execute_query

    execute_query(
        "INSERT INTO media_ai.face_embeddings (photo_id, embedding, confidence) VALUES (%s, %s, %s)",
        (photo_id, embedding, 0.99),
    )


def write_photo_file(storage: Path, event_id: str, name: str, data: bytes, thumb=True) -> tuple[str, str | None]:
    """Write a real file under storage and return (image_rel, thumb_rel)."""
    orig = storage / event_id / "original"
    orig.mkdir(parents=True, exist_ok=True)
    (orig / name).write_bytes(data)
    thumb_rel = None
    if thumb:
        td = storage / event_id / "thumbnails"
        td.mkdir(parents=True, exist_ok=True)
        (td / name).write_bytes(data)
        thumb_rel = f"{event_id}/thumbnails/{name}"
    return f"{event_id}/original/{name}", thumb_rel
