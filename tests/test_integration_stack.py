"""
End-to-end test against the REAL running stack: API + Postgres + CompreFace.

Skipped unless all three respond and COMPREFACE_API_KEY, UPLOAD_API_KEY, ADMIN_API_KEY and
PHOTO_URL_SECRET are set (read from the environment or .env, never from the fake test config),
and FACE_TEST_PHOTOS points at a folder laid out as:

    event\\                    photos
    selfies\\<person>\\        one selfie per person
    labels.csv                columns photo_filename,person_id

It uploads a few labelled photos into a throwaway event, runs the batch, searches with each
selfie, and then deletes everything it created. Photos stay outside the repo and are never
committed. Run with:  pytest -m integration
"""

import csv
import os
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path

import pytest
import requests

from conftest import REAL_ENV, ROOT, make_image_bytes

pytestmark = pytest.mark.integration

API = REAL_ENV.get("INTEGRATION_API_URL", "http://localhost:8000").rstrip("/")
COMPREFACE_UI = REAL_ENV.get("INTEGRATION_COMPREFACE_URL", "http://localhost:8001").rstrip("/")
PEOPLE_TO_TRY = 4


def _pg_connect():
    import psycopg2

    return psycopg2.connect(
        host=REAL_ENV.get("DB_HOST") or "localhost",
        port=int(REAL_ENV.get("DB_PORT") or 5432),
        user=REAL_ENV.get("DB_USER") or "media_user",
        password=REAL_ENV.get("DB_PASS") or "media_pass",
        dbname=REAL_ENV.get("DB_NAME") or "media_ai",
        connect_timeout=3,
    )


@pytest.fixture(scope="module")
def stack():
    missing = [k for k in ("COMPREFACE_API_KEY", "UPLOAD_API_KEY", "ADMIN_API_KEY", "PHOTO_URL_SECRET") if not REAL_ENV.get(k)]
    if missing:
        pytest.skip(f"not set in the environment or .env: {', '.join(missing)}")
    photos = REAL_ENV.get("FACE_TEST_PHOTOS")
    if not photos or not (Path(photos) / "labels.csv").is_file():
        pytest.skip("FACE_TEST_PHOTOS must point at a folder with event\\, selfies\\ and labels.csv")
    try:
        if requests.get(f"{API}/health", timeout=3).status_code != 200:
            raise RuntimeError("API /health is not 200")
    except Exception as e:
        pytest.skip(f"API not responding at {API} ({e.__class__.__name__})")
    try:
        requests.get(COMPREFACE_UI, timeout=3)
    except Exception as e:
        pytest.skip(f"CompreFace not responding at {COMPREFACE_UI} ({e.__class__.__name__})")
    try:
        _pg_connect().close()
    except Exception as e:
        pytest.skip(f"Postgres not reachable ({e.__class__.__name__})")
    return {"photos": Path(photos), "up": REAL_ENV["UPLOAD_API_KEY"], "admin": REAL_ENV["ADMIN_API_KEY"]}


def _cleanup(event_id: str):
    try:
        conn = _pg_connect()
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("DELETE FROM media_ai.search_requests WHERE event_id=%s", (event_id,))
        cur.execute("DELETE FROM media_ai.batch_runs WHERE event_id=%s", (event_id,))
        cur.execute("DELETE FROM media_ai.event_photos WHERE event_id=%s", (event_id,))  # cascades faces/errors
        conn.close()
    except Exception:
        pass
    shutil.rmtree(ROOT / "storage" / event_id, ignore_errors=True)


def test_upload_batch_and_selfie_search_end_to_end(stack):
    photos_dir: Path = stack["photos"]
    labels = defaultdict(list)
    with open(photos_dir / "labels.csv", newline="") as f:
        for row in csv.DictReader(f):
            labels[row["person_id"]].append(row["photo_filename"])
    people = [p for p, files in sorted(labels.items()) if len(files) >= 2 and any((photos_dir / "selfies" / p).glob("*.jpg"))]
    people = people[:PEOPLE_TO_TRY]
    assert len(people) >= 2, "need at least two labelled people with selfies"

    event_id = str(uuid.uuid4())
    up = {"X-API-Key": stack["up"]}
    admin = {"X-API-Key": stack["admin"]}
    owner = {}  # photo_id -> person_id
    try:
        for person in people:
            for name in labels[person]:
                with open(photos_dir / "event" / name, "rb") as fh:
                    r = requests.post(f"{API}/upload", headers=up, data={"event_id": event_id},
                                      files={"file": ("x.jpg", fh, "image/jpeg")}, timeout=30)
                assert r.status_code == 200, r.text
                owner[r.json()["photo_id"]] = person
        r = requests.post(f"{API}/upload", headers=up, data={"event_id": event_id},
                          files={"file": ("blank.jpg", make_image_bytes("JPEG", (300, 300), (128, 128, 128)), "image/jpeg")}, timeout=30)
        blank_id = r.json()["photo_id"]

        # Uploads are not processed until the batch is triggered.
        stats = requests.get(f"{API}/admin/events/{event_id}/stats", headers=admin, timeout=10).json()
        assert stats["total_photos"] == len(owner) + 1 and stats["photos_with_face_embeddings"] == 0

        assert requests.post(f"{API}/admin/batch/{event_id}", headers=admin, timeout=10).status_code == 200
        conn = _pg_connect()
        conn.autocommit = True
        cur = conn.cursor()
        deadline = time.time() + 240
        while time.time() < deadline:
            cur.execute("SELECT count(*) FROM media_ai.event_photos WHERE event_id=%s AND face_attempted_at IS NULL", (event_id,))
            if cur.fetchone()[0] == 0:
                break
            time.sleep(3)
        else:
            pytest.fail("batch did not finish within 240 s")

        cur.execute("SELECT count(*) FROM media_ai.face_embeddings fe JOIN media_ai.event_photos ep USING (photo_id) WHERE ep.event_id=%s", (event_id,))
        assert cur.fetchone()[0] > 0, "no faces were stored"
        cur.execute("SELECT count(*) FROM media_ai.face_embeddings WHERE photo_id=%s", (blank_id,))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM media_ai.processing_errors pe JOIN media_ai.event_photos ep USING (photo_id) WHERE ep.event_id=%s", (event_id,))
        assert cur.fetchone()[0] == 0, "a plain no-face photo must not be logged as an error"
        conn.close()

        found = {}
        for person in people:
            selfie = next((photos_dir / "selfies" / person).glob("*.jpg"))
            with open(selfie, "rb") as fh:
                r = requests.post(f"{API}/search/face", data={"event_id": event_id, "consent": "true"},
                                  files={"selfie": ("me.jpg", fh, "image/jpeg")}, timeout=60)
            assert r.status_code == 200, r.text
            results = r.json()["photos"]
            ids = {p["photo_id"] for p in results}
            wrong = [i for i in ids if owner.get(i) not in (None, person)]
            assert not wrong, f"{person}'s selfie matched someone else's photos"
            assert blank_id not in ids
            found[person] = results
            time.sleep(7)  # stay under the default 10 searches/minute limit

        hits = {p: r for p, r in found.items() if r}
        assert hits, "no selfie matched any of its own photos"
        first = next(iter(hits.values()))[0]
        link = requests.get(f"{API}{first['image_url']}", timeout=20)
        assert link.status_code == 200 and link.headers["content-type"].startswith("image/")
        assert requests.get(f"{API}/events/{event_id}/photos/{first['photo_id']}?size=full", timeout=10).status_code == 401
    finally:
        _cleanup(event_id)
