"""
processing/batch.py against a real throwaway Postgres: ON CONFLICT claiming, stale reclaim,
the stalled-event sweep, processing_errors, and how failures are classified.
CompreFace is mocked (monkeypatching extract_face_embedding or requests.post).
"""

import uuid
from datetime import datetime, timedelta

import pytest
import requests

from conftest import face, insert_photo, write_photo_file
from db.connection import execute_query
from processing import batch
from test_face_search import BAD_KEY, NO_FACE, SERVER_ERROR, compreface_faces, patch_post

EVENT = str(uuid.uuid4())
OTHER = str(uuid.uuid4())


def one(sql, params=()):
    return execute_query(sql, params, fetch="one")[0]


def make_photo(storage, jpeg_bytes, event=EVENT, name=None, **cols):
    name = name or f"{uuid.uuid4().hex}.jpg"
    img, thumb = write_photo_file(storage, event, name, jpeg_bytes)
    return insert_photo(event, img, thumb, **cols)


def attempted(pid):
    return one("SELECT face_attempted_at IS NOT NULL FROM media_ai.event_photos WHERE photo_id=%s", (pid,))


def n_faces(pid):
    return one("SELECT count(*) FROM media_ai.face_embeddings WHERE photo_id=%s", (pid,))


def n_errors(pid=None, stage="face"):
    if pid:
        return one("SELECT count(*) FROM media_ai.processing_errors WHERE photo_id=%s AND stage=%s", (pid, stage))
    return one("SELECT count(*) FROM media_ai.processing_errors")


@pytest.fixture
def fake_embedder(monkeypatch):
    """Replace batch.extract_face_embedding; returns the list of paths it was called with."""
    calls = []

    def install(result):
        def fake(path):
            calls.append(path)
            if isinstance(result, Exception):
                raise result
            return result
        monkeypatch.setattr(batch, "extract_face_embedding", fake)
        return calls

    return install


# ── Claiming (INSERT ... ON CONFLICT) ────────────────────────────────────────

def test_second_claim_is_refused(clean_db):
    assert batch._claim_batch_run(EVENT) is True
    assert batch._claim_batch_run(EVENT) is False
    assert batch._claim_batch_run(OTHER) is True  # other events are independent


def test_release_allows_a_new_claim(clean_db):
    batch._claim_batch_run(EVENT)
    batch._release_batch_run(EVENT)
    assert batch._claim_batch_run(EVENT) is True


def test_stale_claim_is_reclaimed(clean_db, monkeypatch):
    monkeypatch.setattr(batch, "BATCH_RUN_STALE_MINUTES", 60)
    batch._claim_batch_run(EVENT)
    execute_query("UPDATE media_ai.batch_runs SET started_at = NOW() - INTERVAL '61 minutes' WHERE event_id=%s", (EVENT,))
    assert batch._claim_batch_run(EVENT) is True
    # reclaiming refreshes started_at, so an immediate third claim is refused
    assert batch._claim_batch_run(EVENT) is False
    assert one("SELECT started_at > NOW() - INTERVAL '1 minute' FROM media_ai.batch_runs WHERE event_id=%s", (EVENT,))


def test_fresh_claim_is_not_reclaimed(clean_db, monkeypatch):
    monkeypatch.setattr(batch, "BATCH_RUN_STALE_MINUTES", 60)
    batch._claim_batch_run(EVENT)
    execute_query("UPDATE media_ai.batch_runs SET started_at = NOW() - INTERVAL '59 minutes' WHERE event_id=%s", (EVENT,))
    assert batch._claim_batch_run(EVENT) is False


def test_stale_threshold_follows_the_setting(clean_db, monkeypatch):
    monkeypatch.setattr(batch, "BATCH_RUN_STALE_MINUTES", 5)
    batch._claim_batch_run(EVENT)
    execute_query("UPDATE media_ai.batch_runs SET started_at = NOW() - INTERVAL '6 minutes' WHERE event_id=%s", (EVENT,))
    assert batch._claim_batch_run(EVENT) is True


def test_run_skips_when_another_run_holds_the_claim(clean_db, storage_dir, jpeg_bytes, fake_embedder):
    calls = fake_embedder([face(1.0)])
    pid = make_photo(storage_dir, jpeg_bytes)
    batch._claim_batch_run(EVENT)
    assert batch.batch_process_event(EVENT)["skipped"] == "already_running"
    assert calls == [] and not attempted(pid)
    assert one("SELECT count(*) FROM media_ai.batch_runs") == 1  # the other run's claim is untouched


def test_claim_is_released_after_a_run(clean_db, storage_dir, jpeg_bytes, fake_embedder):
    fake_embedder([face(1.0)])
    make_photo(storage_dir, jpeg_bytes)
    batch.batch_process_event(EVENT)
    assert one("SELECT count(*) FROM media_ai.batch_runs") == 0


def test_claim_is_released_even_if_the_run_crashes(clean_db, monkeypatch):
    monkeypatch.setattr(batch, "execute_query", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone"))
                        if "FROM media_ai.event_photos" in a[0] else execute_query(*a, **k))
    with pytest.raises(RuntimeError):
        batch.batch_process_event(EVENT)
    monkeypatch.undo()
    assert one("SELECT count(*) FROM media_ai.batch_runs") == 0


# ── A normal run ─────────────────────────────────────────────────────────────

def test_run_stores_faces_and_marks_attempted(clean_db, storage_dir, jpeg_bytes, fake_embedder):
    fake_embedder([face(1.0), face(0.5)])
    pid = make_photo(storage_dir, jpeg_bytes)
    summary = batch.batch_process_event(EVENT)
    assert summary == {"photos_found": 1, "face_processed": 1}
    assert n_faces(pid) == 2 and attempted(pid) and n_errors() == 0


def test_attempted_photos_are_not_reprocessed(clean_db, storage_dir, jpeg_bytes, fake_embedder):
    calls = fake_embedder([face(1.0)])
    make_photo(storage_dir, jpeg_bytes)
    batch.batch_process_event(EVENT)
    batch.batch_process_event(EVENT)
    assert len(calls) == 1


def test_only_this_event_is_processed(clean_db, storage_dir, jpeg_bytes, fake_embedder):
    fake_embedder([face(1.0)])
    mine = make_photo(storage_dir, jpeg_bytes, EVENT)
    theirs = make_photo(storage_dir, jpeg_bytes, OTHER)
    batch.batch_process_event(EVENT)
    assert attempted(mine) and not attempted(theirs)


def test_empty_event_is_a_noop(clean_db):
    assert batch.batch_process_event(EVENT) == {"photos_found": 0, "face_processed": 0}


def test_photo_with_no_face_is_attempted_without_error(clean_db, storage_dir, jpeg_bytes, fake_embedder):
    fake_embedder([])
    pid = make_photo(storage_dir, jpeg_bytes)
    batch.batch_process_event(EVENT)
    assert attempted(pid) and n_faces(pid) == 0 and n_errors() == 0


def test_missing_file_is_skipped_and_left_for_later(clean_db, fake_embedder):
    calls = fake_embedder([face(1.0)])
    pid = insert_photo(EVENT, f"{EVENT}/original/gone.jpg", None)
    batch.batch_process_event(EVENT)
    assert calls == [] and not attempted(pid)


def test_path_outside_storage_is_never_read(clean_db, tmp_path, fake_embedder):
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")
    calls = fake_embedder([face(1.0)])
    pid = insert_photo(EVENT, str(outside), None)
    batch.batch_process_event(EVENT)
    assert calls == [] and not attempted(pid)


def test_legacy_absolute_path_inside_storage_is_processed(clean_db, storage_dir, jpeg_bytes, fake_embedder):
    calls = fake_embedder([face(1.0)])
    img, _ = write_photo_file(storage_dir, EVENT, "legacy.jpg", jpeg_bytes)
    pid = insert_photo(EVENT, str(storage_dir / img), None)
    batch.batch_process_event(EVENT)
    assert len(calls) == 1 and n_faces(pid) == 1


# ── processing_errors ────────────────────────────────────────────────────────

def test_unexpected_exception_is_recorded_in_processing_errors(clean_db, storage_dir, jpeg_bytes, fake_embedder):
    fake_embedder(RuntimeError("kaboom"))
    pid = make_photo(storage_dir, jpeg_bytes)
    batch.batch_process_event(EVENT)
    assert n_errors(pid) == 1
    assert "kaboom" in one("SELECT error_message FROM media_ai.processing_errors WHERE photo_id=%s", (pid,))


def test_one_bad_photo_does_not_stop_the_rest(clean_db, storage_dir, jpeg_bytes, monkeypatch):
    bad = make_photo(storage_dir, jpeg_bytes)
    good = make_photo(storage_dir, jpeg_bytes)

    def fake(path):
        if bad_name in path:
            raise RuntimeError("only this one fails")
        return [face(1.0)]

    bad_name = one("SELECT image_url FROM media_ai.event_photos WHERE photo_id=%s", (bad,)).split("/")[-1]
    monkeypatch.setattr(batch, "extract_face_embedding", fake)
    batch.batch_process_event(EVENT)
    assert n_faces(good) == 1 and n_errors(bad) == 1


def test_corrupt_image_is_permanent_marked_attempted_and_recorded(clean_db, storage_dir, monkeypatch):
    # real extract_face_embedding; the file is not an image, so CompreFace is never reached
    calls = patch_post(monkeypatch, compreface_faces((1.0, 10, 10)))
    pid = make_photo(storage_dir, b"\xff\xd8\xff\xe0 not a jpeg")
    batch.batch_process_event(EVENT)
    assert attempted(pid) and n_errors(pid) == 1 and calls == []


# ── BUG: infrastructure failures must not mark the photo attempted ───────────

INFRA = [
    requests.exceptions.ConnectionError("refused"),
    requests.exceptions.Timeout("slow"),
    BAD_KEY,
    SERVER_ERROR,
]
INFRA_IDS = ["connection-refused", "timeout", "bad-key", "compreface-500"]


@pytest.mark.xfail(strict=True, reason="BUG: CompreFace down/bad key marks the photo attempted with no error logged")
@pytest.mark.parametrize("failure", INFRA, ids=INFRA_IDS)
def test_compreface_failure_leaves_photo_unattempted_and_logs_error(clean_db, storage_dir, jpeg_bytes, monkeypatch, failure):
    patch_post(monkeypatch, failure)
    pid = make_photo(storage_dir, jpeg_bytes)
    batch.batch_process_event(EVENT)
    assert not attempted(pid)
    assert n_errors(pid) == 1


@pytest.mark.xfail(strict=True, reason="BUG: a missing API key marks every photo attempted")
def test_missing_api_key_leaves_photo_unattempted(clean_db, storage_dir, jpeg_bytes, monkeypatch):
    from processing import face as face_mod

    monkeypatch.setattr(face_mod, "COMPREFACE_API_KEY", "")
    pid = make_photo(storage_dir, jpeg_bytes)
    batch.batch_process_event(EVENT)
    assert not attempted(pid) and n_errors(pid) == 1


def test_real_no_face_response_is_attempted_without_error(clean_db, storage_dir, jpeg_bytes, monkeypatch):
    patch_post(monkeypatch, NO_FACE)  # HTTP 400 code 28, what CompreFace really sends
    pid = make_photo(storage_dir, jpeg_bytes)
    batch.batch_process_event(EVENT)
    assert attempted(pid) and n_faces(pid) == 0 and n_errors() == 0


@pytest.mark.xfail(strict=True, reason="BUG: a photo that failed while CompreFace was down is never retried")
def test_photo_is_retried_once_compreface_is_back(clean_db, storage_dir, jpeg_bytes, monkeypatch):
    pid = make_photo(storage_dir, jpeg_bytes)
    patch_post(monkeypatch, requests.exceptions.ConnectionError("down"))
    batch.batch_process_event(EVENT)
    patch_post(monkeypatch, compreface_faces((1.0, 100, 100)))
    batch.batch_process_event(EVENT)
    assert attempted(pid) and n_faces(pid) == 1


@pytest.mark.xfail(strict=True, reason="BUG: the sweep cannot recover photos that were wrongly marked attempted")
def test_sweep_recovers_photos_after_an_outage(clean_db, storage_dir, jpeg_bytes, monkeypatch):
    monkeypatch.setattr(batch, "SWEEP_GRACE_MINUTES", 0)
    pid = make_photo(storage_dir, jpeg_bytes, uploaded_at=datetime.utcnow() - timedelta(minutes=30))
    patch_post(monkeypatch, requests.exceptions.ConnectionError("down"))
    batch.batch_process_event(EVENT)
    patch_post(monkeypatch, compreface_faces((1.0, 100, 100)))
    batch.sweep_stalled_events()
    assert n_faces(pid) == 1


# ── The sweep ────────────────────────────────────────────────────────────────

@pytest.fixture
def swept(monkeypatch):
    """Record which events sweep_stalled_events() hands to batch_process_event."""
    events = []
    monkeypatch.setattr(batch, "batch_process_event", lambda event_id: events.append(event_id))
    monkeypatch.setattr(batch, "SWEEP_GRACE_MINUTES", 15)
    return events


def old(minutes=30):
    return datetime.utcnow() - timedelta(minutes=minutes)


def test_sweep_picks_up_old_unattempted_photos(clean_db, swept):
    insert_photo(EVENT, uploaded_at=old(30))
    batch.sweep_stalled_events()
    assert swept == [EVENT]


def test_sweep_leaves_recent_uploads_alone(clean_db, swept):
    insert_photo(EVENT, uploaded_at=old(5))  # upload session may still be running
    batch.sweep_stalled_events()
    assert swept == []


def test_sweep_ignores_fully_processed_events(clean_db, swept):
    insert_photo(EVENT, uploaded_at=old(60), face_attempted_at=datetime.utcnow())
    batch.sweep_stalled_events()
    assert swept == []


def test_sweep_is_face_only_since_ocr_is_disabled(clean_db, swept):
    # ocr_attempted_at stays NULL for every photo now; that must not make the sweep re-run events.
    pid = insert_photo(EVENT, uploaded_at=old(60), face_attempted_at=datetime.utcnow())
    assert one("SELECT ocr_attempted_at IS NULL FROM media_ai.event_photos WHERE photo_id=%s", (pid,))
    batch.sweep_stalled_events()
    assert swept == []


def test_sweep_triggers_each_event_once(clean_db, swept):
    for _ in range(3):
        insert_photo(EVENT, uploaded_at=old())
    insert_photo(OTHER, uploaded_at=old())
    batch.sweep_stalled_events()
    assert sorted(swept) == sorted([EVENT, OTHER])


def test_sweep_respects_the_claim(clean_db, storage_dir, jpeg_bytes, fake_embedder, monkeypatch):
    calls = fake_embedder([face(1.0)])
    monkeypatch.setattr(batch, "SWEEP_GRACE_MINUTES", 0)
    make_photo(storage_dir, jpeg_bytes, uploaded_at=old(30))
    batch._claim_batch_run(EVENT)  # another run is active
    batch.sweep_stalled_events()
    assert calls == []


def test_face_unattempted_index_exists_and_is_partial(clean_db):
    definition = one("SELECT indexdef FROM pg_indexes WHERE indexname='idx_event_photos_face_unattempted'")
    assert "face_attempted_at IS NULL" in definition
