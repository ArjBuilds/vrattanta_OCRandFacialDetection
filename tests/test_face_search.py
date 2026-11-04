"""
POST /search/face (real SQL for the vector query, mocked CompreFace) and the
processing.face.extract_face_embedding contract.

CompreFace is mocked by monkeypatching `api.routes.search.extract_face_embedding` or
`requests.post`; it is never contacted.
"""

import json
import uuid
from pathlib import Path

import pytest
import requests

from conftest import face, insert_face, insert_photo, vec
from processing import face as face_mod

EVENT = str(uuid.uuid4())
OTHER_EVENT = str(uuid.uuid4())


# ── helpers ───────────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def compreface_faces(*boxes):
    """A 200 response from /detection/detect with one face per (similarity, w, h)."""
    return FakeResponse(200, {"result": [
        {"box": {"probability": 0.99, "x_min": 0, "y_min": 0, "x_max": w, "y_max": h}, "embedding": vec(sim)}
        for sim, w, h in boxes
    ]})


NO_FACE = FakeResponse(400, {"message": "No face is found in the given image", "code": 28})
BAD_KEY = FakeResponse(404, {"message": "Detection service with API Key x not found", "code": 10})
MISSING_KEY_HEADER = FakeResponse(400, {"message": "Missing header: x-api-key", "code": 20})
SERVER_ERROR = FakeResponse(500, {"message": "Error during synchronization between servers", "code": 41})


def patch_post(monkeypatch, result):
    """Replace requests.post. `result` is a FakeResponse or an exception instance to raise."""
    calls = []

    def fake(url, **kwargs):
        calls.append((url, kwargs))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(requests, "post", fake)
    return calls


def patch_extractor(monkeypatch, faces_or_exc):
    import api.routes.search as search

    seen = {}

    def fake(path):
        seen["path"] = path
        seen["existed_during_call"] = Path(path).exists()
        if isinstance(faces_or_exc, Exception):
            raise faces_or_exc
        return faces_or_exc

    monkeypatch.setattr(search, "extract_face_embedding", fake)
    return seen


def search(client, jpeg_bytes, event_id=EVENT, consent="true"):
    return client.post("/search/face", files={"selfie": ("me.jpg", jpeg_bytes, "image/jpeg")},
                       data={"event_id": event_id, "consent": consent})


def seed(event_id, sim, n=1):
    ids = []
    for _ in range(n):
        pid = insert_photo(event_id)
        insert_face(pid, vec(sim))
        ids.append(pid)
    return ids


# ── Vector query: threshold, ordering, scope, limit ──────────────────────────

def test_returns_matches_above_threshold_best_first(client, clean_db, jpeg_bytes, monkeypatch):
    a, b, c, below, far = (seed(EVENT, s)[0] for s in (0.95, 0.90, 0.72, 0.68, 0.50))
    patch_extractor(monkeypatch, [face(1.0)])
    body = search(client, jpeg_bytes).json()
    assert [p["photo_id"] for p in body["photos"]] == [a, b, c]
    assert [p["confidence"] for p in body["photos"]] == [0.95, 0.9, 0.72]
    assert body["total_matches"] == 3


def test_threshold_is_070(client, clean_db, jpeg_bytes, monkeypatch):
    just_in, just_out = seed(EVENT, 0.7005)[0], seed(EVENT, 0.6995)[0]
    patch_extractor(monkeypatch, [face(1.0)])
    ids = [p["photo_id"] for p in search(client, jpeg_bytes).json()["photos"]]
    assert just_in in ids and just_out not in ids


def test_search_is_scoped_to_the_event(client, clean_db, jpeg_bytes, monkeypatch):
    mine = seed(EVENT, 0.9)[0]
    seed(OTHER_EVENT, 1.0)  # a perfect match, but in another event
    patch_extractor(monkeypatch, [face(1.0)])
    body = search(client, jpeg_bytes).json()
    assert [p["photo_id"] for p in body["photos"]] == [mine]
    assert all(p["event_id"] == EVENT for p in body["photos"])


def test_results_are_limited_to_50(client, clean_db, jpeg_bytes, monkeypatch):
    seed(EVENT, 0.9, n=60)
    patch_extractor(monkeypatch, [face(1.0)])
    assert search(client, jpeg_bytes).json()["total_matches"] == 50


def test_no_matches_is_an_empty_200(client, clean_db, jpeg_bytes, monkeypatch):
    seed(EVENT, 0.3)
    patch_extractor(monkeypatch, [face(1.0)])
    r = search(client, jpeg_bytes)
    assert r.status_code == 200 and r.json()["photos"] == []


def test_result_links_are_signed_and_work(client, clean_db, storage_dir, jpeg_bytes, monkeypatch):
    from conftest import write_photo_file

    img, thumb = write_photo_file(storage_dir, EVENT, "m.jpg", jpeg_bytes)
    pid = insert_photo(EVENT, img, thumb)
    insert_face(pid, vec(0.9))
    patch_extractor(monkeypatch, [face(1.0)])
    photo = search(client, jpeg_bytes).json()["photos"][0]
    assert "&sig=" in photo["image_url"] and "&exp=" in photo["image_url"]
    assert client.get(photo["image_url"]).status_code == 200
    assert client.get(photo["thumbnail_url"]).status_code == 200
    assert "storage" not in photo["image_url"] and str(storage_dir) not in json.dumps(photo)


# ── Consent, temp files, nothing stored ──────────────────────────────────────

def test_consent_row_exists_before_compreface_is_called(client, clean_db, jpeg_bytes, monkeypatch):
    from db.connection import execute_query
    import api.routes.search as search_mod

    rows_at_call = []

    def fake(path):
        rows_at_call.append(execute_query(
            "SELECT consent_given, consent_version, search_image_url FROM media_ai.search_requests", fetch="all"))
        return [face(1.0)]

    monkeypatch.setattr(search_mod, "extract_face_embedding", fake)
    search(client, jpeg_bytes)
    assert rows_at_call == [[(True, "v1", None)]]


def test_consent_row_is_kept_even_if_compreface_fails(client, clean_db, jpeg_bytes, monkeypatch):
    from db.connection import execute_query

    patch_extractor(monkeypatch, RuntimeError("down"))
    with pytest.raises(RuntimeError):
        search(client, jpeg_bytes)
    assert execute_query("SELECT count(*) FROM media_ai.search_requests WHERE consent_given", fetch="one")[0] == 1


def test_no_consent_means_no_row_and_no_compreface_call(client, clean_db, jpeg_bytes, monkeypatch):
    from db.connection import execute_query

    seen = patch_extractor(monkeypatch, [face(1.0)])
    r = search(client, jpeg_bytes, consent="false")
    assert r.status_code == 400
    assert "path" not in seen
    assert execute_query("SELECT count(*) FROM media_ai.search_requests", fetch="one")[0] == 0


def test_invalid_event_id_is_422_and_writes_nothing(client, clean_db, jpeg_bytes, monkeypatch):
    from db.connection import execute_query

    patch_extractor(monkeypatch, [face(1.0)])
    assert search(client, jpeg_bytes, event_id="nope").status_code == 422
    assert execute_query("SELECT count(*) FROM media_ai.search_requests", fetch="one")[0] == 0


def test_selfie_is_not_stored_and_temp_file_is_deleted(client, clean_db, storage_dir, jpeg_bytes, monkeypatch):
    seen = patch_extractor(monkeypatch, [face(1.0)])
    search(client, jpeg_bytes)
    assert seen["existed_during_call"] is True
    assert not Path(seen["path"]).exists()                      # deleted afterwards
    assert storage_dir not in Path(seen["path"]).parents        # never inside STORAGE_PATH
    assert list(storage_dir.rglob("*")) == []                   # nothing written to storage


def test_temp_file_is_deleted_even_when_the_extractor_raises(client, clean_db, jpeg_bytes, monkeypatch):
    seen = patch_extractor(monkeypatch, RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        search(client, jpeg_bytes)
    assert not Path(seen["path"]).exists()


def test_selfie_embedding_is_not_persisted(client, clean_db, jpeg_bytes, monkeypatch):
    from db.connection import execute_query

    patch_extractor(monkeypatch, [face(1.0)])
    search(client, jpeg_bytes)
    assert execute_query("SELECT count(*) FROM media_ai.face_embeddings", fetch="one")[0] == 0


def test_search_fails_closed_without_photo_secret(client, clean_db, jpeg_bytes, monkeypatch):
    from db.connection import execute_query

    monkeypatch.setenv("PHOTO_URL_SECRET", "")
    seen = patch_extractor(monkeypatch, [face(1.0)])
    assert search(client, jpeg_bytes).status_code == 503
    assert "path" not in seen
    assert execute_query("SELECT count(*) FROM media_ai.search_requests", fetch="one")[0] == 0


# ── No face in the selfie -> 422 (real extractor, mocked HTTP) ───────────────

def test_no_face_in_selfie_is_422(client, clean_db, jpeg_bytes, monkeypatch):
    patch_post(monkeypatch, NO_FACE)
    r = search(client, jpeg_bytes)
    assert r.status_code == 422
    assert "No face detected" in r.json()["error"]


def test_undecodable_selfie_is_422_not_500(client, clean_db, jpeg_bytes, monkeypatch):
    patch_extractor(monkeypatch, face_mod.UnreadableImageError("truncated"))
    r = search(client, jpeg_bytes)
    assert r.status_code == 422 and "could not be read" in r.json()["error"]


def test_503_does_not_leak_the_internal_error(client, clean_db, jpeg_bytes, monkeypatch):
    patch_post(monkeypatch, BAD_KEY)
    r = search(client, jpeg_bytes)
    assert r.status_code == 503
    assert "API Key" not in r.text and "10" not in r.json()["detail"].split()
    assert r.headers.get("retry-after") == "30"


# ── CompreFace down must be 503, not "no face" ────────────────────────────────

@pytest.mark.parametrize("failure", [
    requests.exceptions.ConnectionError("refused"),
    requests.exceptions.Timeout("slow"),
    BAD_KEY,
    MISSING_KEY_HEADER,
    SERVER_ERROR,
], ids=["connection-refused", "timeout", "bad-key", "missing-key-header", "compreface-500"])
def test_compreface_failure_is_503_not_422(client, clean_db, jpeg_bytes, monkeypatch, failure):
    patch_post(monkeypatch, failure)
    assert search(client, jpeg_bytes).status_code == 503


def test_missing_compreface_key_is_503(client, clean_db, jpeg_bytes, monkeypatch):
    monkeypatch.setattr(face_mod, "COMPREFACE_API_KEY", "")
    assert search(client, jpeg_bytes).status_code == 503


# ── search must use get_best_face, not faces[0] ──────────────────────────────

def test_search_uses_the_largest_face_in_the_selfie(client, clean_db, jpeg_bytes, monkeypatch):
    background = seed(EVENT, 0.0)[0]       # a photo of the small background person
    wanted = seed(EVENT, 0.9)[0]           # a photo of the selfie's main subject
    small_first = {"embedding": vec(0.0), "confidence": 0.9, "bounding_box": {"x": 0, "y": 0, "w": 20, "h": 20}}
    big_second = {"embedding": vec(1.0), "confidence": 0.99, "bounding_box": {"x": 0, "y": 0, "w": 300, "h": 300}}
    patch_extractor(monkeypatch, [small_first, big_second])
    ids = [p["photo_id"] for p in search(client, jpeg_bytes).json()["photos"]]
    assert wanted in ids and background not in ids


def test_get_best_face_picks_largest_box():
    small = {"bounding_box": {"w": 10, "h": 10}}
    big = {"bounding_box": {"w": 50, "h": 40}}
    assert face_mod.get_best_face([small, big]) is big
    assert face_mod.get_best_face([]) is None


# ── extract_face_embedding contract ──────────────────────────────────────────

@pytest.fixture
def image_file(tmp_path, jpeg_bytes):
    p = tmp_path / "in.jpg"
    p.write_bytes(jpeg_bytes)
    return str(p)


def test_extract_returns_faces_with_box_and_embedding(image_file, monkeypatch):
    calls = patch_post(monkeypatch, compreface_faces((1.0, 100, 80), (0.5, 20, 20)))
    faces = face_mod.extract_face_embedding(image_file)
    assert len(faces) == 2
    assert faces[0]["bounding_box"] == {"x": 0, "y": 0, "w": 100, "h": 80}
    assert len(faces[0]["embedding"]) == 512
    url, kw = calls[0]
    assert url.endswith("/api/v1/detection/detect")
    assert kw["params"]["face_plugins"] == "calculator" and kw["headers"]["x-api-key"] == "test-compreface-key"


def test_extract_sends_a_resized_copy_and_cleans_up(image_file, tmp_path, monkeypatch):
    import tempfile, os

    before = set(os.listdir(tempfile.gettempdir()))
    patch_post(monkeypatch, compreface_faces((1.0, 10, 10)))
    face_mod.extract_face_embedding(image_file)
    assert not [f for f in set(os.listdir(tempfile.gettempdir())) - before if f.endswith(".jpg")]


def test_no_face_returns_empty_list(image_file, monkeypatch):
    patch_post(monkeypatch, NO_FACE)
    assert face_mod.extract_face_embedding(image_file) == []


def test_empty_result_returns_empty_list(image_file, monkeypatch):
    patch_post(monkeypatch, FakeResponse(200, {"result": []}))
    assert face_mod.extract_face_embedding(image_file) == []


@pytest.mark.parametrize("failure", [
    requests.exceptions.ConnectionError("refused"),
    requests.exceptions.Timeout("slow"),
    BAD_KEY,
    MISSING_KEY_HEADER,
    SERVER_ERROR,
], ids=["connection-refused", "timeout", "bad-key", "missing-key-header", "compreface-500"])
def test_infrastructure_errors_raise(image_file, monkeypatch, failure):
    patch_post(monkeypatch, failure)
    with pytest.raises(face_mod.FaceServiceError):
        face_mod.extract_face_embedding(image_file)


def test_missing_api_key_raises(image_file, monkeypatch):
    monkeypatch.setattr(face_mod, "COMPREFACE_API_KEY", "")
    with pytest.raises(face_mod.FaceServiceError):
        face_mod.extract_face_embedding(image_file)


def test_corrupt_image_raises_unreadable(tmp_path, monkeypatch):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xe0 definitely not a jpeg")
    calls = patch_post(monkeypatch, compreface_faces((1.0, 10, 10)))
    with pytest.raises(face_mod.UnreadableImageError):
        face_mod.extract_face_embedding(str(bad))
    assert calls == []  # never sent to CompreFace


# ── one row per photo, best score ────────────────────────────────────────────

def seed_faces(event_id, sims):
    """One photo carrying several faces with the given similarities to the query."""
    pid = insert_photo(event_id)
    for s in sims:
        insert_face(pid, vec(s))
    return pid


def test_photo_with_two_matching_faces_appears_once_with_its_best_score(client, clean_db, jpeg_bytes, monkeypatch):
    pid = seed_faces(EVENT, [0.90, 0.95])
    patch_extractor(monkeypatch, [face(1.0)])
    body = search(client, jpeg_bytes).json()
    assert [p["photo_id"] for p in body["photos"]] == [pid]
    assert body["photos"][0]["confidence"] == 0.95
    assert body["total_matches"] == 1


def test_only_the_matching_face_counts_when_the_other_is_below_threshold(client, clean_db, jpeg_bytes, monkeypatch):
    pid = seed_faces(EVENT, [0.40, 0.88])
    patch_extractor(monkeypatch, [face(1.0)])
    photos = search(client, jpeg_bytes).json()["photos"]
    assert [(p["photo_id"], p["confidence"]) for p in photos] == [(pid, 0.88)]


def test_photos_are_ranked_by_their_best_face(client, clean_db, jpeg_bytes, monkeypatch):
    a = seed_faces(EVENT, [0.80, 0.99])   # best face 0.99
    b = seed_faces(EVENT, [0.95])         # best face 0.95
    c = seed_faces(EVENT, [0.97, 0.96])   # best face 0.97
    patch_extractor(monkeypatch, [face(1.0)])
    photos = search(client, jpeg_bytes).json()["photos"]
    assert [p["photo_id"] for p in photos] == [a, c, b]
    assert [p["confidence"] for p in photos] == [0.99, 0.97, 0.95]


def test_limit_counts_photos_not_faces(client, clean_db, jpeg_bytes, monkeypatch):
    # 120 matching faces in 60 photos, scores interleaved so a photo's two faces are adjacent in the ranking:
    # a LIMIT on faces would cover only 25 photos.
    ids = [seed_faces(EVENT, [0.99 - i * 0.002, 0.99 - i * 0.002 - 0.001]) for i in range(60)]
    patch_extractor(monkeypatch, [face(1.0)])
    photos = search(client, jpeg_bytes).json()["photos"]
    assert len(photos) == 50
    assert len({p["photo_id"] for p in photos}) == 50 and {p["photo_id"] for p in photos} <= set(ids)


# ── threshold default ────────────────────────────────────────────────────────

def test_default_threshold_is_070_when_unset(monkeypatch):
    import importlib
    import api.routes.search as search_mod

    monkeypatch.delenv("MIN_FACE_CONFIDENCE", raising=False)
    try:
        importlib.reload(search_mod)
        assert search_mod.MIN_FACE_CONFIDENCE == 0.70
    finally:
        monkeypatch.setenv("MIN_FACE_CONFIDENCE", "0.70")
        importlib.reload(search_mod)


def test_env_example_matches_the_code_default():
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / ".env.example").read_text()
    assert "\nMIN_FACE_CONFIDENCE=0.70\n" in text
