"""GET /events/{event}/photos/{photo}: signed links, admin key, event ownership and path safety."""

import time
import uuid

import pytest

from api import security
from conftest import insert_photo, make_image_bytes, write_photo_file

EVENT = str(uuid.uuid4())
OTHER_EVENT = str(uuid.uuid4())


@pytest.fixture
def photo(clean_db, storage_dir, jpeg_bytes):
    img_rel, thumb_rel = write_photo_file(storage_dir, EVENT, "p1.jpg", jpeg_bytes)
    return insert_photo(EVENT, img_rel, thumb_rel)


def test_signed_link_serves_the_image(client, photo, jpeg_bytes):
    r = client.get(security.signed_photo_url(EVENT, photo, "full"))
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == jpeg_bytes
    assert "private" in r.headers["cache-control"]
    assert r.headers["x-content-type-options"] == "nosniff"


def test_thumb_and_full_are_separate_signatures(client, photo):
    full = security.signed_photo_url(EVENT, photo, "full")
    assert client.get(full.replace("size=full", "size=thumb")).status_code == 403
    assert client.get(security.signed_photo_url(EVENT, photo, "thumb")).status_code == 200


def test_tampered_signature_rejected(client, photo):
    url = security.signed_photo_url(EVENT, photo, "full")
    assert client.get(url[:-4] + "0000").status_code == 403


def test_expired_link_rejected(client, photo):
    url = security.signed_photo_url(EVENT, photo, "full", now=time.time() - 7200)
    assert client.get(url).status_code == 403


def test_link_cannot_be_replayed_against_another_event(client, photo):
    url = security.signed_photo_url(EVENT, photo, "full").replace(EVENT, OTHER_EVENT)
    assert client.get(url).status_code == 403


def test_valid_signature_but_photo_in_another_event_is_404(client, photo):
    # Sign for (OTHER_EVENT, photo): signature is valid, but the photo does not belong to that event.
    url = security.signed_photo_url(OTHER_EVENT, photo, "full")
    assert client.get(url).status_code == 404


def test_no_credentials_is_401(client, photo):
    assert client.get(f"/events/{EVENT}/photos/{photo}?size=full").status_code == 401


def test_upload_key_does_not_grant_photo_access(client, photo, upload_headers):
    assert client.get(f"/events/{EVENT}/photos/{photo}?size=full", headers=upload_headers).status_code == 401


def test_admin_key_can_fetch_without_a_signature(client, photo, admin_headers):
    assert client.get(f"/events/{EVENT}/photos/{photo}?size=full", headers=admin_headers).status_code == 200


def test_fails_closed_without_secret(client, photo, monkeypatch):
    url = security.signed_photo_url(EVENT, photo, "full")
    monkeypatch.setenv("PHOTO_URL_SECRET", "")
    assert client.get(url).status_code == 503


def test_admin_key_still_works_without_secret(client, photo, admin_headers, monkeypatch):
    monkeypatch.setenv("PHOTO_URL_SECRET", "")
    assert client.get(f"/events/{EVENT}/photos/{photo}?size=full", headers=admin_headers).status_code == 200


def test_bad_ids_and_size_rejected(client, admin_headers):
    assert client.get(f"/events/{EVENT}/photos/not-a-uuid?size=full", headers=admin_headers).status_code == 422
    assert client.get(f"/events/nope/photos/{uuid.uuid4()}?size=full", headers=admin_headers).status_code == 422
    assert client.get(f"/events/{EVENT}/photos/{uuid.uuid4()}?size=huge", headers=admin_headers).status_code == 422


def test_unknown_photo_is_404(client, clean_db, admin_headers):
    assert client.get(f"/events/{EVENT}/photos/{uuid.uuid4()}?size=full", headers=admin_headers).status_code == 404


def test_missing_file_is_404(client, clean_db, admin_headers):
    pid = insert_photo(EVENT, f"{EVENT}/original/gone.jpg", None)
    assert client.get(f"/events/{EVENT}/photos/{pid}?size=full", headers=admin_headers).status_code == 404


def test_missing_thumbnail_is_404(client, clean_db, storage_dir, jpeg_bytes, admin_headers):
    img_rel, _ = write_photo_file(storage_dir, EVENT, "nothumb.jpg", jpeg_bytes, thumb=False)
    pid = insert_photo(EVENT, img_rel, None)
    assert client.get(f"/events/{EVENT}/photos/{pid}?size=thumb", headers=admin_headers).status_code == 404
    assert client.get(f"/events/{EVENT}/photos/{pid}?size=full", headers=admin_headers).status_code == 200


# ── Path safety: a poisoned DB row must never read outside STORAGE_PATH ──────

@pytest.fixture
def secret_file(tmp_path):
    f = tmp_path / "outside" / "secret.jpg"
    f.parent.mkdir()
    f.write_bytes(make_image_bytes("JPEG", color=(1, 2, 3)))
    return f


@pytest.mark.parametrize("style", ["absolute", "dotdot", "symlink"])
def test_paths_outside_storage_are_refused(client, clean_db, storage_dir, secret_file, admin_headers, style, tmp_path):
    if style == "absolute":
        stored = str(secret_file)
    elif style == "dotdot":
        stored = "../outside/secret.jpg"
    else:
        link = storage_dir / EVENT / "original" / "link.jpg"
        link.parent.mkdir(parents=True)
        try:
            link.symlink_to(secret_file)
        except OSError:
            pytest.skip("cannot create symlinks on this system")
        stored = f"{EVENT}/original/link.jpg"
    assert secret_file.exists()
    pid = insert_photo(EVENT, stored, None)
    r = client.get(f"/events/{EVENT}/photos/{pid}?size=full", headers=admin_headers)
    assert r.status_code == 404


def test_legacy_absolute_path_inside_storage_still_served(client, clean_db, storage_dir, jpeg_bytes, admin_headers):
    img_rel, _ = write_photo_file(storage_dir, EVENT, "legacy.jpg", jpeg_bytes, thumb=False)
    pid = insert_photo(EVENT, str(storage_dir / img_rel), None)
    r = client.get(f"/events/{EVENT}/photos/{pid}?size=full", headers=admin_headers)
    assert r.status_code == 200 and r.content == jpeg_bytes


def test_legacy_path_from_another_mount_is_rerooted(client, clean_db, storage_dir, jpeg_bytes, admin_headers):
    img_rel, _ = write_photo_file(storage_dir, EVENT, "legacy2.jpg", jpeg_bytes, thumb=False)
    pid = insert_photo(EVENT, "/somewhere/else/storage/" + img_rel, None)
    r = client.get(f"/events/{EVENT}/photos/{pid}?size=full", headers=admin_headers)
    assert r.status_code == 200


def test_resolve_storage_path_unit(storage_dir, jpeg_bytes):
    img_rel, _ = write_photo_file(storage_dir, EVENT, "u.jpg", jpeg_bytes, thumb=False)
    assert security.resolve_storage_path(img_rel) == (storage_dir / img_rel).resolve()
    assert security.resolve_storage_path("/etc/passwd") is None
    assert security.resolve_storage_path("../../etc/passwd") is None
    assert security.resolve_storage_path("") is None
    assert security.resolve_storage_path(None) is None
