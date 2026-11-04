"""Upload validation: UUID event ids, streaming size cap, content-based type check, pixel limit,
safe stored names and relative paths."""

import io
import os
import re
import uuid

import pytest
from fastapi import HTTPException
from PIL import Image

from api import security
from conftest import make_image_bytes

EVENT = str(uuid.uuid4())


def _post(client, headers, data: bytes, filename="a.jpg", content_type="image/jpeg", event_id=EVENT):
    return client.post("/upload", files={"file": (filename, data, content_type)}, data={"event_id": event_id}, headers=headers)


# ── validate_image (pure) ─────────────────────────────────────────────────────

def test_validate_accepts_jpeg_and_png(jpeg_bytes, png_bytes):
    assert security.validate_image(jpeg_bytes) == ("jpg", 64, 48)
    assert security.validate_image(png_bytes) == ("png", 64, 48)


def test_validate_checks_content_not_declared_type():
    gif = make_image_bytes("GIF")
    with pytest.raises(HTTPException) as e:
        security.validate_image(gif)
    assert e.value.status_code == 415


@pytest.mark.parametrize("data", [b"", b"plain text", b"\xff\xd8\xff\xe0 truncated"])
def test_validate_rejects_garbage(data):
    with pytest.raises(HTTPException) as e:
        security.validate_image(data)
    assert e.value.status_code == 400


def test_validate_rejects_decompression_bomb(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10_000)
    big = make_image_bytes("PNG", size=(200, 200))  # 40,000 px: tiny file, over the limit
    with pytest.raises(HTTPException) as e:
        security.validate_image(big)
    assert e.value.status_code == 413


def test_validate_rejects_pixels_between_limit_and_2x(monkeypatch):
    # Pillow only *warns* between 1x and 2x its limit; we must still refuse.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 30_000)
    img = make_image_bytes("PNG", size=(200, 200))  # 40,000 px = 1.33x
    with pytest.raises(HTTPException) as e:
        security.validate_image(img)
    assert e.value.status_code == 413


def test_max_image_pixels_is_set():
    assert Image.MAX_IMAGE_PIXELS is not None and Image.MAX_IMAGE_PIXELS <= 64_000_000


# ── HTTP: rejected before anything is stored ─────────────────────────────────

def test_non_uuid_event_id_rejected(client, upload_headers, jpeg_bytes, storage_dir):
    for bad in ["../../etc", "abc", ""]:
        r = _post(client, upload_headers, jpeg_bytes, event_id=bad)
        assert r.status_code == 422
    assert list(storage_dir.iterdir()) == []


def test_text_file_with_image_header_rejected(client, upload_headers, storage_dir):
    r = _post(client, upload_headers, b"<?php echo 1; ?>", filename="shell.php.jpg")
    assert r.status_code == 400
    assert list(storage_dir.rglob("*.*")) == []


def test_gif_rejected_even_if_declared_jpeg(client, upload_headers, storage_dir):
    r = _post(client, upload_headers, make_image_bytes("GIF"))
    assert r.status_code == 415
    assert list(storage_dir.rglob("*.*")) == []


def test_size_cap_enforced_while_streaming(client, upload_headers, monkeypatch, storage_dir):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    r = _post(client, upload_headers, os.urandom(2 * 1024 * 1024))
    assert r.status_code == 413
    assert list(storage_dir.rglob("*.*")) == []


def test_read_capped_stops_a_body_that_slips_past_the_header_check(client, upload_headers, monkeypatch):
    # 1.1 MB is within the Content-Length slack, so only the streaming cap can catch it.
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    r = _post(client, upload_headers, os.urandom(int(1.1 * 1024 * 1024)))
    assert r.status_code == 413


def test_file_just_under_cap_is_not_rejected_for_size(client, upload_headers, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    r = _post(client, upload_headers, os.urandom(900 * 1024))
    assert r.status_code == 400  # fails the image check, not the size cap


def test_selfie_has_a_smaller_cap(client, monkeypatch):
    monkeypatch.setenv("MAX_SELFIE_MB", "1")
    r = client.post("/search/face", files={"selfie": ("s.jpg", os.urandom(2 * 1024 * 1024), "image/jpeg")},
                    data={"event_id": EVENT, "consent": "true"})
    assert r.status_code == 413


def test_selfie_content_is_validated(client):
    r = client.post("/search/face", files={"selfie": ("s.jpg", b"not an image", "image/jpeg")},
                    data={"event_id": EVENT, "consent": "true"})
    assert r.status_code == 400


# ── HTTP: accepted uploads (real SQL) ────────────────────────────────────────

def test_upload_stores_uuid_name_and_relative_path(client, upload_headers, jpeg_bytes, clean_db, storage_dir):
    from db.connection import execute_query

    r = _post(client, upload_headers, jpeg_bytes, filename="../../evil name.php")
    assert r.status_code == 200
    body = r.json()
    row = execute_query("SELECT image_url, thumbnail_url, width, height FROM media_ai.event_photos WHERE photo_id=%s",
                        (body["photo_id"],), fetch="one")
    image_rel, thumb_rel, w, h = row
    assert re.fullmatch(rf"{EVENT}/original/[0-9a-f]{{32}}\.jpg", image_rel)
    assert re.fullmatch(rf"{EVENT}/thumbnails/[0-9a-f]{{32}}\.jpg", thumb_rel)
    assert (w, h) == (64, 48)
    assert (storage_dir / image_rel).is_file() and (storage_dir / thumb_rel).is_file()
    assert "evil" not in image_rel and ".php" not in image_rel
    # Response carries unsigned API paths, never filesystem paths.
    assert body["image_url"].startswith(f"/events/{EVENT}/photos/")
    assert str(storage_dir) not in str(body)


def test_png_upload_keeps_png_extension(client, upload_headers, png_bytes, clean_db):
    r = _post(client, upload_headers, png_bytes, filename="x.jpg", content_type="image/jpeg")
    assert r.status_code == 200
    from db.connection import execute_query

    (image_rel,) = execute_query("SELECT image_url FROM media_ai.event_photos WHERE photo_id=%s",
                                 (r.json()["photo_id"],), fetch="one")
    assert image_rel.endswith(".png")


def test_bulk_upload_skips_bad_files_and_keeps_good(client, upload_headers, jpeg_bytes, png_bytes, clean_db, monkeypatch):
    import api.routes.upload as up

    monkeypatch.setattr(up, "batch_process_event", lambda event_id: None)  # don't run the pipeline
    files = [
        ("files", ("a.jpg", jpeg_bytes, "image/jpeg")),
        ("files", ("b.txt", b"hello", "image/jpeg")),
        ("files", ("c.png", png_bytes, "image/png")),
    ]
    r = client.post("/upload/bulk", files=files, data={"event_id": EVENT}, headers=upload_headers)
    assert r.status_code == 200
    assert len(r.json()) == 2
