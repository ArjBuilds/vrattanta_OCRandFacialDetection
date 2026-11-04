import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from PIL import Image, ImageOps

from api.models import UploadResponse
from api.security import (
    max_upload_bytes,
    parse_uuid,
    photo_path,
    read_capped,
    reject_oversized_request,
    relative_storage_path,
    require_upload_key,
    storage_root,
    validate_image,
)
from db.connection import execute_query
from processing.batch import batch_process_event

router = APIRouter(dependencies=[Depends(require_upload_key)])


def _save_photo(content: bytes, event_id: str, ext: str):
    """
    Save the original and a thumbnail as <uuid>.<ext> (the client's filename is never
    used). Returns (original_path, thumbnail_path) as absolute Paths inside STORAGE_PATH.
    """
    root = storage_root()
    original_dir = root / event_id / "original"
    thumb_dir = root / event_id / "thumbnails"
    original_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    name = f"{uuid.uuid4().hex}.{ext}"
    original_path = original_dir / name
    original_path.write_bytes(content)

    # Generate thumbnail (300px max on longest side)
    with Image.open(original_path) as img:
        fmt = img.format
        thumb = ImageOps.exif_transpose(img)  # bake in correct rotation before resizing
        thumb.thumbnail((300, 300), Image.LANCZOS)
        thumb_path = thumb_dir / name
        thumb.save(thumb_path, format=fmt)

    return original_path, thumb_path


def _ingest_single(content: bytes, event_id: str) -> UploadResponse:
    """Core ingestion logic shared by single and bulk upload. `content` is already size-capped."""
    ext, width, height = validate_image(content)  # content-based check, not the client's header
    original_path, thumb_path = _save_photo(content, event_id, ext)

    photo_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Paths are stored relative to STORAGE_PATH.
    execute_query(
        """
        INSERT INTO media_ai.event_photos
            (photo_id, event_id, image_url, thumbnail_url, width, height, captured_at, uploaded_at, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            photo_id,
            event_id,
            relative_storage_path(original_path),
            relative_storage_path(thumb_path),
            width,
            height,
            now,
            now,
            "upload",
        ),
    )

    # Unsigned API paths: fetching them needs the admin key or a signed link from search.
    return UploadResponse(
        photo_id=photo_id,
        event_id=event_id,
        image_url=photo_path(event_id, photo_id, "full"),
        thumbnail_url=photo_path(event_id, photo_id, "thumb"),
        width=width,
        height=height,
        uploaded_at=now,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_photo(
    request: Request,
    file: UploadFile = File(...),
    event_id: str = Form(...),
):
    event_id = parse_uuid(event_id, "event_id")
    reject_oversized_request(request, max_upload_bytes())
    content = await read_capped(file, max_upload_bytes())
    # NOTE: face embedding is intentionally NOT triggered here.
    # Triggering batch processing on every single upload caused multiple
    # overlapping batch runs to fire concurrently, overwhelming CompreFace
    # (and racing with each other on the same photo). Once all photos in an
    # upload session are in, trigger processing once via
    # POST /admin/batch/{event_id} — the photographer portal does this
    # automatically after the last file finishes uploading.
    return _ingest_single(content, event_id)


@router.post("/upload/bulk", response_model=list[UploadResponse])
async def upload_bulk(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    event_id: str = Form(...),
):
    event_id = parse_uuid(event_id, "event_id")
    # The cap is per file; the total request is bounded loosely by file count.
    responses = []
    for file in files:
        try:
            content = await read_capped(file, max_upload_bytes())
            responses.append(_ingest_single(content, event_id))
        except HTTPException as e:
            # Log and skip bad files in bulk upload rather than aborting everything
            print(f"[UPLOAD] Skipping a file: {e.detail}")

    # Trigger face embedding for all uploaded photos once the loop is
    # done. This only fires once per bulk upload call, not once per file.
    background_tasks.add_task(batch_process_event, event_id)
    return responses
