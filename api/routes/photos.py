import mimetypes
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse

from api.security import (
    is_admin_key,
    parse_uuid,
    require_photo_secret,
    resolve_storage_path,
    verify_photo_signature,
)
from db.connection import execute_query

router = APIRouter()


@router.get("/events/{event_id}/photos/{photo_id}")
def get_photo(
    event_id: str,
    photo_id: str,
    size: Literal["thumb", "full"] = "thumb",
    exp: int | None = Query(default=None),
    sig: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None),
):
    """
    Serve one stored photo.

    Access is either a signed, expiring link (the URLs returned by the search
    endpoint) or the admin API key. Anyone else gets 401/403. If PHOTO_URL_SECRET
    is not set, link access fails closed with 503.
    """
    event_id = parse_uuid(event_id, "event_id")
    photo_id = parse_uuid(photo_id, "photo_id")

    if not is_admin_key(x_api_key):
        require_photo_secret()  # 503 when unset
        if exp is None or not sig:
            raise HTTPException(status_code=401, detail="A signed link or admin API key is required")
        if not verify_photo_signature(event_id, photo_id, size, exp, sig):
            raise HTTPException(status_code=403, detail="Link is invalid or has expired")

    row = execute_query(
        """
        SELECT image_url, thumbnail_url
        FROM media_ai.event_photos
        WHERE photo_id = %s AND event_id = %s
        """,
        (photo_id, event_id),
        fetch="one",
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Photo not found")

    stored = row[1] if size == "thumb" else row[0]
    path = resolve_storage_path(stored)  # None if outside STORAGE_PATH
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Photo not found")

    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"},
    )
