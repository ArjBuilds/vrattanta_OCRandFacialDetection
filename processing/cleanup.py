"""
Cleanup task — purges selfie images left over from earlier versions.

Current versions never store a selfie (it is held in memory, plus a temp file
outside STORAGE_PATH that is deleted in a finally block). Earlier versions saved
selfies to STORAGE_PATH/selfies and recorded the path in search_requests, so this job
still:
  1. deletes any file in STORAGE_PATH/selfies older than 24 hours, and
  2. for legacy search_requests rows older than 24 hours, deletes the file (only if it
     is inside STORAGE_PATH/selfies) and sets search_image_url = NULL.

It runs hourly via APScheduler (configured in api/main.py).
"""

import time
from pathlib import Path

from api.security import storage_root
from db.connection import execute_query

MAX_AGE_SECONDS = 24 * 60 * 60


def _selfies_dir() -> Path:
    return storage_root() / "selfies"


def _inside_selfies_dir(path: Path) -> bool:
    try:
        path.resolve().relative_to(_selfies_dir().resolve())
        return True
    except (ValueError, OSError):
        return False


def _purge_selfie_directory() -> int:
    """Delete files in STORAGE_PATH/selfies older than 24h. Returns how many were deleted."""
    selfies_dir = _selfies_dir()
    if not selfies_dir.is_dir():
        return 0
    cutoff = time.time() - MAX_AGE_SECONDS
    deleted = 0
    for f in selfies_dir.iterdir():
        try:
            if f.is_file() and not f.is_symlink() and f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
        except OSError as e:
            print(f"[CLEANUP] Failed to delete {f.name}: {e}")
    return deleted


def delete_expired_selfies() -> int:
    """
    Purge leftover selfies (see module docstring). Returns the number of files deleted.
    """
    deleted_count = _purge_selfie_directory()

    expired = execute_query(
        """
        SELECT search_id, search_image_url
        FROM media_ai.search_requests
        WHERE search_image_url IS NOT NULL
          AND created_at < NOW() - INTERVAL '24 hours'
        """,
        fetch="all",
    )

    for search_id, image_url in expired or []:
        if image_url:
            file_path = Path(image_url)
            # Only ever delete inside STORAGE_PATH/selfies, whatever the DB says.
            if _inside_selfies_dir(file_path) and file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_count += 1
                except OSError as e:
                    print(f"[CLEANUP] Failed to delete {file_path.name}: {e}")

        # Null out the path in DB regardless of whether file existed
        execute_query(
            "UPDATE media_ai.search_requests SET search_image_url = NULL WHERE search_id = %s",
            (str(search_id),),
        )

    if deleted_count:
        print(f"[CLEANUP] Deleted {deleted_count} expired selfie(s)")
    return deleted_count
