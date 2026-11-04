"""
Batch processing pipeline — runs face embedding on all photos for an event.

OCR-DISABLED: BIB OCR used to run here too. It is commented out (not deleted)
so it can be restored; see the OCR-DISABLED note at the top of processing/ocr.py.

Usage:
    python processing/batch.py <event_id>

This is what you run after a photographer bulk-uploads photos post-event.
"""

import os
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.connection import execute_query
# OCR-DISABLED: BIB OCR is off; selfie search is the only active flow.
# from processing.ocr import extract_bib
from processing.face import extract_face_embedding

# OCR-DISABLED: OCR worker pool size.
# # Tesseract runs as a subprocess per call, so OCR calls can run concurrently
# # without fighting over Python's GIL — limited by host CPU, not by an
# # external service.
# OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))

# CompreFace (docker-compose.yml) is configured for UWSGI_PROCESSES=2,
# UWSGI_THREADS=1 — it can only actually work on 2 requests at once
# regardless of how many we send. Raise this only after raising that.
FACE_WORKERS = int(os.getenv("FACE_WORKERS", "2"))

# How old an unattempted photo needs to be before the periodic sweep
# (see sweep_stalled_events) treats it as abandoned rather than "upload
# session still in progress."
SWEEP_GRACE_MINUTES = int(os.getenv("SWEEP_GRACE_MINUTES", "15"))

# How old a batch_runs claim row needs to be before it's treated as
# abandoned (e.g. the process crashed mid-run) and safe to reclaim.
BATCH_RUN_STALE_MINUTES = int(os.getenv("BATCH_RUN_STALE_MINUTES", "60"))


def _record_failure(photo_id: str, stage: str, error: Exception):
    execute_query(
        """
        INSERT INTO media_ai.processing_errors (photo_id, stage, error_message)
        VALUES (%s, %s, %s)
        """,
        (photo_id, stage, str(error)),
    )


def _mark_attempted(photo_id: str, column: str):
    """column must be 'ocr_attempted_at' or 'face_attempted_at' — both are
    hardcoded call sites below, never user input."""
    execute_query(
        f"UPDATE media_ai.event_photos SET {column} = NOW() WHERE photo_id = %s",
        (photo_id,),
    )


# OCR-DISABLED: per-photo OCR step (restore together with the OCR pool in batch_process_event).
# def _process_ocr(photo_id: str, image_path: str) -> bool:
#     """Run OCR for one photo and store results. Returns True on success."""
#     success = False
#     try:
#         bibs = extract_bib(image_path)
#         bib_numbers = [b["bib_number"] for b in bibs]
#         print(f"[OCR] photo {photo_id} — found BIBs: {bib_numbers}")
#
#         for bib in bibs:
#             execute_query(
#                 """
#                 INSERT INTO media_ai.bib_detections
#                     (photo_id, bib_number, confidence, bounding_box)
#                 VALUES (%s, %s, %s, %s)
#                 """,
#                 (
#                     photo_id,
#                     bib["bib_number"],
#                     bib["confidence"],
#                     json.dumps(bib["bounding_box"]),
#                 ),
#             )
#         success = True
#     except Exception as e:
#         print(f"[OCR] photo {photo_id} — failed ({e}), skipping")
#         _record_failure(photo_id, "ocr", e)
#
#     # Mark attempted regardless of outcome — including "found nothing" —
#     # so we never retry this photo's OCR again on a future sweep/re-run.
#     _mark_attempted(photo_id, "ocr_attempted_at")
#     return success


def _process_face(photo_id: str, image_path: str) -> bool:
    """Run face embedding for one photo and store results. Returns True on success."""
    success = False
    try:
        faces = extract_face_embedding(image_path)

        if faces is None:
            print(f"[FACE] photo {photo_id} — no embedding returned, skipping")
        else:
            for face in faces:
                execute_query(
                    """
                    INSERT INTO media_ai.face_embeddings
                        (photo_id, embedding, confidence, face_box)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        photo_id,
                        face["embedding"],
                        face.get("confidence"),
                        json.dumps(face.get("bounding_box")),
                    ),
                )
            print(f"[FACE] photo {photo_id} — stored {len(faces)} face(s)")
            success = True
    except Exception as e:
        print(f"[FACE] photo {photo_id} — failed ({e}), skipping")
        _record_failure(photo_id, "face", e)

    _mark_attempted(photo_id, "face_attempted_at")
    return success


def _claim_batch_run(event_id: str) -> bool:
    """
    Claim the right to run a batch for this event, so two triggers firing
    around the same time (e.g. the periodic sweep and a manual admin
    trigger) can't both spin up a full worker pool for the same
    event at once. A stale claim (older than BATCH_RUN_STALE_MINUTES —
    e.g. a previous run crashed mid-way) is reclaimed automatically.
    Returns False if another run currently holds the claim.
    """
    claimed = execute_query(
        """
        INSERT INTO media_ai.batch_runs (event_id, started_at)
        VALUES (%s, NOW())
        ON CONFLICT (event_id) DO UPDATE
            SET started_at = NOW()
            WHERE media_ai.batch_runs.started_at < NOW() - (%s * INTERVAL '1 minute')
        RETURNING event_id
        """,
        (event_id, BATCH_RUN_STALE_MINUTES),
        fetch="one",
    )
    return claimed is not None


def _release_batch_run(event_id: str):
    execute_query("DELETE FROM media_ai.batch_runs WHERE event_id = %s", (event_id,))


def batch_process_event(event_id: str) -> dict:
    """
    Process all photos for an event: extract face vectors for any photo not
    yet attempted (FACE_WORKERS at a time).

    OCR-DISABLED: this used to also run BIB OCR concurrently in its own pool.

    Safe to call repeatedly for the same event (e.g. from the periodic
    sweep) — photos already attempted are skipped, and only one run per
    event can be active at a time.

    Returns a summary dict with counts.
    """
    if not _claim_batch_run(event_id):
        print(f"[BATCH] Event {event_id} already has a run in progress, skipping")
        # OCR-DISABLED: the "ocr_processed" key was removed from the summary dicts.
        return {"photos_found": 0, "face_processed": 0, "skipped": "already_running"}

    try:
        photos = execute_query(
            """
            SELECT photo_id, image_url, face_attempted_at
            FROM media_ai.event_photos WHERE event_id = %s
            """,
            (event_id,),
            fetch="all",
        )

        if not photos:
            print(f"[BATCH] No photos found for event {event_id}")
            return {"photos_found": 0, "face_processed": 0}

        total = len(photos)
        # OCR-DISABLED: ocr_processed = 0
        face_processed = 0

        print(f"[BATCH] Starting processing for event {event_id} — {total} photos")

        # image_url is stored as the path relative to STORAGE_PATH
        # When running locally, resolve against the project root
        resolved = [
            (
                photo_id,
                image_url if Path(image_url).exists() else str(Path(".") / image_url.lstrip("/")),
                face_attempted_at,
            )
            for photo_id, image_url, face_attempted_at in photos
        ]

        # ── Figure out what actually needs doing ────────────────────────────
        # OCR-DISABLED: ocr_targets = []  (and the per-photo OCR skip checks that filled it)
        face_targets = []
        for photo_id, image_path, face_attempted_at in resolved:
            file_exists = Path(image_path).exists()

            if face_attempted_at is not None:
                print(f"[FACE] photo {photo_id} — already attempted, skipping")
            elif not file_exists:
                print(f"[FACE] photo {photo_id} — image file not found at {image_path}, skipping")
            else:
                face_targets.append((photo_id, image_path))

        # ── Run face embedding ──────────────────────────────────────────────
        # OCR-DISABLED: this block used to run an OCR pool alongside the face
        # pool and print "{n} photo(s) need OCR ({OCR_WORKERS} worker(s))".
        print(f"[BATCH] {len(face_targets)} photo(s) need face embedding ({FACE_WORKERS} worker(s))")

        with ThreadPoolExecutor(max_workers=FACE_WORKERS) as face_pool:
            futures = [
                face_pool.submit(_process_face, photo_id, image_path)
                for photo_id, image_path in face_targets
            ]
            for future in as_completed(futures):
                if future.result():
                    face_processed += 1

        summary = {
            "photos_found": total,
            "face_processed": face_processed,
        }
        print(f"[BATCH] Done — {summary}")
        return summary
    finally:
        _release_batch_run(event_id)


def sweep_stalled_events():
    """
    Catches events with photos that were uploaded but never got
    face-embedded — e.g. the photographer's browser closed before the
    upload portal's auto-trigger (and its retries) could fire. Runs on a
    schedule (see api/main.py).

    Cheap when there's nothing to do: the query below only scans photos
    that haven't been attempted yet (backed by the partial index
    idx_event_photos_face_unattempted in db/schema.sql), and
    batch_process_event() is a near-no-op for events that are already fully
    processed or already mid-run.
    """
    stalled = execute_query(
        """
        SELECT DISTINCT event_id
        FROM media_ai.event_photos
        -- OCR-DISABLED: was (ocr_attempted_at IS NULL OR face_attempted_at IS NULL)
        WHERE face_attempted_at IS NULL
          AND uploaded_at < NOW() - (%s * INTERVAL '1 minute')
        """,
        (SWEEP_GRACE_MINUTES,),
        fetch="all",
    )

    if not stalled:
        return

    print(f"[SWEEP] {len(stalled)} event(s) have unprocessed photos — triggering batch runs")
    for (event_id,) in stalled:
        print(f"[SWEEP] Triggering batch processing for event {event_id}")
        batch_process_event(str(event_id))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python processing/batch.py <event_id>")
        sys.exit(1)

    event_id = sys.argv[1]
    batch_process_event(event_id)
