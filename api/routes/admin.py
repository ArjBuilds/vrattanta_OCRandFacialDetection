from fastapi import APIRouter, BackgroundTasks

from db.connection import execute_query
from processing.batch import batch_process_event

router = APIRouter()


@router.post("/admin/batch/{event_id}")
def trigger_batch(event_id: str, background_tasks: BackgroundTasks):
    """Trigger batch face embedding processing for an event. (OCR-DISABLED: OCR no longer runs.)"""
    background_tasks.add_task(batch_process_event, event_id)
    return {"status": "processing started", "event_id": event_id}


@router.get("/admin/events/{event_id}/stats")
def event_stats(event_id: str):
    """Return processing stats for an event."""
    total_photos = execute_query(
        "SELECT COUNT(*) FROM media_ai.event_photos WHERE event_id = %s",
        (event_id,),
        fetch="one",
    )

    # OCR-DISABLED: BIB detection count.
    #     processed_ocr = execute_query(
    #         """
    #         SELECT COUNT(DISTINCT bd.photo_id)
    #         FROM media_ai.bib_detections bd
    #         JOIN media_ai.event_photos ep ON ep.photo_id = bd.photo_id
    #         WHERE ep.event_id = %s
    #         """,
    #         (event_id,),
    #         fetch="one",
    #     )

    processed_face = execute_query(
        """
        SELECT COUNT(DISTINCT fe.photo_id)
        FROM media_ai.face_embeddings fe
        JOIN media_ai.event_photos ep ON ep.photo_id = fe.photo_id
        WHERE ep.event_id = %s
        """,
        (event_id,),
        fetch="one",
    )

    return {
        "event_id": event_id,
        "total_photos": total_photos[0] if total_photos else 0,
        # OCR-DISABLED: "photos_with_bib_detections": processed_ocr[0] if processed_ocr else 0,
        "photos_with_face_embeddings": processed_face[0] if processed_face else 0,
    }
