import os
import uuid
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from db.connection import execute_query
from processing.face import extract_face_embedding
# OCR-DISABLED: BibSearchResponse and BibDetectionResult are only used by the disabled BIB routes.
from api.models import FaceSearchResponse, PhotoMatch  # , BibSearchResponse, BibDetectionResult

router = APIRouter()

STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "./storage"))
MIN_FACE_CONFIDENCE = float(os.getenv("MIN_FACE_CONFIDENCE", "0.75"))


def _log_search_request(event_id: str, search_type: str, bib_number: str = None, image_url: str = None, consent: bool = False) -> str:
    search_id = str(uuid.uuid4())
    execute_query(
        """
        INSERT INTO media_ai.search_requests
            (search_id, event_id, search_type, bib_number, search_image_url, consent_given, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (search_id, event_id, search_type, bib_number, image_url, consent, datetime.utcnow()),
    )
    return search_id


# OCR-DISABLED: GET /search/bib (BIB number search); see the OCR-DISABLED note at the top of processing/ocr.py.
# @router.get("/search/bib", response_model=BibSearchResponse)
# def search_by_bib(event_id: str, bib: str):
#     rows = execute_query(
#         """
#         SELECT
#             ep.photo_id,
#             ep.event_id,
#             ep.image_url,
#             ep.thumbnail_url,
#             bd.confidence
#         FROM media_ai.bib_detections bd
#         JOIN media_ai.event_photos ep ON ep.photo_id = bd.photo_id
#         WHERE ep.event_id = %s
#           AND bd.bib_number = %s
#         ORDER BY bd.confidence DESC
#         """,
#         (event_id, bib),
#         fetch="all",
#     )
#
#     _log_search_request(event_id=event_id, search_type="bib", bib_number=bib)
#
#     photos = [
#         PhotoMatch(
#             photo_id=str(r[0]),
#             event_id=str(r[1]),
#             image_url=r[2],
#             thumbnail_url=r[3],
#             confidence=float(r[4]) if r[4] is not None else None,
#         )
#         for r in (rows or [])
#     ]
#
#     return BibSearchResponse(
#         event_id=event_id,
#         bib_number=bib,
#         total_matches=len(photos),
#         photos=photos,
#     )


@router.post("/search/face", response_model=FaceSearchResponse)
async def search_by_face(
    selfie: UploadFile = File(...),
    event_id: str = Form(...),
    consent: bool = Form(...),
):
    if not consent:
        raise HTTPException(
            status_code=400,
            detail="Consent is required to perform face search. Please agree to the terms before uploading a selfie.",
        )

    # Save selfie temporarily
    selfies_dir = STORAGE_PATH / "selfies"
    selfies_dir.mkdir(parents=True, exist_ok=True)
    selfie_filename = f"{uuid.uuid4().hex}.jpg"
    selfie_path = selfies_dir / selfie_filename
    selfie_path.write_bytes(await selfie.read())

    search_id = _log_search_request(
        event_id=event_id,
        search_type="face",
        image_url=str(selfie_path),
        consent=True,
    )

    # Extract embedding from selfie
    faces = extract_face_embedding(str(selfie_path))

    if faces is None:
        return JSONResponse(
            status_code=422,
            content={
                "error": "No face detected in selfie",
                "detail": "We couldn't detect a face in your photo. Try a clearer, front-facing selfie with good lighting.",
            },
        )

    # Use the first (best) face embedding for the similarity search
    query_embedding = faces[0]["embedding"]

    rows = execute_query(
        """
        SELECT
            ep.photo_id,
            ep.event_id,
            ep.image_url,
            ep.thumbnail_url,
            1 - (fe.embedding <=> %s::vector) AS confidence
        FROM media_ai.face_embeddings fe
        JOIN media_ai.event_photos ep ON ep.photo_id = fe.photo_id
        WHERE ep.event_id = %s
          AND 1 - (fe.embedding <=> %s::vector) >= %s
        ORDER BY confidence DESC
        LIMIT 50
        """,
        (query_embedding, event_id, query_embedding, MIN_FACE_CONFIDENCE),
        fetch="all",
    )

    photos = [
        PhotoMatch(
            photo_id=str(r[0]),
            event_id=str(r[1]),
            image_url=r[2],
            thumbnail_url=r[3],
            confidence=round(float(r[4]), 4) if r[4] is not None else None,
        )
        for r in (rows or [])
    ]

    return FaceSearchResponse(
        event_id=event_id,
        total_matches=len(photos),
        photos=photos,
    )


# OCR-DISABLED: GET /photos/{photo_id}/bibs; see the OCR-DISABLED note at the top of processing/ocr.py.
# @router.get("/photos/{photo_id}/bibs", response_model=list[BibDetectionResult])
# def get_photo_bibs(photo_id: str):
#     rows = execute_query(
#         """
#         SELECT detection_id, bib_number, confidence, bounding_box
#         FROM media_ai.bib_detections
#         WHERE photo_id = %s
#         ORDER BY confidence DESC
#         """,
#         (photo_id,),
#         fetch="all",
#     )
#
#     return [
#         BibDetectionResult(
#             detection_id=str(r[0]),
#             bib_number=r[1],
#             confidence=float(r[2]) if r[2] is not None else 0.0,
#             bounding_box=r[3],
#         )
#         for r in (rows or [])
#     ]
