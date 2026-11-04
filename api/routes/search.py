import os
import tempfile
import uuid
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from api.security import (
    enforce_search_rate_limit,
    max_selfie_bytes,
    parse_uuid,
    read_capped,
    reject_oversized_request,
    require_photo_secret,
    signed_photo_url,
    validate_image,
)
from db.connection import execute_query
from processing.face import FaceServiceError, UnreadableImageError, extract_face_embedding, get_best_face
# OCR-DISABLED: BibSearchResponse and BibDetectionResult are only used by the disabled BIB routes.
from api.models import FaceSearchResponse, PhotoMatch  # , BibSearchResponse, BibDetectionResult

router = APIRouter()

MIN_FACE_CONFIDENCE = float(os.getenv("MIN_FACE_CONFIDENCE", "0.75"))
CONSENT_VERSION = os.getenv("CONSENT_VERSION", "v1")


def _log_search_request(event_id: str, search_type: str, bib_number: str = None, image_url: str = None, consent: bool = False) -> str:
    """Write a search_requests row. Selfie images are never stored, so image_url is always NULL."""
    search_id = str(uuid.uuid4())
    execute_query(
        """
        INSERT INTO media_ai.search_requests
            (search_id, event_id, search_type, bib_number, search_image_url, consent_given, consent_version, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (search_id, event_id, search_type, bib_number, image_url, consent, CONSENT_VERSION if consent else None, datetime.utcnow()),
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
    request: Request,
    selfie: UploadFile = File(...),
    event_id: str = Form(...),
    consent: bool = Form(...),
):
    enforce_search_rate_limit(request)
    require_photo_secret()  # result links must be signable; fail before doing any work
    event_id = parse_uuid(event_id, "event_id")

    if not consent:
        raise HTTPException(
            status_code=400,
            detail="Consent is required to perform face search. Please agree to the terms before uploading a selfie.",
        )

    reject_oversized_request(request, max_selfie_bytes())
    selfie_bytes = await read_capped(selfie, max_selfie_bytes())
    ext, _, _ = validate_image(selfie_bytes)

    # Consent record is written before the selfie goes to CompreFace. The selfie itself
    # is never stored: search_image_url stays NULL and no selfie embedding is saved.
    _log_search_request(event_id=event_id, search_type="face", image_url=None, consent=True)

    # CompreFace helper needs a file path; use a temp file outside STORAGE_PATH and
    # always delete it.
    fd, tmp_path = tempfile.mkstemp(prefix="selfie_", suffix=f".{ext}")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(selfie_bytes)

        # Extract embedding from selfie
        try:
            faces = extract_face_embedding(tmp_path)
        except FaceServiceError as e:
            # CompreFace is down, misconfigured or erroring. That says nothing about the
            # selfie, so don't tell the user "no face". Details stay in the server log.
            print(f"[SEARCH] face service unavailable: {e}")
            raise HTTPException(
                status_code=503,
                detail="Face search is temporarily unavailable. Please try again in a few minutes.",
                headers={"Retry-After": "30"},
            )
        except UnreadableImageError as e:
            print(f"[SEARCH] selfie could not be decoded: {e}")
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Selfie could not be read",
                    "detail": "We couldn't read that image. Try taking the selfie again.",
                },
            )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not faces:
        return JSONResponse(
            status_code=422,
            content={
                "error": "No face detected in selfie",
                "detail": "We couldn't detect a face in your photo. Try a clearer, front-facing selfie with good lighting.",
            },
        )

    # CompreFace returns faces in no particular order. Search with the most prominent
    # one (largest bounding box) so a bystander in the selfie isn't used by mistake.
    query_embedding = get_best_face(faces)["embedding"]

    rows = execute_query(
        """
        SELECT
            ep.photo_id,
            ep.event_id,
            ep.thumbnail_url IS NOT NULL AS has_thumb,
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
            image_url=signed_photo_url(str(r[1]), str(r[0]), "full"),
            thumbnail_url=signed_photo_url(str(r[1]), str(r[0]), "thumb") if r[2] else None,
            confidence=round(float(r[3]), 4) if r[3] is not None else None,
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
