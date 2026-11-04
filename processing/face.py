import os
import tempfile
import requests
from PIL import Image, ImageOps

COMPREFACE_URL = os.getenv("COMPREFACE_URL", "http://localhost:8001")
COMPREFACE_API_KEY = os.getenv("COMPREFACE_API_KEY", "")

# CompreFace's core service has an internal IMG_LENGTH_LIMIT of 640px.
# Sending full-resolution camera photos (e.g. 4608x3072) directly relies on
# CompreFace's own internal downscaling, which was found to produce a
# degraded image that breaks its detection pipeline — it would detect a face
# internally (visible in compreface-core logs) but then fail with
# "No face is found" or "Something went wrong" right after.
#
# Pre-resizing (and fixing EXIF rotation) on our side before sending fixed
# this completely — confirmed by testing a manually resized copy directly
# through the CompreFace UI.
MAX_DIMENSION = 640

# CompreFace answers "no face in this image" with HTTP 400 and this error code
# (not with 200 and an empty list).
NO_FACE_ERROR_CODE = 28


class FaceServiceError(Exception):
    """
    CompreFace could not be used: unreachable, timed out, wrong or missing API key,
    or it returned an error. The photo itself is fine, so callers should NOT treat the
    photo as processed — it should be retried later.
    """


class UnreadableImageError(Exception):
    """The image file cannot be decoded (corrupt or unsupported). Retrying will not help."""


def _prepare_image_for_compreface(image_path: str) -> str:
    """
    Fix orientation and resize an image to CompreFace's expected max
    dimension before sending it. Returns the path to a temp JPEG file —
    caller is responsible for deleting it after use.

    Raises UnreadableImageError if the file is not a decodable image.
    """
    try:
        with Image.open(image_path) as opened:
            img = ImageOps.exif_transpose(opened)  # correct rotation before anything else
            img = img.convert("RGB")  # CompreFace expects color, not grayscale

            w, h = img.size
            if max(w, h) > MAX_DIMENSION:
                scale = MAX_DIMENSION / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            try:
                img.save(tmp.name, format="JPEG", quality=90)
            except Exception:
                tmp.close()
                os.unlink(tmp.name)
                raise
            tmp.close()
            return tmp.name
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        # UnidentifiedImageError and truncated files are OSErrors.
        raise UnreadableImageError(f"{os.path.basename(image_path)}: {e}") from e


def extract_face_embedding(image_path: str) -> list[dict]:
    """
    Send an image to CompreFace detection endpoint and return all detected
    face embeddings.

    Uses /detection/detect with the calculator plugin, which returns embeddings
    without needing a subject collection — ideal for batch processing event
    photos where we store vectors ourselves in pgvector.

    Returns a list of dicts (one per detected face):
        {
            "embedding": [float, ...],   # 512-dim vector
            "confidence": float,         # face detection probability
            "bounding_box": {x, y, w, h}
        }

    Returns [] when the image was processed fine but contains no face.

    Raises:
        FaceServiceError     CompreFace is down/slow, the API key is missing or wrong, or
                             it returned an error. Not a statement about the image.
        UnreadableImageError the image file itself cannot be decoded (permanent).
    """
    if not COMPREFACE_API_KEY:
        raise FaceServiceError("COMPREFACE_API_KEY is not set")

    prepared_path = _prepare_image_for_compreface(image_path)

    url = f"{COMPREFACE_URL}/api/v1/detection/detect"
    headers = {"x-api-key": COMPREFACE_API_KEY}
    params = {
        "face_plugins": "calculator",  # calculator plugin returns the embedding
        "limit": 0,  # 0 = no limit on number of faces returned (this is also the default)
    }

    try:
        with open(prepared_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(url, headers=headers, params=params, files=files, timeout=30)
    except requests.exceptions.ConnectionError as e:
        raise FaceServiceError(f"could not connect to CompreFace at {COMPREFACE_URL} — is it running?") from e
    except requests.exceptions.Timeout as e:
        raise FaceServiceError(f"CompreFace request timed out for {os.path.basename(image_path)}") from e
    except requests.exceptions.RequestException as e:
        raise FaceServiceError(f"CompreFace request failed: {e}") from e
    finally:
        os.unlink(prepared_path)

    if response.status_code != 200:
        if response.status_code == 400 and _error_code(response) == NO_FACE_ERROR_CODE:
            print(f"[FACE] No faces detected in {os.path.basename(image_path)}")
            return []
        raise FaceServiceError(f"CompreFace returned {response.status_code}: {response.text[:200]}")

    try:
        data = response.json()
    except ValueError as e:
        raise FaceServiceError(f"CompreFace returned a non-JSON 200 response: {response.text[:200]}") from e

    # Response shape from /detection/detect with calculator plugin:
    # {
    #   "result": [
    #     {
    #       "box": {
    #         "probability": 0.9997,
    #         "x_max": 1420, "y_max": 1368,
    #         "x_min": 548,  "y_min": 295
    #       },
    #       "plugins_versions": {...},
    #       "embedding": [0.0231, -0.0412, ...]   <- 512-dim float list
    #     },
    #     ... one entry per detected face
    #   ]
    # }

    results = data.get("result", [])
    if not results:
        print(f"[FACE] No faces detected in {os.path.basename(image_path)}")
        return []

    faces = []
    for face in results:
        embedding = face.get("embedding")
        box = face.get("box", {})
        confidence = box.get("probability", 0.0)

        if not embedding:
            continue

        faces.append({
            "embedding": embedding,
            "confidence": round(float(confidence), 4),
            "bounding_box": {
                "x": box.get("x_min", 0),
                "y": box.get("y_min", 0),
                "w": box.get("x_max", 0) - box.get("x_min", 0),
                "h": box.get("y_max", 0) - box.get("y_min", 0),
            },
        })

    return faces


def _error_code(response) -> int | None:
    try:
        return response.json().get("code")
    except (ValueError, AttributeError):
        return None


def get_best_face(faces: list[dict]) -> dict | None:
    """
    From a list of detected faces, return the most prominent one —
    the largest by bounding box area (usually the runner closest to camera).
    """
    if not faces:
        return None
    return max(faces, key=lambda f: f["bounding_box"]["w"] * f["bounding_box"]["h"])
