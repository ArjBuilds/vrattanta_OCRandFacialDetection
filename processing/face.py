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


def _prepare_image_for_compreface(image_path: str) -> str:
    """
    Fix orientation and resize an image to CompreFace's expected max
    dimension before sending it. Returns the path to a temp JPEG file —
    caller is responsible for deleting it after use.
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)  # correct rotation before anything else
    img = img.convert("RGB")  # CompreFace expects color, not grayscale

    w, h = img.size
    if max(w, h) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format="JPEG", quality=90)
    tmp.close()
    return tmp.name


def extract_face_embedding(image_path: str) -> list[dict] | None:
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

    Returns None if CompreFace is unreachable, returns an error, or detects
    no faces in the image.
    """
    if not COMPREFACE_API_KEY:
        print("[FACE] COMPREFACE_API_KEY not set — skipping face embedding")
        return None

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
    except requests.exceptions.ConnectionError:
        print(f"[FACE] Could not connect to CompreFace at {COMPREFACE_URL} — is it running?")
        return None
    except requests.exceptions.Timeout:
        print(f"[FACE] CompreFace request timed out for {image_path}")
        return None
    finally:
        os.unlink(prepared_path)

    if response.status_code != 200:
        print(f"[FACE] CompreFace returned {response.status_code}: {response.text[:200]}")
        return None

    data = response.json()

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
        print(f"[FACE] No faces detected in {image_path}")
        return None

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

    return faces if faces else None


def get_best_face(faces: list[dict]) -> dict | None:
    """
    From a list of detected faces, return the most prominent one —
    the largest by bounding box area (usually the runner closest to camera).
    """
    if not faces:
        return None
    return max(faces, key=lambda f: f["bounding_box"]["w"] * f["bounding_box"]["h"])
