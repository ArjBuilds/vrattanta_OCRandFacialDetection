# OCR-DISABLED: nothing imports this module any more. It is kept intact so OCR can be
# restored.
#
# Why: OCR was tried first. BIB OCR proved unreliable on real race photos even
# with preprocessing, while face matching found photos that OCR missed, so the
# project's scope moved to selfie search.
#
# To restore: search the repo for "OCR-DISABLED:" and undo each site (batch.py,
# api/routes/search.py, api/routes/admin.py, templates/*.html, Dockerfile,
# requirements.txt, .env.example, docker-compose.yml). The schema was left intact.

import os
import re
import tempfile
from pathlib import Path

import pytesseract

# In Docker, tesseract-ocr (installed via apt) is already on PATH.
# For local dev on Windows, set TESSERACT_CMD in .env if it's not on PATH.
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

from PIL import Image, ImageEnhance, ImageOps

MIN_BIB_CONFIDENCE = float(os.getenv("MIN_BIB_CONFIDENCE", "60"))
MAX_DIMENSION = 1080


def preprocess_image(image_path: str) -> str:
    """
    Prepare an image for OCR:
    - Fix orientation using EXIF metadata (many cameras store rotation as
      metadata rather than physically rotating pixels — without this,
      sideways photos fail OCR even though they look fine to the eye)
    - Convert to grayscale
    - Boost contrast
    - Resize to max 1080px on longest side (preserving aspect ratio)
    - Save to a temp file and return its path
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)  # correct rotation before anything else
    img = img.convert("L")  # grayscale

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    w, h = img.size
    if max(w, h) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


def extract_bib(image_path: str) -> list[dict]:
    """
    Run Tesseract OCR on an image and extract BIB numbers.

    Returns a list of dicts:
        {
            "bib_number": str,          # 3–5 digit number
            "confidence": float,        # Tesseract confidence stored as 0.0–1.0
            "bounding_box": {x, y, w, h}
        }

    Filters:
        - digits only (no letters)
        - length 3–5 digits
        - confidence > MIN_BIB_CONFIDENCE (env, default 60, on Tesseract's 0–100 scale)
    """
    preprocessed_path = preprocess_image(image_path)

    try:
        data = pytesseract.image_to_data(
            preprocessed_path,
            output_type=pytesseract.Output.DICT,
            config="--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789",
        )
    finally:
        Path(preprocessed_path).unlink(missing_ok=True)

    results = []
    seen = set()

    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue

        # digits only, 3–5 characters
        if not re.fullmatch(r"\d{3,5}", text):
            continue

        conf = float(data["conf"][i])
        if conf < MIN_BIB_CONFIDENCE:
            continue

        # deduplicate same bib in same image
        if text in seen:
            continue
        seen.add(text)

        results.append(
            {
                "bib_number": text,
                "confidence": round(conf / 100, 4),
                "bounding_box": {
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i],
                },
            }
        )

    return results
