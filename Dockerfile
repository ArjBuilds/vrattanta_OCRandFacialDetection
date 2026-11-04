FROM python:3.11-slim

# Install system dependencies
# OCR-DISABLED: tesseract-ocr (OCR engine for BIB detection) is not installed.
# Add it back to the apt-get line, and pytesseract to requirements.txt, to restore OCR.
# libgl1: required by opencv-python-headless
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
