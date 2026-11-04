FROM python:3.11-slim

# No system packages are needed. libgl1 was only there for opencv-python-headless, which
# nothing in the code imports.
# OCR-DISABLED: to restore OCR, add here:
#   RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*
# and add pytesseract to requirements.txt.

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
