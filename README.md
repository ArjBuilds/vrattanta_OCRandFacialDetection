# FitFunda Media AI — Photo Discovery

AI-powered race photo discovery using BIB OCR + Face Search.

**Stack:** Python 3.11 · FastAPI · PostgreSQL + pgvector · Tesseract OCR · Docker

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (running)
- Python 3.11+
- Visual Studio Code (recommended)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed locally — only needed for **Local Dev Mode** below (Docker mode installs it automatically). On Windows, if the installer didn't add it to `PATH`, set `TESSERACT_CMD` in `.env` to the full path of `tesseract.exe`.

---

## First Time Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd media-ai

# 2. Create your .env file
cp .env.example .env

# 3. Build and start all services
docker compose up --build
```

The API will be available at **http://localhost:8000**

---

## Load the Database Schema

After `docker compose up`, run the schema once to create all tables.

**PowerShell (Windows):**
```powershell
Get-Content db/schema.sql | docker exec -i media_ai_db psql -U media_user -d media_ai
```

**Bash (Mac/Linux):**
```bash
docker exec -i media_ai_db psql -U media_user -d media_ai < db/schema.sql
```

Verify it worked:
```bash
docker exec -i media_ai_db psql -U media_user -d media_ai -c "\dt media_ai.*"
```

You should see 7 tables: `event_photos`, `bib_detections`, `face_embeddings`, `search_requests`, `search_results`, `processing_errors`, `batch_runs`.

---

## Daily Startup (Local Dev Mode)

For faster iteration, run the API locally (with hot reload) while keeping only the DB in Docker:

```powershell
# 1. Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run once:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. Start the database container only
docker start media_ai_db

# 3. Start the API with hot reload
uvicorn api.main:app --reload
```

---

## Test the API

**Swagger UI (interactive docs):**
```
http://localhost:8000/docs
```

**Upload a photo (curl):**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/photo.jpg" \
  -F "event_id=00000000-0000-0000-0000-000000000001"
```

**BIB search:**
```bash
curl "http://localhost:8000/search/bib?event_id=00000000-0000-0000-0000-000000000001&bib=2451"
```

**Health check:**
```bash
curl http://localhost:8000/health
```

---

## Portals

| Portal | URL |
|---|---|
| Photographer Upload Portal | http://localhost:8000/upload_portal |
| Participant Photo Search | http://localhost:8000/search |
| API Docs (Swagger) | http://localhost:8000/docs |

> **Note:** The templates are served from the `templates/` folder. If you're running in Docker, they're volume-mounted so changes reflect immediately without rebuild.

---

## Run Batch Processing

The photographer upload portal triggers this automatically once every file in
an upload session has finished uploading — no manual step needed for photos
uploaded through the portal. OCR and face embedding are intentionally **not**
run per individual photo upload (that caused overlapping batch runs that
overwhelmed CompreFace); they only run once per event, after the whole
session is done.

If you've ingested photos some other way (direct API calls, restoring from
backup, etc.) and need to trigger this manually:

```bash
python processing/batch.py 00000000-0000-0000-0000-000000000001
```

Replace the UUID with your actual event ID. The script:
1. Skips photos already processed (safe to re-run)
2. Extracts BIB numbers via Tesseract OCR (`OCR_WORKERS` concurrent, default 4)
3. Extracts face embeddings via CompreFace (`FACE_WORKERS` concurrent, default 2; skipped with a log message if `COMPREFACE_API_KEY` isn't set)
4. Logs progress per photo, e.g. `[OCR] photo <id> — found BIBs: [2451]`
5. Records any failures in `media_ai.processing_errors` instead of silently dying — a corrupt photo won't take down the rest of the batch

---

## Admin API Endpoints

**Trigger batch processing manually (async):** this is what the upload portal calls automatically — use this directly only if you uploaded photos some other way.
```bash
curl -X POST http://localhost:8000/admin/batch/00000000-0000-0000-0000-000000000001
```

**Event stats:**
```bash
curl http://localhost:8000/admin/events/00000000-0000-0000-0000-000000000001/stats
```

---

## Selfie Auto-Deletion

Raw selfie images are automatically deleted after 24 hours. Selfie embeddings are not stored at all. A `search_requests` row (event, time, consent flag) is kept after the image is deleted.

The cleanup job runs every hour via APScheduler (starts with the API). To simulate in development, you can call the cleanup function directly:

```python
from processing.cleanup import delete_expired_selfies
delete_expired_selfies()
```

---

## Face Search — Powered by CompreFace

Face search is implemented end-to-end via a self-hosted [CompreFace](https://github.com/exadel-inc/CompreFace) stack, already wired into `docker-compose.yml` (`compreface-core`, `compreface-api`, `compreface-admin`, `compreface-ui`, plus its own Postgres).

**Setup:**
1. `docker compose up --build` to bring up CompreFace alongside the rest of the stack.
2. Open the CompreFace admin UI at **http://localhost:8001**, create an account, then create a **Face Detection** service (not Face Recognition; the code calls `/api/v1/detection/detect`) and copy its API key.
3. Add the key to `.env`:
   ```
   COMPREFACE_API_KEY=your_key_here
   ```
4. Restart the API container (or local `uvicorn` process) so it picks up the key.

Without `COMPREFACE_API_KEY` set, `extract_face_embedding()` in `processing/face.py` logs a message and returns `None` for every photo — OCR/BIB search still works, but face search will report no matches.

`processing/face.py` pre-resizes and EXIF-corrects images before sending them to CompreFace's `/detection/detect` endpoint (with the `calculator` plugin for embeddings) — sending full-resolution camera photos directly was found to break CompreFace's own internal downscaling.

---

## Project Structure

```
media-ai/
├── api/
│   ├── main.py              # FastAPI app, scheduler, startup
│   ├── models.py            # Pydantic models
│   └── routes/
│       ├── upload.py        # POST /upload, POST /upload/bulk
│       ├── search.py        # GET /search/bib, POST /search/face
│       └── admin.py         # POST /admin/batch, GET /admin/events/stats
├── db/
│   ├── schema.sql           # All CREATE TABLE + index statements
│   └── connection.py        # psycopg2 connection pool
├── processing/
│   ├── ocr.py               # Tesseract BIB extraction
│   ├── face.py              # CompreFace face embedding
│   ├── batch.py             # Full batch pipeline script
│   └── cleanup.py           # Selfie auto-deletion
├── templates/
│   ├── search.html          # Participant photo discovery UI
│   └── upload_portal.html   # Photographer bulk upload portal
├── storage/                 # Local photo storage (dev only)
├── static/                  # Static assets
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```
