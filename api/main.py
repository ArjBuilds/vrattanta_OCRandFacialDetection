import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

load_dotenv()

from api.routes import upload, search, admin, photos
from api.security import log_startup_warnings
from processing.cleanup import delete_expired_selfies
from processing.batch import sweep_stalled_events

SWEEP_INTERVAL_MINUTES = int(os.getenv("SWEEP_INTERVAL_MINUTES", "15"))

app = FastAPI(
    title="Selfie Photo Search",
    description="Find your photos in an event by uploading a selfie (face search with CompreFace + pgvector). OCR-DISABLED: BIB number search is off.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Same-origin only unless CORS_ORIGINS (comma-separated) is set. The bundled portals
# are served by this app, so they need no CORS. Credentials are never combined with "*".
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials="*" not in _cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(upload.router, tags=["Upload"])
app.include_router(search.router, tags=["Search"])
app.include_router(admin.router, tags=["Admin"])
app.include_router(photos.router, tags=["Photos"])

# ── Static files ──────────────────────────────────────────────────────────────
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Photos are NOT served from a static mount. They are served by
# GET /events/{event_id}/photos/{photo_id} (api/routes/photos.py) via signed, expiring links.
@app.on_event("startup")
def startup():
    log_startup_warnings()

    # Start scheduler for leftover-selfie cleanup + the stalled-batch safety net
    scheduler = BackgroundScheduler()
    scheduler.add_job(delete_expired_selfies, "interval", hours=1, id="selfie_cleanup")
    scheduler.add_job(sweep_stalled_events, "interval", minutes=SWEEP_INTERVAL_MINUTES, id="batch_sweep")
    scheduler.start()
    print(
        "[STARTUP] APScheduler started — selfie cleanup runs every hour, "
        f"stalled-batch sweep runs every {SWEEP_INTERVAL_MINUTES} minutes"
    )

@app.get("/upload_portal")
async def upload_portal(request: Request):
    return templates.TemplateResponse(
        request,
        "upload_portal.html",
        {"request": request}
    )


@app.get("/search")
async def participant_portal(request: Request):
    return templates.TemplateResponse(
        request,
        "search.html",
        {"request": request}
    )

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
