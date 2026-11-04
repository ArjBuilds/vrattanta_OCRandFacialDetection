import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

load_dotenv()

from api.routes import upload, search, admin
from processing.cleanup import delete_expired_selfies
from processing.batch import sweep_stalled_events

STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "./storage"))
SWEEP_INTERVAL_MINUTES = int(os.getenv("SWEEP_INTERVAL_MINUTES", "15"))

app = FastAPI(
    title="FitFunda Media AI",
    description="AI-powered photo discovery — selfie-based face search (OCR-DISABLED: BIB OCR is off)",
    version="1.0.0",
)

# ── CORS (open for dev) ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(upload.router, tags=["Upload"])
app.include_router(search.router, tags=["Search"])
app.include_router(admin.router, tags=["Admin"])

# ── Static files ──────────────────────────────────────────────────────────────
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# ── Storage served as static so image URLs are accessible ────────────────────
# This mounts after /static to avoid conflict
@app.on_event("startup")
def startup():
    # Create required storage directories
    for subdir in ["selfies"]:
        (STORAGE_PATH / subdir).mkdir(parents=True, exist_ok=True)

    # Mount storage directory so uploaded images are accessible via /storage/...
    if STORAGE_PATH.exists():
        app.mount("/storage", StaticFiles(directory=str(STORAGE_PATH)), name="storage")

    # Start scheduler for selfie cleanup + the stalled-batch safety net
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
