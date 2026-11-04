-- FitFunda Media AI — PostgreSQL Schema
-- Run with: Get-Content db/schema.sql | docker exec -i media_ai_db psql -U media_user -d media_ai
-- Or (bash): docker exec -i media_ai_db psql -U media_user -d media_ai < db/schema.sql

CREATE SCHEMA IF NOT EXISTS media_ai;

SET search_path TO media_ai, public;

CREATE EXTENSION IF NOT EXISTS vector;

-- ─────────────────────────────────────────────
-- Table: event_photos
-- Stores metadata for every uploaded event photo
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS media_ai.event_photos (
    photo_id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id           UUID        NOT NULL,
    image_url          TEXT        NOT NULL,
    thumbnail_url      TEXT,
    width              INT,
    height             INT,
    captured_at        TIMESTAMP,
    uploaded_at        TIMESTAMP   NOT NULL DEFAULT NOW(),
    source             TEXT,
    -- Set once OCR/face embedding has been attempted for this photo,
    -- whether or not anything was detected. Distinguishes "tried, found
    -- nothing" from "never tried" — needed so the periodic sweep below
    -- doesn't repeatedly re-attempt photos that already came back empty.
    ocr_attempted_at   TIMESTAMP,
    face_attempted_at  TIMESTAMP
);

-- Safe to re-run against an existing database that predates these columns.
ALTER TABLE media_ai.event_photos ADD COLUMN IF NOT EXISTS ocr_attempted_at TIMESTAMP;
ALTER TABLE media_ai.event_photos ADD COLUMN IF NOT EXISTS face_attempted_at TIMESTAMP;

-- Face search is scoped to one event: it first selects that event's photos (see db/queries.py),
-- which needs this plain index. Without it the lookup scans every photo row.
CREATE INDEX IF NOT EXISTS idx_event_photos_event_id
    ON media_ai.event_photos (event_id);

-- Partial index keeps the periodic sweep's "find unprocessed photos" query
-- cheap regardless of how many total photos accumulate over time — it only
-- ever has to scan photos that still need attention, not the whole table.
CREATE INDEX IF NOT EXISTS idx_event_photos_unprocessed
    ON media_ai.event_photos (event_id)
    WHERE ocr_attempted_at IS NULL OR face_attempted_at IS NULL;

-- With OCR disabled, ocr_attempted_at stays NULL for every photo, so the
-- partial index above matches every row and stops being selective. The sweep
-- and batch now only look for photos whose face step hasn't run, so index that.
CREATE INDEX IF NOT EXISTS idx_event_photos_face_unattempted
    ON media_ai.event_photos (event_id)
    WHERE face_attempted_at IS NULL;

-- ─────────────────────────────────────────────
-- Table: bib_detections
-- Stores OCR-extracted BIB numbers per photo
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS media_ai.bib_detections (
    detection_id   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_id       UUID        NOT NULL REFERENCES media_ai.event_photos(photo_id) ON DELETE CASCADE,
    bib_number     VARCHAR(10) NOT NULL,
    confidence     NUMERIC(5,2),
    bounding_box   JSONB,
    created_at     TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bib_detections_bib_number
    ON media_ai.bib_detections(bib_number);

CREATE INDEX IF NOT EXISTS idx_bib_detections_photo_id
    ON media_ai.bib_detections(photo_id);

-- ─────────────────────────────────────────────
-- Table: face_embeddings
-- Stores 512-dim face vectors for pgvector similarity search
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS media_ai.face_embeddings (
    face_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_id       UUID        NOT NULL REFERENCES media_ai.event_photos(photo_id) ON DELETE CASCADE,
    embedding      vector(512) NOT NULL,
    confidence     NUMERIC(5,2),
    face_box       JSONB,
    created_at     TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_photo_id
    ON media_ai.face_embeddings(photo_id);

-- IVFFlat index for fast cosine similarity search
-- Note: requires at least 100 rows before the index is useful
CREATE INDEX IF NOT EXISTS idx_face_embeddings_ivfflat
    ON media_ai.face_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ─────────────────────────────────────────────
-- Table: search_requests
-- Tracks every participant search (BIB or selfie)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS media_ai.search_requests (
    search_id        UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          UUID,
    event_id         UUID,
    search_type      VARCHAR(10),        -- 'bib' or 'face'
    bib_number       VARCHAR(10),        -- populated for BIB searches
    search_image_url TEXT,               -- always NULL for new rows (selfies are not stored); legacy rows NULLed after 24h
    consent_given    BOOLEAN   NOT NULL DEFAULT FALSE,
    consent_version  TEXT,               -- which consent text the user agreed to
    created_at       TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Safe to re-run against an existing database that predates this column.
ALTER TABLE media_ai.search_requests ADD COLUMN IF NOT EXISTS consent_version TEXT;

-- ─────────────────────────────────────────────
-- Table: search_results
-- Maps search requests to matching photos
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS media_ai.search_results (
    result_id   UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
    search_id   UUID      NOT NULL REFERENCES media_ai.search_requests(search_id) ON DELETE CASCADE,
    photo_id    UUID      NOT NULL REFERENCES media_ai.event_photos(photo_id) ON DELETE CASCADE,
    confidence  NUMERIC(5,4)
);

-- ─────────────────────────────────────────────
-- Table: processing_errors
-- Records OCR/face processing failures per photo, so a photo that
-- threw an exception is still queryable instead of just disappearing
-- into console logs. A missing bib_detections/face_embeddings row no
-- longer has to mean "failed" — it can also mean "not yet attempted"
-- or "attempted, found nothing." This table disambiguates the failed case.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS media_ai.processing_errors (
    error_id       UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_id       UUID        NOT NULL REFERENCES media_ai.event_photos(photo_id) ON DELETE CASCADE,
    stage          VARCHAR(10) NOT NULL,  -- 'ocr' or 'face'
    error_message  TEXT,
    created_at     TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_processing_errors_photo_id
    ON media_ai.processing_errors(photo_id);

-- ─────────────────────────────────────────────
-- Table: batch_runs
-- Acts as a claim/lock so at most one batch_process_event() run is ever
-- active per event — the row's presence means "currently running." This
-- is what lets the periodic sweep safely re-check stalled events without
-- risking two concurrent runs (and double the CPU/CompreFace load) for the
-- same event.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS media_ai.batch_runs (
    event_id    UUID        PRIMARY KEY,
    started_at  TIMESTAMP   NOT NULL DEFAULT NOW()
);
