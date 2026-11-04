# Design decisions

Why the system looks the way it does, what was measured, and what is still open. Numbers come from
[benchmarks/RESULTS.md](benchmarks/RESULTS.md) (one laptop, Docker with small CPU caps, so relative numbers).

## 1. Why OCR is disabled, and how to restore it

**What happened.** The project started as bib-number search: Tesseract OCR read the number on each runner's bib. On real
race photos that proved unreliable even with preprocessing, while face matching found photos OCR missed. Scope moved to
selfie search, which is now the only active flow. (No failure rates are quoted because no controlled OCR benchmark was run.)

**What was done.** OCR is commented out, not deleted, so it can come back. Every disabled place has a greppable
`# OCR-DISABLED:` marker (HTML comments in the templates). `processing/ocr.py` is intact with a header note and nothing
imports it. The schema is untouched (`bib_detections`, `ocr_attempted_at` remain).

**How to restore it.**

1. `grep -rn "OCR-DISABLED:" .` lists every site.
2. `processing/batch.py`: re-enable the `extract_bib` import, `OCR_WORKERS`, `_process_ocr`, the OCR skip checks and
   `ocr_targets`, the OCR thread pool, the `ocr_processed` summary key; add `ocr_attempted_at` back to the `SELECT` in
   `batch_process_event`; and make `sweep_stalled_events` check `ocr_attempted_at IS NULL OR face_attempted_at IS NULL`
   again (the comment next to the query shows the original). Otherwise OCR would never be marked attempted.
3. `api/routes/search.py`: re-enable `/search/bib`, `/photos/{id}/bibs` and the Bib model imports; `api/routes/admin.py`: the
   bib count in the stats; `api/models.py` already keeps the Bib models.
4. `templates/search.html`: the BIB tab, section and JS; `templates/upload_portal.html`: the wording.
5. `Dockerfile`: install `tesseract-ocr` (the comment shows the line); `requirements.txt`: `pytesseract`; `.env.example` and
   `docker-compose.yml`: `OCR_WORKERS`, `MIN_BIB_CONFIDENCE`, `TESSERACT_CMD`.
6. Rebuild the image. There are no tests for the OCR path; commented-out code cannot be tested, so add them when restoring.

`idx_event_photos_face_unattempted` (added when OCR was disabled) can stay; the older `idx_event_photos_unprocessed` still
covers the OCR-or-face condition.

## 2. Why pgvector

* **One database.** Photos, faces, consent records, the batch claim and the error log already live in PostgreSQL. Keeping the
  vectors in the same database means one backup, one transaction boundary (a photo and its faces are deleted together by
  `ON DELETE CASCADE`), and one fewer service to run than a separate vector database.
* **The workload is small and scoped.** A search only ever compares a selfie with the faces of **one event**. At about 500
  faces per event, exact search in SQL takes a few milliseconds (section 3), so a dedicated vector store would add operations
  without adding speed.
* **Exact beats approximate here.** An approximate index can miss the very photo someone is looking for. The benchmarks showed
  recall dropping to 0.2 to 0.4 at low `probes` once an event filter was applied.
* **Trade-offs accepted.** pgvector search is exact only if no approximate index is used; cost grows with the faces in an
  event, and very large events may eventually want an index. The compose file pins `ankane/pgvector` (PostgreSQL 15.4,
  pgvector 0.5.1), which is no longer maintained: move to a maintained image before real use.

## 3. How the search query works, and what was measured

`db/queries.py` runs, for one selfie embedding: select the event's photos first (`idx_event_photos_event_id`), look up each
photo's faces through `idx_face_embeddings_photo_id`, compute the similarity only for those faces, keep faces at or above the
threshold, return **one row per photo with its best score**, best first, at most 50 photos.

What the benchmarks showed (100,177 synthetic faces in 200 events, 100 queries per row):

| Query | p50 | p99 |
|---|---|---|
| Original query, no event index | 83 ms | 190 ms |
| Original query + plain `event_photos(event_id)` index | **589 ms** | 880 ms |
| Event's photos in a CTE only | 7.6 ms | **681 ms** |
| **Shipped query (CTE + `LATERAL ... OFFSET 0`)** | **8.2 ms** | **51 ms** |

* **Adding the index alone made the query 7x slower:** the planner then computed the similarity for every face and joined
  afterwards. A CTE alone is unstable (hash join over every face at 30,000 to 60,000 photos in tests). The two fences force the
  order, and `tests/test_search_query.py` checks the plan at 10,000 and 60,000 photos.
* **No vector index.** The old schema had an IVFFlat index created on an empty table: nothing used it, and built that way it had
  recall 0.375 at pgvector's default `probes`. With it dropped, results were identical and latency unchanged, so it was removed.
* **If one event grows to tens of thousands of faces,** measure first, then build an IVFFlat index *after* loading data with
  `lists` near sqrt(rows); with an event filter it needed `probes` of 16 or more for recall of about 0.93 to 0.95, where it was
  no faster than exact event-scoped search at 500 faces per event. Untested above that size.
* Caveats: the index study uses synthetic distractors, single runs, and a small machine.

## 4. How batch recovery works

Goal: a photo is never silently lost, and two runs never fight over one event.

* **One run per event.** `batch_runs` holds a claim row per event: `INSERT ... ON CONFLICT (event_id) DO UPDATE ... WHERE
  started_at < now() - BATCH_RUN_STALE_MINUTES RETURNING event_id`. The insert wins only if there is no row, or the row is
  older than `BATCH_RUN_STALE_MINUTES` (60), meaning the previous run crashed. The claim is released in a `finally` block.
* **A photo is marked `face_attempted_at` only when it was really handled:** faces stored, no face in the photo (CompreFace
  answers that with HTTP 400, code 28), or an image that can never be decoded (logged to `processing_errors` and marked, since
  retrying cannot help).
* **Anything else leaves the photo unmarked and logs an error:** CompreFace down, timeout, wrong or missing key, a 5xx, a
  database error. The next batch run or sweep picks it up. This is the fix for the original bug where every failure looked
  like "no face", marked the photo done and was never retried.
* **The sweep** runs every `SWEEP_INTERVAL_MINUTES` (15). It finds events with unprocessed photos older than
  `SWEEP_GRACE_MINUTES` (15, so an upload still in progress is left alone) using the partial index
  `idx_event_photos_face_unattempted`, and runs the same batch (which respects the claim).
* **The portal** triggers the batch once, after the last file, with three retries.
* **Proven** by tests (claim, stale reclaim, sweep, classification) and by a real drill: with CompreFace stopped, 3 photos
  stayed unmarked with 3 errors logged and search answered 503; after CompreFace came back all 3 were processed.
* **Limits:** no retry cap or back-off (a photo that always fails is retried every sweep and logs a row each time); and see
  section 8 for the upload-key limitation.

## 5. How photos are protected

Previously every photo was public through a static `/storage` mount and no route needed authentication.

* The mount is gone. Photos are served only by `GET /events/{event}/photos/{photo}`, which needs either a signed link or the
  admin key.
* A link is an HMAC-SHA256 (`PHOTO_URL_SECRET`) over `event|photo|size|expiry`, valid for `PHOTO_URL_TTL_SECONDS` (1 hour).
  Changing any field (size, event, expiry) invalidates it. The route also checks that the photo belongs to the event and that the
  stored path resolves inside `STORAGE_PATH` (so a poisoned database row cannot read other files; tested with absolute paths
  and `..`). If the secret is unset, links fail closed.
* Only search hands out links, so a person can only reach photos the face search matched.
* Uploads require a key, a UUID event id, a size cap enforced while streaming, a content check with Pillow (not the client's
  header) and a pixel limit; files are renamed to `<uuid>.<ext>`; paths are stored relative to `STORAGE_PATH`.
* **What this does not give you:** per-user access, key rotation, or protection of someone who has a valid link (it can be
  shared until it expires). Signed links are bearer tokens.

## 6. How selfies are handled

The selfie is read into memory, validated, written to a temp file outside `STORAGE_PATH` only because the CompreFace helper
needs a path, and deleted in a `finally`. Nothing is stored: not the image, not its embedding. A `search_requests` row with
`consent_given`, `consent_version`, the event and the time is written **before** the selfie goes to CompreFace. A cleanup job
only purges selfies that earlier versions left in `storage/selfies`.

## 7. The match threshold

`MIN_FACE_CONFIDENCE` is the minimum cosine similarity for a photo to be returned. It was 0.75 and is now **0.70**.
Small test: 10 people, 3 photos each in a 90-photo event, plus one person in none of the photos; the images are 250×250 LFW
portraits, **not race photos**.

| Threshold | Photos of the right person returned (of 30) | Different-person matches (of 300 pairs) | Person not in the event |
|---|---|---|---|
| 0.75 | 12 | 0 | no results |
| **0.70** | **18** | **0** | **no results** |
| 0.50 | 28 | 0 | highest similarity 0.451, below 0.50 |

Same-person similarities ranged from 0.28 to 0.91 (median 0.72); different-person similarities had a median of 0.06, a 95th
percentile of 0.30 and a maximum of 0.48. Top-1 and top-5 were 10/10 at every threshold. At 0.70, through the real endpoint, there were no duplicate photos and
no photo of a different labelled person.

**This is not a calibration.** The threshold still needs calibrating on race-like photos: distance, motion blur, hats,
sunglasses and crowds will lower same-person similarity, and an event with thousands of faces gives impostors many more chances
to score high. A sensible next step is a labelled set of real photos from a few events, then choosing the threshold from the
false-match and miss rates measured there. Until then 0.70 trades a few more potential false matches for more of a person's
photos than 0.75, on evidence from portraits only.

## 8. Known limitation: the upload key cannot trigger processing

The upload portal calls `POST /admin/batch/{event}` when the last file is uploaded, and that route needs `ADMIN_API_KEY`. A
photographer given only `UPLOAD_API_KEY` can upload, but the trigger is refused (the portal says processing will start
automatically). Processing then waits for the sweep: up to about 30 minutes (`SWEEP_GRACE_MINUTES` + `SWEEP_INTERVAL_MINUTES`).
Giving photographers the admin key starts it immediately but also grants them admin routes.

Options if this matters: a narrower route (for example `POST /upload/finish/{event}` accepting the upload key and only starting
the batch for that event), a shorter sweep for new events, or a per-event upload token. Not implemented.

## 9. What the benchmarks showed

* End to end: 57.9 to 66.7 photos/min over four runs (90 photos, 177 faces); CompreFace embedding is the bottleneck.
* More CompreFace workers did not help under a 1-CPU cap: 1.53 / 1.21 / 1.11 calls/s at 1 / 2 / 4 workers. Whether
  `FACE_WORKERS=1` is better in production depends on the CPU CompreFace gets there.
* Search: p50 393 ms / p99 700 ms over HTTP (one CompreFace call dominates); the query itself 8 ms at 100,000 faces.
* Detection: 59 of 60 RBNR race photos and 30 of 30 LFW photos had a face.
* Transient CompreFace "connection reset" errors appeared in several runs; with the recovery design they were logged and retried.
* Limits of the evidence: one machine, small CPU caps, single runs, synthetic index data, and accuracy from portraits rather
  than race photos.

## 10. Open questions

* Calibrating the threshold on real race photos (section 7).
* Whether photographers should get a narrower processing permission (section 8).
* Retry cap and back-off for photos that always fail; rate limiting behind a proxy; key rotation.
* A maintained pgvector image, and an index only if events grow far beyond a few thousand faces.
