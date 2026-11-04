# Benchmarks

`benchmark.py` measures the selfie photo-search pipeline end to end and studies how a
vector index would behave at larger sizes. Results from the last run are in
[`docs/benchmarks/RESULTS.md`](../docs/benchmarks/RESULTS.md) and the matching
`results-<date>.json`.

## What it measures

| Phase | What | How |
|---|---|---|
| `env` | Host CPU, cores, RAM, Docker version, the compose CPU/memory caps actually in effect (`docker inspect`), worker settings, dataset fingerprints | reads the host and the running containers |
| `throughput` | CompreFace calls/s and faces/s, and per-call latency, at 1, 2 and 4 client workers | calls `processing.face.extract_face_embedding` on every event photo, bypassing the API and database |
| `e2e` | Photos per minute from the first upload to the last photo processed | uploads every photo through `/upload` one at a time (as the portal does), triggers `/admin/batch`, polls until nothing is unattempted |
| `search` | Search latency p50/p90/p99 | **full HTTP** (`POST /search/face`, sequential and 4 concurrent clients) and **SQL only** (the production query text on the real tables, plus `EXPLAIN ANALYZE`) |
| `accuracy` | top-1 and top-5, matches and false matches at the 0.75 threshold, out-of-set results, genuine/impostor similarity distributions | uses the **labelled LFW images only** (see below) |
| `index` | exact search vs IVFFlat over a grid of `lists` and `probes`, recall@10 against exact results, latency, with and without an event filter | **synthetic distractors** in a throwaway `bench` schema |

### Data and what it may claim

* The photos are never part of the repo. Point `--photos` (or `FACE_TEST_PHOTOS`) at a folder with
  `event/` (photos), `selfies/<person>/` (one selfie per person, plus an `outsider/` person who is in no
  event photo) and `labels.csv` (`photo_filename,person_id`, LFW images only).
* **RBNR race photos are reported for speed and face detection only.** They have no identity labels.
* **Accuracy comes from the labelled LFW images only.** LFW images are 250x250 portraits, not race photos,
  so accuracy here says little about real race-day photos.
* The report contains counts, timings and hashes. No filenames, no person names, no keys.
  Working state (photo ids, selfie embeddings) goes to `bench_out/`, which is gitignored.

### The index study is synthetic

Real data is far too small to compare indexes, so the study copies the real embeddings into a `bench`
schema and adds distractors: perturbed copies of real vectors (cosine to their source uniform in
[0.2, 0.95]) assigned to events at random with a fixed seed. Recall and latency depend on that
construction, so treat the numbers as a comparison between index settings, not as a forecast.

It runs three things:

1. **Production-shaped tables** (`event_photos` + `face_embeddings`, IVFFlat `lists=100` created on the empty
   table exactly as `db/schema.sql` does): the production query's plan and latency "as is", and again with a
   plain `event_photos(event_id)` index (which makes the search exact and event-scoped).
2. **A grid** on a denormalised table: exact baselines, then IVFFlat with `lists` of N/1000, sqrt(N) and
   N/100 and `probes` of 1 to 32. Recall@10 is measured against exact results, unfiltered and with
   `WHERE event_id = ...`, which shows the loss from filtering after the index scan.
3. **The index as it is today**: created on an empty table, then loaded.

## Run it

```bash
docker compose up -d
pip install -r requirements-dev.txt

# the HTTP search phase would be throttled by the default 10 searches/minute/IP:
#   put SEARCH_RATE_LIMIT_PER_MINUTE=0 in .env, then
docker compose up -d --force-recreate api

export FACE_TEST_PHOTOS=/path/to/face-test-photos     # PowerShell: $env:FACE_TEST_PHOTOS = "..."
python bench/benchmark.py --cleanup                    # all phases, 100,000 synthetic faces, ~25 minutes
```

Useful options:

```
--phases env,throughput,e2e,search,accuracy,index    pick phases (search/accuracy/index need e2e first)
--workers 1,2,4                                      client workers for the throughput phase
--index-size 100000 --events 200 --queries 200       size of the synthetic study
--seed 42                                            fixed seed for the synthetic data and queries
--keep                                               keep the bench schema afterwards
--cleanup                                            delete the bench event (rows and files) at the end
--out docs/benchmarks                                where results-<date>.json is written
```

A partial rerun on the same day updates the phases it ran in that day's JSON and keeps the others.
Put the rate limit back (remove the line from `.env`, recreate `api`) when you are done.

## Reading the numbers

The client runs on the host. Postgres and CompreFace run in Docker under the compose CPU and memory caps
printed in the `environment` section (on Windows that is WSL2). Compare numbers with each other, not with
production capacity.
