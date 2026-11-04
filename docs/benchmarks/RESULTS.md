# Benchmark results

Run date 2026-11-04, one full run of [`bench/benchmark.py`](../../bench/benchmark.py) (raw numbers:
[`results-2026-11-04.json`](results-2026-11-04.json), about 18 minutes). Every number below comes from that file.

**Read this first.** The client ran on the host and Postgres and CompreFace ran in Docker under the small
CPU caps below, so these are *relative* numbers for comparing settings. They are not production capacity.
Each measurement is a single run (no repeats), so differences of a few tens of percent are not reliable.

## Setup

| | |
|---|---|
| Host | 13th Gen Intel Core i7-13650HX, 14 physical / 20 logical cores, 23.7 GiB RAM, Windows 11 |
| Docker | server 29.9.2, Compose 5.6.1; Docker VM: 20 CPUs, 11.5 GiB |
| Compose caps in effect (CPU / memory) | api 0.6 / 2300 MiB · db 0.5 / 1024 MiB · compreface-core 1.0 / 3072 MiB · compreface-api 0.4 / 768 MiB · admin 0.15 / 512 MiB · ui 0.1 / 128 MiB · compreface-db 0.25 / 256 MiB |
| Settings | `FACE_WORKERS=2`, CompreFace `UWSGI_PROCESSES=2`, `UWSGI_THREADS=1`, match threshold 0.75 at the time of this run (the app default is now 0.70, see 4b), PostgreSQL 15.4, pgvector 0.5.1, `shared_buffers=128MB`, `work_mem=4MB` |
| Photos | 90 event photos: **60 RBNR race photos** and **30 LFW photos** (10 people, 3 photos each); 11 selfies (10 people in the event plus 1 outsider who is in none of the event photos). Fingerprints (sha256, first 16 hex): RBNR `b772d050e9fc9cdb`, LFW `db2d69d7fc5edc84`, selfies `959d8d85e265611a` |
| Faces | 177 faces stored for the 90 photos (RBNR 146 in 59 of 60 photos; LFW 31 in 30 of 30 photos) |

### What these photos can and cannot show

* **RBNR race photos are reported for speed and face detection only.** They carry no identity labels.
* **Accuracy comes from the labelled LFW images only.** LFW images are 250x250 portraits. **They are not race
  photos**, so accuracy here says little about real race-day photos (distance, motion blur, hats, bibs, crowds).
* The sample is small: 10 people, 3 photos each, 300 impostor pairs.

## 1. Embedding throughput (CompreFace only, bypassing the API and database)

All 90 photos, `processing.face.extract_face_embedding`, no failed calls in any run.

| Client workers | Wall time | CompreFace calls/s | Faces/s | Call latency p50 | p99 |
|---|---|---|---|---|---|
| 1 | 58.8 s | **1.53** | 3.01 | 486 ms | 2.8 s |
| 2 (= `FACE_WORKERS`) | 74.4 s | 1.21 | 2.38 | 1.39 s | 6.0 s |
| 4 | 80.9 s | 1.11 | 2.19 | 2.80 s | 13.4 s |

Throughput *drops* as client workers increase. CompreFace's core is capped at 1 CPU here, so extra concurrent
requests queue behind it and only add latency. Earlier attempts of this same script in this session (logs not
kept) showed the same ordering (1 worker fastest, 4 workers slowest). Whether `FACE_WORKERS=1` would be faster in
production depends on the production CPU allotment for CompreFace, which this run does not measure.

## 2. End to end: upload to "batch done"

90 photos, sequential uploads through `/upload` (as the portal does), then `/admin/batch`.

| Upload | Batch | Total | Photos/min (end to end) | Photos/min (batch only) | Errors logged | Batch retriggers needed |
|---|---|---|---|---|---|---|
| 9.8 s | 76.4 s | 86.1 s | **62.7** | 70.7 | 0 | 0 |

Face embedding is the bottleneck: batch-only speed (70.7/min) matches the 2-worker CompreFace rate above
(1.21 calls/s = 72.6/min). Uploads are about 551 photos/min.

Across the five runs of this benchmark made in this session, CompreFace returned a transient "connection reset"
(HTTP 500, code 41) twice: once during the direct throughput phase and once during a batch. In the batch case
the fixed code logged the error to `processing_errors` and left that one photo unprocessed for a retry (before
the fix it would have been marked done, with nothing logged, and never retried). Neither failure is in the
numbers above; the run reported here had none.

## 3. Search latency

| | n | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| **Full HTTP** `POST /search/face`, sequential | 150 (all 200 OK) | **393 ms** | 486 ms | **700 ms** | 711 ms |
| Full HTTP, 4 concurrent clients | 100 (all 200 OK) | 1195 ms | 2295 ms | 2479 ms | 2586 ms (2.78 req/s) |
| **SQL only** (production query text, real tables) | 300 | **1.53 ms** | 2.05 ms | **2.88 ms** | 6.5 ms |

The HTTP figure includes the multipart upload, image validation, one CompreFace call, the consent insert,
the vector query and URL signing; the CompreFace call dominates (single-call p50 above is 486 ms). The SQL
figure is client-side on localhost over only 177 faces in the event (query plan: hash join, sequential scans,
0.67 ms server time), so it says nothing about scale. Section 5 does that.

## 4. Accuracy (labelled LFW photos only)

### 4a. First run, threshold 0.75

Ranking and counts use only the 30 labelled LFW photos; RBNR photos are excluded from every number here.

| Measure | Result |
|---|---|
| Top-1 (best-scoring labelled photo belongs to the person) | **10/10** |
| Top-5 (at least one of the person's photos in the top 5) | **10/10** |
| Person's photos returned at 0.75 | **12/30** (per person: 3, 3, 2, 2, 1, 1, 0, 0, 0, 0 of 3) |
| False matches at 0.75 (a labelled photo of a different person) | **0/270** |
| Out-of-set person (in no event photo) | 0/30 labelled photos at 0.75; highest similarity to any labelled photo 0.256; to any stored face 0.451 |

Top-1 and top-5 are easy here because the ranking covers only 30 photos of 10 people. The number that matters
for users is the 12/30: at the current 0.75 threshold, **60% of a person's photos are not returned**.

Similarity distributions (information for threshold calibration; the threshold was not changed):

| | n | min | p25 | median | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| Same person | 30 | 0.280 | 0.633 | 0.720 | | | 0.909 |
| Different person (incl. out-of-set) | 300 | | | 0.060 | 0.298 | 0.393 | 0.478 |

| Threshold | Same-person photos found | Different-person pairs matched |
|---|---|---|
| 0.50 | 28/30 | 0/300 |
| 0.55 | 24/30 | 0/300 |
| 0.60 | 23/30 | 0/300 |
| 0.65 | 22/30 | 0/300 |
| 0.70 | 18/30 | 0/300 |
| **0.75 (current)** | **12/30** | 0/300 |
| 0.80 | 7/30 | 0/300 |
| 0.85 | 4/30 | 0/300 |

In this set the highest different-person similarity (0.478) is far below the lowest threshold tried, so a lower
threshold would have returned more of each person's photos with no false match. That is a statement about
250x250 LFW portraits and 300 impostor pairs only; real events have far more impostor faces and harder
photos.

### 4b. Rerun at the new default, 0.70

The default was then changed from 0.75 to 0.70 (`MIN_FACE_CONFIDENCE`). Same 10 people, same 30 labelled photos,
rerun on 2026-11-04 ([`results-2026-11-04-threshold-0.70.json`](results-2026-11-04-threshold-0.70.json)),
including the 11 selfies searched through the real `/search/face` endpoint with the container running 0.70.

| Measure | 0.75 (first run) | **0.70 (rerun)** | 0.50 (same similarities, not deployed) |
|---|---|---|---|
| Top-1 | 10/10 | **10/10** | 10/10 (ranking does not depend on the threshold) |
| Top-5 | 10/10 | **10/10** | 10/10 |
| Person's photos returned (of 30) | 12 | **18** | 28 |
| False matches: a labelled photo of a different person | 0/270 | **0/270** | 0/300 pairs |
| Out-of-set person: results returned through the API | 0 | **0** | not applicable (highest similarity to any stored face 0.451, below 0.50) |

* At 0.70, per person (photos returned of 3): 3, 1, 2, 0, 1, 0, 3, 3, 3, 2. Through the API, no result was a
  duplicate and none belonged to a different labelled person.
* **The outsider matched nothing at 0.70.** The outsider is in none of the 90 event photos; its highest
  similarity to any stored face is 0.451.
* 0.70 still misses 12 of 30 photos. The "0.50" column is the same similarity table read at 0.50 (28/30, 0/300),
  shown for context only.

**The threshold still needs calibrating on race-like photos.** These numbers come from ten people photographed
as 250x250 portraits, with 300 impostor pairs. They say nothing about distance shots, motion blur, hats,
sunglasses or crowds, and in a large event there are many more impostor faces than here, so the false-match
rate at any threshold is unknown. The first calibration step is a labelled set of real race photos.

## 5. Index study (SYNTHETIC distractors)

> **Synthetic.** The 177 real embeddings were copied into a `bench` schema and 100,000 distractors were added:
> perturbed copies of real embeddings (cosine to their source uniform in [0.2, 0.95]), assigned at random to
> 200 events (about 498 faces per event), fixed seed 42. Absolute numbers depend on this construction.
> In particular the distractors are perturbations of only 188 source vectors (the 177 real embeddings and the 11
> selfies), so they cluster tightly, which makes *unfiltered* approximate search look better than it would on real data.

### 5a. The production query on production-shaped tables

The query is the one in `api/routes/search.py` (event filter, `1 - distance >= 0.75`, `ORDER BY ... DESC`,
`LIMIT 50`). 100 queries per row. A = schema as it is today. B = A plus a plain index on `event_photos(event_id)`,
same query. C = B with the event's photos selected first (`WITH ev AS MATERIALIZED (...)`), same predicate,
same ordering, same limit.

| Faces in table | Scenario | p50 | p99 | Plan (top of tree) |
|---|---|---|---|---|
| 10,177 | A as is | 16.6 ms | 87.9 ms | Nested loop: seq scan `event_photos`, index scan `idx_bench_face_photo` |
| 10,177 | B + event index | 17.0 ms | 84.0 ms | Nested loop: bitmap scan on the new index |
| 10,177 | C event first | 10.1 ms | 79.1 ms | CTE on the new index, nested loop |
| 100,177 | A as is | **79.9 ms** | 164.5 ms | Gather Merge, nested loop: seq scan `event_photos` |
| 100,177 | B + event index | **603.5 ms** | 903.6 ms | **Hash join with a seq scan of all of `face_embeddings`** |
| 100,177 | C event first | **9.6 ms** | 712.5 ms (p90 60.1 ms) | CTE on the new index, nested loop |

* **The IVFFlat index is never used by the production query** (in the captured plans, `indexes_used` contains only the `photo_id` and
  new event indexes). It would only apply to `ORDER BY embedding <=> q`, and the query orders by
  `1 - distance DESC` with a similarity filter.
* **A plain `event_photos(event_id)` index alone made the query 7.5x slower at 100k faces**: the planner then
  chose to compute the cosine similarity for every face. Restructuring the query so the event's photos are
  selected first (C) fixed that: p50 9.6 ms at 100k faces and 10.1 ms at 10k, so the cost follows the event
  size, not the table size.
* B and C returned the same photos as A for all 100 queries at both sizes, but the queries return few rows on
  this data (0.35 on average at 100k), so that check has limited power.
* C's p99 (712 ms) comes from a handful of slow queries (p90 is 60 ms); the cause was not investigated, and with
  100 queries per row p99 is set by one or two samples.

### 5b. IVFFlat grid vs exact search (100,177 faces, 200 queries, recall@10 against exact results)

Exact baselines: unfiltered sequential scan **601 ms** p50 (908 ms p99). Exact within one event (about 498
faces): sequential scan 11.5 ms p50, **with a btree on `event_id` 5.95 ms p50 (39 ms p99)** and recall 1.0 by
definition.

IVFFlat built after loading the data. Cells: recall@10 and p50 latency; "short" = queries that returned fewer
rows than the exact search (out of 200).

| `lists` (build time) | `probes` | Unfiltered recall | Unfiltered p50 | Event-filtered recall | Event-filtered p50 | Filtered short |
|---|---|---|---|---|---|---|
| 100 = N/1000 (7.6 s) | 1 | 1.00 | 3.8 ms | 0.412 | 3.5 ms | 128 |
| | 4 | 1.00 | 6.6 ms | 0.719 | 7.3 ms | 4 |
| | 8 | 1.00 | 11.7 ms | 0.845 | 13.6 ms | 0 |
| | 16 | 1.00 | 20.4 ms | 0.940 | 23.1 ms | 0 |
| | 32 | 1.00 | 78.0 ms | 0.990 | 82.3 ms | 0 |
| 317 = sqrt(N) (12.9 s) | 1 | 1.00 | 2.9 ms | 0.279 | 2.6 ms | 189 |
| | 4 | 1.00 | 3.6 ms | 0.585 | 4.0 ms | 60 |
| | 8 | 1.00 | 6.8 ms | 0.794 | 7.5 ms | 2 |
| | 16 | 1.00 | 11.1 ms | 0.934 | 11.1 ms | 0 |
| | 32 | 1.00 | 18.0 ms | 0.986 | 19.1 ms | 0 |
| 1001 = N/100 (48.5 s) | 1 | 1.00 | 2.8 ms | 0.203 | 2.6 ms | 200 |
| | 4 | 1.00 | 3.6 ms | 0.429 | 4.7 ms | 155 |
| | 8 | 1.00 | 4.8 ms | 0.625 | 6.9 ms | 63 |
| | 16 | 1.00 | 6.4 ms | 0.852 | 7.5 ms | 4 |
| | 32 | 1.00 | 11.7 ms | 0.979 | 10.8 ms | 0 |

(probes 2 is in the JSON.) Unfiltered recall is 1.00 everywhere only because of the clustered synthetic data;
do not rely on it. The event-filtered column is the one that matters for this app: the index returns its
candidates first and the event filter is applied afterwards, so with about 1 face in 200 belonging to the event
the recall at `probes = 1` is 0.20 to 0.41 and at `probes = 1` most queries come back with fewer than 10 rows.

### 5c. The index as it exists today (created on an empty table, then loaded)

`lists = 100`, created before any rows existed (as `db/schema.sql` does), then the 100,177 faces inserted
(50 s). Unfiltered recall@10, 200 queries, same data as 5b:

| `probes` | Recall@10 | Rows returned of 10 |
|---|---|---|
| 1 (pgvector default) | **0.375** | 10.0 |
| 10 | 0.926 | 10.0 |
| 32 | 0.998 | 10.0 |

On the same data, the index built after loading (5b, `lists = 100`) has recall 1.00 at `probes = 1`.

## Findings and options (nothing below has been applied)

1. **Today's query does not use the vector index.** At 100k faces it takes about 80 ms p50 and relies on a
   parallel nested loop; it has no plain index on `event_photos(event_id)`.
2. **Do not just add `event_photos(event_id)`.** On its own it made the query 7.5x slower at 100k faces
   (604 ms). It only helps if the query also selects the event's photos first.
3. **Option 1, recommended to evaluate: exact event-scoped search** (event index + event-first query). 9.6 ms p50
   at 100k faces, recall 1.0, no approximate index to tune. Cost grows with faces *per event* (about 498 here);
   it was not tested for events with tens of thousands of faces.
4. **Option 2: IVFFlat.** Only worth it if single events become large. It must be built after the data is loaded
   (the current empty-table build has recall 0.375 at default probes), and with an event filter it needs
   `probes` of about 16 or more to reach recall 0.93 to 0.95, at which point it is no faster than the exact
   event-scoped search above (5.95 ms exact vs 7.5 to 23 ms).
5. **Drop the empty-table IVFFlat index** from the schema if Option 1 is chosen: nothing uses it.
6. **Other things the run showed.** `FACE_WORKERS=2` was slower than 1 under the 1-CPU CompreFace cap; at the
   0.75 threshold 60% of a person's LFW photos were not returned while a lower threshold produced no false
   matches in this small set; a search that matches two faces of the same photo returns that photo twice
   (checked in an ad-hoc test that is not committed, not part of this run).

## Data sources

* **RBNR** (60 race photos used for speed and face detection only): Ben-Ami, Basha, Avidan, "Racing Bib
  Numbers Recognition", BMVC 2012 (citation from memory; verify before publishing). Downloaded from
  <https://people.csail.mit.edu/talidekel/Data/RBNR/RBNR_Datasets.zip>.
* **LFW** (30 labelled photos and 11 selfies): Huang, Ramesh, Berg, Learned-Miller, "Labeled Faces in the Wild:
  A Database for Studying Face Recognition in Unconstrained Environments", University of Massachusetts Amherst
  Technical Report 07-49, October 2007. The canonical site (<https://vis-www.cs.umass.edu/lfw/>) did not
  resolve on 2026-11-04, so the images are the original, unresized JPEGs (`lfw_funneled`) fetched through
  scikit-learn's figshare mirror.

The photos are not in this repository and are not redistributed.
