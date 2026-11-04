#!/usr/bin/env python
"""
Benchmarks for the selfie photo-search pipeline. See bench/README.md.

Phases (run all, or pick with --phases):
    env         host / docker / compose limits / dataset fingerprints
    throughput  CompreFace calls/s and faces/s at several worker counts
    e2e         upload -> batch done, photos per minute (creates the bench event)
    search      search latency p50/p99: full HTTP and SQL-only
    accuracy    top-1 / top-5, false matches, out-of-set (labelled LFW images only)
    index       exact vs IVFFlat grid, with and without an event filter (SYNTHETIC distractors)

The report contains counts, timings and hashes only: no filenames, no person names, no
secrets. Working state (photo ids, selfie embeddings) goes to bench_out/, which is gitignored.

Requires the docker stack to be up, a CompreFace key in .env, and a local photo folder
(--photos or FACE_TEST_PHOTOS) laid out as  event/  selfies/<person>/  labels.csv .
For the HTTP search phase set SEARCH_RATE_LIMIT_PER_MINUTE=0 in .env and recreate the api
container (the default limit of 10/min would throttle the benchmark).
"""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psycopg2
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_dot = dotenv_values(ROOT / ".env")
ENV = {**_dot, **{k: v for k, v in os.environ.items() if k in _dot or k.startswith("FACE_")}}
for _k in ("COMPREFACE_URL", "COMPREFACE_API_KEY"):
    if ENV.get(_k):
        os.environ[_k] = ENV[_k]
os.environ.setdefault("COMPREFACE_URL", "http://localhost:8001")

from db.queries import FACE_SEARCH_SQL  # noqa: E402
from processing.face import FaceServiceError, extract_face_embedding, get_best_face  # noqa: E402

API = ENV.get("BENCH_API_URL", "http://localhost:8000").rstrip("/")
OUT_DIR = ROOT / "bench_out"
THRESHOLD = float(ENV.get("MIN_FACE_CONFIDENCE") or 0.70)
CONTAINERS = ["media_ai_api", "media_ai_db", "compreface_core", "compreface_api",
              "compreface_admin", "compreface_ui", "compreface_db"]

# CTE only (no LATERAL fence): kept to show the plan is not stable without the fence.
EVENT_FIRST_QUERY = """
WITH ev AS MATERIALIZED (
    SELECT photo_id, event_id, thumbnail_url FROM {s}.event_photos WHERE event_id = %(event_id)s
)
SELECT ev.photo_id, ev.event_id, ev.thumbnail_url IS NOT NULL AS has_thumb,
       1 - (fe.embedding <=> %(q)s::vector) AS confidence
FROM ev
JOIN {s}.face_embeddings fe ON fe.photo_id = ev.photo_id
WHERE 1 - (fe.embedding <=> %(q)s::vector) >= %(thr)s
ORDER BY confidence DESC
LIMIT 50
"""

# The query as it was before the event-first rewrite.
LEGACY_QUERY = """
SELECT ep.photo_id, ep.event_id, ep.thumbnail_url IS NOT NULL AS has_thumb,
       1 - (fe.embedding <=> %(q)s::vector) AS confidence
FROM {s}.face_embeddings fe
JOIN {s}.event_photos ep ON ep.photo_id = fe.photo_id
WHERE ep.event_id = %(event_id)s
  AND 1 - (fe.embedding <=> %(q)s::vector) >= %(thr)s
ORDER BY confidence DESC
LIMIT 50
"""

# What the app runs now (db/queries.py), pointed at another schema.
CURRENT_QUERY = FACE_SEARCH_SQL.replace("media_ai.", "{s}.")


# ── small helpers ─────────────────────────────────────────────────────────────

def sh(*cmd, timeout=60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_hash(paths) -> str:
    return hashlib.sha256("".join(sorted(sha(p) for p in paths)).encode()).hexdigest()[:16]


def stats_ms(xs) -> dict:
    a = np.array(xs, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": round(float(a.mean()), 2), "p50": round(float(np.percentile(a, 50)), 2),
            "p90": round(float(np.percentile(a, 90)), 2), "p99": round(float(np.percentile(a, 99)), 2),
            "max": round(float(a.max()), 2)}


def connect(autocommit=True):
    conn = psycopg2.connect(host=ENV.get("DB_HOST") or "localhost", port=int(ENV.get("DB_PORT") or 5432),
                            user=ENV.get("DB_USER") or "media_user", password=ENV.get("DB_PASS") or "media_pass",
                            dbname=ENV.get("DB_NAME") or "media_ai", options="-c search_path=media_ai,public")
    conn.autocommit = autocommit
    return conn


def vlit(vec) -> str:
    return "[" + ",".join(f"{float(x):.6g}" for x in vec) + "]"


def log(msg):
    print(f"[bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state() -> dict:
    p = OUT_DIR / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(state: dict):
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "state.json").write_text(json.dumps(state))


class Dataset:
    """The local test photos. Names are only held in memory / bench_out, never in the report."""

    def __init__(self, root: Path):
        self.root = root
        self.labels = {}
        with open(root / "labels.csv", newline="") as f:
            for row in csv.DictReader(f):
                self.labels[row["photo_filename"]] = row["person_id"]
        self.event = sorted((root / "event").glob("*.jpg"))
        self.lfw = [p for p in self.event if p.name in self.labels]
        self.rbnr = [p for p in self.event if p.name not in self.labels]
        self.selfies = {d.name: sorted(d.glob("*.jpg"))[0] for d in sorted((root / "selfies").iterdir()) if d.is_dir()}

    def group(self, p: Path) -> str:
        return "lfw" if p.name in self.labels else "rbnr"


# ── phase: env ────────────────────────────────────────────────────────────────

def phase_env(ds: Dataset) -> dict:
    log("env")
    cpu = ""
    if platform.system() == "Windows":
        cpu = sh("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
        cores = sh("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum")
        try:
            import ctypes

            class MEM(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                            ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
            m = MEM(); m.l = ctypes.sizeof(MEM); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            ram = m.total
        except Exception:
            ram = None
    else:
        cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
        cores = ""
        ram = next((int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.startswith("MemTotal")), None)

    caps = {}
    for c in CONTAINERS:
        out = sh("docker", "inspect", c, "--format", "{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}}")
        try:
            nano, mem = (int(x) for x in out.split())
            caps[c] = {"cpus": round(nano / 1e9, 2) or None, "memory_mib": round(mem / 1048576) or None}
        except ValueError:
            caps[c] = None
    envs = {}
    for c, keys in (("media_ai_api", ("FACE_WORKERS",)), ("compreface_core", ("UWSGI_PROCESSES", "UWSGI_THREADS"))):
        raw = sh("docker", "inspect", c, "--format", "{{range .Config.Env}}{{println .}}{{end}}")
        envs.update({k: v for k, v in (l.split("=", 1) for l in raw.splitlines() if "=" in l) if k in keys})

    conn = connect()
    cur = conn.cursor()
    cur.execute("select extversion from pg_extension where extname='vector'"); pgv = cur.fetchone()[0]
    cur.execute("show server_version"); pgver = cur.fetchone()[0]
    for k in ("shared_buffers", "work_mem", "maintenance_work_mem"):
        cur.execute(f"show {k}"); envs["postgres_" + k] = cur.fetchone()[0]
    conn.close()

    return {
        "date": dt.date.today().isoformat(),
        "host": {"os": platform.platform(), "cpu": cpu, "physical_cores": int(cores) if str(cores).isdigit() else None,
                 "logical_cpus": os.cpu_count(), "ram_gib": round(ram / 2**30, 1) if ram else None},
        "docker": {"server_version": sh("docker", "version", "--format", "{{.Server.Version}}"),
                   "compose_version": sh("docker", "compose", "version", "--short"),
                   "vm_cpus": sh("docker", "info", "--format", "{{.NCPU}}"),
                   "vm_memory_gib": round(int(sh("docker", "info", "--format", "{{.MemTotal}}") or 0) / 2**30, 1)},
        "compose_caps_in_effect": caps,
        "settings": {**envs, "match_threshold": THRESHOLD, "postgres": pgver, "pgvector": pgv},
        "dataset": {
            "event_photos": len(ds.event), "rbnr_race_photos": len(ds.rbnr), "lfw_labelled_photos": len(ds.lfw),
            "lfw_people_in_event": len(set(ds.labels.values())), "selfies": len(ds.selfies),
            "fingerprint_sha256_16": {"rbnr": group_hash(ds.rbnr), "lfw": group_hash(ds.lfw),
                                      "selfies": group_hash(ds.selfies.values())},
        },
        "notes": "Client runs on the host; db and CompreFace run in Docker (WSL2) under the caps above, so numbers are relative, not production capacity.",
    }


# ── phase: throughput ─────────────────────────────────────────────────────────

def phase_throughput(ds: Dataset, workers_list) -> dict:
    log("throughput (CompreFace direct)")
    paths = ds.event
    for p in paths[:3]:  # warm-up, not counted
        extract_face_embedding(str(p))
    out = {}
    for w in workers_list:
        lat, nfaces, groups = [], 0, defaultdict(lambda: {"photos": 0, "with_face": 0, "faces": 0})

        errors = defaultdict(int)

        def one(p):
            t = time.perf_counter()
            try:
                faces = extract_face_embedding(str(p))
            except FaceServiceError as e:  # count it; a benchmark should report failures, not die on one
                errors[" ".join(str(e).split())[:90]] += 1
                return p, None, 0
            return p, (time.perf_counter() - t) * 1000, len(faces)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=w) as pool:
            for p, ms, n in pool.map(one, paths):
                if ms is not None:
                    lat.append(ms)
                nfaces += n
                g = groups[ds.group(p)]; g["photos"] += 1; g["with_face"] += n > 0; g["faces"] += n
        wall = time.perf_counter() - t0
        ok = len(lat)
        out[f"workers_{w}"] = {"photos": len(paths), "calls_ok": ok, "calls_failed": len(paths) - ok, "errors": dict(errors),
                               "faces": nfaces, "wall_s": round(wall, 2),
                               "compreface_calls_per_s": round(ok / wall, 2), "faces_per_s": round(nfaces / wall, 2),
                               "call_latency_ms": stats_ms(lat)}
        out[f"workers_{w}"]["detection_by_group"] = {k: dict(v) for k, v in groups.items()}
        log(f"  workers={w}: {out[f'workers_{w}']['compreface_calls_per_s']} calls/s, {out[f'workers_{w}']['faces_per_s']} faces/s, failed={len(paths) - ok}")
    return out


# ── phase: e2e ────────────────────────────────────────────────────────────────

def phase_e2e(ds: Dataset, state: dict) -> dict:
    log("e2e: upload + batch")
    up = {"X-API-Key": ENV["UPLOAD_API_KEY"]}
    adm = {"X-API-Key": ENV["ADMIN_API_KEY"]}
    event = str(uuid.uuid4())
    mapping = {}
    s = requests.Session()
    t0 = time.perf_counter()
    for p in ds.event:
        with open(p, "rb") as f:
            r = s.post(f"{API}/upload", headers=up, data={"event_id": event}, files={"file": ("x.jpg", f, "image/jpeg")}, timeout=60)
        r.raise_for_status()
        mapping[r.json()["photo_id"]] = {"group": ds.group(p), "person": ds.labels.get(p.name)}
    t_upload = time.perf_counter() - t0

    conn = connect(); cur = conn.cursor()
    t1 = time.perf_counter()
    s.post(f"{API}/admin/batch/{event}", headers=adm, timeout=30).raise_for_status()
    # A photo that hits a transient CompreFace error is logged and left unattempted so it is retried.
    # In production the sweep retries it after SWEEP_GRACE_MINUTES; here, when a run has ended with photos
    # left over, trigger the batch again (and count it) instead of waiting for the sweep.
    retriggers = 0
    while True:
        cur.execute("select count(*) from media_ai.event_photos where event_id=%s and face_attempted_at is null", (event,))
        if cur.fetchone()[0] == 0:
            break
        if time.perf_counter() - t1 > 900:
            raise RuntimeError("batch did not finish in 15 minutes")
        cur.execute("select count(*) from media_ai.batch_runs where event_id=%s", (event,))
        if cur.fetchone()[0] == 0 and time.perf_counter() - t1 > 5:  # no run active, photos remain
            if retriggers >= 5:
                raise RuntimeError("photos still unattempted after 5 batch retriggers")
            retriggers += 1
            log(f"  run ended with photos left over (transient CompreFace error); retriggering batch ({retriggers})")
            s.post(f"{API}/admin/batch/{event}", headers=adm, timeout=30).raise_for_status()
            time.sleep(2)
        time.sleep(1)
    t_batch = time.perf_counter() - t1
    cur.execute("select count(*) from media_ai.face_embeddings fe join media_ai.event_photos ep using(photo_id) where ep.event_id=%s", (event,))
    faces = cur.fetchone()[0]
    cur.execute("select count(*) from media_ai.processing_errors pe join media_ai.event_photos ep using(photo_id) where ep.event_id=%s", (event,))
    errors = cur.fetchone()[0]
    conn.close()

    state.update({"event_id": event, "photos": mapping})
    save_state(state)
    n = len(ds.event)
    total = t_upload + t_batch
    res = {"photos": n, "faces_stored": faces, "processing_errors_logged": errors, "batch_retriggers_needed": retriggers, "upload_s": round(t_upload, 1), "batch_s": round(t_batch, 1),
           "total_s": round(total, 1), "upload_photos_per_min": round(n / t_upload * 60, 1),
           "batch_photos_per_min": round(n / t_batch * 60, 1), "end_to_end_photos_per_min": round(n / total * 60, 1),
           "face_workers": ENV.get("FACE_WORKERS") or "2 (container setting)",
           "note": "upload is sequential, one HTTP request per file, as the portal does; batch starts when the upload ends"}
    log(f"  end-to-end {res['end_to_end_photos_per_min']} photos/min (upload {res['upload_s']}s + batch {res['batch_s']}s)")
    return res


# ── selfie embeddings (cached) ────────────────────────────────────────────────

def selfie_embeddings(ds: Dataset) -> dict:
    cache = OUT_DIR / "selfie_embeddings.json"
    if cache.exists():
        return json.loads(cache.read_text())
    emb = {}
    for person, path in ds.selfies.items():
        best = get_best_face(extract_face_embedding(str(path)))
        if best is None:
            raise RuntimeError("no face found in a selfie")
        emb[person] = best["embedding"]
    OUT_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(emb))
    return emb


# ── phase: search ─────────────────────────────────────────────────────────────

def explain_nodes(plan) -> dict:
    nodes, idx, ops = [], [], []

    def walk(n, depth=0):
        nodes.append(n["Node Type"])
        label = n["Node Type"]
        if "Index Name" in n:
            idx.append(n["Index Name"])
            label += f" using {n['Index Name']}"
        if "Relation Name" in n:
            label += f" on {n['Relation Name']}"
        ops.append("  " * depth + label)
        for c in n.get("Plans", []):
            walk(c, depth + 1)
    walk(plan["Plan"])
    return {"nodes": nodes, "plan_tree": ops, "indexes_used": idx,
            "execution_ms": plan.get("Execution Time"), "planning_ms": plan.get("Planning Time")}


def phase_search(ds: Dataset, state: dict, n_http=150, n_sql=300) -> dict:
    log("search latency")
    event = state["event_id"]
    # ── full HTTP ──
    sess = requests.Session()
    files = list(ds.selfies.values())

    def http_once(i):
        with open(files[i % len(files)], "rb") as f:
            t = time.perf_counter()
            r = sess.post(f"{API}/search/face", data={"event_id": event, "consent": "true"}, files={"selfie": ("s.jpg", f, "image/jpeg")}, timeout=60)
            return (time.perf_counter() - t) * 1000, r.status_code

    first = http_once(0)
    if first[1] == 429:
        raise SystemExit("Search is rate limited (HTTP 429). Set SEARCH_RATE_LIMIT_PER_MINUTE=0 in .env, recreate the api container, and rerun.")
    seq = [http_once(i) for i in range(n_http)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        t0 = time.perf_counter()
        conc = list(pool.map(http_once, range(100)))
        conc_wall = time.perf_counter() - t0
    http = {
        "sequential": {"requests": len(seq), "status_codes": dict(Counter_(c for _, c in seq)), "latency_ms": stats_ms([ms for ms, c in seq if c == 200])},
        "concurrent_4_clients": {"requests": len(conc), "status_codes": dict(Counter_(c for _, c in conc)), "latency_ms": stats_ms([ms for ms, c in conc if c == 200]),
                                 "throughput_req_per_s": round(len(conc) / conc_wall, 2)},
        "includes": "multipart upload, image validation, one CompreFace call, consent insert, vector query, URL signing",
    }

    # ── SQL only (the production query text, real tables) ──
    emb = selfie_embeddings(ds)
    qs = list(emb.values())
    conn = connect(); cur = conn.cursor()
    lat, rows = [], []
    for i in range(n_sql):
        q = vlit(qs[i % len(qs)])
        t = time.perf_counter()
        cur.execute(CURRENT_QUERY.format(s="media_ai"), {"q": q, "event_id": event, "thr": THRESHOLD})
        r = cur.fetchall()
        lat.append((time.perf_counter() - t) * 1000); rows.append(len(r))
    cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + CURRENT_QUERY.format(s="media_ai"), {"q": vlit(qs[0]), "event_id": event, "thr": THRESHOLD})
    plan = explain_nodes(cur.fetchone()[0][0])
    cur.execute("select count(*) from media_ai.face_embeddings fe join media_ai.event_photos ep using(photo_id) where ep.event_id=%s", (event,))
    n_faces = cur.fetchone()[0]
    conn.close()
    sql = {"queries": n_sql, "faces_in_event": n_faces, "avg_rows_returned": round(statistics.mean(rows), 2), "latency_ms": stats_ms(lat),
           "plan": plan, "note": "client-side time on localhost including round trip; tiny table, see the index study for scale"}
    log(f"  http p50={http['sequential']['latency_ms'].get('p50')} ms, sql p50={sql['latency_ms']['p50']} ms")
    return {"http": http, "sql_only": sql}


def Counter_(it):
    c = defaultdict(int)
    for x in it:
        c[x] += 1
    return c


# ── phase: accuracy ───────────────────────────────────────────────────────────

def phase_accuracy(ds: Dataset, state: dict) -> dict:
    log("accuracy (labelled LFW only)")
    emb = selfie_embeddings(ds)
    event, mapping = state["event_id"], state["photos"]
    conn = connect(); cur = conn.cursor()

    # per-photo score = best face similarity, no threshold
    scores = {}
    for person, e in emb.items():
        cur.execute("""
            SELECT ep.photo_id, MAX(1 - (fe.embedding <=> %s::vector)) AS sim
            FROM media_ai.face_embeddings fe JOIN media_ai.event_photos ep ON ep.photo_id = fe.photo_id
            WHERE ep.event_id = %s GROUP BY ep.photo_id""", (vlit(e), event))
        scores[person] = {str(pid): float(sim) for pid, sim in cur.fetchall()}
    cur.execute("""SELECT ep.photo_id, count(fe.face_id) FROM media_ai.event_photos ep LEFT JOIN media_ai.face_embeddings fe USING (photo_id)
                   WHERE ep.event_id=%s GROUP BY ep.photo_id""", (event,))
    faces_per_photo = {str(p): int(n) for p, n in cur.fetchall()}
    conn.close()

    labelled = {pid: m["person"] for pid, m in mapping.items() if m["group"] == "lfw"}
    people = sorted({p for p in labelled.values()})
    genuine, impostor, per_person = [], [], []
    top1 = top5 = 0
    tp_at_thr = fp_at_thr = 0
    for person in people:
        ranked = sorted(((scores[person].get(pid, -1.0), pid) for pid in labelled), reverse=True)
        own = [pid for pid, p in labelled.items() if p == person]
        top1 += labelled[ranked[0][1]] == person
        top5 += any(labelled[pid] == person for _, pid in ranked[:5])
        for pid, p in labelled.items():
            s = scores[person].get(pid, -1.0)
            (genuine if p == person else impostor).append(s)
            if s >= THRESHOLD:
                tp_at_thr += p == person
                fp_at_thr += p != person
        per_person.append({"person": hashlib.sha256(person.encode()).hexdigest()[:8], "own_photos": len(own),
                           "found_at_threshold": sum(scores[person].get(pid, -1) >= THRESHOLD for pid in own)})

    outsider = [p for p in emb if p not in people]
    out_stats = [{"labelled_photos_at_or_above_threshold": sum(scores[o].get(pid, -1) >= THRESHOLD for pid in labelled),
                  "max_similarity_to_labelled_photo": round(max(scores[o].get(pid, -1) for pid in labelled), 4),
                  "max_similarity_to_any_photo": round(max(scores[o].values()), 4)} for o in outsider]
    out_imp = [scores[o].get(pid, -1.0) for o in outsider for pid in labelled]

    def tbl(vals_g, vals_i):
        rows = []
        for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
            rows.append({"threshold": t, "genuine_pairs_found": f"{sum(v >= t for v in vals_g)}/{len(vals_g)}",
                         "impostor_pairs_matched": f"{sum(v >= t for v in vals_i)}/{len(vals_i)}"})
        return rows

    # The same selfies through the real endpoint, so the numbers reflect the deployed threshold, the
    # one-row-per-photo query and the signed-URL response. (Respects the default 10 searches/minute limit.)
    sess = requests.Session()
    api_rows, outsider_results = [], []
    for person, path in ds.selfies.items():
        with open(path, "rb") as f:
            r = sess.post(f"{API}/search/face", data={"event_id": event, "consent": "true"},
                          files={"selfie": ("s.jpg", f, "image/jpeg")}, timeout=60)
        r.raise_for_status()
        ids = [x["photo_id"] for x in r.json()["photos"]]
        if person in people:
            api_rows.append({"person": hashlib.sha256(person.encode()).hexdigest()[:8], "returned": len(ids), "distinct": len(set(ids)),
                             "correct_labelled": sum(labelled.get(i) == person for i in ids),
                             "wrong_person_labelled": sum(labelled.get(i) not in (None, person) for i in ids),
                             "not_in_labelled_set": sum(i not in labelled for i in ids)})
        else:
            outsider_results.append({"returned": len(ids)})
        time.sleep(6.5)
    api_check = {
        "container_MIN_FACE_CONFIDENCE": sh("docker", "exec", "media_ai_api", "sh", "-c", "echo ${MIN_FACE_CONFIDENCE:-unset}"),
        "people": api_rows,
        "totals": {"correct_labelled": f"{sum(x['correct_labelled'] for x in api_rows)}/{len(labelled)}",
                   "wrong_person_labelled": sum(x["wrong_person_labelled"] for x in api_rows),
                   "duplicate_photos_in_any_result": sum(x["returned"] - x["distinct"] for x in api_rows)},
        "outsider_results_returned": outsider_results,
    }
    pct = lambda a, q: round(float(np.percentile(a, q)), 4)
    det = defaultdict(lambda: {"photos": 0, "with_face": 0, "faces": 0})
    for pid, m in mapping.items():
        d = det[m["group"]]; d["photos"] += 1; d["with_face"] += faces_per_photo.get(pid, 0) > 0; d["faces"] += faces_per_photo.get(pid, 0)
    return {
        "scope": "ranking and counts use only the labelled LFW photos in the event; RBNR photos are excluded from every accuracy number. LFW images are 250x250 and are NOT race photos.",
        "queries": len(people), "labelled_photos": len(labelled),
        "photos_per_person": sorted({sum(1 for v in labelled.values() if v == p) for p in people}),
        "top1_accuracy": f"{top1}/{len(people)}", "top5_accuracy": f"{top5}/{len(people)}",
        "at_threshold": {"threshold": THRESHOLD, "true_matches": f"{tp_at_thr}/{len(genuine)}",
                         "false_matches_other_labelled_person": f"{fp_at_thr}/{len(impostor)}"},
        "per_person": per_person,
        "out_of_set": {"people": len(outsider), "results": out_stats, "pairs_at_or_above_threshold": sum(v >= THRESHOLD for v in out_imp), "pairs": len(out_imp)},
        "similarity": {"genuine": {"n": len(genuine), "min": pct(genuine, 0), "p25": pct(genuine, 25), "median": pct(genuine, 50), "max": pct(genuine, 100)},
                       "impostor": {"n": len(impostor) + len(out_imp), "median": pct(impostor + out_imp, 50), "p95": pct(impostor + out_imp, 95),
                                    "p99": pct(impostor + out_imp, 99), "max": pct(impostor + out_imp, 100)}},
        "threshold_table": tbl(genuine, impostor + out_imp),
        "through_the_api": api_check,
        "face_detection": {k: dict(v) for k, v in det.items()},
    }


# ── phase: index study (synthetic distractors) ───────────────────────────────

def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def perturb(rng, base, cos):
    """Vector(s) whose cosine similarity to `base` is `cos`. base: (d,) or (n,d); cos: scalar or (n,)."""
    base = unit(np.atleast_2d(base))
    n = base.shape[0]
    noise = rng.standard_normal(base.shape)
    noise = unit(noise - (noise * base).sum(1, keepdims=True) * base)
    cos = np.broadcast_to(np.asarray(cos, dtype=float), (n,))[:, None]
    return (cos * base + np.sqrt(1 - cos ** 2) * noise).astype(np.float32)


def copy_rows(cur, table, columns, rows_iter):
    buf = io.StringIO()
    for r in rows_iter:
        buf.write("\t".join(r) + "\n")
    buf.seek(0)
    cur.copy_expert(f"COPY {table} ({','.join(columns)}) FROM STDIN", buf)


def vec_text(a):
    return "[" + ",".join(f"{x:.5g}" for x in a) + "]"


def recall(approx, exact):
    return (len(set(approx) & set(exact)) / len(exact)) if exact else 1.0


def phase_index(state: dict, n_synth: int, n_events: int, seed: int, n_queries: int, keep: bool, parts=("prod", "grid")) -> dict:
    log(f"index study: {n_synth} synthetic distractors across {n_events} events (seed {seed})")
    rng = np.random.default_rng(seed)
    conn = connect(); cur = conn.cursor()
    cur.execute("SET maintenance_work_mem = '640MB'")

    # real embeddings from the bench event
    cur.execute("""SELECT fe.photo_id, ep.event_id, fe.embedding::text FROM media_ai.face_embeddings fe
                   JOIN media_ai.event_photos ep USING (photo_id) WHERE ep.event_id = %s ORDER BY fe.face_id""", (state["event_id"],))
    real = cur.fetchall()
    real_vecs = unit(np.array([json.loads(r[2]) for r in real], dtype=np.float32))
    n_real = len(real)
    emb = selfie_embeddings_cached()
    base_pool = np.vstack([real_vecs, unit(np.array(list(emb.values()), dtype=np.float32))])
    events = [str(uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63)), version=4)) for _ in range(n_events)]

    cur.execute("DROP SCHEMA IF EXISTS bench CASCADE")
    cur.execute("CREATE SCHEMA bench")

    # ---------------- A. production-shaped tables: plan + latency ----------------
    cur.execute("""CREATE TABLE bench.event_photos (photo_id uuid PRIMARY KEY, event_id uuid NOT NULL, image_url text, thumbnail_url text)""")
    cur.execute("""CREATE TABLE bench.face_embeddings (face_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                   photo_id uuid NOT NULL REFERENCES bench.event_photos(photo_id) ON DELETE CASCADE, embedding vector(512) NOT NULL,
                   confidence numeric(5,2), face_box jsonb, created_at timestamp NOT NULL DEFAULT now())""")
    cur.execute("CREATE INDEX idx_bench_face_photo ON bench.face_embeddings(photo_id)")
    # exactly as db/schema.sql does it: IVFFlat (lists=100) created while the table is EMPTY
    cur.execute("CREATE INDEX idx_bench_ivfflat ON bench.face_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")

    def load_prod(vectors, evs, real_rows=None):
        photo_ids = [str(uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63)), version=4)) for _ in range(len(vectors))]
        copy_rows(cur, "bench.event_photos", ["photo_id", "event_id", "image_url", "thumbnail_url"],
                  ((p, e, f"{e}/original/{p}.jpg", f"{e}/thumbnails/{p}.jpg") for p, e in zip(photo_ids, evs)))
        copy_rows(cur, "bench.face_embeddings", ["photo_id", "embedding"], ((p, vec_text(v)) for p, v in zip(photo_ids, vectors)))

    def synth(n):
        base_idx = rng.integers(0, len(base_pool), size=n)
        cos = rng.uniform(0.2, 0.95, size=n)
        vecs = np.vstack([perturb(rng, base_pool[base_idx[i:i + 2000]], cos[i:i + 2000]) for i in range(0, n, 2000)])
        evs = [events[i] for i in rng.integers(0, n_events, size=n)]
        return vecs, evs

    # real faces go in as one more "event" (the bench event)
    load_prod(real_vecs, [state["event_id"]] * n_real)
    prod = []
    n_q = 100
    qbase = base_pool[rng.integers(0, len(base_pool), size=n_q)]
    qvecs = perturb(rng, qbase, 0.9)

    results_by_scenario = {}

    def measure_prod(label, total, evs_q, query=LEGACY_QUERY):
        cur.execute("ANALYZE bench.event_photos"); cur.execute("ANALYZE bench.face_embeddings")
        lat, nrows, res = [], [], []
        for q, ev in zip(qvecs, evs_q):
            t = time.perf_counter()
            cur.execute(query.format(s="bench"), {"q": vlit(q), "event_id": ev, "thr": THRESHOLD})
            rows = cur.fetchall()
            lat.append((time.perf_counter() - t) * 1000); nrows.append(len(rows)); res.append([str(r[0]) for r in rows])
        cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + query.format(s="bench"), {"q": vlit(qvecs[0]), "event_id": evs_q[0], "thr": THRESHOLD})
        plan = explain_nodes(cur.fetchone()[0][0])
        prod.append({"scenario": label, "faces_in_table": total, "avg_faces_per_event": round(total / (n_events + 1)),
                     "avg_rows_returned": round(statistics.mean(nrows), 2), "latency_ms": stats_ms(lat), "plan": plan})
        results_by_scenario[(label, total)] = res
        log(f"  {label} @ {total}: p50={prod[-1]['latency_ms']['p50']} ms p99={prod[-1]['latency_ms']['p99']} ms plan={plan['plan_tree'][:5]}")
        return res

    loaded = n_real
    sizes = sorted({min(10_000, n_synth), n_synth})
    for target in sizes:
        cur.execute("DROP INDEX IF EXISTS bench.idx_bench_ep_event")
        t = time.perf_counter()
        remaining = target - (loaded - n_real)
        for i in range(0, remaining, 10_000):
            vecs, evs = synth(min(10_000, remaining - i))
            load_prod(vecs, evs)
        loaded = n_real + target
        load_s = round(time.perf_counter() - t, 1)
        evs_q = [events[i] for i in rng.integers(0, n_events, size=n_q)]
        base = measure_prod("A. previous production query, no event_id index (IVFFlat built on empty table)", loaded, evs_q)
        prod[-1]["incremental_load_s"] = load_s
        cur.execute("CREATE INDEX idx_bench_ep_event ON bench.event_photos(event_id)")
        r_b = measure_prod("B. A + plain index on event_photos(event_id), same query", loaded, evs_q)
        r_c = measure_prod("C. B + event's photos selected first (CTE only, no LATERAL fence)", loaded, evs_q, EVENT_FIRST_QUERY)
        r_d = measure_prod("D. CURRENT production query (CTE + LATERAL fence) + event index", loaded, evs_q, CURRENT_QUERY)
        prod[-3]["same_results_as_A"] = r_b == base
        prod[-2]["same_results_as_A"] = r_c == base
        prod[-1]["same_results_as_A"] = r_d == base
        prod[-1]["non_empty_result_queries"] = sum(1 for r in r_d if r)
        if target == sizes[-1]:
            cur.execute("DROP INDEX bench.idx_bench_ivfflat")
            r_e = measure_prod("E. D with the unused IVFFlat index dropped", loaded, evs_q, CURRENT_QUERY)
            prod[-1]["same_results_as_D"] = r_e == r_d
    cur.execute("DROP INDEX IF EXISTS bench.idx_bench_ep_event")

    if "grid" not in parts:
        result = {"SYNTHETIC": "Distractors are perturbed copies of real embeddings (cosine to their source uniform in [0.2, 0.95]), assigned to events at random, fixed seed.",
                  "seed": seed, "synthetic_faces": n_synth, "real_faces": n_real, "events": n_events + 1, "queries": n_queries,
                  "production_shaped_tables": prod}
        if not keep:
            cur.execute("DROP SCHEMA bench CASCADE")
        conn.close()
        return result

    # ---------------- B. IVFFlat grid on a denormalised table ----------------
    log("index grid: loading denormalised table")
    cur.execute("CREATE TABLE bench.faces (face_id bigserial PRIMARY KEY, event_id uuid NOT NULL, embedding vector(512) NOT NULL)")
    t = time.perf_counter()
    copy_rows(cur, "bench.faces", ["event_id", "embedding"], ((state["event_id"], vec_text(v)) for v in real_vecs))
    for i in range(0, n_synth, 10_000):
        vecs, evs = synth(min(10_000, n_synth - i))
        copy_rows(cur, "bench.faces", ["event_id", "embedding"], ((e, vec_text(v)) for e, v in zip(evs, vecs)))
    cur.execute("ANALYZE bench.faces")
    cur.execute("SELECT count(*) FROM bench.faces"); N = cur.fetchone()[0]
    load_s = round(time.perf_counter() - t, 1)
    log(f"  {N} rows in {load_s}s")

    gq = perturb(rng, base_pool[rng.integers(0, len(base_pool), size=n_queries)], 0.9)
    gev = [events[i] for i in rng.integers(0, n_events, size=n_queries)]

    def topk(q, ev=None, k=10):
        if ev is None:
            cur.execute("SELECT face_id FROM bench.faces ORDER BY embedding <=> %s::vector LIMIT %s", (vlit(q), k))
        else:
            cur.execute("SELECT face_id FROM bench.faces WHERE event_id = %s ORDER BY embedding <=> %s::vector LIMIT %s", (ev, vlit(q), k))
        return [r[0] for r in cur.fetchall()]

    # exact ground truth + exact latency (no index yet => sequential scan)
    cur.execute("SET enable_indexscan = off")
    t_un, t_f = [], []
    exact_un, exact_f = [], []
    for q, ev in zip(gq, gev):
        s0 = time.perf_counter(); exact_un.append(topk(q)); t_un.append((time.perf_counter() - s0) * 1000)
        s0 = time.perf_counter(); exact_f.append(topk(q, ev)); t_f.append((time.perf_counter() - s0) * 1000)
    cur.execute("RESET enable_indexscan")
    exact = {"rows": N, "unfiltered_seq_scan_ms": stats_ms(t_un), "event_filtered_seq_scan_ms": stats_ms(t_f),
             "avg_rows_in_event": round(N / (n_events + 1))}
    cur.execute("CREATE INDEX idx_bench_faces_event ON bench.faces(event_id)")
    cur.execute("ANALYZE bench.faces")
    t_fb = []
    for q, ev in zip(gq, gev):
        s0 = time.perf_counter(); got = topk(q, ev); t_fb.append((time.perf_counter() - s0) * 1000)
    exact["event_filtered_btree_exact_ms"] = stats_ms(t_fb)
    cur.execute("DROP INDEX bench.idx_bench_faces_event")

    def run_cfg(label, lists, build_s):
        rows = []
        cur.execute("SET enable_seqscan = off"); cur.execute("SET enable_bitmapscan = off")
        cur.execute("EXPLAIN (FORMAT JSON) SELECT face_id FROM bench.faces WHERE event_id = %s ORDER BY embedding <=> %s::vector LIMIT 10", (gev[0], vlit(gq[0])))
        used = explain_nodes({"Plan": cur.fetchone()[0][0]["Plan"]})["indexes_used"]
        for probes in (1, 2, 4, 8, 16, 32):
            cur.execute(f"SET ivfflat.probes = {probes}")
            rec = {"index": label, "lists": lists, "probes": probes, "index_used_for_filtered_query": bool(used), "build_s": build_s}
            for mode in ("unfiltered", "event_filtered"):
                lat, rc, short = [], [], 0
                for j, (q, ev) in enumerate(zip(gq, gev)):
                    s0 = time.perf_counter()
                    got = topk(q) if mode == "unfiltered" else topk(q, ev)
                    lat.append((time.perf_counter() - s0) * 1000)
                    truth = exact_un[j] if mode == "unfiltered" else exact_f[j]
                    rc.append(recall(got, truth)); short += len(got) < len(truth)
                rec[mode] = {"recall_at_10": round(float(np.mean(rc)), 3), "queries_returning_fewer_than_exact": short, "latency_ms": stats_ms(lat)}
            rows.append(rec)
            log(f"  {label} lists={lists} probes={probes}: recall unf={rec['unfiltered']['recall_at_10']} filt={rec['event_filtered']['recall_at_10']} "
                f"p50 unf={rec['unfiltered']['latency_ms']['p50']}ms")
        cur.execute("RESET enable_seqscan"); cur.execute("RESET enable_bitmapscan")
        return rows

    grid = []
    for lists in sorted({max(1, int(N / 1000)), int(round(math.sqrt(N))), max(1, int(N / 100))}):
        t = time.perf_counter()
        cur.execute(f"CREATE INDEX idx_bench_faces_ivf ON bench.faces USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})")
        build_s = round(time.perf_counter() - t, 1)
        grid += run_cfg("IVFFlat built after load", lists, build_s)
        cur.execute("DROP INDEX bench.idx_bench_faces_ivf")

    # production-style: index created first on the EMPTY table, rows inserted afterwards (same face_ids as bench.faces)
    log("index grid: production-style (IVFFlat created on an empty table, then loaded)")
    cur.execute("CREATE TABLE bench.faces_empty_first (face_id bigint PRIMARY KEY, event_id uuid NOT NULL, embedding vector(512) NOT NULL)")
    cur.execute("CREATE INDEX idx_bench_faces_ef ON bench.faces_empty_first USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
    t = time.perf_counter()
    cur.execute("INSERT INTO bench.faces_empty_first SELECT face_id, event_id, embedding FROM bench.faces")
    ef_load = round(time.perf_counter() - t, 1)
    cur.execute("ANALYZE bench.faces_empty_first")
    prod_style = []
    cur.execute("SET enable_seqscan = off"); cur.execute("SET enable_bitmapscan = off")
    for probes in (1, 10, 32):
        cur.execute(f"SET ivfflat.probes = {probes}")
        rc, nret = [], []
        for j, q in enumerate(gq):
            cur.execute("SELECT face_id FROM bench.faces_empty_first ORDER BY embedding <=> %s::vector LIMIT 10", (vlit(q),))
            got = [r[0] for r in cur.fetchall()]
            rc.append(recall(got, exact_un[j])); nret.append(len(got))
        prod_style.append({"probes": probes, "recall_at_10_unfiltered": round(float(np.mean(rc)), 3), "avg_rows_returned_of_10": round(float(np.mean(nret)), 2)})
        log(f"  empty-first lists=100 probes={probes}: recall={prod_style[-1]['recall_at_10_unfiltered']}")
    cur.execute("RESET enable_seqscan"); cur.execute("RESET enable_bitmapscan")

    result = {
        "SYNTHETIC": "Distractors are perturbed copies of real embeddings (cosine to their source uniform in [0.2, 0.95]), assigned to events at random, fixed seed. Absolute numbers depend on this construction.",
        "seed": seed, "synthetic_faces": n_synth, "real_faces": n_real, "events": n_events + 1, "queries": n_queries,
        "production_shaped_tables": prod, "exact_baseline": exact, "ivfflat_grid": grid,
        "ivfflat_built_on_empty_table": {"lists": 100, "insert_s": ef_load, "by_probes": prod_style,
                                         "note": "same data and queries as the grid; recall vs exact unfiltered top-10"},
        "denormalised_table_rows": N, "load_s": load_s,
    }
    if not keep:
        cur.execute("DROP SCHEMA bench CASCADE")
        log("  dropped schema bench")
    conn.close()
    return result


def selfie_embeddings_cached():
    return json.loads((OUT_DIR / "selfie_embeddings.json").read_text())


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", default=ENV.get("FACE_TEST_PHOTOS"), help="folder with event/, selfies/, labels.csv")
    ap.add_argument("--phases", default="env,throughput,e2e,search,accuracy,index")
    ap.add_argument("--workers", default="1,2,4", help="CompreFace client worker counts for the throughput phase")
    ap.add_argument("--index-size", type=int, default=100_000, help="synthetic distractor faces")
    ap.add_argument("--events", type=int, default=200, help="synthetic events the distractors are spread over")
    ap.add_argument("--queries", type=int, default=200, help="queries per index configuration")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep", action="store_true", help="keep the bench schema after the index phase")
    ap.add_argument("--index-parts", default="prod,grid", help="index phase parts: prod (production-shaped tables), grid (IVFFlat grid)")
    ap.add_argument("--label", default="", help="suffix for the results file name, e.g. event-first -> results-<date>-event-first.json")
    ap.add_argument("--cleanup", action="store_true", help="delete the bench event (rows and files) at the end of the run")
    ap.add_argument("--out", default=str(ROOT / "docs" / "benchmarks"))
    args = ap.parse_args()
    if not args.photos or not (Path(args.photos) / "labels.csv").is_file():
        raise SystemExit("--photos (or FACE_TEST_PHOTOS) must point at a folder with event/, selfies/ and labels.csv")
    for k in ("COMPREFACE_API_KEY", "UPLOAD_API_KEY", "ADMIN_API_KEY"):
        if not ENV.get(k):
            raise SystemExit(f"{k} is not set in .env")

    OUT_DIR.mkdir(exist_ok=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"results-{dt.date.today().isoformat()}{'-' + args.label if args.label else ''}.json"
    ds = Dataset(Path(args.photos))
    phases = args.phases.split(",")
    state = load_state()
    report = {"phases_run": phases}
    t_start = time.time()

    def flush():
        """Write the report after every phase so a crash late in the run loses nothing."""
        merged = json.loads(path.read_text()) if path.exists() else {}
        merged.update(report)
        merged["phases_run"] = sorted(set(merged.get("phases_run", [])) | set(phases))
        path.write_text(json.dumps(merged, indent=2))

    if "env" in phases:
        report["environment"] = phase_env(ds); flush()
    if "throughput" in phases:
        report["embedding_throughput"] = phase_throughput(ds, [int(w) for w in args.workers.split(",")]); flush()
    if "e2e" in phases:
        report["end_to_end"] = phase_e2e(ds, state); flush()
    if {"search", "accuracy", "index"} & set(phases):
        if not state.get("event_id"):
            raise SystemExit("run the e2e phase first (it creates the bench event)")
        selfie_embeddings(ds)
    if "search" in phases:
        report["search_latency"] = phase_search(ds, state); flush()
    if "accuracy" in phases:
        report["accuracy"] = phase_accuracy(ds, state); flush()
    if "index" in phases:
        report["index_study"] = phase_index(state, args.index_size, args.events, args.seed, args.queries, args.keep, tuple(args.index_parts.split(','))); flush()
    report["total_run_s"] = round(time.time() - t_start)

    if args.cleanup and state.get("event_id"):
        c = connect(); cur = c.cursor()
        cur.execute("DELETE FROM media_ai.search_requests WHERE event_id=%s", (state["event_id"],))
        cur.execute("DELETE FROM media_ai.event_photos WHERE event_id=%s", (state["event_id"],))
        c.close()
        import shutil
        shutil.rmtree(ROOT / "storage" / state["event_id"], ignore_errors=True)
        log("removed the bench event")

    flush()
    log(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()
