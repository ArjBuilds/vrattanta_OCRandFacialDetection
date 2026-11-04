"""The face-search SQL (db/queries.py): event-first plan, the event_id index, and equivalence with the
query it replaced. Real SQL against the throwaway test database."""

import io
import random
import uuid

import pytest

from conftest import vec
from db.connection import execute_query, get_conn, release_conn
from db.queries import FACE_SEARCH_SQL

# The query as it was before the event-first rewrite, kept here only as a reference for equivalence tests.
LEGACY_SQL = """
SELECT ep.photo_id, ep.event_id, ep.thumbnail_url IS NOT NULL AS has_thumb,
       1 - (fe.embedding <=> %(q)s::vector) AS confidence
FROM media_ai.face_embeddings fe
JOIN media_ai.event_photos ep ON ep.photo_id = fe.photo_id
WHERE ep.event_id = %(event_id)s
  AND 1 - (fe.embedding <=> %(q)s::vector) >= %(thr)s
ORDER BY confidence DESC
LIMIT 50
"""


def vlit(v):
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def bulk_load(rows):
    """rows: iterable of (event_id, [similarities...]) -> one photo per row with one face per similarity."""
    photos, faces = io.StringIO(), io.StringIO()
    n = 0
    for event_id, sims in rows:
        pid = str(uuid.uuid4())
        photos.write(f"{pid}\t{event_id}\t{event_id}/original/{pid}.jpg\t{event_id}/thumbnails/{pid}.jpg\n")
        for s in sims:
            faces.write(f"{pid}\t{vlit(vec(s))}\n")
        n += 1
    photos.seek(0); faces.seek(0)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.copy_expert("COPY media_ai.event_photos (photo_id, event_id, image_url, thumbnail_url) FROM STDIN", photos)
            cur.copy_expert("COPY media_ai.face_embeddings (photo_id, embedding) FROM STDIN", faces)
            cur.execute("ANALYZE media_ai.event_photos")
            cur.execute("ANALYZE media_ai.face_embeddings")
        conn.commit()
    finally:
        release_conn(conn)
    return n


def plan_of(params):
    plan = execute_query("EXPLAIN (FORMAT JSON) " + FACE_SEARCH_SQL, params, fetch="one")[0][0]["Plan"]
    nodes = []

    def walk(n):
        nodes.append((n["Node Type"], n.get("Relation Name"), n.get("Index Name")))
        for c in n.get("Plans", []):
            walk(c)
    walk(plan)
    return nodes


def test_event_id_index_exists(test_db):
    row = execute_query("SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_event_photos_event_id'", fetch="one")
    assert row is not None and "(event_id)" in row[0]
    assert "WHERE" not in row[0]  # a plain index, not partial


def test_query_keeps_its_two_fences():
    # Removing either lets the planner pick a hash join over a sequential scan of every face.
    assert "MATERIALIZED" in FACE_SEARCH_SQL
    assert "LATERAL" in FACE_SEARCH_SQL and "OFFSET 0" in FACE_SEARCH_SQL
    assert FACE_SEARCH_SQL.index("event_photos") < FACE_SEARCH_SQL.index("face_embeddings")


def test_same_rows_and_order_as_the_query_it_replaced(clean_db):
    rng = random.Random(7)
    events = [str(uuid.uuid4()) for _ in range(12)]
    # distinct similarities so ordering is deterministic; some photos have two faces
    bulk_load((rng.choice(events), [rng.uniform(0.3, 1.0) for _ in range(rng.choice([1, 1, 2]))]) for _ in range(600))
    for thr in (0.5, 0.7, 0.9):
        for ev in events:
            p = {"event_id": ev, "q": vec(1.0), "thr": thr}
            old = [(str(r[0]), round(float(r[3]), 6)) for r in execute_query(LEGACY_SQL, p, fetch="all")]
            new = [(str(r[0]), round(float(r[3]), 6)) for r in execute_query(FACE_SEARCH_SQL, p, fetch="all")]
            assert new == old and len(new) <= 50


def test_unknown_event_returns_nothing(clean_db):
    bulk_load([(str(uuid.uuid4()), [0.9])])
    assert execute_query(FACE_SEARCH_SQL, {"event_id": str(uuid.uuid4()), "q": vec(1.0), "thr": 0.5}, fetch="all") == []


@pytest.mark.parametrize("n_photos,n_events", [(10_000, 20), (60_000, 120)])
def test_planner_uses_the_event_index_and_never_scans_all_faces(clean_db, n_photos, n_events):
    """The plan must start from idx_event_photos_event_id and reach faces through idx_face_embeddings_photo_id, at
    the sizes where a plain join was seen to turn into a hash join over every face."""
    events = [str(uuid.uuid4()) for _ in range(n_events)]
    rng = random.Random(3)
    bulk_load((events[i % n_events], [rng.uniform(0.0, 1.0)]) for i in range(n_photos))
    nodes = plan_of({"event_id": events[0], "q": vec(1.0), "thr": 0.7})
    indexes = {n[2] for n in nodes if n[2]}
    assert "idx_event_photos_event_id" in indexes, nodes
    assert "idx_face_embeddings_photo_id" in indexes, nodes
    assert ("Seq Scan", "face_embeddings", None) not in nodes, nodes
    assert ("Seq Scan", "event_photos", None) not in nodes, nodes
