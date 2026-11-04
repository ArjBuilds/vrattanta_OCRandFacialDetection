"""SQL used by the request path. Kept in one place so tests and the benchmark exercise the same text."""

# Face search, scoped to one event.
#
# Shape: select the event's photos first (idx_event_photos_event_id), then look up each photo's faces
# through idx_face_embeddings_photo_id, and only then compute the similarity. Two fences enforce that
# order instead of leaving it to the planner:
#   * the MATERIALIZED CTE keeps the event lookup from being merged into the join, and
#   * LATERAL (...) OFFSET 0 stops the planner flattening the face lookup into an ordinary join, which
#     it otherwise sometimes turns into a hash join over a sequential scan of ALL faces, computing the
#     cosine similarity for every row. Measured: adding only the event index made the old query ~7.5x
#     slower at 100,000 faces (80 ms -> 604 ms), and a CTE alone still produced that hash join at
#     30,000 to 60,000 photos in tests (docs/benchmarks/RESULTS.md).
#
# The work is exact (no approximate index) and limited to one event, so its cost grows with the number
# of faces in that event, not with the size of the table.
#
# One row per PHOTO: a photo with several faces that match returns once, with its best score, and
# LIMIT 50 counts photos (not faces).
#
# Named parameters: event_id, q (query embedding), thr (minimum similarity).
FACE_SEARCH_SQL = """
WITH ev AS MATERIALIZED (
    SELECT photo_id, event_id, thumbnail_url
    FROM media_ai.event_photos
    WHERE event_id = %(event_id)s
)
SELECT
    ev.photo_id,
    ev.event_id,
    ev.thumbnail_url IS NOT NULL AS has_thumb,
    MAX(1 - (f.embedding <=> %(q)s::vector)) AS confidence
FROM ev
CROSS JOIN LATERAL (
    SELECT fe.embedding
    FROM media_ai.face_embeddings fe
    WHERE fe.photo_id = ev.photo_id
    OFFSET 0
) f
WHERE 1 - (f.embedding <=> %(q)s::vector) >= %(thr)s
GROUP BY ev.photo_id, ev.event_id, ev.thumbnail_url
ORDER BY confidence DESC
LIMIT 50
"""
