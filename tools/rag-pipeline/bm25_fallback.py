"""BM25 keyword search over the ChromaDB chunks.

Two consumers:
- Hybrid search (search.py): BM25 as one ranked list in the RRF fusion
  (mark_degraded=False, wait=False - never block a live query).
- Degraded mode (server.py): if the embedding API is down, search falls
  back to BM25-only results (mark_degraded=True, wait=True).

The index lives in RAM and is rebuilt in the background whenever the
collection's chunk count changes; queries are served from the stale
index meanwhile.
"""
import logging
import re
import threading
import time

from rank_bm25 import BM25Okapi

from config import COLLECTION_KNOWLEDGE
from indexer import get_chroma_client, get_or_create_collection

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9äöüß]+")
_FETCH_BATCH = 2000
_BUILD_WAIT_SECONDS = 25

_indexes: dict[str, dict] = {}
_building: set[str] = set()
_lock = threading.Lock()


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _build(collection_name: str) -> dict:
    client = get_chroma_client()
    collection = get_or_create_collection(client, collection_name)
    total = collection.count()
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    offset = 0
    while offset < total:
        batch = collection.get(
            include=["documents", "metadatas"], limit=_FETCH_BATCH, offset=offset,
        )
        got = batch.get("documents") or []
        if not got:
            break
        ids.extend(batch.get("ids") or [""] * len(got))
        docs.extend(got)
        metas.extend(batch.get("metadatas") or [{}] * len(got))
        offset += len(got)

    index = {
        "bm25": None, "ids": ids, "docs": docs, "metas": metas,
        "count": total, "built_at": time.time(),
    }
    if docs:
        t0 = time.time()
        index["bm25"] = BM25Okapi([_tokenize(d) for d in docs])
        logger.info(
            "BM25 index for %s built: %d chunks in %.1fs",
            collection_name, len(docs), time.time() - t0,
        )
    return index


def _rebuild_async(collection_name: str) -> None:
    with _lock:
        if collection_name in _building:
            return
        _building.add(collection_name)

    def worker():
        try:
            index = _build(collection_name)
            with _lock:
                _indexes[collection_name] = index
        except Exception:
            logger.exception("BM25 index build failed for %s", collection_name)
        finally:
            with _lock:
                _building.discard(collection_name)

    threading.Thread(target=worker, daemon=True, name=f"bm25-build-{collection_name}").start()


def ensure_index_async(collection_name: str = COLLECTION_KNOWLEDGE) -> None:
    """Warm the index in the background (call once at server start)."""
    with _lock:
        have = collection_name in _indexes
    if not have:
        _rebuild_async(collection_name)


def _get_or_wait_for_index(collection_name: str) -> dict | None:
    """Return the current index; if a build is running, wait briefly for it."""
    deadline = time.time() + _BUILD_WAIT_SECONDS
    while True:
        with _lock:
            index = _indexes.get(collection_name)
            building = collection_name in _building
        if index is not None or not building or time.time() > deadline:
            return index
        time.sleep(0.5)


def search(
    query: str, collection_name: str = COLLECTION_KNOWLEDGE, n_results: int = 5,
    mark_degraded: bool = True, wait: bool = True,
) -> list[dict]:
    """Keyword search returning results in the vector-search shape
    (text/source/metadata/distance + chunk_id). Distance is a
    pseudo-distance derived from the BM25 score (lower = better);
    the hybrid layer replaces it with the true cosine distance.

    wait=False never blocks: if the index is still building, it returns
    [] and the caller proceeds without the BM25 list."""
    # Trigger a background rebuild if the collection changed since the
    # index was built; the stale index keeps serving in the meantime.
    try:
        client = get_chroma_client()
        current = get_or_create_collection(client, collection_name).count()
        with _lock:
            index = _indexes.get(collection_name)
        if index is None or index["count"] != current:
            _rebuild_async(collection_name)
    except Exception:
        logger.exception("BM25 count check failed for %s", collection_name)

    if wait:
        index = _get_or_wait_for_index(collection_name)
    else:
        with _lock:
            index = _indexes.get(collection_name)
    if index is None or index["bm25"] is None:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = index["bm25"].get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    items = []
    for i in order[:n_results]:
        score = float(scores[i])
        if score <= 0:
            break
        meta = dict(index["metas"][i] or {})
        meta["_bm25_score"] = round(score, 3)
        if mark_degraded:
            meta["_degraded"] = True
        items.append({
            "text": index["docs"][i],
            "source": meta.get("source", meta.get("book_title", "unknown")),
            "metadata": meta,
            "distance": round(1.0 / (1.0 + score), 4),
            "chunk_id": index["ids"][i],
        })
    return items
