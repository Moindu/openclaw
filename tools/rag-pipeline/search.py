"""RAG search module - queries ChromaDB collections."""
import sys
from pathlib import Path

from config import COLLECTION_KNOWLEDGE, COLLECTION_PRODUCTS, COLLECTION_RECIPES
import bm25_fallback
from embeddings import get_embedding, get_image_embedding
from indexer import get_chroma_client, get_or_create_collection


def _detect_task_type(query: str) -> str:
    """Choose embedding task type based on query structure.

    Questions and longer queries use QUESTION_ANSWERING (optimized for
    finding answer-containing documents). Short keyword queries use
    RETRIEVAL_QUERY (optimized for general search).
    """
    question_indicators = ["?", "wie ", "was ", "warum ", "welche ", "welcher ",
                           "welches ", "wann ", "wo ", "wer ", "how ", "what ",
                           "why ", "which ", "when ", "where ", "who "]
    query_lower = query.lower()
    if any(q in query_lower for q in question_indicators):
        return "QUESTION_ANSWERING"
    if len(query.split()) <= 3:
        return "RETRIEVAL_QUERY"
    return "QUESTION_ANSWERING"


def _query_collection(
    collection, query_embedding: list[float], n_results: int = 5,
    where: dict | None = None, max_distance: float | None = None,
) -> list[dict]:
    """Run a vector query against a ChromaDB collection.

    Args:
        max_distance: Optional threshold - only return results with
            cosine distance below this value. ChromaDB cosine distance
            ranges from 0.0 (identical) to 2.0 (opposite).
            Recommended: 0.8 for strict, 1.2 for relaxed.
    """
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    items = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        if max_distance is not None and dist > max_distance:
            continue
        items.append({
            "text": doc,
            "source": meta.get("source", meta.get("book_title", "unknown")),
            "metadata": meta,
            "distance": dist,
        })

    return items


def _diversify_results(
    results: list[dict], n_results: int, max_per_source: int = 2,
) -> list[dict]:
    """Select diverse results: max N chunks per source document.

    Takes a larger pool of results sorted by relevance and picks
    the best from each unique source, ensuring broad coverage across
    the knowledge base.
    """
    source_counts: dict[str, int] = {}
    diverse = []

    for r in results:
        source = r["source"]
        count = source_counts.get(source, 0)
        if count < max_per_source:
            diverse.append(r)
            source_counts[source] = count + 1
            if len(diverse) >= n_results:
                break

    return diverse


def _filename_search(collection, query, query_embedding, n_results=5):
    """Find chunks by content/filename match as HNSW fallback.

    HNSW approximate search can miss recently added or poorly-connected
    chunks. This uses ChromaDB full-text search and filesystem scanning
    to find documents that vector search missed.
    """
    import logging
    import numpy as np
    from config import DOCUMENTS_DIR

    logger = logging.getLogger(__name__)
    words = [w.lower() for w in query.split() if len(w) >= 3]
    if not words:
        return []

    items = []
    qvec = np.array(query_embedding)
    seen_ids = set()

    def _compute_and_add(results):
        if not results or not results["ids"]:
            return
        for cid, doc, meta, emb in zip(
            results["ids"],
            results.get("documents", []),
            results.get("metadatas", []),
            results.get("embeddings", []),
        ):
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            evec = np.array(emb)
            norm_e = np.linalg.norm(evec)
            norm_q = np.linalg.norm(qvec)
            if norm_e > 0 and norm_q > 0:
                dist = 1.0 - float(np.dot(evec, qvec) / (norm_e * norm_q))
            else:
                dist = 1.0
            items.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "metadata": meta,
                "distance": dist,
            })

    # Strategy 1: Full-text search in document content
    for word in words[:3]:
        try:
            r = collection.get(
                where_document={"$contains": word},
                include=["documents", "metadatas", "embeddings"],
                limit=20,
            )
            _compute_and_add(r)
        except Exception as e:
            logger.debug("Full-text search for '%s' failed: %s", word, e)

    # Strategy 2: Scan filesystem for files whose name matches query words,
    # then fetch their chunks from ChromaDB by source metadata
    try:
        docs_dir = Path(DOCUMENTS_DIR)
        if docs_dir.exists():
            for path in docs_dir.rglob("*"):
                if not path.is_file():
                    continue
                fname_lower = path.name.lower()
                if any(w in fname_lower for w in words):
                    try:
                        r = collection.get(
                            where={"source": path.name},
                            include=["documents", "metadatas", "embeddings"],
                            limit=10,
                        )
                        _compute_and_add(r)
                    except Exception as e:
                        logger.debug("Source fetch for '%s' failed: %s", path.name, e)
    except Exception as e:
        logger.debug("Filesystem scan failed: %s", e)

    items.sort(key=lambda x: x["distance"])
    return items[:n_results]


# --- Hybrid search: weighted Reciprocal Rank Fusion ---
#
# Four ranked lists are fused: vector similarity, BM25 keyword ranking,
# filename matches and cross-modal image results. RRF replaces the old
# ad-hoc distance scaling (x0.5 for images/filename hits): ranking comes
# from fused rank positions, while `distance` keeps its meaning as the
# true cosine distance (BM25-only hits get theirs computed on the fly).
RRF_K = 60
WEIGHT_VECTOR = 1.0
WEIGHT_BM25 = 0.7
WEIGHT_FILENAME = 0.5
WEIGHT_IMAGE = 0.4


def _result_key(item: dict) -> str:
    meta = item.get("metadata", {})
    return item["source"] + "_" + str(meta.get("chunk_index", meta.get("page_start", "")))


def _rrf_fuse(ranked_lists: list[tuple[float, list[dict]]], k: int = RRF_K) -> list[dict]:
    """Weighted Reciprocal Rank Fusion over multiple ranked lists.

    ranked_lists: [(weight, items_best_first), ...]. Deduplicates by
    source+chunk position; returns items sorted by fused score."""
    scores: dict[str, float] = {}
    best: dict[str, dict] = {}
    for weight, items in ranked_lists:
        for rank, item in enumerate(items, 1):
            key = _result_key(item)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            prev = best.get(key)
            if prev is None or item["distance"] < prev["distance"]:
                best[key] = item
    fused = []
    for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        item = best[key]
        item["metadata"]["_rrf_score"] = round(score, 5)
        fused.append(item)
    return fused


def _attach_true_distances(collection, items: list[dict], query_embedding: list[float]) -> None:
    """Replace BM25 pseudo-distances with real cosine distances by
    fetching the stored embeddings of the hit chunks."""
    import numpy as np

    ids = [it["chunk_id"] for it in items if it.get("chunk_id")]
    if not ids:
        return
    try:
        got = collection.get(ids=ids, include=["embeddings"])
    except Exception:
        return
    emb_by_id = dict(zip(got.get("ids", []), got.get("embeddings", [])))
    q = np.array(query_embedding)
    qn = np.linalg.norm(q)
    if qn == 0:
        return
    for it in items:
        emb = emb_by_id.get(it.get("chunk_id"))
        if emb is None:
            continue
        e = np.array(emb)
        en = np.linalg.norm(e)
        if en > 0:
            it["distance"] = round(1.0 - float(np.dot(e, q) / (en * qn)), 4)


def search_collection(
    query: str, collection_name: str, n_results: int = 5,
    max_distance: float | None = None, diverse: bool = False,
    max_per_source: int = 2, query_embedding: list[float] | None = None,
) -> list[dict]:
    """Hybrid search over a single ChromaDB collection.

    Vector similarity and BM25 keyword ranking are fused via weighted
    RRF, together with filename matches (HNSW-miss fallback) and
    cross-modal image results.

    Args:
        diverse: If True, diversify across sources, limiting each source
            to max_per_source chunks.
        max_per_source: Max chunks per source document (only with diverse=True).
        query_embedding: Pre-computed embedding vector. If None, will be generated.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client, collection_name)
    if query_embedding is None:
        query_embedding = get_embedding(query, task_type=_detect_task_type(query))

    fetch_n = n_results * 3 if diverse else max(n_results * 5, 50)

    # List 1: vector similarity (text chunks only)
    text_items = _query_collection(
        collection, query_embedding, fetch_n,
        where={"chunk_type": {"$ne": "image"}},
    )

    # List 2: BM25 keyword ranking (never blocks; empty while index builds)
    bm25_items = bm25_fallback.search(
        query, collection_name, fetch_n, mark_degraded=False, wait=False,
    )
    _attach_true_distances(collection, bm25_items, query_embedding)

    # List 3: filename matches (catches docs HNSW approximate search misses)
    query_words = [w.lower() for w in query.split() if len(w) >= 3]
    filename_items = [
        it for it in _filename_search(collection, query, query_embedding, n_results)
        if any(w in it["source"].lower() for w in query_words)
    ]

    # List 4: cross-modal image results
    image_items = _query_collection(
        collection, query_embedding, max(3, n_results),
        where={"chunk_type": "image"},
    )
    for item in image_items:
        item["metadata"]["_image_result"] = True

    fused = _rrf_fuse([
        (WEIGHT_VECTOR, text_items),
        (WEIGHT_BM25, bm25_items),
        (WEIGHT_FILENAME, filename_items),
        (WEIGHT_IMAGE, image_items),
    ])

    if max_distance is not None:
        fused = [i for i in fused if i["distance"] <= max_distance]

    if diverse:
        return _diversify_results(fused, n_results, max_per_source)
    return fused[:n_results]


def search_knowledge(
    query: str, n_results: int = 5, max_distance: float | None = None,
    diverse: bool = False, max_per_source: int = 2,
) -> list[dict]:
    """Search the knowledge (documents) collection."""
    return search_collection(
        query, COLLECTION_KNOWLEDGE, n_results,
        max_distance=max_distance, diverse=diverse, max_per_source=max_per_source,
    )


def search_products(
    query: str, n_results: int = 5, max_distance: float | None = None,
    diverse: bool = False, max_per_source: int = 2,
) -> list[dict]:
    """Search the products collection."""
    return search_collection(
        query, COLLECTION_PRODUCTS, n_results,
        max_distance=max_distance, diverse=diverse, max_per_source=max_per_source,
    )


def search_all(
    query: str, n_results: int = 5, max_distance: float | None = None,
    diverse: bool = False, max_per_source: int = 2,
) -> list[dict]:
    """Search both knowledge and products, merge by relevance."""
    # Generate embedding once and reuse for both collections
    shared_embedding = get_embedding(query, task_type=_detect_task_type(query))
    knowledge = search_collection(
        query, COLLECTION_KNOWLEDGE, n_results,
        max_distance=max_distance, diverse=diverse, max_per_source=max_per_source,
        query_embedding=shared_embedding,
    )
    products = search_collection(
        query, COLLECTION_PRODUCTS, n_results,
        max_distance=max_distance, diverse=diverse, max_per_source=max_per_source,
        query_embedding=shared_embedding,
    )

    combined = knowledge + products
    # Both lists are RRF-ordered; merge by fused score so BM25-only hits
    # (worse cosine distance, better rank) keep their position.
    combined.sort(key=lambda x: -x["metadata"].get("_rrf_score", 0.0))

    if diverse:
        return _diversify_results(combined, n_results, max_per_source)
    return combined[:n_results]


# --- Recipe book search (new) ---


def search_recipes(
    query: str, n_results: int = 5, filters: dict | None = None,
    max_distance: float | None = None,
) -> list[dict]:
    """Search the recipe books collection with optional metadata filters.

    Args:
        query: Search query text.
        n_results: Number of results to return.
        filters: Optional ChromaDB where-filter, e.g.:
            {"leather_type": "Rindsleder"}
            {"tanning_method": "pflanzlich"}
            {"book_title": "Hein Gerberei-Handbuch 1923"}
        max_distance: Optional similarity threshold (0.0-2.0).

    Returns:
        List of result dicts with text, metadata, distance.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client, COLLECTION_RECIPES)
    query_embedding = get_embedding(query, task_type=_detect_task_type(query))
    return _query_collection(collection, query_embedding, n_results, where=filters, max_distance=max_distance)


def search_recipes_by_image(
    image_path: str | Path, n_results: int = 5
) -> list[dict]:
    """Search recipes by image similarity (cross-modal search).

    Upload a photo of a recipe page to find similar pages in the database.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client, COLLECTION_RECIPES)
    img_embedding = get_image_embedding(image_path, task_type="RETRIEVAL_QUERY")
    return _query_collection(collection, img_embedding, n_results)


# --- Formatting ---


def format_context(results: list[dict]) -> str:
    """Format search results as context for the LLM."""
    if not results:
        return "Keine relevanten Informationen gefunden."

    parts = []
    for i, r in enumerate(results, 1):
        source = r["source"]
        text = r["text"]
        meta = r.get("metadata", {})

        header = f"[Quelle {i}: {source}"
        if meta.get("page_number"):
            header += f", Seite {meta['page_number']}"
        if meta.get("book_title"):
            header += f", {meta['book_title']}"
        header += "]"

        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


def format_recipe_results(results: list[dict]) -> str:
    """Format recipe search results with rich metadata."""
    if not results:
        return "Keine Rezepturen gefunden."

    parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        text = r["text"]

        header_parts = [f"Ergebnis {i}"]
        if meta.get("book_title"):
            header_parts.append(meta["book_title"])
        if meta.get("page_number"):
            header_parts.append(f"Seite {meta['page_number']}")
        if meta.get("chunk_type"):
            header_parts.append(f"Typ: {meta['chunk_type']}")

        header = " | ".join(header_parts)
        parts.append(f"### {header}\n\n{text}")

        if meta.get("source_image_path"):
            parts.append(f"\n_Originalbild: {meta['source_image_path']}_")

    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "dickes Leder fuer Pferdesattel"
    results = search_all(query)
    print(f"Suche: {query}\n")
    print(format_context(results))
