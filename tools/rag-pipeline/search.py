"""RAG search module - queries ChromaDB collections."""
import sys
from pathlib import Path

from config import COLLECTION_KNOWLEDGE, COLLECTION_PRODUCTS, COLLECTION_RECIPES
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


def search_collection(
    query: str, collection_name: str, n_results: int = 5,
    max_distance: float | None = None, diverse: bool = False,
    max_per_source: int = 2, query_embedding: list[float] | None = None,
) -> list[dict]:
    """Search a single ChromaDB collection by text query.

    Automatically includes image results with cross-modal distance adjustment.
    Image embeddings are in the same vector space but have systematically higher
    distances to text queries (~0.55-0.65 vs ~0.30-0.40 for text-text).
    We compensate by searching images separately and scaling their distances.

    Args:
        diverse: If True, fetch 3x more results and diversify across sources,
            limiting each source to max_per_source chunks.
        max_per_source: Max chunks per source document (only with diverse=True).
        query_embedding: Pre-computed embedding vector. If None, will be generated.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client, collection_name)
    if query_embedding is None:
        query_embedding = get_embedding(query, task_type=_detect_task_type(query))

    fetch_n = n_results * 3 if diverse else max(n_results * 5, 50)

    # Search text results (exclude images to avoid them being pushed out)
    text_items = _query_collection(
        collection, query_embedding, fetch_n,
        where={"chunk_type": {"$ne": "image"}},
        max_distance=max_distance,
    )

    # Search image results separately with cross-modal distance scaling
    # Images are scaled to be comparable with text distances
    IMAGE_DISTANCE_SCALE = 0.5  # Multiply image distances to compensate cross-modal gap
    image_items = _query_collection(
        collection, query_embedding, max(3, n_results),
        where={"chunk_type": "image"},
    )
    for item in image_items:
        item["distance"] = item["distance"] * IMAGE_DISTANCE_SCALE
        item["metadata"]["_distance_scaled"] = True

    # Filename-based search: catch documents that HNSW approximate search misses
    filename_items = _filename_search(collection, query, query_embedding, n_results)

    # Merge all results, deduplicating by source+chunk_index
    seen = set()
    for item in text_items:
        key = item["source"] + "_" + str(item["metadata"].get("chunk_index", item["metadata"].get("page_start", "")))
        seen.add(key)

    # Filename matches get a distance bonus: if the source filename contains
    # query words, the document is highly likely relevant even if the vector
    # distance is moderate (e.g. table-heavy PDFs with sparse text)
    FILENAME_MATCH_SCALE = 0.5
    for item in filename_items:
        key = item["source"] + "_" + str(item["metadata"].get("chunk_index", item["metadata"].get("page_start", "")))
        if key not in seen:
            source_lower = item["source"].lower()
            query_words = [w.lower() for w in query.split() if len(w) >= 3]
            if any(w in source_lower for w in query_words):
                item["distance"] = item["distance"] * FILENAME_MATCH_SCALE
                item["metadata"]["_filename_boosted"] = True
            text_items.append(item)
            seen.add(key)

    items = text_items + image_items
    items.sort(key=lambda x: x["distance"])

    # Apply max_distance filter after scaling
    if max_distance is not None:
        items = [i for i in items if i["distance"] <= max_distance]

    if diverse:
        return _diversify_results(items, n_results, max_per_source)
    return items[:n_results]


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
    combined.sort(key=lambda x: x["distance"])

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
