"""Google Gemini Embedding module using gemini-embedding-2-preview.

Task types (from Google docs):
  RETRIEVAL_DOCUMENT  - Indexing documents (articles, books, web pages)
  RETRIEVAL_QUERY     - General search queries (pair with RETRIEVAL_DOCUMENT)
  QUESTION_ANSWERING  - Chatbot queries finding answer docs (pair with RETRIEVAL_DOCUMENT)
  SEMANTIC_SIMILARITY - Comparing text similarity
  CLASSIFICATION      - Categorizing texts by labels
  CLUSTERING          - Grouping texts by similarity
  CODE_RETRIEVAL_QUERY - Finding code from natural language
  FACT_VERIFICATION   - Retrieving evidence for claims

Supported modalities (gemini-embedding-2-preview):
  Text:  up to 8192 tokens
  Image: max 6 per request (PNG, JPEG)
  Audio: up to 80s (MP3, WAV)
  Video: up to 128s (MP4, MOV)
  PDF:   max 6 pages

Output dimensionality: 128-3072 (recommended: 768, 1536, 3072)
Default output: 3072 dims (we use 768 via MRL truncation)
"""
import logging
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)

MAX_RETRIES = 8
RETRY_BASE_DELAY = 10  # seconds

_client = None


def _get_client() -> genai.Client:
    """Lazy singleton for the Gemini API client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize an embedding vector to unit length.

    Improves cosine similarity search quality by ensuring all vectors
    lie on the unit hypersphere (adapted from Arnie936/multimodal-rag).
    """
    a = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm > 0:
        a = a / norm
    return a.tolist()


def _normalize_batch(vecs: list[list[float]]) -> list[list[float]]:
    """L2-normalize a batch of embedding vectors."""
    return [_normalize(v) for v in vecs]


def _retry_embed(func, *args, **kwargs):
    """Retry wrapper with exponential backoff for rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("Rate limit hit, retrying in %ds (attempt %d/%d)", delay, attempt + 1, MAX_RETRIES)
                time.sleep(delay)
            else:
                raise
    raise Exception(f"Failed after {MAX_RETRIES} retries (RESOURCE_EXHAUSTED / 429 rate limit)")


def get_embedding(text: str, task_type: str = "QUESTION_ANSWERING") -> list[float]:
    """Generate embedding for a single text using Gemini.

    Default task_type is QUESTION_ANSWERING (optimized for chatbot queries
    finding answer-containing documents - pair with RETRIEVAL_DOCUMENT).

    For general keyword search, use task_type="RETRIEVAL_QUERY".
    """
    client = _get_client()

    def _embed():
        return client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

    result = _retry_embed(_embed)
    return _normalize(result.embeddings[0].values)


def get_embeddings_batch(
    texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:
    """Generate embeddings for a batch of texts using Gemini.

    Default task_type is RETRIEVAL_DOCUMENT (optimized for indexing
    searchable documents like articles, books, web pages).
    """
    if not texts:
        return []

    client = _get_client()

    all_embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        def _embed_batch(b=batch):
            return client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=b,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIMENSIONS,
                ),
            )

        result = _retry_embed(_embed_batch)
        all_embeddings.extend(_normalize(e.values) for e in result.embeddings)

        # Pace requests
        if i + batch_size < len(texts):
            time.sleep(2)

    return all_embeddings


def get_image_embedding(
    image_path: str | Path, task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[float]:
    """Generate embedding for an image using Gemini.

    The image is embedded in the same vector space as text,
    enabling cross-modal search (text query finds relevant images).
    Supported formats: PNG, JPEG.
    """
    client = _get_client()
    image_path = Path(image_path)

    image_data = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(suffix, "image/jpeg")

    image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)

    def _embed_img():
        return client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[image_part],
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

    result = _retry_embed(_embed_img)
    return _normalize(result.embeddings[0].values)


def get_interleaved_embedding(
    text: str, image_path: str | Path | None = None,
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[float]:
    """Generate embedding from combined text + image content.

    Gemini Embedding 2 processes both modalities together in a single request,
    producing richer embeddings that capture cross-modal relationships
    (e.g., visual layout + textual content of a recipe book page).
    """
    client = _get_client()
    parts = []

    if image_path:
        image_path = Path(image_path)
        if image_path.exists():
            image_data = image_path.read_bytes()
            suffix = image_path.suffix.lower()
            mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }
            mime_type = mime_types.get(suffix, "image/jpeg")
            parts.append(types.Part.from_bytes(data=image_data, mime_type=mime_type))

    parts.append(text)

    def _embed():
        return client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=parts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

    result = _retry_embed(_embed)
    return _normalize(result.embeddings[0].values)


def get_pdf_embedding(
    pdf_path: str | Path, task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[float]:
    """Generate embedding for a PDF file using Gemini (max 6 pages).

    Gemini Embedding 2 can embed PDFs directly in the same vector space
    as text and images. For PDFs > 6 pages, use get_pdf_embedding_bytes
    with chunked sub-PDFs.
    """
    client = _get_client()
    pdf_path = Path(pdf_path)
    pdf_data = pdf_path.read_bytes()
    return get_pdf_embedding_bytes(pdf_data, task_type=task_type)


def get_pdf_embedding_bytes(
    pdf_bytes: bytes, task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[float]:
    """Generate embedding for PDF bytes using Gemini (max 6 pages).

    Accepts raw PDF bytes - useful for sub-PDF chunks split from larger PDFs.
    Gemini Embedding 2 natively processes the PDF layout, tables, formulas
    and images, producing richer embeddings than text-only extraction.
    """
    client = _get_client()
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

    def _embed_pdf():
        return client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[pdf_part],
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

    result = _retry_embed(_embed_pdf)
    return _normalize(result.embeddings[0].values)
