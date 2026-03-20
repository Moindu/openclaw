"""ChromaDB indexer for documents, products, and recipe books."""
import hashlib
import logging
from pathlib import Path

import chromadb

from config import (
    AUTO_DESCRIBE_IMAGES,
    CHROMA_HOST,
    CHROMA_PORT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_KNOWLEDGE,
    COLLECTION_RECIPES,
    PDF_STRATEGY,
    USE_INTERLEAVED,
)
from embeddings import (
    get_embedding,
    get_embeddings_batch,
    get_image_embedding,
    get_interleaved_embedding,
    get_pdf_embedding_bytes,
)
from parser import IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS, chunk_text, is_image, is_video, parse_document

logger = logging.getLogger(__name__)


def get_chroma_client() -> chromadb.HttpClient:
    """Create ChromaDB HTTP client."""
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def get_or_create_collection(client: chromadb.HttpClient, name: str):
    """Get or create a ChromaDB collection."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# --- Existing knowledge document indexing (unchanged logic) ---


def _detect_doc_type(path: Path) -> str:
    """Detect document type from directory structure."""
    parts = [p.lower() for p in path.parts]
    if any("rezeptur" in p for p in parts):
        return "rezeptur"
    if any("sonderdruck" in p for p in parts):
        return "sonderdruck"
    if any("e-book" in p or "ebook" in p for p in parts):
        return "fachbuch"
    return "notiz"


def _detect_language(path: Path) -> str:
    """Detect language from directory structure or filename."""
    name = path.stem.lower()
    parts = [p.lower() for p in path.parts]
    de_indicators = ["einfluss", "probleme", "gerbung", "saeuren", "salzen", "pflanzlich"]
    if any(ind in name for ind in de_indicators) or any("sonderdruck" in p for p in parts):
        return "de"
    return "en"


def _load_description(path: Path) -> str:
    """Load optional description from sidecar .desc.txt file.

    If a file like 'photo.jpg.desc.txt' exists next to 'photo.jpg',
    its content is used as searchable document text in ChromaDB.
    This dramatically improves search quality for images since the
    visual embedding alone can't distinguish domain-specific details
    (e.g., Geschirrleder vs Blankleder).
    """
    desc_path = path.parent / f"{path.name}.desc.txt"
    if desc_path.exists():
        try:
            text = desc_path.read_text(encoding="utf-8").strip()
            if text:
                logger.info("Found description for %s: %s", path.name, text[:80])
                return text
        except Exception as e:
            logger.warning("Could not read description file %s: %s", desc_path, e)
    return ""


def _get_image_part(path: Path):
    """Create a Gemini Part from an image file."""
    from google.genai import types as genai_types
    suffix = path.suffix.lower()
    mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_types.get(suffix, "image/jpeg")
    return genai_types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def _auto_describe_image(path: Path) -> str:
    """Generate a brief description for an image using Gemini Flash-Lite.

    Produces a 1-2 sentence German description focused on leather/tanning content.
    Used as fallback when no .desc.txt sidecar file exists.
    """
    from google import genai
    from google.genai import types as genai_types
    from config import GOOGLE_API_KEY, OCR_MODEL_STANDARD

    client = genai.Client(api_key=GOOGLE_API_KEY)
    image_part = _get_image_part(path)

    result = client.models.generate_content(
        model=OCR_MODEL_STANDARD,
        contents=[
            image_part,
            "Beschreibe dieses Bild in 1-2 Saetzen auf Deutsch. "
            "Fokus auf Leder, Gerberei, Werkzeuge oder Materialien falls sichtbar.",
        ],
        config=genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return result.text.strip() if result.text else ""


def _ocr_image(path: Path) -> str:
    """Extract full text from an image (book page, document scan) via Gemini OCR.

    Returns the complete extracted text, not just a summary.
    Used for photos of book pages, documents, or any text-heavy images.
    """
    from google import genai
    from google.genai import types as genai_types
    from config import GOOGLE_API_KEY, OCR_MODEL_COMPLEX

    client = genai.Client(api_key=GOOGLE_API_KEY)
    image_part = _get_image_part(path)

    result = client.models.generate_content(
        model=OCR_MODEL_COMPLEX,
        contents=[
            image_part,
            "Extrahiere den VOLLSTAENDIGEN Text aus diesem Bild. "
            "Gib den Text exakt so wieder wie er auf dem Bild steht, "
            "inklusive Ueberschriften, Absaetze und Aufzaehlungen. "
            "Korrigiere offensichtliche OCR-Fehler. "
            "Gib NUR den extrahierten Text zurueck, keine Beschreibung des Bildes.",
        ],
        config=genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return result.text.strip() if result.text else ""


def _looks_like_document_scan(path: Path, description: str) -> bool:
    """Heuristic: does this image look like a scanned document/book page?

    Checks filename and description for indicators of text-heavy content.
    """
    name_lower = path.stem.lower()
    desc_lower = description.lower() if description else ""
    indicators = [
        "seite", "page", "kapitel", "chapter", "buchseite", "scan",
        "rezeptur", "tabelle", "dokument", "document", "grundlagen",
        "text", "buch", "abschnitt", "inhaltsverzeichnis", "gerbung",
        "verfahren", "protokoll", "anleitung",
    ]
    return any(ind in name_lower or ind in desc_lower for ind in indicators)


def index_image(path: Path, collection) -> int:
    """Embed and store a single image file. Returns chunk count.

    Uses interleaved multimodal embedding (image + text combined) when text
    context is available (description or OCR), producing richer embeddings
    that capture both visual and textual information in one vector.

    For document scans, also performs full OCR and stores extracted text as
    additional searchable text chunks.
    """
    file_hash = hashlib.md5(path.read_bytes()).hexdigest()[:8]
    img_id = f"img_{path.stem}_{file_hash}"

    # Step 1: Get description from sidecar file or auto-generate
    description = _load_description(path)
    if not description and AUTO_DESCRIBE_IMAGES:
        try:
            description = _auto_describe_image(path)
            if description:
                logger.info("Auto-described %s: %s", path.name, description[:80])
        except Exception as e:
            logger.warning("Auto-describe failed for %s: %s", path.name, e)

    # Step 2: For document scans, extract full OCR text BEFORE embedding
    is_scan = _looks_like_document_scan(path, description)
    ocr_text = ""
    if is_scan:
        try:
            ocr_text = _ocr_image(path)
            if ocr_text and len(ocr_text) > 50:
                logger.info("OCR extracted %d chars from %s", len(ocr_text), path.name)
            else:
                ocr_text = ""
        except Exception as e:
            logger.warning("OCR failed for %s: %s", path.name, e)

    # Step 3: Build interleaved embedding (image + best available text)
    # Priority: OCR text (richest) > description > filename only
    embed_text = ocr_text or description or ""
    if embed_text:
        # Interleaved: image + text in one vector — captures cross-modal relationships
        embedding = get_interleaved_embedding(embed_text[:2000], path)
        logger.info("Interleaved embedding for %s (%d chars text + image)", path.name, len(embed_text[:2000]))
    else:
        # Fallback: image-only embedding when no text available
        embedding = get_image_embedding(path)

    if description:
        document_text = f"[Bild: {path.name}] {description}"
    else:
        document_text = f"[Bild: {path.name}]"

    metadata = {
        "source": path.name,
        "doc_type": "buchseite" if is_scan else "bild",
        "language": "de" if (ocr_text or is_scan) else "unknown",
        "chunk_type": "image",
        "has_description": bool(description),
        "has_ocr": bool(ocr_text),
        "embedding_type": "interleaved" if embed_text else "image_only",
    }
    if description:
        metadata["description"] = description[:500]

    collection.upsert(
        ids=[img_id],
        documents=[document_text],
        embeddings=[embedding],
        metadatas=[metadata],
    )
    chunk_count = 1

    # Step 4: For document scans with OCR, also store text as separate searchable chunks
    if ocr_text:
        chunks = chunk_text(ocr_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if chunks:
            embeddings = get_embeddings_batch(chunks)
            ids = [f"{path.stem}_{file_hash}_ocr_chunk{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": path.name,
                    "chunk_type": "ocr_text",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "doc_type": "buchseite",
                    "language": "de",
                }
                for i in range(len(chunks))
            ]
            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            chunk_count += len(chunks)
            logger.info("OCR indexed %s: %d text chunks", path.name, len(chunks))

    return chunk_count


def _transcribe_video(path: Path) -> str:
    """Extract full transcript/description from a video using Gemini.

    Gemini can natively process video files and extract:
    - Spoken text (transcription)
    - Visible text (OCR from frames)
    - Content description
    """
    from google import genai
    from google.genai import types as genai_types
    from config import GOOGLE_API_KEY, OCR_MODEL_COMPLEX

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Read video bytes and determine MIME type
    video_data = path.read_bytes()
    mime_types = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }
    mime_type = mime_types.get(path.suffix.lower(), "video/mp4")
    video_part = genai_types.Part.from_bytes(data=video_data, mime_type=mime_type)

    result = client.models.generate_content(
        model=OCR_MODEL_COMPLEX,
        contents=[
            video_part,
            "Analysiere dieses Video vollstaendig. Extrahiere:\n"
            "1. Allen gesprochenen Text (Transkription)\n"
            "2. Allen sichtbaren Text auf Schildern, Dokumenten, Bildschirmen\n"
            "3. Eine detaillierte Beschreibung was im Video passiert\n\n"
            "Strukturiere die Ausgabe so:\n"
            "## Beschreibung\n[Was im Video zu sehen ist]\n\n"
            "## Transkription\n[Gesprochener Text, falls vorhanden]\n\n"
            "## Sichtbarer Text\n[Text der im Video zu lesen ist, falls vorhanden]\n\n"
            "Gib alles auf Deutsch aus. Wenn ein Abschnitt leer ist, schreibe 'Nicht vorhanden'.",
        ],
        config=genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return result.text.strip() if result.text else ""


def index_video(path: Path, collection) -> int:
    """Index a video file by extracting transcript and description via Gemini.

    Creates text chunks from the video's content (speech, visible text, description)
    for searchable retrieval.
    """
    file_hash = hashlib.md5(path.read_bytes()[:65536]).hexdigest()[:8]  # Hash first 64KB (videos are large)
    video_id = f"video_{path.stem}_{file_hash}"

    # Check file size — Gemini has limits (~2GB, but practical limit ~100MB for API)
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 100:
        logger.warning("Video %s is %.0f MB — may be too large for Gemini API", path.name, file_size_mb)

    # Extract content from video
    logger.info("Transcribing video %s (%.1f MB)...", path.name, file_size_mb)
    transcript = _transcribe_video(path)

    if not transcript or len(transcript) < 20:
        logger.warning("No meaningful content extracted from video %s", path.name)
        # Store at least a reference entry
        embedding = get_embedding(f"Video: {path.name}")
        collection.upsert(
            ids=[video_id],
            documents=[f"[Video: {path.name}] Kein Text extrahiert"],
            embeddings=[embedding],
            metadatas={
                "source": path.name,
                "doc_type": "video",
                "chunk_type": "video_ref",
                "language": "de",
            },
        )
        return 1

    logger.info("Video transcript: %d chars from %s", len(transcript), path.name)

    # Store summary chunk with video reference
    summary = transcript[:2000]
    summary_embedding = get_embedding(summary, task_type="RETRIEVAL_DOCUMENT")
    collection.upsert(
        ids=[video_id],
        documents=[f"[Video: {path.name}]\n\n{summary}"],
        embeddings=[summary_embedding],
        metadatas={
            "source": path.name,
            "doc_type": "video",
            "chunk_type": "video_summary",
            "language": "de",
            "transcript_length": len(transcript),
        },
    )
    chunk_count = 1

    # Chunk full transcript for detailed search
    chunks = chunk_text(transcript, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if chunks:
        embeddings = get_embeddings_batch(chunks)
        ids = [f"{path.stem}_{file_hash}_video_chunk{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": path.name,
                "chunk_type": "video_transcript",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "doc_type": "video",
                "language": "de",
            }
            for i in range(len(chunks))
        ]
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        chunk_count += len(chunks)
        logger.info("Video indexed %s: %d text chunks", path.name, len(chunks))

    return chunk_count


def index_document(path: Path, collection) -> int:
    """Parse, chunk, embed and store a single document. Returns chunk count."""
    if is_image(path):
        return index_image(path, collection)

    if is_video(path):
        return index_video(path, collection)

    # PDFs: use native Gemini PDF embedding (understands layout, tables, images)
    # unless PDF_STRATEGY is set to "dual" for legacy text-chunk + native behavior
    if path.suffix.lower() == ".pdf" and PDF_STRATEGY == "native":
        file_hash = hashlib.md5(path.read_bytes()).hexdigest()[:8]
        return _index_pdf_native(path, collection, file_hash)

    text = parse_document(path)
    if not text.strip():
        return 0

    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        return 0

    embeddings = get_embeddings_batch(chunks)
    file_hash = hashlib.md5(path.read_bytes()).hexdigest()[:8]

    doc_type = _detect_doc_type(path)
    language = _detect_language(path)
    ids = [f"{path.stem}_{file_hash}_chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": path.name,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "doc_type": doc_type,
            "language": language,
            "chunk_type": "text",
        }
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Legacy dual approach: also embed PDF pages natively (only with PDF_STRATEGY="dual")
    if path.suffix.lower() == ".pdf" and PDF_STRATEGY == "dual":
        try:
            pdf_chunks = _index_pdf_native(path, collection, file_hash)
            logger.info("PDF native: %s -> %d page-group embeddings", path.name, pdf_chunks)
        except Exception as e:
            logger.warning("PDF native embedding failed for %s: %s", path.name, e)

    return len(chunks)


def _index_pdf_native(path: Path, collection, file_hash: str) -> int:
    """Index a PDF using native Gemini PDF embedding (5-page sub-PDFs).

    Complements text-based indexing: Gemini Embedding 2 'sees' the actual
    PDF layout (tables, formulas, images) and produces richer embeddings.
    Sub-PDFs are limited to 5 pages (Gemini max is 6).
    """
    import time

    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        return 0

    chunk_size = 5  # Gemini limit: 6 pages, use 5 for safety
    doc_type = _detect_doc_type(path)
    language = _detect_language(path)
    count = 0

    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)

        # Extract sub-PDF bytes
        sub_doc = fitz.open()
        sub_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
        pdf_bytes = sub_doc.tobytes()

        # Also extract text for the document field (for full-text search)
        text_parts = []
        for page in sub_doc:
            text = page.get_text()
            if text.strip():
                text_parts.append(text.strip())
        sub_doc.close()

        display_text = "\n\n".join(text_parts) if text_parts else f"[PDF Seiten {start+1}-{end}]"

        try:
            vec = get_pdf_embedding_bytes(pdf_bytes, task_type="RETRIEVAL_DOCUMENT")
        except Exception as e:
            logger.warning("PDF native embed failed for %s pages %d-%d: %s", path.name, start+1, end, e)
            continue

        chunk_id = f"{path.stem}_{file_hash}_pdf_p{start+1}-{end}"
        metadata = {
            "source": path.name,
            "chunk_type": "pdf_native",
            "page_start": start + 1,
            "page_end": end,
            "total_pages": total_pages,
            "doc_type": doc_type,
            "language": language,
            "chunk_type": "text",
        }

        collection.upsert(
            ids=[chunk_id],
            documents=[display_text],
            embeddings=[vec],
            metadatas=[metadata],
        )
        count += 1

        # Pace API requests
        if start + chunk_size < total_pages:
            time.sleep(2)

    doc.close()
    return count


def index_documents_dir(docs_dir: Path, collection) -> dict:
    """Index all supported documents in a directory."""
    stats = {"files": 0, "chunks": 0, "skipped": 0}

    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                count = index_document(path, collection)
                stats["files"] += 1
                stats["chunks"] += count
                print(f"  Indexed: {path.name} ({count} chunks)")
            except Exception as e:
                stats["skipped"] += 1
                print(f"  Skipped: {path.name} ({e})")
        elif path.is_file():
            stats["skipped"] += 1

    return stats


# --- Recipe book indexing (new) ---


def index_recipe_chunks(chunks: list, collection) -> dict:
    """Index semantic chunks from a recipe book into ChromaDB.

    Args:
        chunks: List of Chunk objects from chunker.py.
        collection: ChromaDB collection for recipes.

    Returns:
        Stats dict with counts.
    """
    if not chunks:
        return {"text_chunks": 0, "image_embeddings": 0, "errors": 0}

    stats = {"text_chunks": 0, "image_embeddings": 0, "errors": 0}

    # Generate embeddings: interleaved (image+text) for chunks with images, batch for text-only
    if USE_INTERLEAVED:
        embeddings = []
        batch_indices = []  # indices of chunks without images (for batch processing)
        batch_texts = []

        for i, chunk in enumerate(chunks):
            img_path = chunk.source_image_path
            if img_path and Path(img_path).exists():
                try:
                    emb = get_interleaved_embedding(chunk.text, img_path)
                    embeddings.append((i, emb))
                except Exception as e:
                    logger.warning("Interleaved embedding failed for chunk %s, falling back to text: %s", chunk.chunk_id, e)
                    batch_indices.append(i)
                    batch_texts.append(chunk.text)
            else:
                batch_indices.append(i)
                batch_texts.append(chunk.text)

        # Batch-embed text-only chunks
        if batch_texts:
            try:
                batch_embs = get_embeddings_batch(batch_texts, task_type="RETRIEVAL_DOCUMENT")
                for idx, emb in zip(batch_indices, batch_embs):
                    embeddings.append((idx, emb))
            except Exception as e:
                logger.error("Batch embedding failed: %s", e)
                return {"text_chunks": 0, "image_embeddings": 0, "errors": len(chunks)}

        # Sort by original index and extract just the vectors
        embeddings.sort(key=lambda x: x[0])
        embeddings = [emb for _, emb in embeddings]
    else:
        # Legacy: batch all as text-only
        texts = [c.text for c in chunks]
        try:
            embeddings = get_embeddings_batch(texts, task_type="RETRIEVAL_DOCUMENT")
        except Exception as e:
            logger.error("Batch embedding failed: %s", e)
            return {"text_chunks": 0, "image_embeddings": 0, "errors": len(chunks)}

    ids = [c.chunk_id for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    stats["text_chunks"] = len(chunks)

    # Image embeddings for pages that have source images
    seen_images = set()
    for chunk in chunks:
        image_path = chunk.source_image_path
        if not image_path or image_path in seen_images:
            continue
        seen_images.add(image_path)

        img_path = Path(image_path)
        if not img_path.exists():
            continue

        try:
            img_embedding = get_image_embedding(img_path)
            img_id = f"img_{img_path.stem}_{chunk.book_title}"
            collection.upsert(
                ids=[img_id],
                documents=[f"[Bild: {img_path.name}] {chunk.book_title} Seite {chunk.page_number}"],
                embeddings=[img_embedding],
                metadatas=[{
                    "chunk_type": "image",
                    "book_title": chunk.book_title,
                    "page_number": chunk.page_number,
                    "source_image_path": str(img_path),
                    "language": chunk.language,
                }],
            )
            stats["image_embeddings"] += 1
        except Exception as e:
            logger.warning("Image embedding failed for %s: %s", img_path.name, e)
            stats["errors"] += 1

    return stats


# --- Entry points ---


def run_knowledge_index():
    """Main entry point: index all documents in the knowledge directory."""
    from config import DOCUMENTS_DIR

    docs_dir = Path(DOCUMENTS_DIR)
    if not docs_dir.exists():
        print(f"Documents directory not found: {docs_dir}")
        return

    client = get_chroma_client()
    collection = get_or_create_collection(client, COLLECTION_KNOWLEDGE)

    print(f"Indexing documents from {docs_dir}...")
    stats = index_documents_dir(docs_dir, collection)
    print(f"Done: {stats['files']} files, {stats['chunks']} chunks, {stats['skipped']} skipped")


if __name__ == "__main__":
    run_knowledge_index()
