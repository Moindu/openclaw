"""RAG Pipeline configuration for Kobel Gerberei."""
import os

# ChromaDB
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

# Google Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Embedding
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-2-preview")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))

# OCR-Modelle (austauschbar via .env falls Modelle deprecated werden)
OCR_MODEL_STANDARD = os.environ.get("OCR_MODEL_STANDARD", "gemini-3.1-flash-lite-preview")
OCR_MODEL_COMPLEX = os.environ.get("OCR_MODEL_COMPLEX", "gemini-3-flash-preview")
OCR_MODEL_CLASSIFIER = os.environ.get("OCR_MODEL_CLASSIFIER", "gemini-3.1-flash-lite-preview")

# Dokumente (bestehende Wissensdatenbank)
DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "/Users/kobel/.openclaw/lederberater/documents")

# Buecher-Pipeline
BOOKS_DIR = os.environ.get("BOOKS_DIR", "/data/books")

# Shopware
SHOPWARE_BASE_URL = os.environ.get("SHOPWARE_BASE_URL", "https://kobelshop.de/store-api")
SHOPWARE_ACCESS_KEY = os.environ.get("SHOPWARE_ACCESS_KEY", "")

# Collections
COLLECTION_PRODUCTS = "products"
COLLECTION_KNOWLEDGE = "knowledge"
COLLECTION_RECIPES = "gerberei_rezepturen"

# PDF-Indexierung: "native" (nur Gemini PDF-Embedding) oder "dual" (zusaetzlich Text-Chunks)
PDF_STRATEGY = os.environ.get("PDF_STRATEGY", "native")

# Interleaved Embeddings: Bild + Text kombiniert fuer reichhaltigere Rezeptbuch-Embeddings
USE_INTERLEAVED = os.environ.get("USE_INTERLEAVED", "true").lower() == "true"

# Auto-Bildbeschreibung via Gemini Flash wenn kein .desc.txt vorhanden
AUTO_DESCRIBE_IMAGES = os.environ.get("AUTO_DESCRIBE_IMAGES", "true").lower() == "true"

# Chunking (fuer bestehende knowledge-Pipeline)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))
