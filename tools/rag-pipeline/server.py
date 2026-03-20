"""HTTP API server for RAG queries, book processing, and file manager integration."""
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

from search import (
    format_context,
    format_recipe_results,
    search_all,
    search_knowledge,
    search_products,
    search_recipes,
    search_recipes_by_image,
)
from websearch import format_web_results, search_web
from status_bridge import read_status

logger = logging.getLogger(__name__)

# Cache for /indexed endpoint (used by Dateimanager)
_indexed_cache = {"data": None, "timestamp": 0}
_cache_lock = threading.Lock()
CACHE_TTL = 30

# Retry lock
_retry_running = False
_retry_lock = threading.Lock()

# Reindex lock
_reindex_running = False
_reindex_lock = threading.Lock()


def _get_indexed_cached():
    """Get list of indexed files with caching and content-type breakdown."""
    with _cache_lock:
        now = time.time()
        if _indexed_cache["data"] is not None and (now - _indexed_cache["timestamp"]) < CACHE_TTL:
            return _indexed_cache["data"]

    try:
        from config import COLLECTION_KNOWLEDGE
        from indexer import get_chroma_client, get_or_create_collection

        client = get_chroma_client()
        collection = get_or_create_collection(client, COLLECTION_KNOWLEDGE)
        result = collection.get(include=["metadatas"])

        sources = set()
        content_types = {}  # doc_type -> count of unique files
        languages = {}  # language -> count of unique files
        chunk_types = {"text": 0, "pdf_native": 0}  # chunk_type breakdown
        type_sources = {}  # doc_type -> set of source filenames
        file_details = {}  # source -> {chunks, chunk_types, doc_type, language}

        if result and result.get("metadatas"):
            for meta in result["metadatas"]:
                if not meta:
                    continue
                source = meta.get("source", "")
                if source:
                    sources.add(source)

                # Count chunk types
                ct = meta.get("chunk_type", "text")
                chunk_types[ct] = chunk_types.get(ct, 0) + 1

                # Track doc_type per unique source
                doc_type = meta.get("doc_type", "unknown")
                if source:
                    if doc_type not in type_sources:
                        type_sources[doc_type] = set()
                    type_sources[doc_type].add(source)

                # Track language per unique source
                lang = meta.get("language", "unknown")
                if source:
                    languages.setdefault(lang, set())
                    languages[lang].add(source)

                # Per-file detail tracking
                if source:
                    if source not in file_details:
                        file_details[source] = {
                            "chunks": 0,
                            "chunk_types": {},
                            "doc_type": doc_type,
                            "language": lang,
                        }
                    file_details[source]["chunks"] += 1
                    file_details[source]["chunk_types"][ct] = file_details[source]["chunk_types"].get(ct, 0) + 1

        # Convert sets to counts
        content_types = {k: len(v) for k, v in type_sources.items()}
        language_counts = {k: len(v) for k, v in languages.items()}

        filenames = sorted(set(s.split("/")[-1] for s in sources))
        data = {
            "indexed": filenames,
            "total_chunks": collection.count(),
            "content_types": content_types,
            "languages": language_counts,
            "chunk_types": chunk_types,
            "file_details": file_details,
        }

        with _cache_lock:
            _indexed_cache["data"] = data
            _indexed_cache["timestamp"] = time.time()
        return data
    except Exception as e:
        with _cache_lock:
            if _indexed_cache["data"] is not None:
                return _indexed_cache["data"]
        return {"indexed": [], "total_chunks": 0, "error": str(e)}


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class RAGHandler(BaseHTTPRequestHandler):
    """Handle RAG search, pipeline, and file manager requests."""

    def do_GET(self):
        path = self.path.rstrip("/")

        if path == "/indexed":
            self._send_json(200, _get_indexed_cached())
            return

        if path == "/status":
            self._handle_status()
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        path = self.path.rstrip("/")

        # Upload must be handled BEFORE reading body (needs raw rfile for multipart)
        if path == "/upload":
            self._handle_upload()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        # Shopware product sync
        if path == "/sync":
            self._handle_sync()
            return

        # Retry failed files
        if path == "/retry":
            self._handle_retry()
            return

        # Clear indexing errors
        if path == "/clear-errors":
            from status_bridge import clear_errors
            clear_errors()
            self._send_json(200, {"status": "errors_cleared"})
            return

        # Re-index files
        if path == "/reindex":
            data = json.loads(body)
            self._handle_reindex(data)
            return

        # Web search
        if path == "/websearch":
            data = json.loads(body)
            query = data.get("query", "")
            n_results = data.get("n_results", 5)
            language = data.get("language", "de")
            results = search_web(query, n_results, language)
            self._send_json(200, {
                "query": query,
                "results": results,
                "context": format_web_results(results),
            })
            return

        # Recipe search (text)
        if path == "/recipes":
            data = json.loads(body)
            query = data.get("query", "")
            n_results = data.get("n_results", 5)
            filters = data.get("filters")
            max_distance = data.get("max_distance")
            results = search_recipes(query, n_results, filters, max_distance=max_distance)
            self._send_json(200, {
                "query": query,
                "results": results,
                "context": format_recipe_results(results),
            })
            return

        # Recipe search (image)
        if path == "/recipes/image":
            data = json.loads(body)
            image_path = data.get("image_path", "")
            n_results = data.get("n_results", 5)
            if not image_path or not Path(image_path).exists():
                self._send_json(400, {"error": "image_path not found"})
                return
            results = search_recipes_by_image(image_path, n_results)
            self._send_json(200, {
                "image_path": image_path,
                "results": results,
                "context": format_recipe_results(results),
            })
            return

        # Process a new book (async)
        if path == "/process":
            data = json.loads(body)
            input_path = data.get("input_path", "")
            book_name = data.get("book_name", "")
            book_year = data.get("book_year", "")
            if not input_path or not book_name:
                self._send_json(400, {"error": "input_path and book_name required"})
                return
            self._handle_process(input_path, book_name, book_year)
            return

        # Default: RAG search (existing behavior)
        data = json.loads(body)
        query = data.get("query", "")
        n_results = data.get("n_results", 5)
        collection = data.get("collection", "all")
        diverse = data.get("diverse", True)  # Default: diverse search for better coverage
        max_per_source = data.get("max_per_source", 2)

        max_distance = data.get("max_distance")

        if collection == "products":
            results = search_products(query, n_results, max_distance=max_distance,
                                      diverse=diverse, max_per_source=max_per_source)
        elif collection == "knowledge":
            results = search_knowledge(query, n_results, max_distance=max_distance,
                                       diverse=diverse, max_per_source=max_per_source)
        elif collection == "recipes":
            results = search_recipes(query, n_results, max_distance=max_distance)
        else:
            results = search_all(query, n_results, max_distance=max_distance,
                                 diverse=diverse, max_per_source=max_per_source)

        unique_sources = len(set(r["source"] for r in results))
        self._send_json(200, {
            "query": query,
            "results": results,
            "context": format_context(results),
            "total_results": len(results),
            "unique_sources": unique_sources,
        })

    def _send_json(self, status: int, data: dict):
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _handle_sync(self):
        """Trigger Shopware product sync in background."""
        def run_sync():
            try:
                from shopware import index_products
                index_products()
            except Exception as e:
                logger.error("Shopware sync error: %s", e)

        threading.Thread(target=run_sync, daemon=True).start()
        self._send_json(202, {"status": "sync_started"})

    def _handle_retry(self):
        """Retry indexing failed files."""
        global _retry_running
        with _retry_lock:
            if _retry_running:
                self._send_json(409, {"status": "retry_already_running"})
                return
            _retry_running = True

        def run_retry():
            global _retry_running
            try:
                from retry_failed import retry_failed_files
                retry_failed_files()
            except Exception as e:
                logger.error("Retry error: %s", e)
            finally:
                with _retry_lock:
                    _retry_running = False

        threading.Thread(target=run_retry, daemon=True).start()
        self._send_json(202, {"status": "retry_started"})

    def _handle_reindex(self, data):
        """Re-index specified files or all files."""
        global _reindex_running
        with _reindex_lock:
            if _reindex_running:
                self._send_json(409, {"status": "reindex_already_running"})
                return
            _reindex_running = True

        files = data.get("files", [])
        reindex_all = data.get("all", False)

        def run_reindex():
            global _reindex_running
            try:
                from reindex import reindex_files
                reindex_files(files=files, reindex_all=reindex_all)
            except Exception as e:
                logger.error("Reindex error: %s", e)
            finally:
                with _reindex_lock:
                    _reindex_running = False
                # Invalidate cache so next poll gets fresh data
                with _cache_lock:
                    _indexed_cache["data"] = None
                    _indexed_cache["timestamp"] = 0

        threading.Thread(target=run_reindex, daemon=True).start()
        self._send_json(202, {
            "status": "reindex_started",
            "files": files if not reindex_all else "all",
        })

    def _handle_upload(self):
        """Accept file upload (multipart/form-data) and save to DOCUMENTS_DIR.

        Expects:
          - file: the file to upload (required)
          - description: optional text description (creates .desc.txt sidecar for images)
          - filename: optional override for the filename
        The watcher will automatically detect and index the new file.
        """
        from config import DOCUMENTS_DIR
        from parser import SUPPORTED_EXTENSIONS

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json(400, {"error": "Content-Type must be multipart/form-data"})
            return

        try:
            # Read full body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Extract boundary from Content-Type
            boundary = None
            for part in content_type.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary = part[len("boundary="):].strip().strip('"')
            if not boundary:
                self._send_json(400, {"error": "Missing boundary in Content-Type"})
                return

            # Parse multipart manually
            sep = f"--{boundary}".encode()
            parts = body.split(sep)
            fields = {}  # name -> (data, filename_or_None)

            for part in parts:
                if not part or part == b"--\r\n" or part == b"--":
                    continue
                # Split headers from body
                if b"\r\n\r\n" in part:
                    header_section, file_data = part.split(b"\r\n\r\n", 1)
                elif b"\n\n" in part:
                    header_section, file_data = part.split(b"\n\n", 1)
                else:
                    continue

                # Remove trailing \r\n from data
                if file_data.endswith(b"\r\n"):
                    file_data = file_data[:-2]

                # Parse Content-Disposition
                header_str = header_section.decode("utf-8", errors="replace")
                name = None
                filename = None
                for line in header_str.split("\n"):
                    line = line.strip()
                    if line.lower().startswith("content-disposition:"):
                        # Extract name
                        m = re.search(r'name="([^"]*)"', line)
                        if m:
                            name = m.group(1)
                        m = re.search(r'filename="([^"]*)"', line)
                        if m:
                            filename = m.group(1)

                if name:
                    fields[name] = (file_data, filename)

            # Validate file field
            if "file" not in fields or not fields["file"][0]:
                self._send_json(400, {"error": "No file uploaded"})
                return

            file_data, original_filename = fields["file"]

            # Determine filename
            override_name = ""
            if "filename" in fields:
                override_name = fields["filename"][0].decode("utf-8", errors="replace").strip()
            final_name = override_name or original_filename or "uploaded_file"
            # Sanitize filename: keep only safe characters
            safe_name = re.sub(r'[^\w\s\-.]', '_', final_name).strip()
            if not safe_name:
                self._send_json(400, {"error": "Invalid filename"})
                return

            # Check extension
            ext = Path(safe_name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                self._send_json(400, {
                    "error": f"Unsupported file type: {ext}",
                    "supported": sorted(SUPPORTED_EXTENSIONS),
                })
                return

            # Save file
            docs_dir = Path(DOCUMENTS_DIR)
            dest = docs_dir / safe_name
            counter = 1
            stem = dest.stem
            while dest.exists():
                dest = docs_dir / f"{stem}_{counter}{ext}"
                counter += 1

            dest.write_bytes(file_data)
            logger.info("Uploaded file: %s (%d bytes)", dest.name, len(file_data))

            # Optional description sidecar for images and videos
            description = ""
            if "description" in fields:
                description = fields["description"][0].decode("utf-8", errors="replace").strip()
            image_exts = {".jpg", ".jpeg", ".png", ".webp"}
            video_exts = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
            if description and ext in (image_exts | video_exts):
                desc_path = docs_dir / f"{dest.name}.desc.txt"
                desc_path.write_text(description, encoding="utf-8")
                logger.info("Created description sidecar: %s", desc_path.name)

            # Invalidate indexed cache
            with _cache_lock:
                _indexed_cache["data"] = None
                _indexed_cache["timestamp"] = 0

            self._send_json(201, {
                "status": "uploaded",
                "filename": dest.name,
                "size": len(file_data),
                "description": description or None,
                "message": f"Datei '{dest.name}' gespeichert. Wird automatisch indexiert.",
            })

        except Exception as e:
            logger.error("Upload error: %s", e, exc_info=True)
            self._send_json(500, {"error": f"Upload fehlgeschlagen: {str(e)}"})

    def _handle_process(self, input_path: str, book_name: str, book_year: str):
        """Process a book in the background."""
        def run_process():
            try:
                from chunker import chunk_book
                from config import COLLECTION_RECIPES
                from indexer import get_chroma_client, get_or_create_collection, index_recipe_chunks
                from input_handler import process_input

                path = Path(input_path)
                if not path.exists():
                    logger.error("Input path not found: %s", input_path)
                    return

                logger.info("Processing book: %s from %s", book_name, input_path)

                # Step 1: Extract text (OCR or direct)
                page_results = process_input(path, book_name)
                logger.info("Extracted %d pages", len(page_results))

                # Step 2: Semantic chunking
                chunks = chunk_book(page_results, book_title=book_name, book_year=book_year)
                logger.info("Created %d chunks", len(chunks))

                # Step 3: Embed and store
                client = get_chroma_client()
                collection = get_or_create_collection(client, COLLECTION_RECIPES)
                stats = index_recipe_chunks(chunks, collection)
                logger.info(
                    "Indexed: %d text chunks, %d image embeddings, %d errors",
                    stats["text_chunks"],
                    stats["image_embeddings"],
                    stats["errors"],
                )
            except Exception as e:
                logger.error("Book processing failed: %s", e, exc_info=True)

        threading.Thread(target=run_process, daemon=True).start()
        self._send_json(202, {
            "status": "processing_started",
            "book_name": book_name,
            "input_path": input_path,
        })

    def _handle_status(self):
        """Return pipeline status (used by Dateimanager + pipeline CLI).

        Returns immediately from the status file without blocking on
        ChromaDB queries. The /indexed endpoint handles slow DB queries
        separately (polled less frequently by the frontend).
        """
        # Read live status from watcher via shared file — instant, no DB
        status = read_status()

        # Use cached indexed data ONLY if already in cache (non-blocking)
        with _cache_lock:
            cached = _indexed_cache["data"]
            cache_age = time.time() - _indexed_cache["timestamp"]
        if cached is not None and cache_age < CACHE_TTL * 10:  # Use stale cache up to 5 min
            status["indexed_count"] = len(cached.get("indexed", []))
            status["total_chunks"] = cached.get("total_chunks", 0)
        # else: keep indexed_count/total_chunks from status file (watcher tracks these)

        # Recipe info — collection.count() is fast (metadata only)
        from config import BOOKS_DIR, COLLECTION_RECIPES
        try:
            from indexer import get_chroma_client
            client = get_chroma_client()
            try:
                recipe_col = client.get_collection(COLLECTION_RECIPES)
                status["recipe_chunks"] = recipe_col.count()
            except Exception:
                status["recipe_chunks"] = 0
        except Exception:
            pass

        books_dir = Path(BOOKS_DIR)
        originals = books_dir / "originals"
        if originals.exists():
            status["books"] = [d.name for d in originals.iterdir() if d.is_dir()]
        else:
            status["books"] = []

        self._send_json(200, status)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def run_server(host: str = "0.0.0.0", port: int = 8100):
    """Start the RAG API server."""
    server = ThreadingHTTPServer((host, port), RAGHandler)
    print(f"RAG API server listening on http://{host}:{port}", flush=True)
    print("Endpoints:", flush=True)
    print("  GET  /indexed     - Indexed files list (Dateimanager)", flush=True)
    print("  GET  /status      - Pipeline + indexing status", flush=True)
    print('  POST /            - RAG search {"query": "...", "collection": "all|products|knowledge|recipes"}', flush=True)
    print('  POST /recipes     - Recipe search {"query": "...", "filters": {...}}', flush=True)
    print('  POST /recipes/image - Image search {"image_path": "..."}', flush=True)
    print('  POST /process     - Process book {"input_path": "...", "book_name": "..."}', flush=True)
    print('  POST /websearch   - Web search {"query": "..."}', flush=True)
    print("  POST /sync        - Shopware product sync", flush=True)
    print("  POST /retry       - Retry failed files", flush=True)
    print('  POST /reindex     - Re-index files {"files": [...]} or {"all": true}', flush=True)
    print("  POST /upload      - Upload file to documents (multipart/form-data)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    run_server()
