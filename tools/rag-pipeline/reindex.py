"""Re-index specified files: delete existing chunks from ChromaDB, then re-index."""
import time
from pathlib import Path

from config import COLLECTION_KNOWLEDGE, DOCUMENTS_DIR
from indexer import get_chroma_client, get_or_create_collection
from parser import SUPPORTED_EXTENSIONS
from status_bridge import write_status, add_error
from watcher import index_with_timeout, IndexTimeoutError, _calc_timeout


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] REINDEX: {msg}", flush=True)


def _delete_chunks_for_file(collection, filename: str) -> int:
    """Delete all chunks where metadata source matches filename.

    Uses metadata filtering instead of ID patterns -- robust across all
    chunk types (text, pdf_native, image) since all share source=filename.
    """
    try:
        result = collection.get(where={"source": filename}, include=[])
        if result and result.get("ids"):
            ids_to_delete = result["ids"]
            collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
    except Exception as e:
        log(f"  Fehler beim Loeschen von Chunks fuer {filename}: {e}")
    return 0


def reindex_files(files: list[str] | None = None, reindex_all: bool = False):
    """Re-index specified files or all files in the documents directory."""
    docs_dir = Path(DOCUMENTS_DIR)
    client = get_chroma_client()
    collection = get_or_create_collection(client, COLLECTION_KNOWLEDGE)

    if reindex_all:
        file_paths = sorted(
            p for p in docs_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
            and not p.name.endswith(".desc.txt")
        )
    else:
        filenames = files or []
        file_paths = []
        for fn in filenames:
            matches = list(docs_dir.rglob(fn))
            if matches:
                file_paths.append(matches[0])
            else:
                log(f"  Datei nicht gefunden: {fn}")
                add_error(fn, "Datei nicht gefunden beim Reindex")

    total = len(file_paths)
    log(f"Re-indexiere {total} Dateien...")
    first_file = file_paths[0].name if file_paths else None
    write_status(is_indexing=True, current_file=first_file, pending_count=total)

    success = 0
    failed = 0

    for i, file_path in enumerate(file_paths, 1):
        filename = file_path.name
        log(f"  [{i}/{total}] {filename}")
        write_status(current_file=filename, pending_count=total - i)

        # Step 1: Delete existing chunks
        deleted = _delete_chunks_for_file(collection, filename)
        log(f"  [{i}/{total}] {deleted} alte Chunks geloescht")

        # Step 2: Re-index
        timeout = max(600, _calc_timeout(file_path))
        try:
            count = index_with_timeout(file_path, collection, timeout=timeout)
            log(f"  [{i}/{total}] OK: {filename} ({count} neue Chunks)")
            success += 1
        except IndexTimeoutError:
            log(f"  [{i}/{total}] TIMEOUT: {filename}")
            add_error(filename, f"Reindex-Timeout nach {timeout}s")
            failed += 1
        except Exception as e:
            log(f"  [{i}/{total}] FEHLER: {filename}: {e}")
            add_error(filename, str(e))
            failed += 1

        time.sleep(2)  # Pace API requests

    write_status(is_indexing=False, current_file=None, pending_count=0)
    log(f"Reindex fertig: {success} OK, {failed} Fehler")
