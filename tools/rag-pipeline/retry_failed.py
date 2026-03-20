"""Retry failed files from the last indexing run."""
import time
from pathlib import Path

from config import COLLECTION_KNOWLEDGE, DOCUMENTS_DIR
from indexer import get_chroma_client, get_or_create_collection, index_document
from parser import SUPPORTED_EXTENSIONS
from status_bridge import read_status, write_status, add_error
from watcher import index_with_timeout, IndexTimeoutError, _calc_timeout

# Laengerer Timeout fuer Retry (nachts, keine Konkurrenz)
RETRY_TIMEOUT_MIN = 600  # 10 Minuten Minimum


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] RETRY: {msg}", flush=True)


def retry_failed_files():
    """Re-index all files that previously failed.

    Uses thread-based timeout from watcher.py (works in background threads,
    unlike the previous SIGALRM approach).
    """
    status = read_status()
    errors = status.get("errors", [])

    if not errors:
        log("Keine fehlerhaften Dateien zum Retry.")
        return

    failed_filenames = [e["file"] for e in errors if "file" in e]
    log(f"{len(failed_filenames)} fehlerhafte Dateien gefunden: {', '.join(failed_filenames)}")

    docs_dir = Path(DOCUMENTS_DIR)
    client = get_chroma_client()
    collection = get_or_create_collection(client, COLLECTION_KNOWLEDGE)

    # Fehlerliste leeren fuer den Retry
    write_status(errors=[])

    success = 0
    still_failed = 0

    for i, filename in enumerate(failed_filenames, 1):
        # Datei im Verzeichnis finden
        matches = list(docs_dir.rglob(filename))
        if not matches:
            log(f"  [{i}/{len(failed_filenames)}] Datei nicht gefunden: {filename}")
            add_error(filename, "Datei nicht gefunden beim Retry")
            still_failed += 1
            continue

        file_path = matches[0]
        timeout = max(RETRY_TIMEOUT_MIN, _calc_timeout(file_path))
        log(f"  [{i}/{len(failed_filenames)}] Retry: {filename} ({file_path.stat().st_size / 1024 / 1024:.1f} MB, timeout {timeout}s)...")
        write_status(is_indexing=True, current_file=filename)

        try:
            count = index_with_timeout(file_path, collection, timeout=timeout)
            log(f"  [{i}/{len(failed_filenames)}] OK: {filename} ({count} Chunks)")
            success += 1
        except IndexTimeoutError:
            log(f"  [{i}/{len(failed_filenames)}] TIMEOUT: {filename} (>{timeout}s)")
            add_error(filename, f"Retry-Timeout nach {timeout}s")
            still_failed += 1
        except Exception as e:
            log(f"  [{i}/{len(failed_filenames)}] FEHLER: {filename}: {e}")
            add_error(filename, str(e))
            still_failed += 1

        # Kurze Pause zwischen Dateien
        time.sleep(2)

    write_status(is_indexing=False, current_file=None)
    log(f"Retry fertig: {success} erfolgreich, {still_failed} weiterhin fehlerhaft")
