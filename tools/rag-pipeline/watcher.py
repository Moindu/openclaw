"""File watcher for automatic document re-indexing + scheduled Shopware sync.

Robustness features:
- Thread-based timeout (works in both main and background threads, unlike SIGALRM)
- Per-document retry with backoff (1 retry before marking as failed)
- Heartbeat during rate-limit pauses (prevents stale status detection)
- Rate-limit detection from both direct errors and exhausted retry exceptions
- Safe status writes (I/O errors never abort the indexing loop)
- Auto-recovery from stuck states
"""
import os
import concurrent.futures
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from config import COLLECTION_KNOWLEDGE, DOCUMENTS_DIR
from indexer import get_chroma_client, get_or_create_collection, index_document
from parser import SUPPORTED_EXTENSIONS
from status_bridge import write_status, add_error

SYNC_INTERVAL = int(os.environ.get("SHOPWARE_SYNC_INTERVAL", str(6 * 3600)))
POLL_INTERVAL = int(os.environ.get("WATCHER_POLL_INTERVAL", "10"))
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_INDEX_FILE_SIZE_MB", "250"))
FILE_TIMEOUT_BASE = int(os.environ.get("INDEX_FILE_TIMEOUT", "300"))  # 5 min base timeout
FILE_TIMEOUT_PER_MB = int(os.environ.get("INDEX_TIMEOUT_PER_MB", "10"))  # +10s per MB
MAX_RETRIES_PER_DOC = 2  # Retry each document once before giving up
RETRY_DOC_DELAY = 30  # seconds between per-document retries


class IndexTimeoutError(Exception):
    pass


def _calc_timeout(file_path: Path) -> int:
    """Calculate timeout based on file size: base + 10s per MB."""
    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
    except OSError:
        size_mb = 0
    return FILE_TIMEOUT_BASE + int(size_mb * FILE_TIMEOUT_PER_MB)


def index_with_timeout(file_path, collection, timeout=None):
    """Index a single document with a thread-based timeout.

    Uses concurrent.futures.ThreadPoolExecutor instead of SIGALRM,
    so this works from any thread (main or background/watchdog).
    Writes heartbeat status every 30s to prevent stale detection.
    """
    if timeout is None:
        timeout = _calc_timeout(file_path)

    # Heartbeat thread keeps status fresh during long indexing
    heartbeat_stop = threading.Event()

    def _heartbeat():
        start = time.time()
        while not heartbeat_stop.wait(30):
            elapsed = int(time.time() - start)
            elapsed_min = elapsed // 60
            _safe_write_status(
                is_indexing=True,
                current_file=f"{file_path.name} ({elapsed_min}:{elapsed % 60:02d} vergangen)",
            )

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(index_document, file_path, collection)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise IndexTimeoutError(f"Timeout nach {timeout}s fuer {file_path.name}")
    finally:
        heartbeat_stop.set()
        hb_thread.join(timeout=5)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _safe_write_status(**kwargs):
    """Write status, catching I/O errors so they never abort indexing."""
    try:
        write_status(**kwargs)
    except Exception as e:
        log(f"  WARNUNG: Status-Schreiben fehlgeschlagen: {e}")


def _safe_add_error(filename, error):
    """Add error, catching I/O errors so they never abort indexing."""
    try:
        add_error(filename, error)
    except Exception as e:
        log(f"  WARNUNG: Fehler-Log fehlgeschlagen: {e}")


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is related to API rate limiting.

    Checks the exception message, its __cause__, and __context__
    to catch rate limits even when wrapped by retry exhaustion.
    """
    for ex in [exc, getattr(exc, '__cause__', None), getattr(exc, '__context__', None)]:
        if ex is not None:
            err = str(ex)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate" in err.lower():
                return True
    # Also detect retry exhaustion (which is always caused by rate limits in _retry_embed)
    if "Failed after" in str(exc) and "retries" in str(exc):
        return True
    return False


def _sleep_with_heartbeat(duration_seconds: int, status_msg: str = None):
    """Sleep for a duration, writing heartbeat status every 30 seconds.

    Prevents the status file from going stale (300s threshold) during
    long rate-limit pauses.
    """
    elapsed = 0
    interval = 30
    while elapsed < duration_seconds:
        chunk = min(interval, duration_seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        remaining_min = (duration_seconds - elapsed) // 60
        if status_msg:
            _safe_write_status(
                current_file=f"{status_msg} (noch {remaining_min} Min)",
            )


class DocumentHandler(FileSystemEventHandler):
    def __init__(self, collection):
        self.collection = collection
        self._debounce: dict[str, float] = {}

    def _should_process(self, path: str) -> bool:
        p = Path(path)
        # Skip .desc.txt sidecar files (they are metadata, not documents)
        if p.name.endswith(".desc.txt"):
            return False
        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return False
        now = time.time()
        if path in self._debounce and now - self._debounce[path] < 5:
            return False
        self._debounce[path] = now
        return True

    def _is_description_file(self, path: str) -> bool:
        """Check if a file is a .desc.txt sidecar for an image."""
        return Path(path).name.endswith(".desc.txt")

    def _get_image_for_description(self, desc_path: str) -> Path | None:
        """Find the image file that a .desc.txt sidecar belongs to."""
        p = Path(desc_path)
        # e.g., photo.jpg.desc.txt -> photo.jpg
        image_name = p.name.removesuffix(".desc.txt")
        image_path = p.parent / image_name
        if image_path.exists() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return image_path
        return None

    def _check_file_size(self, path: Path) -> bool:
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
        except OSError:
            return False
        if size_mb > MAX_FILE_SIZE_MB:
            log(f"  SKIP (zu gross): {path.name} ({size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB)")
            return False
        return True

    def _index_file(self, path: Path, action: str):
        """Index a file with retry logic, safe for background threads."""
        _safe_write_status(is_indexing=True, current_file=path.name)

        last_error = None
        for attempt in range(MAX_RETRIES_PER_DOC):
            try:
                count = index_with_timeout(path, self.collection)
                log(f"  {action}: {path.name} ({count} Chunks)")
                _safe_write_status(is_indexing=False, current_file=None)
                return
            except IndexTimeoutError as e:
                last_error = e
                timeout = _calc_timeout(path)
                log(f"  TIMEOUT: {path.name} (>{timeout}s) [Versuch {attempt + 1}/{MAX_RETRIES_PER_DOC}]")
                if attempt < MAX_RETRIES_PER_DOC - 1:
                    time.sleep(RETRY_DOC_DELAY)
            except Exception as e:
                last_error = e
                log(f"  FEHLER bei {path.name}: {e} [Versuch {attempt + 1}/{MAX_RETRIES_PER_DOC}]")
                if attempt < MAX_RETRIES_PER_DOC - 1:
                    if _is_rate_limit_error(e):
                        time.sleep(60)  # Longer pause for rate limits
                    else:
                        time.sleep(RETRY_DOC_DELAY)

        # All retries exhausted
        _safe_add_error(path.name, last_error)
        _safe_write_status(is_indexing=False, current_file=None)

    def on_created(self, event):
        if event.is_directory:
            return
        # Handle .desc.txt sidecar: re-index the parent image
        if self._is_description_file(event.src_path):
            image_path = self._get_image_for_description(event.src_path)
            if image_path:
                log(f"Beschreibung hinzugefuegt: {image_path.name}")
                self._index_file(image_path, "Re-indiziert (mit Beschreibung)")
            return
        if self._should_process(event.src_path):
            path = Path(event.src_path)
            if not self._check_file_size(path):
                return
            log(f"Neue Datei: {path.name}")
            self._index_file(path, "Indiziert")

    def on_modified(self, event):
        if event.is_directory:
            return
        # Handle .desc.txt sidecar: re-index the parent image
        if self._is_description_file(event.src_path):
            image_path = self._get_image_for_description(event.src_path)
            if image_path:
                log(f"Beschreibung aktualisiert: {image_path.name}")
                self._index_file(image_path, "Re-indiziert (Beschreibung aktualisiert)")
            return
        if self._should_process(event.src_path):
            path = Path(event.src_path)
            if not self._check_file_size(path):
                return
            log(f"Datei geaendert: {path.name}")
            self._index_file(path, "Re-indiziert")


def initial_index(collection):
    docs_dir = Path(DOCUMENTS_DIR)
    log(f"Starte initialen Scan von {docs_dir}...")

    try:
        existing = collection.get()
        existing_sources = set()
        if existing and existing.get("metadatas"):
            for meta in existing["metadatas"]:
                if meta and "source" in meta:
                    existing_sources.add(meta["source"])
    except Exception:
        existing_sources = set()

    all_files = []
    for file_path in sorted(docs_dir.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            if file_path.name not in existing_sources:
                all_files.append(file_path)

    already_indexed = len(existing_sources)
    total_pending = len(all_files)
    total_files = already_indexed + total_pending

    log(f"Gefunden: {already_indexed} bereits indiziert, {total_pending} ausstehend")
    _safe_write_status(
        is_indexing=total_pending > 0,
        current_file=None,
        pending_count=total_pending,
        indexed_count=already_indexed,
        total_files=total_files,
        files_indexed_this_session=0,
        errors=[],
    )

    if total_pending == 0:
        log("Keine neuen Dateien.")
        _safe_write_status(is_indexing=False)
        return

    indexed_count = 0
    skipped_count = 0
    error_count = 0
    consecutive_rate_limits = 0
    MAX_CONSECUTIVE_RATE_LIMITS = 3
    RATE_LIMIT_PAUSE = int(os.environ.get("RATE_LIMIT_PAUSE", "1800"))  # 30 min default

    for i, file_path in enumerate(all_files, 1):
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
        except OSError:
            continue

        remaining = total_pending - i

        if size_mb > MAX_FILE_SIZE_MB:
            log(f"  [{i}/{total_pending}] SKIP (zu gross): {file_path.name} ({size_mb:.1f} MB)")
            skipped_count += 1
            _safe_write_status(pending_count=remaining)
            continue

        log(f"  [{i}/{total_pending}] Indiziere: {file_path.name} ({size_mb:.1f} MB)...")
        _safe_write_status(current_file=file_path.name, pending_count=remaining)

        # Per-document retry loop
        success = False
        last_error = None
        for attempt in range(MAX_RETRIES_PER_DOC):
            try:
                count = index_with_timeout(file_path, collection)
                log(f"  [{i}/{total_pending}] OK: {file_path.name} ({count} Chunks)")
                indexed_count += 1
                consecutive_rate_limits = 0
                _safe_write_status(
                    indexed_count=already_indexed + indexed_count,
                    files_indexed_this_session=indexed_count,
                )
                success = True
                break
            except IndexTimeoutError as e:
                last_error = e
                timeout = _calc_timeout(file_path)
                log(f"  [{i}/{total_pending}] TIMEOUT: {file_path.name} (>{timeout}s) "
                    f"[Versuch {attempt + 1}/{MAX_RETRIES_PER_DOC}]")
                if attempt < MAX_RETRIES_PER_DOC - 1:
                    log(f"  Warte {RETRY_DOC_DELAY}s vor erneutem Versuch...")
                    time.sleep(RETRY_DOC_DELAY)
            except Exception as e:
                last_error = e
                log(f"  [{i}/{total_pending}] FEHLER: {file_path.name}: {e} "
                    f"[Versuch {attempt + 1}/{MAX_RETRIES_PER_DOC}]")
                if attempt < MAX_RETRIES_PER_DOC - 1:
                    delay = 60 if _is_rate_limit_error(e) else RETRY_DOC_DELAY
                    log(f"  Warte {delay}s vor erneutem Versuch...")
                    time.sleep(delay)

        if not success:
            error_count += 1
            _safe_add_error(file_path.name, last_error)

            # Check if this is a rate-limit related failure
            is_rate_limit = (
                isinstance(last_error, IndexTimeoutError)
                or (last_error is not None and _is_rate_limit_error(last_error))
            )

            if is_rate_limit:
                consecutive_rate_limits += 1
                if consecutive_rate_limits >= MAX_CONSECUTIVE_RATE_LIMITS:
                    pause_min = RATE_LIMIT_PAUSE // 60
                    log(f"  Rate-Limit erkannt: {consecutive_rate_limits} Fehler in Folge. "
                        f"Pausiere {pause_min} Min. ({remaining} Dateien verbleiben)")
                    _safe_write_status(
                        is_indexing=True,  # Keep True so UI knows we're still working
                        current_file=f"PAUSIERT - Rate-Limit ({pause_min} Min Pause, {remaining} verbleibend)",
                    )
                    _sleep_with_heartbeat(
                        RATE_LIMIT_PAUSE,
                        f"PAUSIERT - Rate-Limit ({remaining} verbleibend)"
                    )
                    consecutive_rate_limits = 0
                    log(f"  Pause beendet, setze Indexierung fort...")
                    _safe_write_status(is_indexing=True, current_file=None)
            else:
                consecutive_rate_limits = 0

    log(f"Scan fertig: {indexed_count} neu, {skipped_count} uebersprungen, {error_count} Fehler")
    _safe_write_status(is_indexing=False, current_file=None, pending_count=0)


def shopware_sync_loop():
    from config import SHOPWARE_ACCESS_KEY
    if not SHOPWARE_ACCESS_KEY:
        log("SHOPWARE_ACCESS_KEY nicht gesetzt - Sync deaktiviert.")
        return
    time.sleep(30)
    while True:
        try:
            from shopware import index_products
            log("Starte Shopware Sync...")
            index_products()
            log(f"Shopware Sync fertig. Naechster in {SYNC_INTERVAL // 3600}h.")
        except Exception as e:
            log(f"Shopware Sync Fehler: {e}")
        time.sleep(SYNC_INTERVAL)


def run_watcher():
    docs_dir = Path(DOCUMENTS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)

    log(f"RAG Watcher gestartet (Max: {MAX_FILE_SIZE_MB} MB, Timeout: {FILE_TIMEOUT_BASE}s + {FILE_TIMEOUT_PER_MB}s/MB, Polling: {POLL_INTERVAL}s)")

    client = get_chroma_client()
    collection = get_or_create_collection(client, COLLECTION_KNOWLEDGE)
    initial_index(collection)

    handler = DocumentHandler(collection)
    observer = PollingObserver(timeout=POLL_INTERVAL)
    observer.schedule(handler, str(docs_dir), recursive=True)
    observer.start()
    log(f"Ueberwache {docs_dir}...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    run_watcher()
