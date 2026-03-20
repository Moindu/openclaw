"""Shared status file bridge between watcher and server processes."""
import json
import time
import threading
from pathlib import Path

STATUS_FILE = Path("/tmp/rag_indexing_status.json")
_lock = threading.Lock()

def write_status(**kwargs):
    """Write indexing status from watcher process."""
    with _lock:
        try:
            existing = read_status()
        except Exception:
            existing = {}
        existing.update(kwargs)
        existing["updated_at"] = time.time()
        try:
            STATUS_FILE.write_text(json.dumps(existing, ensure_ascii=False))
        except OSError:
            pass  # Never let I/O errors propagate to callers
    
def add_error(filename, error):
    """Append an error to the errors list."""
    with _lock:
        data = read_status()
        errors = data.get("errors", [])
        errors.append({
            "file": filename,
            "error": str(error)[:200],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        if len(errors) > 50:
            errors = errors[-50:]
        data["errors"] = errors
        data["updated_at"] = time.time()
        try:
            STATUS_FILE.write_text(json.dumps(data, ensure_ascii=False))
        except OSError:
            pass  # Never let I/O errors propagate to callers

def clear_errors():
    """Clear all errors from the status file."""
    with _lock:
        data = read_status()
        data["errors"] = []
        data["updated_at"] = time.time()
        try:
            STATUS_FILE.write_text(json.dumps(data, ensure_ascii=False))
        except OSError:
            pass

def read_status():
    """Read indexing status from server process."""
    try:
        if STATUS_FILE.exists():
            data = json.loads(STATUS_FILE.read_text())
            # Consider stale if not updated in 60 seconds
            if time.time() - data.get("updated_at", 0) > 300:
                data["is_indexing"] = False
                data["current_file"] = None
            return data
    except Exception:
        pass
    return {
        "is_indexing": False,
        "current_file": None,
        "pending_count": 0,
        "indexed_count": 0,
        "total_files": 0,
        "errors": [],
        "files_indexed_this_session": 0,
    }
