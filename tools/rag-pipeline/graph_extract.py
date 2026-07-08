"""Wissensgraph-Extraktion für die Kobel-Wissensdatenbank.

Extrahiert pro Quelldokument Fachentitäten (Ledertypen, Gerbverfahren,
Gerbstoffe, ...) und Relationen via Gemini Flash-Lite und speichert sie
in SQLite (/data/rag-graph/graph.db). Dokument-Ebene, ein API-Call pro
Quelle - resumefähig, bereits extrahierte Quellen werden übersprungen.

Usage: graph_extract.py [--limit N] [--source NAME] [--status]
"""
import argparse
import json
import logging
import os
import re as _re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/openclaw/tools/rag-pipeline")

from google import genai

from config import COLLECTION_KNOWLEDGE, GOOGLE_API_KEY
from indexer import get_chroma_client, get_or_create_collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("graph")

DB_PATH = Path("/data/rag-graph/graph.db")
GEN_BACKEND = os.environ.get("GRAPH_LLM", "gemini")  # gemini | ollama
GEN_MODEL = os.environ.get("GRAPH_GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("GRAPH_OLLAMA_MODEL", "gemma4:12b-it-qat")
GEN_SLEEP = 5.0 if GEN_BACKEND == "gemini" else 0.0  # Free-Tier-Drossel nur für die API

ENTITY_TYPES = [
    "Ledertyp", "Gerbverfahren", "Gerbstoff", "Chemikalie", "Prozessschritt",
    "Produkt", "Werkzeug", "Organisation", "Person", "Epoche", "Ort", "Thema",
]

PROMPT = """Du baust einen Wissensgraphen für die Wissensdatenbank einer \
Ledermanufaktur/Gerberei (wie die Graph-Ansicht in Obsidian).

Extrahiere aus dem Dokumentanfang unten die wichtigsten Fachentitäten und \
ihre Beziehungen.

Entitätstypen (nur diese): {types}

Regeln:
- Max. 12 Entitäten, max. 8 Relationen. Nur fachlich Relevantes, keine Trivia.
- Namen normalisiert: Deutsch (wo üblich), Singular, ohne Artikel,
  z.B. "Grubengerbung", "Eichenlohe", "Blankleder", "Chromgerbung".
- Produktnamen exakt wie geschrieben (z.B. "NOVALTAN DPA").
- Relationstyp ist ein kurzes deutsches Verb/Prädikat, z.B. "verwendet",
  "gehört zu", "ersetzt", "hergestellt von", "behandelt".

Antworte NUR mit JSON:
{{"entities": [{{"name": "...", "type": "..."}}],
  "relations": [{{"from": "...", "to": "...", "type": "..."}}]}}

Dateiname: {source}

Dokumentanfang:
{head}"""

_client = None


def _gen_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY, http_options={"timeout": 60000})
    return _client


def _generate_ollama(prompt: str) -> str:
    """Lokale Extraktion via Ollama (gemma4) - quota-unabhängig, offline-tauglich."""
    import urllib.request
    body = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json",
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


def _generate(prompt: str, attempts: int = 5) -> str:
    if GEN_BACKEND == "ollama":
        return _generate_ollama(prompt)
    for attempt in range(attempts):
        try:
            resp = _gen_client().models.generate_content(model=GEN_MODEL, contents=prompt)
            return resp.text or ""
        except Exception as e:
            msg = str(e)
            m = _re.search(r"retry in ([\d.]+)s", msg, _re.IGNORECASE)
            delay = float(m.group(1)) + 2 if m else 30 * (attempt + 1)
            logger.info("  warte %.0fs (%s)", delay, msg[:80])
            time.sleep(delay)
    raise RuntimeError("generate_content fehlgeschlagen")


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            norm TEXT NOT NULL UNIQUE,
            type TEXT NOT NULL,
            mentions INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS mentions (
            entity_id INTEGER NOT NULL REFERENCES entities(id),
            source TEXT NOT NULL,
            UNIQUE(entity_id, source)
        );
        CREATE TABLE IF NOT EXISTS relations (
            src INTEGER NOT NULL REFERENCES entities(id),
            dst INTEGER NOT NULL REFERENCES entities(id),
            type TEXT NOT NULL,
            weight INTEGER DEFAULT 1,
            UNIQUE(src, dst, type)
        );
        CREATE TABLE IF NOT EXISTS done_sources (
            source TEXT PRIMARY KEY,
            extracted_at REAL
        );
    """)
    return db


def _norm(name: str) -> str:
    return " ".join(name.lower().split())


def _upsert_entity(db, name: str, etype: str) -> int:
    norm = _norm(name)
    row = db.execute("SELECT id FROM entities WHERE norm=?", (norm,)).fetchone()
    if row:
        return row[0]
    cur = db.execute(
        "INSERT INTO entities (name, norm, type) VALUES (?,?,?)", (name.strip(), norm, etype),
    )
    return cur.lastrowid


def _doc_head(collection, source: str, max_chars: int = 6000) -> str:
    got = collection.get(where={"source": source}, include=["documents", "metadatas"])
    rows = list(zip(got.get("documents", []), got.get("metadatas", [])))

    def order_key(r):
        m = r[1] or {}
        for key in ("chunk_index", "page_start", "page_number"):
            v = m.get(key)
            if isinstance(v, (int, float)):
                return (0, v)
        return (1, 0)

    rows.sort(key=order_key)
    head = ""
    for doc, meta in rows:
        if (meta or {}).get("chunk_type") == "image":
            continue
        head += (doc or "") + "\n"
        if len(head) >= max_chars:
            break
    return head[:max_chars]


def extract_source(db, collection, source: str) -> dict:
    head = _doc_head(collection, source)
    if len(head) < 100:
        db.execute("INSERT OR REPLACE INTO done_sources VALUES (?, ?)", (source, time.time()))
        db.commit()
        return {"entities": 0, "relations": 0, "skipped": "zu wenig Text"}

    raw = _generate(PROMPT.format(types=", ".join(ENTITY_TYPES), source=source, head=head))
    m = _re.search(r"\{.*\}", raw, _re.DOTALL)
    if not m:
        raise ValueError(f"kein JSON in Antwort: {raw[:120]!r}")
    data = json.loads(m.group(0))

    ids_by_norm = {}
    n_ent = 0
    for ent in data.get("entities", [])[:15]:
        name = str(ent.get("name", "")).strip()
        etype = str(ent.get("type", "Thema")).strip()
        if not name or len(name) > 80:
            continue
        if etype not in ENTITY_TYPES:
            etype = "Thema"
        eid = _upsert_entity(db, name, etype)
        ids_by_norm[_norm(name)] = eid
        changed = db.execute(
            "INSERT OR IGNORE INTO mentions (entity_id, source) VALUES (?,?)", (eid, source),
        ).rowcount
        if changed:
            db.execute("UPDATE entities SET mentions = mentions + 1 WHERE id=?", (eid,))
        n_ent += 1

    n_rel = 0
    for rel in data.get("relations", [])[:10]:
        src_id = ids_by_norm.get(_norm(str(rel.get("from", ""))))
        dst_id = ids_by_norm.get(_norm(str(rel.get("to", ""))))
        rtype = str(rel.get("type", "bezieht sich auf")).strip()[:40] or "bezieht sich auf"
        if not src_id or not dst_id or src_id == dst_id:
            continue
        cur = db.execute(
            "INSERT INTO relations (src, dst, type) VALUES (?,?,?) "
            "ON CONFLICT(src, dst, type) DO UPDATE SET weight = weight + 1",
            (src_id, dst_id, rtype),
        )
        n_rel += 1

    db.execute("INSERT OR REPLACE INTO done_sources VALUES (?, ?)", (source, time.time()))
    db.commit()
    return {"entities": n_ent, "relations": n_rel}


def all_sources(collection) -> list[str]:
    got = collection.get(include=["metadatas"])
    sources = set()
    for meta in got["metadatas"]:
        meta = meta or {}
        src = meta.get("source", meta.get("book_title"))
        if src:
            sources.add(src)
    return sorted(sources)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    db = get_db()
    if args.status:
        e = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        r = db.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        d = db.execute("SELECT COUNT(*) FROM done_sources").fetchone()[0]
        print(f"Entitäten: {e}, Relationen: {r}, Quellen verarbeitet: {d}")
        return

    client = get_chroma_client()
    collection = get_or_create_collection(client, COLLECTION_KNOWLEDGE)

    if args.source:
        sources = [args.source]
    else:
        done = {row[0] for row in db.execute("SELECT source FROM done_sources")}
        sources = [s for s in all_sources(collection) if s not in done]
        if args.limit:
            sources = sources[: args.limit]

    logger.info("%d Quellen zu extrahieren", len(sources))
    for k, source in enumerate(sources, 1):
        try:
            stats = extract_source(db, collection, source)
            logger.info("[%d/%d] %s: %s", k, len(sources), source[:60], stats)
        except Exception:
            logger.exception("[%d/%d] %s fehlgeschlagen", k, len(sources), source[:60])
        time.sleep(GEN_SLEEP)


if __name__ == "__main__":
    main()
