"""Contextual Retrieval (Anthropic-Methode) für die knowledge-Collection.

Jeder Chunk bekommt 1-2 Sätze Dokument-Kontext vorangestellt und wird neu
embedded - in die Collection `knowledge_ctx`, die als vollständige Kopie
der bestehenden Chunks startet (inkl. vorhandener Embeddings, kein Re-OCR,
keine Embedding-Kosten für die Kopie). Quellen werden schrittweise
kontextualisiert; der Fortschritt steht in den Metadaten (context_added),
das Skript ist beliebig unterbrech- und wiederaufnehmbar.

Modi:
  --copy               knowledge -> knowledge_ctx komplett kopieren (einmalig)
  --sources a.pdf b.pdf   diese Quellen kontextualisieren
  --sources-from-eval  die erwarteten Quellen des Eval-Sets kontextualisieren
  --all                alle noch offenen Quellen kontextualisieren
  --status             Fortschritt anzeigen
"""
import argparse
import json
import logging
import re as _re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/openclaw/tools/rag-pipeline")

from google import genai

from config import COLLECTION_KNOWLEDGE, GOOGLE_API_KEY
from embeddings import get_embeddings_batch
from indexer import get_chroma_client, get_or_create_collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("contextual")

CTX_COLLECTION = "knowledge_ctx"
GEN_MODEL = "gemini-3.1-flash-lite-preview"
CHUNKS_PER_CALL = 10
GEN_SLEEP = 5.0  # ~12 Calls/Minute -> bleibt unter dem Free-Tier-Limit (15 RPM)
COPY_BATCH = 500
FETCH_BATCH = 2000

SUMMARY_PROMPT = """Fasse in 3-4 Sätzen auf Deutsch zusammen, worum es in diesem \
Dokument geht: Dokumenttyp (z.B. Produktdatenblatt, Fachbuch, Rechnung, Studie), \
Thema, behandelte Produkte/Verfahren. Gib NUR die Zusammenfassung aus.

Dateiname: {source}

Anfang des Dokuments:
{head}"""

CONTEXT_PROMPT = """Dokument: {source}
Zusammenfassung: {summary}

Unten stehen nummerierte Ausschnitte aus diesem Dokument. Schreibe für JEDEN \
Ausschnitt 1-2 kurze Sätze auf Deutsch, die ihn im Dokument verorten (welches \
Produkt/Verfahren/Thema er behandelt und in welchem Zusammenhang). Diese Sätze \
werden dem Ausschnitt für eine Suchmaschine vorangestellt - keine Einleitungen \
wie "Dieser Ausschnitt".

Antworte NUR mit JSON: {{"1": "...", "2": "..."}}

{chunks}"""

_client = None


def _gen_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY, http_options={"timeout": 60000})
    return _client


def _generate(prompt: str, attempts: int = 5) -> str:
    for attempt in range(attempts):
        try:
            resp = _gen_client().models.generate_content(model=GEN_MODEL, contents=prompt)
            return resp.text or ""
        except Exception as e:
            msg = str(e)
            m = _re.search(r"retry in ([\d.]+)s", msg, _re.IGNORECASE)
            delay = float(m.group(1)) + 2 if m else 30 * (attempt + 1)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                logger.info("  Rate-Limit, warte %.0fs...", delay)
            else:
                logger.warning("  generate fehlgeschlagen (%s), warte %.0fs", msg[:120], delay)
            time.sleep(delay)
    raise RuntimeError("generate_content nach mehreren Versuchen fehlgeschlagen")


def _collections():
    client = get_chroma_client()
    src = get_or_create_collection(client, COLLECTION_KNOWLEDGE)
    dst = get_or_create_collection(client, CTX_COLLECTION)
    return src, dst


def do_copy():
    """knowledge -> knowledge_ctx vollständig kopieren (mit Embeddings)."""
    src, dst = _collections()
    total = src.count()
    have = dst.count()
    if have >= total:
        logger.info("Kopie existiert bereits (%d Chunks)", have)
        return
    offset = 0
    copied = 0
    while offset < total:
        batch = src.get(
            include=["documents", "metadatas", "embeddings"],
            limit=FETCH_BATCH, offset=offset,
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        docs = batch["documents"]
        metas = [dict(m or {}) for m in batch["metadatas"]]
        embs = batch["embeddings"]
        # add() in kleineren Batches (Chroma-Limit)
        for j in range(0, len(ids), COPY_BATCH):
            sl = slice(j, j + COPY_BATCH)
            try:
                dst.add(ids=ids[sl], documents=docs[sl], metadatas=metas[sl], embeddings=embs[sl])
            except Exception:
                # Existierende IDs (Resume): einzeln upserten
                dst.upsert(ids=ids[sl], documents=docs[sl], metadatas=metas[sl], embeddings=embs[sl])
        copied += len(ids)
        offset += len(ids)
        logger.info("kopiert: %d / %d", copied, total)
    logger.info("Kopie fertig: %d Chunks in %s", dst.count(), CTX_COLLECTION)


def _load_source_chunks(dst, source: str):
    """Alle Text-Chunks einer Quelle aus knowledge_ctx, in Dokumentreihenfolge."""
    got = dst.get(where={"source": source}, include=["documents", "metadatas"])
    rows = []
    for cid, doc, meta in zip(got.get("ids", []), got["documents"], got["metadatas"]):
        meta = meta or {}
        rows.append({"id": cid, "text": doc or "", "meta": meta})

    def order_key(r):
        m = r["meta"]
        for key in ("chunk_index", "page_start", "page_number"):
            v = m.get(key)
            if isinstance(v, (int, float)):
                return (0, v)
        return (1, r["id"])

    rows.sort(key=order_key)
    return rows


def contextualize_source(dst, source: str) -> dict:
    rows = _load_source_chunks(dst, source)
    todo = [
        r for r in rows
        if not r["meta"].get("context_added") and r["meta"].get("chunk_type") != "image"
        and len(r["text"]) >= 80
    ]
    if not todo:
        return {"source": source, "chunks": 0, "skipped": True}

    # 1) Dokument-Zusammenfassung aus dem Anfang des Dokuments
    head = ""
    for r in rows:
        if r["meta"].get("chunk_type") == "image":
            continue
        head += r["text"] + "\n"
        if len(head) > 6000:
            break
    summary = _generate(SUMMARY_PROMPT.format(source=source, head=head[:6000])).strip()
    time.sleep(GEN_SLEEP)

    # 2) Kontexte in 10er-Batches generieren
    done = 0
    for i in range(0, len(todo), CHUNKS_PER_CALL):
        batch = todo[i : i + CHUNKS_PER_CALL]
        listing = "\n\n".join(
            f"[{j+1}] {' '.join(r['text'].split())[:700]}" for j, r in enumerate(batch)
        )
        raw = _generate(CONTEXT_PROMPT.format(source=source, summary=summary, chunks=listing))
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        contexts = {}
        if m:
            try:
                contexts = json.loads(m.group(0))
            except json.JSONDecodeError:
                logger.warning("  JSON-Parse-Fehler bei %s Batch %d", source, i)
        time.sleep(GEN_SLEEP)

        new_texts, ids, metas = [], [], []
        for j, r in enumerate(batch):
            ctx = str(contexts.get(str(j + 1), "")).strip()
            if not ctx:
                ctx = f"Aus dem Dokument {source}. {summary.split('.')[0]}."
            new_text = f"{ctx}\n\n{r['text']}"
            meta = dict(r["meta"])
            meta["context_added"] = True
            meta["context"] = ctx[:500]
            new_texts.append(new_text)
            ids.append(r["id"])
            metas.append(meta)

        # 3) Neu embedden + upserten
        embeddings = get_embeddings_batch(new_texts, task_type="RETRIEVAL_DOCUMENT")
        dst.upsert(ids=ids, documents=new_texts, metadatas=metas, embeddings=embeddings)
        done += len(batch)
        logger.info("  %s: %d/%d Chunks kontextualisiert", source, done, len(todo))

    return {"source": source, "chunks": done, "skipped": False}


def open_sources(dst) -> list[str]:
    got = dst.get(include=["metadatas"])
    pending: dict[str, int] = {}
    for meta in got["metadatas"]:
        meta = meta or {}
        if meta.get("context_added") or meta.get("chunk_type") == "image":
            continue
        src = meta.get("source", meta.get("book_title", "unknown"))
        pending[src] = pending.get(src, 0) + 1
    return sorted(pending, key=pending.get)  # kleine Quellen zuerst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--sources-from-eval", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    src, dst = _collections()

    if args.copy:
        do_copy()
        return

    if args.status:
        got = dst.get(include=["metadatas"])
        total = len(got["metadatas"])
        done = sum(1 for m in got["metadatas"] if (m or {}).get("context_added"))
        images = sum(1 for m in got["metadatas"] if (m or {}).get("chunk_type") == "image")
        print(f"{CTX_COLLECTION}: {total} Chunks, {done} kontextualisiert, {images} Bilder (bleiben wie sie sind), {total-done-images} offen")
        return

    if dst.count() == 0:
        logger.error("knowledge_ctx ist leer - zuerst --copy ausführen")
        sys.exit(1)

    if args.sources_from_eval:
        queries = json.loads(Path("/opt/openclaw/tools/rag-pipeline/eval/queries.json").read_text())
        sources = sorted({q["expected_source"] for q in queries})
    elif args.sources:
        sources = args.sources
    elif args.all:
        sources = open_sources(dst)
    else:
        ap.print_help()
        return

    logger.info("%d Quellen zu kontextualisieren", len(sources))
    for k, source in enumerate(sources, 1):
        logger.info("[%d/%d] %s", k, len(sources), source)
        try:
            contextualize_source(dst, source)
        except Exception:
            logger.exception("Quelle %s fehlgeschlagen - weiter mit der nächsten", source)


if __name__ == "__main__":
    main()
