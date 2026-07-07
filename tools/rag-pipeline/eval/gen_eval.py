"""Generate an eval set for the knowledge RAG: (German question, expected source).

Samples one sufficiently long text chunk from each of N distinct source
documents and lets Gemini Flash-Lite formulate a realistic user question
that the chunk answers. Deterministic sampling (seed) for reproducibility.
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/openclaw/tools/rag-pipeline")

from google import genai

from config import COLLECTION_KNOWLEDGE, GOOGLE_API_KEY
from indexer import get_chroma_client, get_or_create_collection

N_QUESTIONS = 40
MIN_CHUNK_CHARS = 400
MODEL = "gemini-3.1-flash-lite-preview"
OUT = Path("/opt/openclaw/tools/rag-pipeline/eval/queries.json")

PROMPT = """Du erstellst Testfragen für die Wissensdatenbank einer Ledermanufaktur \
(Gerberei, Lederverarbeitung, Lederpflege, historische Fachbücher).

Formuliere EINE realistische Frage auf Deutsch, die ein Nutzer (Kunde, Sattler \
oder Gerber) stellen könnte und die sich mit dem folgenden Textausschnitt \
beantworten lässt.

Regeln:
- Die Frage muss eigenständig verständlich sein (ohne den Text zu kennen).
- Kopiere keine wörtlichen Formulierungen aus dem Text.
- Nenne KEINE Metaangaben (Buchtitel, Seite, Autor, "im Text").
- Auch wenn der Text englisch ist: Die Frage ist auf Deutsch.
- Gib NUR die Frage aus, sonst nichts.

Textausschnitt:
{chunk}"""


def main():
    client = get_chroma_client()
    collection = get_or_create_collection(client, COLLECTION_KNOWLEDGE)
    total = collection.count()

    by_source: dict[str, list[str]] = {}
    offset = 0
    while offset < total:
        batch = collection.get(include=["documents", "metadatas"], limit=2000, offset=offset)
        docs = batch.get("documents") or []
        if not docs:
            break
        for doc, meta in zip(docs, batch.get("metadatas") or []):
            meta = meta or {}
            if meta.get("chunk_type") == "image":
                continue
            if not doc or len(doc) < MIN_CHUNK_CHARS:
                continue
            src = meta.get("source", meta.get("book_title"))
            if src:
                by_source.setdefault(src, []).append(doc)
        offset += len(docs)

    print(f"{len(by_source)} Quellen mit brauchbaren Chunks (von {total} Chunks)")
    rng = random.Random(42)
    sources = rng.sample(sorted(by_source), min(N_QUESTIONS, len(by_source)))

    gclient = genai.Client(api_key=GOOGLE_API_KEY)
    queries = []
    for i, src in enumerate(sources, 1):
        chunk = rng.choice(by_source[src])[:1500]
        for attempt in range(3):
            try:
                resp = gclient.models.generate_content(
                    model=MODEL, contents=PROMPT.format(chunk=chunk),
                )
                question = (resp.text or "").strip().strip('"')
                break
            except Exception as e:
                print(f"  Versuch {attempt+1} fehlgeschlagen: {e}")
                time.sleep(5)
        else:
            continue
        if not question or len(question) < 15:
            print(f"  übersprungen ({src}): leere/kurze Frage")
            continue
        queries.append({
            "id": i,
            "question": question,
            "expected_source": src,
            "chunk_preview": chunk[:200],
            "generator_model": MODEL,
        })
        print(f"  {i:2d}. [{src[:45]}] {question[:80]}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(queries)} Fragen -> {OUT}")


if __name__ == "__main__":
    main()
