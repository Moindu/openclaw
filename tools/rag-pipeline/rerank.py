"""Listwise reranking of search results via Gemini Flash-Lite.

The hybrid search (RRF fusion) produces a solid candidate list; a small
LLM then reads query + candidates together and reorders them, which is
markedly more precise than embedding distance alone. Any failure
(quota, timeout, unparseable output) silently falls back to the hybrid
order - reranking must never break or noticeably delay a search.
"""
import json
import logging
import os
import re

from google import genai

from config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

RERANK_ENABLED = os.environ.get("RERANK", "on").lower() in ("on", "true", "1")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "gemini-3.1-flash-lite-preview")
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "20"))
RERANK_TIMEOUT_MS = int(os.environ.get("RERANK_TIMEOUT_MS", "10000"))
_SNIPPET_CHARS = 400

_client = None

PROMPT = """Du bist ein Ranking-Modul der Wissensdatenbank einer Ledermanufaktur \
(Gerberei, Lederverarbeitung, historische Fachbücher).

Sortiere die nummerierten Textausschnitte nach Relevanz für die Suchanfrage. \
Relevant ist, was die Anfrage direkt beantwortet oder das gesuchte \
Produkt/Verfahren konkret behandelt - nicht, was nur dieselben Wörter enthält.

Suchanfrage: {query}

Textausschnitte:
{candidates}

Antworte NUR mit einem JSON-Array der Ausschnitt-Nummern, relevanteste zuerst, \
maximal {top_n} Einträge. Beispiel: [3, 1, 7]"""


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options={"timeout": RERANK_TIMEOUT_MS},
        )
    return _client


def rerank(query: str, items: list[dict], n_results: int) -> list[dict]:
    """Reorder items for the query; fall back to the given order on any error."""
    if not RERANK_ENABLED or len(items) <= 1:
        return items[:n_results]

    candidates = items[:RERANK_CANDIDATES]
    lines = []
    for i, item in enumerate(candidates, 1):
        snippet = " ".join((item.get("text") or "").split())[:_SNIPPET_CHARS]
        lines.append(f"[{i}] (Quelle: {item['source']}) {snippet}")

    try:
        resp = _get_client().models.generate_content(
            model=RERANK_MODEL,
            contents=PROMPT.format(
                query=query, candidates="\n".join(lines), top_n=n_results,
            ),
        )
        text = resp.text or ""
        match = re.search(r"\[[\d,\s]+\]", text)
        if not match:
            raise ValueError(f"keine Indexliste in Antwort: {text[:120]!r}")
        order = json.loads(match.group(0))
        picked = []
        seen = set()
        for num in order:
            idx = int(num) - 1
            if 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                candidates[idx]["metadata"]["_rerank_pos"] = len(picked) + 1
                picked.append(candidates[idx])
        # Fill up from the hybrid order if the model returned too few
        for idx, item in enumerate(candidates):
            if len(picked) >= n_results:
                break
            if idx not in seen:
                picked.append(item)
        return picked[:n_results]
    except Exception as e:
        logger.warning("Rerank fehlgeschlagen (%s) - nutze Hybrid-Reihenfolge", e)
        return items[:n_results]
