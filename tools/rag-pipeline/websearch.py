"""Web search via SearXNG for supplementing RAG results."""
import os

import httpx

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8888")


def search_web(query: str, n_results: int = 5, language: str = "de") -> list[dict]:
    """Search the web via SearXNG and return formatted results."""
    try:
        response = httpx.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "language": language,
                "categories": "general",
            },
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Web search error: {e}")
        return []

    results = []
    for r in data.get("results", [])[:n_results]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        })

    return results


def format_web_results(results: list[dict]) -> str:
    """Format web search results as context."""
    if not results:
        return "Keine Webrecherche-Ergebnisse gefunden."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[Web-Quelle {i}: {r['title']}]\n{r['snippet']}\nURL: {r['url']}")

    return "\n\n---\n\n".join(parts)
