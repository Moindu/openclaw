"""Measure retrieval quality of the knowledge RAG against eval/queries.json.

Metrics (source-level): Recall@1, Recall@5, MRR. Runs against the live
HTTP API so the whole path (embedding, fusion, fallback) is measured.

Usage: run_eval.py --label baseline [--endpoint http://localhost:8100/]
                   [--n 5] [--no-diverse]
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

BASE = Path("/opt/openclaw/tools/rag-pipeline/eval")


def search(endpoint: str, query: str, n: int, diverse: bool) -> dict:
    body = json.dumps({
        "query": query, "collection": "knowledge",
        "n_results": n, "diverse": diverse,
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--endpoint", default="http://localhost:8100/")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--no-diverse", action="store_true")
    args = ap.parse_args()

    queries = json.loads((BASE / "queries.json").read_text(encoding="utf-8"))
    diverse = not args.no_diverse

    per_query = []
    hits1 = hits5 = 0
    mrr = 0.0
    degraded_seen = False
    t0 = time.time()

    for q in queries:
        data = search(args.endpoint, q["question"], args.n, diverse)
        degraded_seen = degraded_seen or bool(data.get("degraded"))
        sources = [r["source"] for r in data.get("results", [])]
        rank = None
        for idx, src in enumerate(sources, 1):
            if src == q["expected_source"]:
                rank = idx
                break
        if rank == 1:
            hits1 += 1
        if rank is not None and rank <= 5:
            hits5 += 1
        mrr += (1.0 / rank) if rank else 0.0
        per_query.append({
            "id": q["id"], "question": q["question"],
            "expected_source": q["expected_source"],
            "rank": rank, "top_sources": sources[:5],
        })
        mark = f"rank {rank}" if rank else "MISS"
        print(f"  {q['id']:2d}. [{mark:>7}] {q['question'][:70]}")

    n = len(queries)
    aggregate = {
        "recall_at_1": round(hits1 / n, 3),
        "recall_at_5": round(hits5 / n, 3),
        "mrr": round(mrr / n, 3),
        "queries": n,
        "duration_s": round(time.time() - t0, 1),
        "degraded_seen": degraded_seen,
    }
    print(f"\n== {args.label} ==")
    print(json.dumps(aggregate, indent=2))

    out = BASE / "results" / f"{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "label": args.label,
        "params": {"n": args.n, "diverse": diverse, "endpoint": args.endpoint},
        "aggregate": aggregate,
        "per_query": per_query,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
