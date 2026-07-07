"""Eval direkt gegen eine benannte Collection (z.B. knowledge_ctx).

Repliziert exakt den Server-Suchpfad (Hybrid-RRF + Rerank), nur mit
wählbarem Collection-Namen - für den A/B-Vergleich knowledge vs.
knowledge_ctx, bevor umgeschaltet wird.

Usage: eval_ctx.py --collection knowledge_ctx --label ctx-pilot [--sleep 4.5]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/openclaw/tools/rag-pipeline")

from rerank import rerank
from search import search_collection

BASE = Path("/opt/openclaw/tools/rag-pipeline/eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    queries = json.loads((BASE / "queries.json").read_text(encoding="utf-8"))
    per_query = []
    hits1 = hits5 = 0
    mrr = 0.0
    t0 = time.time()

    for q in queries:
        if args.sleep:
            time.sleep(args.sleep)
        items = search_collection(
            q["question"], args.collection, args.candidates,
            diverse=True, max_per_source=2,
        )
        items = rerank(q["question"], items, args.n)
        sources = [r["source"] for r in items]
        rank = next((i for i, s in enumerate(sources, 1) if s == q["expected_source"]), None)
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
        print(f"  {q['id']:2d}. [{mark:>7}] {q['question'][:70]}", flush=True)

    n = len(queries)
    aggregate = {
        "recall_at_1": round(hits1 / n, 3),
        "recall_at_5": round(hits5 / n, 3),
        "mrr": round(mrr / n, 3),
        "queries": n,
        "duration_s": round(time.time() - t0, 1),
        "collection": args.collection,
    }
    print(f"\n== {args.label} ==")
    print(json.dumps(aggregate, indent=2))

    out = BASE / "results" / f"{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "label": args.label,
        "params": {"n": args.n, "candidates": args.candidates, "collection": args.collection},
        "aggregate": aggregate,
        "per_query": per_query,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
