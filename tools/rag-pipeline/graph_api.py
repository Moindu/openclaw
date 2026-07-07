"""Lese-API für den Wissensgraphen (SQLite, siehe graph_extract.py)."""
import sqlite3
from pathlib import Path

DB_PATH = Path("/data/rag-graph/graph.db")


def _db() -> sqlite3.Connection | None:
    if not DB_PATH.exists():
        return None
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def graph_overview(min_mentions: int = 1, limit: int = 400) -> dict:
    """Knoten (Entitäten) + Kanten (Relationen) für die Graph-Ansicht."""
    db = _db()
    if db is None:
        return {"nodes": [], "edges": [], "sources_done": 0, "available": False}

    nodes = [
        dict(r) for r in db.execute(
            "SELECT id, name, type, mentions FROM entities "
            "WHERE mentions >= ? ORDER BY mentions DESC LIMIT ?",
            (min_mentions, limit),
        )
    ]
    ids = {n["id"] for n in nodes}
    edges = [
        dict(r) for r in db.execute(
            "SELECT src, dst, type, weight FROM relations ORDER BY weight DESC LIMIT 2000",
        )
        if r["src"] in ids and r["dst"] in ids
    ]
    done = db.execute("SELECT COUNT(*) FROM done_sources").fetchone()[0]
    db.close()
    return {"nodes": nodes, "edges": edges, "sources_done": done, "available": True}


def entity_detail(name: str) -> dict:
    """Detail zu einer Entität: Quellen + Nachbar-Entitäten."""
    db = _db()
    if db is None:
        return {"available": False}

    norm = " ".join(name.lower().split())
    row = db.execute("SELECT * FROM entities WHERE norm=?", (norm,)).fetchone()
    if row is None:
        db.close()
        return {"available": True, "found": False}

    eid = row["id"]
    sources = [r["source"] for r in db.execute(
        "SELECT source FROM mentions WHERE entity_id=? ORDER BY source", (eid,),
    )]
    neighbors = [
        dict(r) for r in db.execute(
            """SELECT e.name, e.type, e.mentions, rel.type AS relation, rel.weight,
                      CASE WHEN rel.src=? THEN 'out' ELSE 'in' END AS direction
               FROM relations rel
               JOIN entities e ON e.id = CASE WHEN rel.src=? THEN rel.dst ELSE rel.src END
               WHERE rel.src=? OR rel.dst=?
               ORDER BY rel.weight DESC, e.mentions DESC LIMIT 50""",
            (eid, eid, eid, eid),
        )
    ]
    db.close()
    return {
        "available": True, "found": True,
        "entity": {"id": eid, "name": row["name"], "type": row["type"], "mentions": row["mentions"]},
        "sources": sources,
        "neighbors": neighbors,
    }
