"""One-time migration: replace Ollama nomic-embed-text with Gemini Embedding 2.

Deletes existing ChromaDB collections and re-indexes everything with the new
embedding model. Run once after switching to Gemini Embedding 2.

Usage:
    python migrate.py
    # or via Docker:
    docker compose run --rm rag-api python migrate.py
"""
from pathlib import Path

from config import COLLECTION_KNOWLEDGE, COLLECTION_PRODUCTS, DOCUMENTS_DIR
from indexer import get_chroma_client, get_or_create_collection, index_documents_dir


def reset_collection(client, name: str):
    """Delete and recreate a ChromaDB collection."""
    try:
        client.delete_collection(name)
        print(f"  Deleted old collection: {name}")
    except Exception:
        print(f"  Collection {name} did not exist, creating fresh")
    return get_or_create_collection(client, name)


def run_migration():
    """Delete old collections and re-index everything with Gemini Embedding 2."""
    print("=" * 60)
    print("Migration: Ollama nomic-embed-text -> Gemini Embedding 2")
    print("=" * 60)

    client = get_chroma_client()

    # 1. Reset and re-index knowledge documents
    print("\n[1/2] Re-indexing knowledge documents...")
    collection = reset_collection(client, COLLECTION_KNOWLEDGE)
    docs_dir = Path(DOCUMENTS_DIR)
    if docs_dir.exists():
        stats = index_documents_dir(docs_dir, collection)
        print(f"  Done: {stats['files']} files, {stats['chunks']} chunks, {stats['skipped']} skipped")
    else:
        print(f"  Documents directory not found: {docs_dir}")

    # 2. Reset and re-index Shopware products
    print("\n[2/2] Re-indexing Shopware products...")
    reset_collection(client, COLLECTION_PRODUCTS)
    try:
        from shopware import index_products
        index_products()
    except Exception as e:
        print(f"  Product re-index skipped: {e}")

    print("\n" + "=" * 60)
    print("Migration complete!")
    print("Test with: curl -s -X POST http://localhost:8100 \\")
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"query": "pflanzliche Gerbung", "collection": "knowledge"}\'')
    print("=" * 60)


if __name__ == "__main__":
    run_migration()
