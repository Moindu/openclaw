"""CLI interface for the Rezepturbuecher-RAG-Pipeline.

Usage:
    python pipeline.py process --input /data/books/incoming/buch/ --book-name "Hein 1923"
    python pipeline.py process --input /data/books/incoming/video.mp4 --book-name "Stroehlein"
    python pipeline.py status
    python pipeline.py search --query "pflanzliche Gerbung Rindsleder"
    python pipeline.py search --image /path/to/foto.jpg
    python pipeline.py migrate  # One-time: re-index all with Gemini Embedding 2
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def cmd_process(args):
    """Process a book through the full pipeline."""
    from chunker import chunk_book
    from config import COLLECTION_RECIPES
    from indexer import get_chroma_client, get_or_create_collection, index_recipe_chunks
    from input_handler import process_input

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)

    book_name = args.book_name
    book_year = args.book_year or ""

    logger.info("=" * 60)
    logger.info("Processing: %s", book_name)
    logger.info("Input: %s", input_path)
    logger.info("=" * 60)

    # Step 1: Extract text
    logger.info("[1/3] Extracting text...")
    page_results = process_input(input_path, book_name)
    logger.info("Extracted %d pages", len(page_results))

    # Save extracted text as JSON
    from config import BOOKS_DIR
    extracted_dir = Path(BOOKS_DIR) / "extracted" / book_name
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with open(extracted_dir / "pages.json", "w", encoding="utf-8") as f:
        json.dump(page_results, f, ensure_ascii=False, indent=2)
    logger.info("Saved extracted text to %s", extracted_dir / "pages.json")

    # Step 2: Semantic chunking
    logger.info("[2/3] Creating semantic chunks...")
    chunks = chunk_book(page_results, book_title=book_name, book_year=book_year)
    logger.info("Created %d chunks", len(chunks))

    # Log chunk types
    type_counts = {}
    for c in chunks:
        type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
    for ct, count in sorted(type_counts.items()):
        logger.info("  %s: %d", ct, count)

    # Step 3: Embed and store
    logger.info("[3/3] Embedding and indexing...")
    client = get_chroma_client()
    collection = get_or_create_collection(client, COLLECTION_RECIPES)
    stats = index_recipe_chunks(chunks, collection)

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("  Text chunks indexed: %d", stats["text_chunks"])
    logger.info("  Image embeddings: %d", stats["image_embeddings"])
    logger.info("  Errors: %d", stats["errors"])
    logger.info("=" * 60)

    # Move input to processed
    processed_dir = Path(BOOKS_DIR) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    if input_path.is_file():
        import shutil
        dest = processed_dir / input_path.name
        shutil.move(str(input_path), str(dest))
        logger.info("Moved input to %s", dest)


def cmd_search(args):
    """Search the recipe database."""
    from search import (
        format_context,
        format_recipe_results,
        search_all,
        search_recipes,
        search_recipes_by_image,
    )

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error("Image not found: %s", image_path)
            sys.exit(1)
        logger.info("Bildsuche: %s", image_path)
        results = search_recipes_by_image(image_path, n_results=args.n_results)
        print(format_recipe_results(results))
    elif args.query:
        logger.info("Suche: %s", args.query)
        if args.collection == "recipes":
            filters = {}
            if args.leather_type:
                filters["leather_type"] = args.leather_type
            if args.tanning_method:
                filters["tanning_method"] = args.tanning_method
            results = search_recipes(
                args.query, n_results=args.n_results,
                filters=filters if filters else None,
            )
            print(format_recipe_results(results))
        else:
            results = search_all(args.query, n_results=args.n_results)
            print(format_context(results))
    else:
        logger.error("--query or --image required")
        sys.exit(1)


def cmd_status(args):
    """Show pipeline status."""
    from config import BOOKS_DIR, COLLECTION_RECIPES
    from indexer import get_chroma_client

    client = get_chroma_client()

    print("Collections:")
    for name in ["knowledge", "products", COLLECTION_RECIPES]:
        try:
            col = client.get_collection(name)
            print(f"  {name}: {col.count()} entries")
        except Exception:
            print(f"  {name}: not found")

    books_dir = Path(BOOKS_DIR)
    originals = books_dir / "originals"
    if originals.exists():
        books = [d.name for d in originals.iterdir() if d.is_dir()]
        print(f"\nProcessed books ({len(books)}):")
        for b in sorted(books):
            pages = len(list((originals / b).glob("page-*")))
            print(f"  {b}: {pages} pages")
    else:
        print(f"\nNo books processed yet (dir: {books_dir})")

    incoming = books_dir / "incoming"
    if incoming.exists():
        files = list(incoming.rglob("*"))
        files = [f for f in files if f.is_file()]
        if files:
            print(f"\nPending in incoming/ ({len(files)} files):")
            for f in sorted(files)[:10]:
                print(f"  {f.name}")


def cmd_migrate(args):
    """Run the one-time embedding migration."""
    from migrate import run_migration
    run_migration()


def main():
    parser = argparse.ArgumentParser(
        description="Rezepturbuecher-RAG-Pipeline fuer Kobel Gerberei"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # process
    p_process = subparsers.add_parser("process", help="Process a book or document")
    p_process.add_argument("--input", "-i", required=True, help="Path to file or directory")
    p_process.add_argument("--book-name", "-n", required=True, help="Name of the book")
    p_process.add_argument("--book-year", "-y", default="", help="Publication year")
    p_process.set_defaults(func=cmd_process)

    # search
    p_search = subparsers.add_parser("search", help="Search the knowledge base")
    p_search.add_argument("--query", "-q", help="Search query text")
    p_search.add_argument("--image", help="Search by image path")
    p_search.add_argument("--collection", "-c", default="recipes",
                          choices=["all", "recipes", "knowledge", "products"])
    p_search.add_argument("--n-results", "-n", type=int, default=5)
    p_search.add_argument("--leather-type", help="Filter by leather type")
    p_search.add_argument("--tanning-method", help="Filter by tanning method")
    p_search.set_defaults(func=cmd_search)

    # status
    p_status = subparsers.add_parser("status", help="Show pipeline status")
    p_status.set_defaults(func=cmd_status)

    # migrate
    p_migrate = subparsers.add_parser("migrate", help="Run embedding migration")
    p_migrate.set_defaults(func=cmd_migrate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
