"""Semantic chunking for leather tanning recipe books.

Instead of splitting by fixed word count, this chunker uses the structured
OCR output to create semantically meaningful chunks:
- Each recipe = one chunk (with ingredients, process steps, notes)
- Procedure descriptions = own chunks
- Tables = own chunks with context header
- General text = overlapping chunks of ~150 tokens
"""
import uuid
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A semantically meaningful text chunk with metadata."""
    text: str
    chunk_type: str  # rezeptur, verfahren, tabelle, text
    page_number: int
    book_title: str = ""
    book_year: str = ""
    language: str = "de"
    recipe_name: str = ""
    leather_type: str = ""
    tanning_method: str = ""
    chemicals: list[str] = field(default_factory=list)
    source_image_path: str = ""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_metadata(self) -> dict:
        """Convert to ChromaDB-compatible metadata dict."""
        meta = {
            "chunk_id": self.chunk_id,
            "book_title": self.book_title,
            "book_year": self.book_year,
            "page_number": self.page_number,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "source_image_path": self.source_image_path,
        }
        if self.recipe_name:
            meta["recipe_name"] = self.recipe_name
        if self.leather_type:
            meta["leather_type"] = self.leather_type
        if self.tanning_method:
            meta["tanning_method"] = self.tanning_method
        if self.chemicals:
            meta["chemicals"] = ", ".join(self.chemicals)
        return meta


def _chunk_recipe(recipe: dict, page_number: int, image_path: str) -> Chunk:
    """Create a chunk from a structured recipe."""
    parts = []
    name = recipe.get("name", "Unbenannte Rezeptur")
    parts.append(f"Rezeptur: {name}")

    leather_type = recipe.get("leather_type", "")
    if leather_type:
        parts.append(f"Lederart: {leather_type}")

    tanning = recipe.get("tanning_method", "")
    if tanning:
        parts.append(f"Gerbverfahren: {tanning}")

    ingredients = recipe.get("ingredients", [])
    if ingredients:
        parts.append("\nZutaten:")
        for ing in ingredients:
            ing_name = ing.get("name", "")
            amount = ing.get("amount", "")
            unit = ing.get("unit", "")
            parts.append(f"  - {ing_name}: {amount} {unit}".strip())

    steps = recipe.get("process_steps", [])
    if steps:
        parts.append("\nVerfahren:")
        for i, step in enumerate(steps, 1):
            parts.append(f"  {i}. {step}")

    notes = recipe.get("notes", "")
    if notes:
        parts.append(f"\nHinweise: {notes}")

    chemicals = [ing.get("name", "") for ing in ingredients if ing.get("name")]

    return Chunk(
        text="\n".join(parts),
        chunk_type="rezeptur",
        page_number=page_number,
        recipe_name=name,
        leather_type=leather_type,
        tanning_method=tanning,
        chemicals=chemicals,
        source_image_path=image_path,
    )


def _chunk_table(table: dict | str, page_number: int, image_path: str, title: str = "") -> Chunk:
    """Create a chunk from a table."""
    if isinstance(table, str):
        text = table
    else:
        text = str(table)

    if title:
        text = f"{title}\n\n{text}"

    return Chunk(
        text=text,
        chunk_type="tabelle",
        page_number=page_number,
        source_image_path=image_path,
    )


def _chunk_text_overlapping(
    text: str, page_number: int, image_path: str,
    chunk_size: int = 150, overlap: int = 30
) -> list[Chunk]:
    """Split general text into overlapping chunks by word count."""
    words = text.split()
    if len(words) <= chunk_size:
        return [Chunk(
            text=text,
            chunk_type="text",
            page_number=page_number,
            source_image_path=image_path,
        )]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        if chunk_text.strip():
            chunks.append(Chunk(
                text=chunk_text,
                chunk_type="text",
                page_number=page_number,
                source_image_path=image_path,
            ))
        start = end - overlap

    return chunks


def _detect_language(text: str) -> str:
    """Simple heuristic to detect German vs English text."""
    de_words = {"der", "die", "das", "und", "ist", "ein", "eine", "fuer", "mit", "von",
                "werden", "wird", "bei", "nach", "durch", "gerbung", "leder"}
    words_lower = set(text.lower().split()[:100])
    de_count = len(words_lower & de_words)
    return "de" if de_count >= 3 else "en"


def chunk_page(page_result: dict, book_title: str = "", book_year: str = "") -> list[Chunk]:
    """Create semantic chunks from a single page result.

    Args:
        page_result: Dict from input_handler with keys:
            page_number, text, source_image_path, ocr_result
        book_title: Title of the source book.
        book_year: Publication year (estimated).

    Returns:
        List of Chunk objects for this page.
    """
    page_num = page_result.get("page_number", 0)
    image_path = page_result.get("source_image_path", "")
    ocr_result = page_result.get("ocr_result")
    raw_text = page_result.get("text", "")

    if not raw_text.strip():
        return []

    language = _detect_language(raw_text)
    chunks = []

    # If we have structured OCR output, use it for semantic chunking
    if ocr_result and isinstance(ocr_result, dict):
        # Extract recipes as individual chunks
        recipes = ocr_result.get("recipes", [])
        for recipe in recipes:
            if recipe and recipe.get("name"):
                chunk = _chunk_recipe(recipe, page_num, image_path)
                chunk.book_title = book_title
                chunk.book_year = book_year
                chunk.language = language
                chunks.append(chunk)

        # Extract tables as individual chunks
        tables = ocr_result.get("tables", [])
        title = ocr_result.get("title", "")
        for table in tables:
            if table:
                chunk = _chunk_table(table, page_num, image_path, title)
                chunk.book_title = book_title
                chunk.book_year = book_year
                chunk.language = language
                chunks.append(chunk)

        # If no structured content was extracted, fall back to text chunking
        if not chunks:
            page_type = ocr_result.get("page_type", "sonstiges")
            chunk_type = "verfahren" if page_type == "verfahrensbeschreibung" else "text"
            for chunk in _chunk_text_overlapping(raw_text, page_num, image_path):
                chunk.chunk_type = chunk_type
                chunk.book_title = book_title
                chunk.book_year = book_year
                chunk.language = language
                chunks.append(chunk)
    else:
        # No OCR structure (digital extraction): plain text chunking
        for chunk in _chunk_text_overlapping(raw_text, page_num, image_path):
            chunk.book_title = book_title
            chunk.book_year = book_year
            chunk.language = language
            chunks.append(chunk)

    return chunks


def chunk_book(
    page_results: list[dict], book_title: str = "", book_year: str = ""
) -> list[Chunk]:
    """Create semantic chunks for an entire book.

    Args:
        page_results: List of page result dicts from input_handler.
        book_title: Title of the book.
        book_year: Publication year.

    Returns:
        List of all Chunk objects for the book.
    """
    all_chunks = []
    for page_result in page_results:
        page_chunks = chunk_page(page_result, book_title, book_year)
        all_chunks.extend(page_chunks)

    return all_chunks
