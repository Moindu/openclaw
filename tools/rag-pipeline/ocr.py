"""OCR module using Gemini with Smart Model Routing.

Routes documents to the most cost-effective model based on complexity:
- Simple/modern print -> Gemini 3.1 Flash-Lite (cheap)
- Fraktur/handwriting/poor quality -> Gemini 3 Flash (better)
"""
import json
import logging
from pathlib import Path

from google import genai

from config import (
    GOOGLE_API_KEY,
    OCR_MODEL_CLASSIFIER,
    OCR_MODEL_COMPLEX,
    OCR_MODEL_STANDARD,
)

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def _load_image_part(image_path: Path) -> genai.types.Part:
    """Load an image file as a Gemini Part."""
    suffix = image_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(suffix, "image/jpeg")
    return genai.types.Part.from_bytes(
        data=image_path.read_bytes(), mime_type=mime_type
    )


def classify_document(image_path: Path) -> dict:
    """Classify a document image to determine OCR routing.

    Uses Gemini 3.1 Flash-Lite for cheap classification (~$0.001 per image).
    Returns classification dict with print_type, has_tables, etc.
    """
    client = _get_client()
    image_part = _load_image_part(image_path)

    result = client.models.generate_content(
        model=OCR_MODEL_CLASSIFIER,
        contents=[
            image_part,
            (
                "Klassifiziere dieses Dokumentbild. Antworte NUR mit einem JSON:\n"
                "{\n"
                '  "print_type": "modern_print|old_print|fraktur|handwriting|mixed",\n'
                '  "has_tables": true/false,\n'
                '  "has_formulas": true/false,\n'
                '  "image_quality": "good|medium|poor",\n'
                '  "confidence": 0.0-1.0\n'
                "}"
            ),
        ],
        config=genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
        ),
    )

    try:
        return json.loads(result.text)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Classification failed for %s, defaulting to complex model", image_path.name)
        return {
            "print_type": "mixed",
            "has_tables": True,
            "has_formulas": False,
            "image_quality": "medium",
            "confidence": 0.0,
        }


def _select_ocr_model(classification: dict) -> str:
    """Select the OCR model based on document classification."""
    print_type = classification.get("print_type", "mixed")
    quality = classification.get("image_quality", "medium")
    confidence = classification.get("confidence", 0.0)

    # Use complex model for difficult documents
    if print_type in ("fraktur", "handwriting", "mixed"):
        return OCR_MODEL_COMPLEX
    if quality == "poor":
        return OCR_MODEL_COMPLEX
    if confidence < 0.7:
        return OCR_MODEL_COMPLEX

    # Standard model for clear, modern print
    return OCR_MODEL_STANDARD


OCR_PROMPT = """\
Du bist ein Experte fuer historische und moderne Gerberei-Fachliteratur.
Extrahiere den VOLLSTAENDIGEN Text dieser Buchseite. Bewahre exakt:
- Struktur (Ueberschriften, Absaetze, Listen)
- Tabellen (als Markdown-Tabellen)
- Chemische Formeln und Mengenangaben (exakte Zahlen und Einheiten)
- Fachbegriffe der Lederherstellung (auch veraltete deutsche Begriffe)
- Handschriftliche Randnotizen (markiere als [Handnotiz: ...])

Falls die Schrift Fraktur/Kurrentschrift ist, transkribiere in moderne Lateinschrift.

Gib den extrahierten Text als strukturiertes JSON zurueck:
{
  "page_type": "rezeptur|verfahrensbeschreibung|tabelle|inhaltsverzeichnis|sonstiges",
  "title": "Ueberschrift der Seite falls vorhanden",
  "raw_text": "Der vollstaendige extrahierte Text",
  "recipes": [
    {
      "name": "Name der Rezeptur",
      "leather_type": "Lederart",
      "tanning_method": "Gerbverfahren",
      "ingredients": [
        {"name": "Zutat", "amount": "Menge", "unit": "Einheit"}
      ],
      "process_steps": ["Schritt 1", "Schritt 2"],
      "notes": "Besondere Hinweise"
    }
  ],
  "tables": [],
  "handwritten_notes": []
}"""


def ocr_image(image_path: Path, classification: dict | None = None) -> dict:
    """Perform OCR on a single image using the appropriate Gemini model.

    Args:
        image_path: Path to the image file.
        classification: Pre-computed classification (skips classify step if provided).

    Returns:
        Structured JSON dict with extracted text and metadata.
    """
    if classification is None:
        classification = classify_document(image_path)

    model = _select_ocr_model(classification)
    logger.info(
        "OCR: %s -> %s (type=%s, quality=%s)",
        image_path.name,
        model,
        classification.get("print_type"),
        classification.get("image_quality"),
    )

    client = _get_client()
    image_part = _load_image_part(image_path)

    result = client.models.generate_content(
        model=model,
        contents=[image_part, OCR_PROMPT],
        config=genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
        ),
    )

    try:
        ocr_result = json.loads(result.text)
    except (json.JSONDecodeError, AttributeError):
        logger.error("OCR JSON parse failed for %s, returning raw text", image_path.name)
        ocr_result = {
            "page_type": "sonstiges",
            "title": "",
            "raw_text": result.text if result.text else "",
            "recipes": [],
            "tables": [],
            "handwritten_notes": [],
        }

    ocr_result["_classification"] = classification
    ocr_result["_model_used"] = model
    return ocr_result


def ocr_images_batch(
    image_paths: list[Path], skip_classification: bool = False
) -> list[dict]:
    """Process multiple images through OCR.

    Args:
        image_paths: List of image file paths.
        skip_classification: If True, use complex model for all (safer but costlier).

    Returns:
        List of OCR result dicts, one per image.
    """
    results = []
    for path in image_paths:
        try:
            if skip_classification:
                classification = {"print_type": "mixed", "confidence": 0.0}
            else:
                classification = classify_document(path)
            result = ocr_image(path, classification)
            results.append(result)
        except Exception as e:
            logger.error("OCR failed for %s: %s", path.name, e)
            results.append({
                "page_type": "error",
                "title": "",
                "raw_text": "",
                "recipes": [],
                "tables": [],
                "handwritten_notes": [],
                "_error": str(e),
                "_source": str(path),
            })
    return results
