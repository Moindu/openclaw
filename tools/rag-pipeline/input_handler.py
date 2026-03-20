"""Input handler for the book processing pipeline.

Detects input type and routes to the appropriate processing path:
- Single images (JPG/PNG/TIFF) -> OCR
- Scanned PDFs (images as PDF) -> extract pages as images -> OCR
- Digital PDFs (machine-readable text) -> direct text extraction (no OCR)
- Mixed PDFs (some pages digital, some scanned) -> per-page decision
- EPUBs/MOBIs -> direct text extraction
- Videos (MP4/MOV) -> frame extraction -> OCR
"""
import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import imagehash
import numpy as np
from PIL import Image

from config import BOOKS_DIR
from ocr import classify_document, ocr_image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS = {".pdf"}
EBOOK_EXTENSIONS = {".epub"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def detect_input_type(path: Path) -> str:
    """Detect the input type of a file or directory.

    Returns: 'images', 'pdf', 'ebook', 'video', or 'directory'.
    """
    if path.is_dir():
        return "directory"
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in EBOOK_EXTENSIONS:
        return "ebook"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unsupported input type: {path}")


def _ensure_book_dirs(book_name: str) -> dict[str, Path]:
    """Create the directory structure for a book."""
    base = Path(BOOKS_DIR)
    dirs = {
        "originals": base / "originals" / book_name,
        "extracted": base / "extracted" / book_name,
        "processed": base / "processed",
        "failed": base / "failed",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# --- Image quality check ---


def sharpness_score(image_path: Path) -> float:
    """Calculate Laplacian variance as sharpness score. Higher = sharper."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()


# --- PDF processing ---


def _page_has_text(page: fitz.Page, min_chars: int = 30) -> bool:
    """Check if a PDF page has extractable text (not scanned)."""
    text = page.get_text().strip()
    return len(text) >= min_chars


def process_pdf(
    pdf_path: Path, book_name: str
) -> list[dict]:
    """Process a PDF file, routing each page to OCR or direct extraction.

    Returns a list of page results, each containing:
    - page_number, text, source_image_path, extraction_method
    """
    dirs = _ensure_book_dirs(book_name)
    doc = fitz.open(str(pdf_path))
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_label = f"page-{page_num + 1:03d}"

        # Always render page image for reference and image embedding
        pix = page.get_pixmap(dpi=200)
        image_path = dirs["originals"] / f"{page_label}.jpg"
        pix.save(str(image_path))

        if _page_has_text(page):
            # Digital page: extract text directly (free, no API cost)
            text = page.get_text()
            results.append({
                "page_number": page_num + 1,
                "text": text,
                "source_image_path": str(image_path),
                "extraction_method": "digital",
                "ocr_result": None,
            })
            logger.info("Page %d: digital text extraction", page_num + 1)
        else:
            # Scanned page: OCR via Gemini
            try:
                ocr_result = ocr_image(image_path)
                results.append({
                    "page_number": page_num + 1,
                    "text": ocr_result.get("raw_text", ""),
                    "source_image_path": str(image_path),
                    "extraction_method": "ocr",
                    "ocr_result": ocr_result,
                })
                logger.info(
                    "Page %d: OCR (%s)",
                    page_num + 1,
                    ocr_result.get("_model_used", "unknown"),
                )
            except Exception as e:
                logger.error("Page %d: OCR failed: %s", page_num + 1, e)
                results.append({
                    "page_number": page_num + 1,
                    "text": "",
                    "source_image_path": str(image_path),
                    "extraction_method": "failed",
                    "ocr_result": None,
                    "error": str(e),
                })

    doc.close()
    return results


# --- EPUB processing ---


def process_epub(epub_path: Path, book_name: str) -> list[dict]:
    """Process an EPUB file by extracting text from chapters."""
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    dirs = _ensure_book_dirs(book_name)
    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    results = []
    chapter_num = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if text.strip() and len(text.strip()) > 50:
            chapter_num += 1
            title = item.get_name().rsplit("/", 1)[-1].rsplit(".", 1)[0]
            results.append({
                "page_number": chapter_num,
                "text": f"[Kapitel {chapter_num}: {title}]\n{text}",
                "source_image_path": None,
                "extraction_method": "digital",
                "ocr_result": None,
            })

    return results


# --- Image processing ---


def process_images(
    image_paths: list[Path], book_name: str
) -> list[dict]:
    """Process individual image files through OCR."""
    dirs = _ensure_book_dirs(book_name)
    results = []

    for i, img_path in enumerate(sorted(image_paths)):
        page_label = f"page-{i + 1:03d}"

        # Copy original to archive
        dest = dirs["originals"] / f"{page_label}{img_path.suffix}"
        if img_path != dest:
            shutil.copy2(img_path, dest)

        # Check sharpness
        score = sharpness_score(img_path)
        if score < 50:
            logger.warning("Low sharpness (%.1f) for %s", score, img_path.name)

        try:
            ocr_result = ocr_image(dest)
            results.append({
                "page_number": i + 1,
                "text": ocr_result.get("raw_text", ""),
                "source_image_path": str(dest),
                "extraction_method": "ocr",
                "ocr_result": ocr_result,
                "sharpness": score,
            })
        except Exception as e:
            logger.error("OCR failed for %s: %s", img_path.name, e)
            results.append({
                "page_number": i + 1,
                "text": "",
                "source_image_path": str(dest),
                "extraction_method": "failed",
                "error": str(e),
            })

    return results


# --- Video processing ---


def _perceptual_hash(image_path: Path) -> imagehash.ImageHash:
    """Compute perceptual hash for duplicate detection."""
    return imagehash.phash(Image.open(image_path))


def process_video(video_path: Path, book_name: str) -> list[dict]:
    """Extract frames from video, detect page changes, OCR unique pages.

    Uses FFmpeg for frame extraction and OpenCV for scene detection.
    """
    dirs = _ensure_book_dirs(book_name)
    frames_dir = dirs["originals"] / "_frames"
    frames_dir.mkdir(exist_ok=True)

    # Extract frames at 2 fps (enough for page-turning videos)
    logger.info("Extracting frames from %s...", video_path.name)
    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vf", "fps=2",
            "-q:v", "2",
            str(frames_dir / "frame-%04d.jpg"),
        ],
        capture_output=True,
        check=True,
    )

    frame_paths = sorted(frames_dir.glob("frame-*.jpg"))
    if not frame_paths:
        logger.error("No frames extracted from %s", video_path.name)
        return []

    # Scene detection: find frames where content changes significantly
    logger.info("Detecting page changes in %d frames...", len(frame_paths))
    prev_gray = None
    scenes = []  # list of (start_frame_idx, end_frame_idx)
    scene_start = 0
    threshold = 30.0  # Pixel difference threshold for scene change

    for i, fp in enumerate(frame_paths):
        gray = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            mean_diff = np.mean(diff)
            if mean_diff > threshold:
                scenes.append((scene_start, i - 1))
                scene_start = i
        prev_gray = gray

    scenes.append((scene_start, len(frame_paths) - 1))

    # For each scene, pick the sharpest frame
    logger.info("Found %d scenes, selecting best frames...", len(scenes))
    best_frames = []
    seen_hashes = set()

    for start, end in scenes:
        scene_frames = frame_paths[start : end + 1]
        if not scene_frames:
            continue

        # Find sharpest frame in scene
        best_path = max(scene_frames, key=sharpness_score)

        # Duplicate check via perceptual hash
        phash = _perceptual_hash(best_path)
        is_duplicate = any(phash - h < 5 for h in seen_hashes)
        if is_duplicate:
            continue
        seen_hashes.add(phash)
        best_frames.append(best_path)

    logger.info("Selected %d unique pages from video", len(best_frames))

    # Copy best frames and rename as pages
    page_paths = []
    for i, fp in enumerate(best_frames):
        dest = dirs["originals"] / f"page-{i + 1:03d}.jpg"
        shutil.copy2(fp, dest)
        page_paths.append(dest)

    # Clean up temp frames
    shutil.rmtree(frames_dir, ignore_errors=True)

    # OCR the unique pages
    return process_images(page_paths, book_name)


# --- Directory processing ---


def process_directory(dir_path: Path, book_name: str) -> list[dict]:
    """Process a directory of images."""
    image_files = sorted(
        p for p in dir_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No images found in {dir_path}")
    return process_images(image_files, book_name)


# --- Main dispatcher ---


def process_input(input_path: Path, book_name: str) -> list[dict]:
    """Main entry point: detect input type and process accordingly.

    Args:
        input_path: Path to file or directory.
        book_name: Human-readable name for the book.

    Returns:
        List of page results with text, image paths, and metadata.
    """
    input_type = detect_input_type(input_path)
    logger.info("Processing %s as %s: %s", input_path.name, input_type, book_name)

    if input_type == "directory":
        return process_directory(input_path, book_name)
    elif input_type == "image":
        return process_images([input_path], book_name)
    elif input_type == "pdf":
        return process_pdf(input_path, book_name)
    elif input_type == "ebook":
        return process_epub(input_path, book_name)
    elif input_type == "video":
        return process_video(input_path, book_name)
    else:
        raise ValueError(f"Unknown input type: {input_type}")
