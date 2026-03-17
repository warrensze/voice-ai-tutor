import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

try:
    import pypdfium2 as pdfium
except Exception:  # pragma: no cover - optional dependency resolution
    pdfium = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency resolution
    pytesseract = None

PAGE_MARKER_TEMPLATE = "=== PAGE {page_number} ==="


@dataclass
class OCRResult:
    input_pdf: Path
    output_txt: Path
    pages_total: int
    pages_processed: int
    pages_with_text: int


def default_ocr_output_path(input_pdf: Path) -> Path:
    """Create a sidecar text path for OCR output."""
    return input_pdf.with_name(f"{input_pdf.stem}.ocr.txt")


def _require_ocr_dependencies():
    if pdfium is None or pytesseract is None:
        raise RuntimeError(
            "Missing OCR dependencies. Install with: pip install pypdfium2 pytesseract pillow"
        )


def _ensure_tesseract_available():
    try:
        pytesseract.get_tesseract_version()
    except Exception as error:
        raise RuntimeError(
            "Tesseract executable is not available. Install Tesseract OCR or pass "
            "--tesseract-cmd with the full executable path."
        ) from error


def ocr_pdf_to_text(
    input_pdf: str | Path,
    output_txt: str | Path | None = None,
    *,
    dpi: int = 220,
    language: str = "eng",
    max_pages: int | None = None,
    overwrite: bool = False,
    tesseract_cmd: str | None = None,
) -> OCRResult:
    """Run OCR over a PDF and write extracted text into a sidecar .ocr.txt file."""
    _require_ocr_dependencies()

    source_pdf = Path(input_pdf)
    if not source_pdf.exists():
        raise FileNotFoundError(f"PDF not found: {source_pdf}")

    target_txt = Path(output_txt) if output_txt else default_ocr_output_path(source_pdf)
    if target_txt.exists() and not overwrite:
        raise FileExistsError(
            f"OCR output already exists: {target_txt}. Use --overwrite to replace it."
        )

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    _ensure_tesseract_available()

    target_txt.parent.mkdir(parents=True, exist_ok=True)

    document = pdfium.PdfDocument(str(source_pdf))
    page_total = len(document)
    page_limit = page_total if max_pages is None else min(page_total, max_pages)
    processed = 0
    pages_with_text = 0

    with target_txt.open("w", encoding="utf-8") as output_file:
        for page_index in range(page_limit):
            page = document[page_index]
            bitmap = page.render(scale=dpi / 72)
            image = bitmap.to_pil()

            extracted = pytesseract.image_to_string(image, lang=language)
            cleaned = (extracted or "").strip()
            if cleaned:
                pages_with_text += 1

            output_file.write(PAGE_MARKER_TEMPLATE.format(page_number=page_index + 1))
            output_file.write("\n")
            output_file.write(cleaned)
            output_file.write("\n\n")

            processed += 1
            image.close()
            bitmap.close()
            page.close()

    return OCRResult(
        input_pdf=source_pdf,
        output_txt=target_txt,
        pages_total=page_total,
        pages_processed=processed,
        pages_with_text=pages_with_text,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="OCR a PDF into embeddable text")
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument("--output", help="Path to output .txt file")
    parser.add_argument("--dpi", type=int, default=220, help="Render DPI for OCR")
    parser.add_argument("--language", default="eng", help="OCR language code")
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Optional page limit for quick tests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing OCR output file",
    )
    parser.add_argument(
        "--tesseract-cmd",
        help="Optional full path to tesseract executable",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        result = ocr_pdf_to_text(
            input_pdf=args.input,
            output_txt=args.output,
            dpi=args.dpi,
            language=args.language,
            max_pages=args.max_pages,
            overwrite=args.overwrite,
            tesseract_cmd=args.tesseract_cmd,
        )
    except Exception as error:
        print(f"[OCR] Failed: {error}")
        sys.exit(1)

    print(f"[OCR] Input: {result.input_pdf}")
    print(f"[OCR] Output: {result.output_txt}")
    print(f"[OCR] Pages processed: {result.pages_processed}/{result.pages_total}")
    print(f"[OCR] Pages with text: {result.pages_with_text}")


if __name__ == "__main__":
    main()
