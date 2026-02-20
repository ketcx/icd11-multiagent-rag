"""Extracts structured text from the ICD-11 PDF preserving metadata."""

import fitz  # type: ignore # PyMuPDF
import re
from pathlib import Path
from typing import Generator

# from core.schemas.session import DocumentChunk # Will be defined later


def _extract_headings(blocks: list[dict]) -> list[str]:
    # Placeholder for actual heading detection logic
    headings = []
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                for span in line.get("spans", []):
                    if span.get("size", 0) > 12:  # Example threshold
                        headings.append(span["text"].strip())
    return headings


def _extract_cie11_codes(text: str) -> list[str]:
    # Detect ICD-11 codes (regex: \d[A-Z]\d{2}(\.\d+)?)
    pattern = r"\d[A-Z]\d{2}(?:\.\d+)?"
    return re.findall(pattern, text)


def extract_pages(pdf_path: Path) -> Generator[dict, None, None]:
    """Extracts text page by page with metadata.

    Yields:
        dict with keys: page_number, text, headings, codes
    """
    doc = fitz.open(str(pdf_path))
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")
        blocks = page.get_text("dict")["blocks"]

        # Detect headings based on font size
        headings = _extract_headings(blocks)
        # Detect ICD-11 codes
        codes = _extract_cie11_codes(text)

        yield {
            "page_number": page_num + 1,
            "text": text,
            "headings": headings,
            "codes": codes,
            "source_pdf": pdf_path.name,
        }
    doc.close()
