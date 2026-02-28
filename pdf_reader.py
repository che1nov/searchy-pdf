"""PDF extraction helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(file_path: Path, logger: logging.Logger) -> str | None:
    """Extract text from a PDF file. Returns None for unreadable PDFs."""
    try:
        reader = PdfReader(str(file_path))
        chunks: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                chunks.append(text)
    except Exception as exc:
        logger.warning("Failed to read PDF %s: %s", file_path, exc)
        return None

    full_text = "\n".join(chunks).strip()
    return full_text or None
