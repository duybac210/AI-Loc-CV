"""
modules/pdf_processor.py
Extracts raw text from PDF files uploaded via Streamlit (BytesIO or file path).
Uses pdfplumber as primary extractor (better layout & table handling) with
PyPDF2 as fallback.  Splits text into overlapping chunks for semantic search.
"""
from __future__ import annotations

import io
import logging
import re
from typing import Union

import pdfplumber
import PyPDF2

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def extract_text_from_pdf(source: Union[bytes, str]) -> str:
    """
    Extract all text from a PDF.

    Strategy
    --------
    1. Try pdfplumber (handles more layout types, tables, rotated pages)
    2. If pdfplumber yields < 50 chars, fall back to PyPDF2
    3. Return error string if both fail (no silent empty result)

    Parameters
    ----------
    source : bytes | str
        Raw bytes of the PDF (from st.file_uploader) or a file path string.

    Returns
    -------
    str
        Concatenated text from all pages, or an error description on failure.
    """
    raw: bytes
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        with open(source, "rb") as fh:
            raw = fh.read()

    # --- attempt 1: pdfplumber ---
    text = _extract_pdfplumber(raw)

    # --- attempt 2: PyPDF2 fallback ---
    if len(text.strip()) < 50:
        text_fallback = _extract_pypdf2(raw)
        if len(text_fallback.strip()) > len(text.strip()):
            text = text_fallback

    if not text.strip():
        return "[PDF extraction failed: the document may be a scanned image or encrypted.]"

    return text


def _extract_pdfplumber(raw: bytes) -> str:
    """Extract text using pdfplumber."""
    try:
        pages: list[str] = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                pages.append(page_text)
        return "\n".join(pages)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfplumber extraction failed: %s", exc)
        return ""


def _extract_pypdf2(raw: bytes) -> str:
    """Extract text using PyPDF2 (fallback)."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(raw))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PyPDF2 extraction failed: %s", exc)
        return ""


def clean_text(text: str) -> str:
    """
    Normalise whitespace and remove repeated blank lines while keeping
    meaningful line breaks that separate CV sections.
    """
    # Collapse runs of spaces/tabs to a single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ consecutive newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping character-level chunks, preferring to
    break at sentence/paragraph boundaries when possible.

    Parameters
    ----------
    text       : str
    chunk_size : int  – target chunk length in characters
    overlap    : int  – how many characters the next chunk shares with the previous

    Returns
    -------
    list[str]  – list of non-empty text chunks
    """
    text = clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at a natural boundary (newline or sentence end)
        if end < len(text):
            boundary = _find_boundary(text, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end])
        start = end - overlap

    return [c.strip() for c in chunks if c.strip()]


def _find_boundary(text: str, pos: int, window: int = 80) -> int:
    """
    Search backwards from *pos* (up to *window* chars) for the nearest
    sentence/paragraph boundary.  Returns *pos* if nothing found.
    """
    search_start = max(0, pos - window)
    segment = text[search_start:pos]
    # Prefer double newline, then single newline, then sentence end
    for delimiter in ("\n\n", "\n", ". ", "! ", "? "):
        idx = segment.rfind(delimiter)
        if idx != -1:
            return search_start + idx + len(delimiter)
    return pos
