"""
modules/pdf_processor.py
Extracts raw text from PDF files uploaded via Streamlit (BytesIO or file path).
Splits text into overlapping chunks for semantic search.
"""
from __future__ import annotations

import io
import re
from typing import Union

import PyPDF2

from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(source: Union[bytes, str]) -> str:
    """
    Extract all text from a PDF.

    Parameters
    ----------
    source : bytes | str
        Raw bytes of the PDF (from st.file_uploader) or a file path string.

    Returns
    -------
    str
        Concatenated text from all pages, or empty string on failure.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            file_like = io.BytesIO(source)
        else:
            file_like = open(source, "rb")

        reader = PyPDF2.PdfReader(file_like)
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        if isinstance(source, str):
            file_like.close()

        return "\n".join(pages)
    except Exception as exc:  # noqa: BLE001
        return f"[PDF extraction error: {exc}]"


def clean_text(text: str) -> str:
    """Basic whitespace normalisation."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping character-level chunks.

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
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return [c.strip() for c in chunks if c.strip()]
