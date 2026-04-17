"""
modules/cv_analyzer.py
Core analysis logic:
  - compute overall semantic match score between JD and a CV
  - find evidence chunks
  - detect present / missing skills via keyword matching
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from config import SKILL_KEYWORDS, TOP_K_EVIDENCE
from modules.embedding_manager import encode, top_k_chunks
from modules.pdf_processor import chunk_text


@dataclass
class CVResult:
    """Holds all analysis results for one CV."""

    filename: str
    full_text: str
    score: float                          # 0-1 overall semantic match
    evidence: list[tuple[str, float]]     # [(chunk, similarity), ...]
    skills_found: list[str]
    skills_missing: list[str]
    chunks: list[str] = field(default_factory=list)


def _extract_skills(text: str, skill_keywords: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """
    Scan *text* for skill keywords.

    Returns
    -------
    (found_skills, missing_skills_from_jd)  – both lists of skill names
    NOTE: 'missing' is computed later by the caller once the JD skill list is known.
    """
    lower_text = text.lower()
    found: list[str] = []
    for skill_name, keywords in skill_keywords.items():
        for kw in keywords:
            pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            if pattern.search(lower_text):
                found.append(skill_name)
                break
    return found


def extract_jd_skills(jd_text: str) -> list[str]:
    """Return which skills from SKILL_KEYWORDS appear in the Job Description."""
    return _extract_skills(jd_text, SKILL_KEYWORDS)


def analyze_cv(
    filename: str,
    cv_text: str,
    jd_embedding: np.ndarray,
    jd_skills: list[str],
    k_evidence: int = TOP_K_EVIDENCE,
) -> CVResult:
    """
    Analyse a single CV against the job description embedding.

    Parameters
    ----------
    filename      : str         original file name
    cv_text       : str         raw extracted text of the CV
    jd_embedding  : np.ndarray  embedding of the full JD text (1-D, normalised)
    jd_skills     : list[str]   skills detected in the JD
    k_evidence    : int         number of evidence chunks to return

    Returns
    -------
    CVResult
    """
    # --- chunk & embed CV ---
    chunks = chunk_text(cv_text)
    if not chunks:
        return CVResult(
            filename=filename,
            full_text=cv_text,
            score=0.0,
            evidence=[],
            skills_found=[],
            skills_missing=jd_skills[:],
            chunks=[],
        )

    chunk_embeddings = encode(chunks)

    # --- overall score: mean of top-3 chunk similarities ---
    k_for_score = min(3, len(chunks))
    scores = chunk_embeddings @ jd_embedding
    top_scores = np.sort(scores)[::-1][:k_for_score]
    overall_score = float(np.mean(top_scores))
    # clamp to [0, 1]
    overall_score = max(0.0, min(1.0, overall_score))

    # --- evidence ---
    evidence = top_k_chunks(jd_embedding, chunk_embeddings, chunks, k=k_evidence)

    # --- skill detection ---
    cv_skills_found = _extract_skills(cv_text, SKILL_KEYWORDS)
    # Keep only skills that are also required by JD
    skills_found = [s for s in cv_skills_found if s in jd_skills]
    skills_missing = [s for s in jd_skills if s not in cv_skills_found]

    return CVResult(
        filename=filename,
        full_text=cv_text,
        score=overall_score,
        evidence=evidence,
        skills_found=skills_found,
        skills_missing=skills_missing,
        chunks=chunks,
    )


def rank_results(results: list[CVResult]) -> list[CVResult]:
    """Return results sorted by score descending."""
    return sorted(results, key=lambda r: r.score, reverse=True)
