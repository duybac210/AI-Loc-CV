"""
modules/cv_analyzer.py
Core analysis logic:
  - compute composite match score (semantic + skill coverage)
  - find evidence chunks
  - detect present / missing skills via keyword matching
  - generate a short AI-driven candidate summary
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from config import SKILL_KEYWORDS, TOP_K_EVIDENCE, WEIGHT_SEMANTIC, WEIGHT_SKILL
from modules.embedding_manager import encode, top_k_chunks
from modules.pdf_processor import chunk_text


@dataclass
class CVResult:
    """Holds all analysis results for one CV."""

    filename: str
    full_text: str
    score: float                          # 0-1 composite score
    semantic_score: float                 # 0-1 pure semantic similarity
    skill_score: float                    # 0-1 skill coverage fraction
    evidence: list[tuple[str, float]]     # [(chunk, similarity), ...]
    skills_found: list[str]
    skills_missing: list[str]
    summary: str = ""                     # short AI-driven candidate summary
    chunks: list[str] = field(default_factory=list)


def _extract_skills(text: str, skill_keywords: dict[str, list[str]]) -> list[str]:
    """
    Scan *text* for skill keywords.

    Returns
    -------
    list[str]  – names of skills detected in text
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


def _build_summary(
    filename: str,
    skills_found: list[str],
    skills_missing: list[str],
    semantic_score: float,
    skill_score: float,
) -> str:
    """
    Produce a short human-readable summary for a candidate.
    This is rule-based (no external LLM needed).
    """
    name = filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()
    total = len(skills_found) + len(skills_missing)
    coverage_pct = round(skill_score * 100)
    sem_pct = round(semantic_score * 100)

    if skill_score >= 0.8 and semantic_score >= 0.65:
        verdict = "Ứng viên xuất sắc – rất phù hợp với yêu cầu công việc."
    elif skill_score >= 0.5 and semantic_score >= 0.50:
        verdict = "Ứng viên tiềm năng – đáp ứng phần lớn yêu cầu."
    elif skill_score >= 0.3 or semantic_score >= 0.40:
        verdict = "Ứng viên có thể cần đào tạo thêm một số kỹ năng."
    else:
        verdict = "Ứng viên chưa đáp ứng được yêu cầu tối thiểu."

    found_str = (", ".join(skills_found[:5]) + ("…" if len(skills_found) > 5 else "")) if skills_found else "—"
    missing_str = (", ".join(skills_missing[:3]) + ("…" if len(skills_missing) > 3 else "")) if skills_missing else "—"

    return (
        f"{verdict} "
        f"Độ tương đồng ngữ nghĩa: {sem_pct}%. "
        f"Bao phủ kỹ năng: {coverage_pct}% ({len(skills_found)}/{total if total else '?'}). "
        f"Kỹ năng có: {found_str}. "
        f"Thiếu: {missing_str}."
    )


def analyze_cv(
    filename: str,
    cv_text: str,
    jd_embedding: np.ndarray,
    jd_skills: list[str],
    k_evidence: int = TOP_K_EVIDENCE,
) -> CVResult:
    """
    Analyse a single CV against the job description embedding.

    Composite score
    ---------------
    score = WEIGHT_SEMANTIC * semantic_score + WEIGHT_SKILL * skill_score

    semantic_score = mean of top-3 chunk cosine similarities
    skill_score    = len(skills_found) / len(jd_skills)  (0 if no JD skills)

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
        summary = _build_summary(filename, [], jd_skills, 0.0, 0.0)
        return CVResult(
            filename=filename,
            full_text=cv_text,
            score=0.0,
            semantic_score=0.0,
            skill_score=0.0,
            evidence=[],
            skills_found=[],
            skills_missing=jd_skills[:],
            summary=summary,
            chunks=[],
        )

    chunk_embeddings = encode(chunks)

    # --- semantic score: mean of top-k chunk similarities ---
    k_for_score = min(3, len(chunks))
    raw_scores = chunk_embeddings @ jd_embedding
    top_scores = np.sort(raw_scores)[::-1][:k_for_score]
    semantic_score = float(np.clip(np.mean(top_scores), 0.0, 1.0))

    # --- evidence ---
    evidence = top_k_chunks(jd_embedding, chunk_embeddings, chunks, k=k_evidence)

    # --- skill detection ---
    cv_skills = _extract_skills(cv_text, SKILL_KEYWORDS)
    skills_found = [s for s in cv_skills if s in jd_skills]
    skills_missing = [s for s in jd_skills if s not in cv_skills]

    # --- skill coverage score ---
    if jd_skills:
        skill_score = len(skills_found) / len(jd_skills)
    else:
        skill_score = 0.0

    # --- composite score ---
    if jd_skills:
        composite = WEIGHT_SEMANTIC * semantic_score + WEIGHT_SKILL * skill_score
    else:
        # No JD skills detected → rely entirely on semantic similarity
        composite = semantic_score

    composite = float(np.clip(composite, 0.0, 1.0))

    summary = _build_summary(filename, skills_found, skills_missing, semantic_score, skill_score)

    return CVResult(
        filename=filename,
        full_text=cv_text,
        score=composite,
        semantic_score=semantic_score,
        skill_score=skill_score,
        evidence=evidence,
        skills_found=skills_found,
        skills_missing=skills_missing,
        summary=summary,
        chunks=chunks,
    )


def rank_results(results: list[CVResult]) -> list[CVResult]:
    """Return results sorted by composite score descending."""
    return sorted(results, key=lambda r: r.score, reverse=True)
