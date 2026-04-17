"""
modules/cv_analyzer.py
Core analysis logic:
  - compute composite match score (semantic + skill coverage + experience)
  - find evidence chunks
  - detect present / missing skills via keyword matching
  - generate a short AI-driven candidate summary
  - extract experience years and project presence
  - generate quick-summary tags
  - detect red flags
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from config import (
    SKILL_KEYWORDS, TOP_K_EVIDENCE, WEIGHT_SEMANTIC, WEIGHT_SKILL,
    MAX_EXPERIENCE_YEARS, EXPERIENCE_NORMALIZATION_YEARS,
    MIN_CV_LENGTH, MAX_SKILLS_WITHOUT_PROJECT, MAX_SKILL_DENSITY,
)
from modules.embedding_manager import encode, top_k_chunks
from modules.pdf_processor import chunk_text


@dataclass
class CVResult:
    """Holds all analysis results for one CV."""

    filename: str
    full_text: str
    score: float                          # 0-1 composite score (default weights)
    semantic_score: float                 # 0-1 pure semantic similarity
    skill_score: float                    # 0-1 skill coverage fraction
    evidence: list[tuple[str, float]]     # [(chunk, similarity), ...]
    skills_found: list[str]
    skills_missing: list[str]
    summary: str = ""                     # short AI-driven candidate summary
    chunks: list[str] = field(default_factory=list)
    # ---- new fields ----
    experience_years: int = 0             # max years of experience found in CV
    experience_score: float = 0.0        # 0-1 normalised experience (÷5 years)
    has_projects: bool = False            # CV mentions concrete projects
    tags: list[str] = field(default_factory=list)        # quick-summary badges
    red_flags: list[str] = field(default_factory=list)   # quality warning flags


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
    experience_years: int = 0,
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
    exp_str = f"{experience_years} năm" if experience_years > 0 else "Không rõ"

    return (
        f"{verdict} "
        f"Độ tương đồng ngữ nghĩa: {sem_pct}%. "
        f"Bao phủ kỹ năng: {coverage_pct}% ({len(skills_found)}/{total if total else '?'}). "
        f"Kinh nghiệm: {exp_str}. "
        f"Kỹ năng có: {found_str}. "
        f"Thiếu: {missing_str}."
    )


# ---------------------------------------------------------------------------
# New helpers: experience, projects, tags, red flags
# ---------------------------------------------------------------------------

_CV_EXP_PATTERNS: list[str] = [
    r"(\d+)\+?\s*(?:năm|years?|yrs?)(?:\s*kinh\s*nghiệm|\s*experience)?",
    r"(\d+)\s*[-–]\s*\d+\s*(?:năm|years?)",
    r"kinh\s*nghiệm\s*(\d+)\s*năm",
    r"(\d+)\s*năm\s*kinh\s*nghiệm",
    r"over\s*(\d+)\s*(?:years?|yrs?)",
    r"more\s*than\s*(\d+)\s*(?:years?|yrs?)",
]

_PROJECT_KEYWORDS: list[str] = [
    "project", "dự án", "built", "developed", "xây dựng",
    "triển khai", "implement", "created", "launched", "designed",
    "phát triển", "xây", "làm dự án",
]


def _extract_experience_years(text: str) -> int:
    """Extract maximum stated years of experience from CV text."""
    max_years = 0
    for pat in _CV_EXP_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                y = int(m.group(1))
                if 0 < y <= MAX_EXPERIENCE_YEARS:
                    max_years = max(max_years, y)
            except (ValueError, IndexError):
                pass
    return max_years


def _check_has_projects(text: str) -> bool:
    """Return True if the CV text indicates at least one concrete project."""
    lower = text.lower()
    return any(kw in lower for kw in _PROJECT_KEYWORDS)


def _generate_tags(
    skills_found: list[str],
    skills_missing: list[str],
    experience_years: int,
    has_proj: bool,
    cv_text: str,
) -> list[str]:
    """
    Generate a list of quick-read tags for a candidate.

    Examples: "Senior", "Strong Python", "Missing Docker", "Has Projects"
    """
    tags: list[str] = []

    # Level tag derived from experience
    if experience_years >= 5:
        tags.append("Senior")
    elif experience_years >= 2:
        tags.append("Mid")
    elif experience_years == 1:
        tags.append("Junior")
    elif experience_years == 0:
        tags.append("Fresher/Unknown")

    # Project presence
    tags.append("Has Projects" if has_proj else "No Projects")

    # Strong skill: keyword appears ≥ 3 times in CV
    lower_cv = cv_text.lower()
    for skill in skills_found:
        keyword_list = SKILL_KEYWORDS.get(skill, [])
        count = sum(
            len(re.findall(r"\b" + re.escape(kw) + r"\b", lower_cv, re.IGNORECASE))
            for kw in keyword_list
        )
        if count >= 3:
            tags.append(f"Strong {skill}")

    # Missing top skills
    for skill in skills_missing[:3]:
        tags.append(f"Missing {skill}")

    return tags


def _detect_red_flags(
    cv_text: str,
    experience_years: int,
    has_proj: bool,
) -> list[str]:
    """
    Detect potential quality issues in a CV.

    Returns
    -------
    list[str]  – human-readable warning messages (empty = no red flags)
    """
    flags: list[str] = []
    clean = cv_text.strip()

    # Very short CV
    if len(clean) < MIN_CV_LENGTH:
        flags.append(f"CV rất ngắn (< {MIN_CV_LENGTH} ký tự)")

    # Many skills claimed but no projects described
    all_cv_skills = _extract_skills(cv_text, SKILL_KEYWORDS)
    if len(all_cv_skills) > MAX_SKILLS_WITHOUT_PROJECT and not has_proj:
        flags.append(f"Nhiều kỹ năng (>{MAX_SKILLS_WITHOUT_PROJECT}) nhưng không có dự án thực tế")

    # Abnormally high skill density relative to text length
    word_count = len(clean.split())
    if word_count > 0 and len(all_cv_skills) / max(word_count, 1) > MAX_SKILL_DENSITY:
        flags.append("Mật độ kỹ năng bất thường (có thể liệt kê thiếu context)")

    # No experience information at all
    if experience_years == 0:
        flags.append("Không tìm thấy thông tin số năm kinh nghiệm")

    return flags


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

    semantic_score    = mean of top-3 chunk cosine similarities
    skill_score       = len(skills_found) / len(jd_skills)  (0 if no JD skills)
    experience_score  = min(1.0, experience_years / 5.0)   (stored, used by UI)

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
        exp_years = _extract_experience_years(cv_text)
        exp_score = min(1.0, exp_years / 5.0)
        has_proj = _check_has_projects(cv_text)
        tags = _generate_tags([], jd_skills, exp_years, has_proj, cv_text)
        red_flags = _detect_red_flags(cv_text, exp_years, has_proj)
        summary = _build_summary(filename, [], jd_skills, 0.0, 0.0, exp_years)
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
            experience_years=exp_years,
            experience_score=exp_score,
            has_projects=has_proj,
            tags=tags,
            red_flags=red_flags,
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

    # --- experience ---
    experience_years = _extract_experience_years(cv_text)
    experience_score = min(1.0, experience_years / EXPERIENCE_NORMALIZATION_YEARS)

    # --- project detection ---
    has_projects = _check_has_projects(cv_text)

    # --- composite score (default weights; UI can recompute with custom weights) ---
    if jd_skills:
        composite = WEIGHT_SEMANTIC * semantic_score + WEIGHT_SKILL * skill_score
    else:
        # No JD skills detected → rely entirely on semantic similarity
        composite = semantic_score

    composite = float(np.clip(composite, 0.0, 1.0))

    # --- tags & red flags ---
    tags = _generate_tags(skills_found, skills_missing, experience_years, has_projects, cv_text)
    red_flags = _detect_red_flags(cv_text, experience_years, has_projects)

    summary = _build_summary(filename, skills_found, skills_missing, semantic_score, skill_score, experience_years)

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
        experience_years=experience_years,
        experience_score=experience_score,
        has_projects=has_projects,
        tags=tags,
        red_flags=red_flags,
    )


def rank_results(results: list[CVResult]) -> list[CVResult]:
    """Return results sorted by composite score descending."""
    return sorted(results, key=lambda r: r.score, reverse=True)
