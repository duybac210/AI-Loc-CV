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
  - extract contact info (name, email, phone)
  - must-have / nice-to-have weighted scoring
  - section-aware experience and skill scoring
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from config import (
    SKILL_KEYWORDS, TOP_K_EVIDENCE, WEIGHT_SEMANTIC, WEIGHT_SKILL,
    MAX_EXPERIENCE_YEARS, EXPERIENCE_NORMALIZATION_YEARS,
    MIN_CV_LENGTH, MAX_SKILLS_WITHOUT_PROJECT, MAX_SKILL_DENSITY,
)
from modules.embedding_manager import encode, top_k_chunks
from modules.pdf_processor import chunk_text

if TYPE_CHECKING:
    from modules.jd_parser import JDSummary


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
    # ---- experience / projects ----
    experience_years: int = 0             # max years of experience found in CV
    experience_score: float = 0.0        # 0-1 normalised experience (÷5 years)
    has_projects: bool = False            # CV mentions concrete projects
    tags: list[str] = field(default_factory=list)        # quick-summary badges
    red_flags: list[str] = field(default_factory=list)   # quality warning flags
    # ---- contact info (new) ----
    candidate_name: str = ""             # extracted from CV header
    email: str = ""                      # extracted via regex
    phone: str = ""                      # extracted via regex
    # ---- must-have / nice-to-have (new) ----
    must_have_missing: list[str] = field(default_factory=list)   # missing must-have skills
    nice_to_have_missing: list[str] = field(default_factory=list)  # missing nice-to-have skills
    # ---- job-hopping (new) ----
    job_hopping: bool = False            # detected from experience section
    job_count: int = 0                   # estimated number of distinct positions
    # ---- experience source ----
    experience_source: str = ""          # "stated" | "inferred_from_dates" | ""
    # ---- LLM-enriched fields ----
    ats_score: float = 0.0               # ATS keyword score (0-100) from LLM
    potential_level: str = ""            # "High" | "Medium" | "Low" from LLM
    culture_fit: str = ""                # culture fit note from LLM


# ---------------------------------------------------------------------------
# Contact info extraction
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?:"
    r"(?:\+84|0084|0)\d{9,10}"          # Vietnamese mobile: 0xxxxxxxxx or +84xxxxxxxxx
    r"|(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}"  # international
    r")"
)
_NAME_BLACKLIST = re.compile(
    r"\b(?:email|phone|address|curriculum\s*vitae|resume|cv|tel|mobile|"
    r"họ\s*tên|tên|địa\s*chỉ|điện\s*thoại|liên\s*hệ|contact)\b",
    re.IGNORECASE,
)


def extract_contact_info(cv_text: str) -> tuple[str, str, str]:
    """
    Extract (candidate_name, email, phone) from raw CV text.

    Strategy
    --------
    - email  : first match of RFC-5321-ish pattern
    - phone  : first match of Vietnamese/international phone pattern
    - name   : first non-blank, non-keyword line in the top 10 lines that
               looks like a name (all words capitalised, length 2–50 chars,
               no digits)

    Returns
    -------
    tuple[str, str, str]  – (name, email, phone) — empty string if not found
    """
    # --- email ---
    email_match = _EMAIL_RE.search(cv_text)
    email = email_match.group(0) if email_match else ""

    # --- phone ---
    phone_match = _PHONE_RE.search(cv_text)
    phone = phone_match.group(0).strip() if phone_match else ""

    # --- name ---
    name = ""
    top_lines = cv_text.split("\n")[:15]
    for raw_line in top_lines:
        line = raw_line.strip()
        if not line:
            continue
        if len(line) < 2 or len(line) > 60:
            continue
        if re.search(r"\d", line):
            continue
        if _NAME_BLACKLIST.search(line):
            continue
        # Prefer lines where every word starts with an uppercase letter
        words = line.split()
        if len(words) < 1:
            continue
        if all(w[0].isupper() for w in words if w):
            name = line
            break

    return name, email, phone


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

# Month name → number mapping (English abbreviations + full)
_MONTH_MAP: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Named month alternation (used in both _DATE_ENDPOINT_RE and _DR)
_MONTH_NAMES_RE = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

# Patterns for a single date endpoint: MM/YYYY, YYYY, Mon YYYY, tháng N/YYYY
# Note: use [ \t]+ (horizontal space only) for Mon YYYY to prevent cross-line matches.
_DATE_ENDPOINT_RE = re.compile(
    r"(?:"
    r"(?:tháng\s*(\d{1,2})[,/\s]+)(\d{4})"             # tháng N/YYYY or tháng N YYYY
    r"|(\d{1,2})[/\-](\d{4})"                            # MM/YYYY or MM-YYYY
    r"|(" + _MONTH_NAMES_RE + r")\.?[ \t]+(\d{4})"      # Mon YYYY  (Jan 2019)
    r"|(\d{4})"                                           # bare YYYY
    r")",
    re.IGNORECASE,
)

# Separator between two date endpoints (dash only — slash is used inside dates like MM/YYYY)
_DATE_SEP_RE = re.compile(r"\s*[-–—]\s*")

# "present / now / hiện tại / hiện nay / current"
_PRESENT_RE = re.compile(r"\b(present|now|current|hiện\s*(?:tại|nay))\b", re.IGNORECASE)

# Single date-endpoint fragment used in the date-range regex below.
# Matches: tháng N/YYYY  |  MM/YYYY  |  Jan 2019  |  2019
_DATE_PART_PAT = (
    r"(?:tháng\s*\d{1,2}[,/\s]+\d{4}"
    r"|\d{1,2}[/\-]\d{4}"
    r"|" + _MONTH_NAMES_RE + r"\.?[ \t]+\d{4}"
    r"|\d{4})"
)

# Full date-range regex: <date> – <date|present>.
# Named-month alternation (not generic [a-zA-Z]+) and horizontal whitespace in the
# month-year form prevent cross-line false matches like "engineer\n2023".
_DATE_RANGE_FULL_RE = re.compile(
    _DATE_PART_PAT
    + r"\s*[-–—]\s*"
    + r"(?:present|now|current|hiện\s*(?:tại|nay)|" + _DATE_PART_PAT + r")",
    re.IGNORECASE,
)


def _parse_date_endpoint(text: str) -> tuple[int, int] | None:
    """
    Parse a single date string into (year, month).
    Returns None if unparseable.
    """
    text = text.strip()
    m = _DATE_ENDPOINT_RE.match(text)
    if not m:
        return None
    # tháng N/YYYY
    if m.group(1) and m.group(2):
        return int(m.group(2)), int(m.group(1))
    # MM/YYYY
    if m.group(3) and m.group(4):
        mon, yr = int(m.group(3)), int(m.group(4))
        if 1 <= mon <= 12:
            return yr, mon
        return None
    # Mon YYYY (named month)
    if m.group(5) and m.group(6):
        mon_str = m.group(5).lower().rstrip(".")
        mon = _MONTH_MAP.get(mon_str)
        if mon:
            return int(m.group(6)), mon
        return None
    # bare YYYY
    if m.group(7):
        return int(m.group(7)), 1
    return None


def _abs_month(year: int, month: int) -> int:
    """Convert (year, month) to an absolute month index for arithmetic."""
    return year * 12 + month


def _infer_experience_from_date_ranges(text: str) -> int:
    """
    Infer total years of work experience by parsing date ranges from *text*.

    Algorithm
    ---------
    1. Find all date-range patterns (start – end / start – present).
    2. Convert each to (abs_start_month, abs_end_month).
    3. Merge overlapping / adjacent intervals.
    4. Sum total months across merged intervals.
    5. Return floor(total_months / 12).

    Returns 0 if no valid date ranges are found.
    """
    import datetime
    now = datetime.date.today()
    now_abs = _abs_month(now.year, now.month)

    intervals: list[tuple[int, int]] = []
    for match in _DATE_RANGE_FULL_RE.finditer(text):
        span_text = match.group(0)
        # Split at the separator
        sep_m = _DATE_SEP_RE.search(span_text)
        if not sep_m:
            continue
        start_str = span_text[: sep_m.start()]
        end_str = span_text[sep_m.end():]

        start_ep = _parse_date_endpoint(start_str)
        if start_ep is None:
            continue
        start_abs = _abs_month(*start_ep)

        if _PRESENT_RE.match(end_str.strip()):
            end_abs = now_abs
        else:
            end_ep = _parse_date_endpoint(end_str)
            if end_ep is None:
                continue
            end_abs = _abs_month(*end_ep)

        if end_abs < start_abs:
            continue  # malformed range
        intervals.append((start_abs, end_abs))

    if not intervals:
        return 0

    # Merge overlapping or adjacent intervals.
    # We treat intervals 1 month apart as adjacent (e.g. a job ending in Dec 2022
    # and the next starting in Jan 2023) to avoid gaps from imprecise date rounding.
    intervals.sort()
    merged: list[tuple[int, int]] = []
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end + 1:  # overlapping or adjacent (within 1 month)
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    total_months = sum(e - s for s, e in merged)
    return max(0, total_months // 12)


def _extract_experience_years(text: str) -> tuple[int, str]:
    """
    Extract years of experience from *text*.

    Returns
    -------
    tuple[int, str]
        (years, source) where source is "stated" or "inferred_from_dates".
        years == 0 means not found by either method.
    """
    max_years = 0
    for pat in _CV_EXP_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                y = int(m.group(1))
                if 0 < y <= MAX_EXPERIENCE_YEARS:
                    max_years = max(max_years, y)
            except (ValueError, IndexError):
                pass
    if max_years > 0:
        return max_years, "stated"

    inferred = _infer_experience_from_date_ranges(text)
    if inferred > 0:
        return min(inferred, MAX_EXPERIENCE_YEARS), "inferred_from_dates"

    return 0, ""


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


# ---------------------------------------------------------------------------
# Must-have / nice-to-have weighted skill score
# ---------------------------------------------------------------------------

_MUST_HAVE_PENALTY = 0.15   # penalty per missing must-have skill
_NICE_TO_HAVE_PENALTY = 0.05  # penalty per missing nice-to-have skill


def _compute_skill_score(
    skills_found: list[str],
    skills_missing: list[str],
    jd_skills: list[str],
    jd_summary: "JDSummary | None" = None,
) -> float:
    """
    Compute a weighted skill coverage score.

    Without JDSummary: simple fraction = found / total
    With JDSummary:
      - Start at base fraction
      - Apply extra penalty for each missing must-have skill
      - Apply smaller penalty for each missing nice-to-have skill
    """
    if not jd_skills:
        return 0.0

    base = len(skills_found) / len(jd_skills)

    if jd_summary is None:
        return float(np.clip(base, 0.0, 1.0))

    penalty = 0.0
    for skill in skills_missing:
        if skill in jd_summary.must_have:
            penalty += _MUST_HAVE_PENALTY
        elif skill in jd_summary.nice_to_have:
            penalty += _NICE_TO_HAVE_PENALTY

    return float(np.clip(base - penalty, 0.0, 1.0))


def analyze_cv(
    filename: str,
    cv_text: str,
    jd_embedding: np.ndarray,
    jd_skills: list[str],
    k_evidence: int = TOP_K_EVIDENCE,
    jd_summary: "JDSummary | None" = None,
) -> CVResult:
    """
    Analyse a single CV against the job description embedding.

    Composite score
    ---------------
    score = WEIGHT_SEMANTIC * semantic_score + WEIGHT_SKILL * skill_score

    semantic_score    = mean of top-3 chunk cosine similarities
    skill_score       = weighted skill coverage using must_have/nice_to_have
    experience_score  = min(1.0, experience_years / 5.0)   (stored, used by UI)

    Parameters
    ----------
    filename      : str          original file name
    cv_text       : str          raw extracted text of the CV
    jd_embedding  : np.ndarray   embedding of the full JD text (1-D, normalised)
    jd_skills     : list[str]    skills detected in the JD
    k_evidence    : int          number of evidence chunks to return
    jd_summary    : JDSummary    optional parsed JD with must_have/nice_to_have lists

    Returns
    -------
    CVResult
    """
    from modules.cv_section_parser import parse_cv_sections

    # --- parse CV sections ---
    cv_sections = parse_cv_sections(cv_text)
    job_hopping = cv_sections.job_hopping
    job_count = cv_sections.job_count

    # Text to use for experience and skill scoring (section-aware)
    exp_text = cv_sections.experience_text or cv_text
    skills_text = (
        "\n\n".join(t for t in (cv_sections.skills_text, cv_sections.projects_text) if t)
        or cv_text
    )

    # --- chunk & embed CV ---
    chunks = chunk_text(cv_text)
    if not chunks:
        exp_years, exp_source = _extract_experience_years(exp_text)
        exp_score = min(1.0, exp_years / EXPERIENCE_NORMALIZATION_YEARS)
        has_proj = _check_has_projects(cv_text)
        tags = _generate_tags([], jd_skills, exp_years, has_proj, cv_text)
        red_flags = _detect_red_flags(cv_text, exp_years, has_proj)
        summary = _build_summary(filename, [], jd_skills, 0.0, 0.0, exp_years)
        candidate_name, email, phone = extract_contact_info(cv_text)
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
            candidate_name=candidate_name,
            email=email,
            phone=phone,
            must_have_missing=list(jd_summary.must_have) if jd_summary else [],
            nice_to_have_missing=list(jd_summary.nice_to_have) if jd_summary else [],
            job_hopping=job_hopping,
            job_count=job_count,
            experience_source=exp_source,
        )

    chunk_embeddings = encode(chunks)

    # --- semantic score: mean of top-k chunk similarities ---
    k_for_score = min(3, len(chunks))
    raw_scores = chunk_embeddings @ jd_embedding
    top_scores = np.sort(raw_scores)[::-1][:k_for_score]
    semantic_score = float(np.clip(np.mean(top_scores), 0.0, 1.0))

    # --- evidence ---
    evidence = top_k_chunks(jd_embedding, chunk_embeddings, chunks, k=k_evidence)

    # --- skill detection (section-aware) ---
    cv_skills_all = _extract_skills(cv_text, SKILL_KEYWORDS)
    cv_skills_skills_section = _extract_skills(skills_text, SKILL_KEYWORDS)
    # Use full-text for "skills_found" so we don't miss skills mentioned in experience text,
    # but prefer skills_section for skill_score computation
    skills_found = [s for s in cv_skills_all if s in jd_skills]
    skills_missing = [s for s in jd_skills if s not in cv_skills_all]

    # --- skill coverage score with must_have / nice_to_have weighting ---
    skill_score = _compute_skill_score(
        skills_found=skills_found,
        skills_missing=skills_missing,
        jd_skills=jd_skills,
        jd_summary=jd_summary,
    )

    # --- must-have / nice-to-have gap lists ---
    must_have_missing: list[str] = []
    nice_to_have_missing: list[str] = []
    if jd_summary:
        must_have_missing = [s for s in jd_summary.must_have if s in skills_missing]
        nice_to_have_missing = [s for s in jd_summary.nice_to_have if s in skills_missing]

    # --- experience (from experience section text) ---
    experience_years, experience_source = _extract_experience_years(exp_text)
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
    if job_hopping:
        red_flags.append(f"Job hopping: ~{job_count} vị trí (có thể không ổn định)")

    summary = _build_summary(filename, skills_found, skills_missing, semantic_score, skill_score, experience_years)

    # --- contact info ---
    candidate_name, email, phone = extract_contact_info(cv_text)

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
        candidate_name=candidate_name,
        email=email,
        phone=phone,
        must_have_missing=must_have_missing,
        nice_to_have_missing=nice_to_have_missing,
        job_hopping=job_hopping,
        job_count=job_count,
        experience_source=experience_source,
    )


def rank_results(results: list[CVResult]) -> list[CVResult]:
    """Return results sorted by composite score descending."""
    return sorted(results, key=lambda r: r.score, reverse=True)


def extract_experience_from_text(text: str) -> tuple[int, str]:
    """
    Public API: extract years-of-experience from arbitrary text.

    Returns
    -------
    tuple[int, str]  – (years, source)
        source is "stated" | "inferred_from_dates" | ""
    """
    return _extract_experience_years(text)


def extract_cv_skills(cv_text: str) -> list[str]:
    """
    Public API: extract all skill names from *cv_text* using the SKILL_KEYWORDS catalogue.

    Returns
    -------
    list[str]  – skill names that appear in the text
    """
    return _extract_skills(cv_text, SKILL_KEYWORDS)
