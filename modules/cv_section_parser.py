"""
modules/cv_section_parser.py

Parse a raw CV text into structured sections (Work Experience, Skills, Education…)
so that experience_score and skill_score are computed from the correct sections
instead of the entire document.

Also provides job-hopping detection from the Work Experience section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Section heading patterns (Vietnamese + English)
# ---------------------------------------------------------------------------

_SECTION_PATTERNS: dict[str, list[str]] = {
    "experience": [
        r"kinh\s*nghi[eệ]m\s*(làm\s*vi[eệ]c|ngh[eề]\s*nghi[eệ]p|công\s*vi[eệ]c)?",
        r"work\s*(experience|history|background|exp\.?)",
        r"professional\s*(experience|background)",
        r"employment\s*(history|record)",
        r"career\s*(history|summary|background|profile)",
        r"experience",
        r"work\s*exp\.?",
        r"qu[aá]\s*tr[iì]nh\s*(c[oô]ng\s*t[aá]c|l[aà]m\s*vi[eệ]c)",
        r"l[iị]ch\s*s[uử]\s*c[oô]ng\s*vi[eệ]c",
        r"kinh\s*nghi[eệ]m",
    ],
    "skills": [
        r"k[yỹ]\s*n[aă]ng",
        r"technical\s*skills?",
        r"core\s*skills?",
        r"skills?\s*(summary|set)?",
        r"technologies",
        r"tech\s*stack",
        r"competencies",
        r"expertise",
        r"proficiencies",
        r"tools?\s*(&|and)?\s*technologies",
        r"programming\s*languages?",
    ],
    "education": [
        r"h[oọ]c\s*v[aấ]n",
        r"tr[iì]nh\s*[dđ][oộ]",
        r"education",
        r"academic\s*(background|qualifications?)",
        r"degrees?",
        r"certifications?",
        r"ch[uứ]ng\s*ch[iỉ]",
        r"b[aằ]ng\s*c[aấ]p",
    ],
    "projects": [
        r"d[uự]\s*[aá]n",
        r"projects?",
        r"personal\s*projects?",
        r"side\s*projects?",
        r"portfolio",
        r"notable\s*projects?",
    ],
    "summary": [
        r"gi[oớ]i\s*thi[eệ]u",
        r"t[oó]m\s*t[aắ]t",
        r"objective",
        r"summary",
        r"profile",
        r"about\s*me",
        r"career\s*(objective|summary|goal)",
        r"professional\s*summary",
    ],
}

# Compile each pattern group into a single regex.
# We allow optional leading ordinal/number prefix (e.g. "III. ", "2. ", "A. ")
# and do NOT require the heading to end at EOL so lines like "Kinh nghiệm làm việc:"
# are also matched.
# 100 chars is generous enough to include long section titles with sub-titles while
# still excluding body text lines (which typically span full paragraphs).
_MAX_HEADING_LENGTH = 100
_ORDINAL_PREFIX = r"(?:\d+[\.\)]\s*|[IVXivxA-Za-z]{1,5}[\.\)]\s*)?"
_COMPILED: dict[str, re.Pattern] = {
    section: re.compile(
        r"^\s*" + _ORDINAL_PREFIX + r"(?:" + "|".join(patterns) + r")\s*[:\-–—]?\s*",
        re.IGNORECASE | re.MULTILINE,
    )
    for section, patterns in _SECTION_PATTERNS.items()
}

# Date range pattern to detect work periods (for job-hopping detection)
_DATE_RANGE_RE = re.compile(
    r"(?:"
    r"\d{1,2}[/\-]\d{4}"          # MM/YYYY or MM-YYYY
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}"
    r"|(?:tháng\s*\d{1,2}[,\s]+)?\d{4}"     # Vietnamese: tháng X, YYYY
    r")"
    r"\s*[-–—/]\s*"
    r"(?:present|now|hiện\s*(?:tại|nay)|current|\d{1,2}[/\-]\d{4}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}"
    r"|(?:tháng\s*\d{1,2}[,\s]+)?\d{4})",
    re.IGNORECASE,
)

# Company name heuristic: line that starts with a capital or is bold-like
_COMPANY_LINE_RE = re.compile(
    r"(?:công\s*ty|company|corp\.?|ltd\.?|inc\.?|llc|co\.|tập\s*đoàn|agency)",
    re.IGNORECASE,
)


@dataclass
class CVSections:
    """Structured sections extracted from a CV."""
    experience_text: str = ""
    skills_text: str = ""
    education_text: str = ""
    projects_text: str = ""
    summary_text: str = ""
    other_text: str = ""

    # Derived
    job_count: int = 0          # estimated number of distinct jobs
    job_hopping: bool = False   # True if ≥4 jobs or avg tenure <1 year


def parse_cv_sections(cv_text: str) -> CVSections:
    """
    Split *cv_text* into named sections based on heading-line detection.

    Returns
    -------
    CVSections
        Each field contains the raw text of that section (empty string if not found).
    """
    lines = cv_text.split("\n")
    buckets: dict[str, list[str]] = {s: [] for s in _SECTION_PATTERNS}
    buckets["other"] = []

    current_section = "other"

    for line in lines:
        stripped = line.strip()

        # Check if this line is a section heading
        matched_section: str | None = None
        if 2 <= len(stripped) <= _MAX_HEADING_LENGTH:
            for section, pattern in _COMPILED.items():
                if pattern.match(stripped):
                    matched_section = section
                    break
            # Fallback: try without ordinal prefix
            if not matched_section:
                for section, patterns in _SECTION_PATTERNS.items():
                    combined = re.compile(
                        r"^" + _ORDINAL_PREFIX + r"(?:" + "|".join(patterns) + r")\s*[:\-–—]?\s*",
                        re.IGNORECASE,
                    )
                    if combined.match(stripped):
                        matched_section = section
                        break

        if matched_section:
            current_section = matched_section
        else:
            buckets[current_section].append(line)

    sections = CVSections(
        experience_text="\n".join(buckets.get("experience", [])).strip(),
        skills_text="\n".join(buckets.get("skills", [])).strip(),
        education_text="\n".join(buckets.get("education", [])).strip(),
        projects_text="\n".join(buckets.get("projects", [])).strip(),
        summary_text="\n".join(buckets.get("summary", [])).strip(),
        other_text="\n".join(buckets.get("other", [])).strip(),
    )

    # Job-hopping analysis from experience section
    exp_text = sections.experience_text or cv_text
    sections.job_count = _estimate_job_count(exp_text)
    sections.job_hopping = sections.job_count >= 4

    return sections


def _estimate_job_count(exp_text: str) -> int:
    """
    Estimate number of distinct jobs from the experience section by counting
    date ranges (each job typically has a date range).

    Falls back to counting company-name indicators.
    """
    date_matches = _DATE_RANGE_RE.findall(exp_text)
    if date_matches:
        return len(date_matches)

    # Fallback: count lines that look like company names
    company_lines = sum(1 for line in exp_text.split("\n") if _COMPANY_LINE_RE.search(line))
    return company_lines


def get_experience_text(cv_text: str) -> str:
    """Convenience: return only the Work Experience section text (or full text if not found)."""
    sections = parse_cv_sections(cv_text)
    return sections.experience_text or cv_text


def get_skills_text(cv_text: str) -> str:
    """Convenience: return Skills + Projects sections combined."""
    sections = parse_cv_sections(cv_text)
    parts = [s for s in (sections.skills_text, sections.projects_text) if s]
    return "\n\n".join(parts) or cv_text
