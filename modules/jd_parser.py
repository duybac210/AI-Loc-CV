"""
modules/jd_parser.py

JD Intelligence Engine – parse a job description to extract:
  - must_have skills
  - nice_to_have skills
  - min_experience (years)
  - level (junior / mid / senior / lead)
  - position_title
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import SKILL_KEYWORDS


@dataclass
class JDSummary:
    """Structured metadata extracted from a Job Description."""

    position_title: str = ""
    level: str = ""           # junior / mid / senior / lead / ""
    min_experience: int = 0   # years, 0 = not found
    must_have: list[str] = field(default_factory=list)
    nice_to_have: list[str] = field(default_factory=list)
    all_skills: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Keyword patterns (Vietnamese + English)
# ---------------------------------------------------------------------------

_MUST_PATTERNS: list[str] = [
    r"yêu\s*cầu",
    r"bắt\s*buộc",
    r"required?",
    r"must[\s\-]*have",
    r"cần\s*có",
    r"cần\s*thiết",
    r"mandatory",
    r"essential",
    r"require[sd]",
]

_NICE_PATTERNS: list[str] = [
    r"ưu\s*tiên",
    r"preferred?",
    r"nice[\s\-]*to[\s\-]*have",
    r"là\s*lợi\s*thế",
    r"\bbonus\b",
    r"advantage",
    r"khuyến\s*khích",
    r"would\s*be\s*a\s*plus",
    r"tốt\s*hơn\s*nếu",
    r"\ba\s*plus\b",
]

_LEVEL_MAP: dict[str, list[str]] = {
    "lead":   [r"\blead\b", r"tech\s*lead", r"principal", r"trưởng\s*nhóm", r"team\s*lead"],
    "senior": [r"\bsenior\b", r"\bsr\.?\b", r"5\+\s*(?:năm|year)"],
    "mid":    [r"\bmid\b", r"\bmiddle\b", r"\bintermediate\b",
               r"[23]\+\s*(?:năm|year)", r"3[-–]\d+\s*(?:năm|year)"],
    "junior": [r"\bjunior\b", r"\bjr\.?\b", r"\bfresher\b",
               r"entry[\s\-]?level", r"mới\s*ra\s*trường", r"fresh\s*graduate"],
}

_EXP_PATTERNS: list[str] = [
    r"(\d+)\+?\s*(?:năm|years?|yrs?)(?:\s*kinh\s*nghiệm|\s*experience)?",
    r"at\s*least\s*(\d+)\s*(?:years?|yrs?)",
    r"tối\s*thiểu\s*(\d+)\s*năm",
    r"minimum\s*(\d+)\s*(?:years?|yrs?)",
    r"(\d+)\s*[-–]\s*\d+\s*(?:năm|years?)",
]

_TITLE_PREFIXES: list[str] = [
    r"(?:vị\s*trí|position|role|job\s*title|chức\s*danh)\s*[:\-]?\s*(.+)",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_skills_in_text(text: str) -> list[str]:
    """Return ordered list of skill names from SKILL_KEYWORDS found in *text*."""
    lower = text.lower()
    found: list[str] = []
    for name, keywords in SKILL_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower, re.IGNORECASE):
                found.append(name)
                break
    return found


def _split_must_nice_other(text: str) -> tuple[str, str, str]:
    """
    Walk the JD line-by-line and route each line into a must / nice / other
    bucket based on section header keywords.

    Returns
    -------
    tuple[str, str, str]  – (must_text, nice_text, other_text)
    """
    must_lines: list[str] = []
    nice_lines: list[str] = []
    other_lines: list[str] = []
    current = "other"

    for line in text.split("\n"):
        lo = line.lower()
        if any(re.search(p, lo) for p in _MUST_PATTERNS):
            current = "must"
        elif any(re.search(p, lo) for p in _NICE_PATTERNS):
            current = "nice"

        if current == "must":
            must_lines.append(line)
        elif current == "nice":
            nice_lines.append(line)
        else:
            other_lines.append(line)

    return "\n".join(must_lines), "\n".join(nice_lines), "\n".join(other_lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_jd(jd_text: str) -> JDSummary:
    """
    Parse a job description and return a :class:`JDSummary` with extracted
    metadata.

    Parameters
    ----------
    jd_text : str
        Raw job description text (Vietnamese, English, or mixed).

    Returns
    -------
    JDSummary
    """
    summary = JDSummary()

    # ---- position title ----
    for pat in _TITLE_PREFIXES:
        m = re.search(pat, jd_text, re.IGNORECASE)
        if m:
            summary.position_title = m.group(1).strip()[:80]
            break
    if not summary.position_title:
        for line in jd_text.split("\n"):
            s = line.strip()
            if len(s) > 3:
                summary.position_title = s[:80]
                break

    # ---- level ----
    lower = jd_text.lower()
    for lvl, patterns in _LEVEL_MAP.items():
        if any(re.search(p, lower) for p in patterns):
            summary.level = lvl
            break

    # ---- min_experience ----
    for pat in _EXP_PATTERNS:
        for m in re.finditer(pat, jd_text, re.IGNORECASE):
            try:
                y = int(m.group(1))
                if 0 < y <= 40:
                    summary.min_experience = max(summary.min_experience, y)
            except (ValueError, IndexError):
                pass

    # ---- skill classification ----
    must_text, nice_text, other_text = _split_must_nice_other(jd_text)

    must_skills = _detect_skills_in_text(must_text)
    nice_skills = _detect_skills_in_text(nice_text)
    other_skills = _detect_skills_in_text(other_text)
    all_skills = _detect_skills_in_text(jd_text)

    # nice_to_have = explicitly in nice section
    # must_have    = explicitly in must section + rest (other) that are not nice
    summary.nice_to_have = list(dict.fromkeys(nice_skills))
    summary.must_have = list(dict.fromkeys(
        must_skills + [s for s in other_skills if s not in summary.nice_to_have]
    ))

    # Fallback: no section markers detected → all skills → must_have
    if not summary.must_have and not summary.nice_to_have:
        summary.must_have = all_skills

    summary.all_skills = all_skills
    return summary
