"""
modules/export_manager.py
Builds a pandas DataFrame from CVResult objects and provides helpers
for CSV and Excel export (Streamlit download buttons).
"""
from __future__ import annotations

import io
import re

import pandas as pd

from modules.cv_analyzer import CVResult

# Matches characters that are illegal in Excel/openpyxl worksheets
# (control chars except tab, newline, carriage-return)
_ILLEGAL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize(value: object) -> object:
    """Strip illegal Excel characters from string values."""
    if isinstance(value, str):
        return _ILLEGAL_CHARS_RE.sub("", value)
    return value


def results_to_dataframe(results: list[CVResult]) -> pd.DataFrame:
    """
    Convert a ranked list of CVResults into a tidy DataFrame.

    Columns
    -------
    Rank | Filename | Match Score (%) | Semantic (%) | Skill Coverage (%)
    | Skills Found | Skills Missing | Summary | Top Evidence
    """
    rows = []
    for rank, r in enumerate(results, start=1):
        evidence_text = " | ".join(chunk for chunk, _ in r.evidence[:3])
        rows.append(
            {
                "Rank": rank,
                "Filename": r.filename,
                "Match Score (%)": round(r.score * 100, 1),
                "Semantic (%)": round(r.semantic_score * 100, 1),
                "Skill Coverage (%)": round(r.skill_score * 100, 1),
                "Skills Found": ", ".join(r.skills_found) if r.skills_found else "—",
                "Skills Missing": ", ".join(r.skills_missing) if r.skills_missing else "—",
                "Summary": r.summary,
                "Top Evidence": evidence_text,
            }
        )
    return pd.DataFrame(rows)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return the DataFrame encoded as UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Return the DataFrame encoded as an Excel (.xlsx) file in memory."""
    # Sanitize all string cells to remove characters illegal in Excel worksheets
    clean_df = df.apply(lambda col: col.map(_sanitize))
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        clean_df.to_excel(writer, index=False, sheet_name="CV Rankings")
    return buffer.getvalue()
