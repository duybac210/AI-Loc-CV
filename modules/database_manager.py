"""
modules/database_manager.py
SQLite-based persistence layer for CV screening sessions.

Schema
------
sessions  – one row per analysis run (JD + timestamp)
cv_results – one row per CV analysed in a session
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from config import DATABASE_PATH
from modules.cv_analyzer import CVResult


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    jd_text     TEXT    NOT NULL,
    jd_snippet  TEXT    NOT NULL,   -- first 120 chars for display
    jd_skills   TEXT    NOT NULL,   -- JSON list of skill names
    cv_count    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS cv_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    rank                INTEGER NOT NULL,
    filename            TEXT    NOT NULL,
    score               REAL    NOT NULL,
    semantic_score      REAL    NOT NULL,
    skill_score         REAL    NOT NULL,
    skills_found        TEXT    NOT NULL,   -- JSON list
    skills_missing      TEXT    NOT NULL,   -- JSON list
    summary             TEXT    NOT NULL,
    full_text           TEXT    NOT NULL,
    -- contact info
    candidate_name      TEXT    NOT NULL DEFAULT '',
    email               TEXT    NOT NULL DEFAULT '',
    phone               TEXT    NOT NULL DEFAULT '',
    -- extended fields
    experience_years    INTEGER NOT NULL DEFAULT 0,
    tags                TEXT    NOT NULL DEFAULT '[]',     -- JSON list
    red_flags           TEXT    NOT NULL DEFAULT '[]',     -- JSON list
    -- HR workflow
    decision            TEXT    DEFAULT NULL,  -- 'Shortlist'|'Consider'|'Reject'|NULL
    hr_notes            TEXT    NOT NULL DEFAULT '',
    interview_questions TEXT    NOT NULL DEFAULT '[]',     -- JSON list
    -- scoring detail
    must_have_missing   TEXT    NOT NULL DEFAULT '[]',     -- JSON list
    nice_to_have_missing TEXT   NOT NULL DEFAULT '[]',     -- JSON list
    -- job-hopping
    job_hopping         INTEGER NOT NULL DEFAULT 0,        -- 0/1 boolean
    job_count           INTEGER NOT NULL DEFAULT 0,
    -- timestamp
    scored_at           TEXT    NOT NULL DEFAULT ''
);
"""

# Migration: columns added after initial schema — safe to run repeatedly
_MIGRATIONS = [
    "ALTER TABLE cv_results ADD COLUMN candidate_name TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE cv_results ADD COLUMN email TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE cv_results ADD COLUMN phone TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE cv_results ADD COLUMN experience_years INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE cv_results ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE cv_results ADD COLUMN red_flags TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE cv_results ADD COLUMN decision TEXT DEFAULT NULL",
    "ALTER TABLE cv_results ADD COLUMN hr_notes TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE cv_results ADD COLUMN interview_questions TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE cv_results ADD COLUMN must_have_missing TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE cv_results ADD COLUMN nice_to_have_missing TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE cv_results ADD COLUMN job_hopping INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE cv_results ADD COLUMN job_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE cv_results ADD COLUMN scored_at TEXT NOT NULL DEFAULT ''",
]


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they do not already exist, then run any pending migrations."""
    with _get_conn() as conn:
        conn.executescript(_DDL)
        # Run migrations — ignore "duplicate column" errors gracefully
        for sql in _MIGRATIONS:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_session(
    jd_text: str,
    jd_skills: list[str],
    ranked_results: list[CVResult],
) -> int:
    """
    Persist a complete analysis session.

    Returns
    -------
    int  – the new session_id
    """
    jd_snippet = jd_text[:120].replace("\n", " ")
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scored_at = created_at

    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO sessions (created_at, jd_text, jd_snippet, jd_skills, cv_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (created_at, jd_text, jd_snippet, json.dumps(jd_skills), len(ranked_results)),
        )
        session_id = cur.lastrowid

        for rank, result in enumerate(ranked_results, start=1):
            conn.execute(
                """
                INSERT INTO cv_results
                    (session_id, rank, filename, score, semantic_score, skill_score,
                     skills_found, skills_missing, summary, full_text,
                     candidate_name, email, phone,
                     experience_years, tags, red_flags,
                     must_have_missing, nice_to_have_missing,
                     job_hopping, job_count, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    rank,
                    result.filename,
                    result.score,
                    result.semantic_score,
                    result.skill_score,
                    json.dumps(result.skills_found),
                    json.dumps(result.skills_missing),
                    result.summary,
                    result.full_text,
                    result.candidate_name,
                    result.email,
                    result.phone,
                    result.experience_years,
                    json.dumps(result.tags),
                    json.dumps(result.red_flags),
                    json.dumps(result.must_have_missing),
                    json.dumps(result.nice_to_have_missing),
                    1 if result.job_hopping else 0,
                    result.job_count,
                    scored_at,
                ),
            )

    return session_id


def update_decision(cv_result_id: int, decision: str, hr_notes: str = "") -> None:
    """Update the HR decision and notes for a single cv_result row."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE cv_results SET decision = ?, hr_notes = ? WHERE id = ?",
            (decision or None, hr_notes, cv_result_id),
        )


def update_interview_questions(cv_result_id: int, questions: list[str]) -> None:
    """Persist LLM-generated interview questions for a cv_result."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE cv_results SET interview_questions = ? WHERE id = ?",
            (json.dumps(questions), cv_result_id),
        )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def list_sessions(limit: int = 50) -> list[dict]:
    """
    Return the most recent *limit* sessions (metadata only, no full text).
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, jd_snippet, jd_skills, cv_count
            FROM sessions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result = []
    for row in rows:
        result.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "jd_snippet": row["jd_snippet"],
                "jd_skills": json.loads(row["jd_skills"]),
                "cv_count": row["cv_count"],
            }
        )
    return result


def get_session_results(session_id: int) -> list[dict]:
    """
    Return all CV results for a given session, ordered by rank.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, rank, filename, score, semantic_score, skill_score,
                   skills_found, skills_missing, summary,
                   candidate_name, email, phone, experience_years,
                   tags, red_flags, decision, hr_notes, interview_questions,
                   must_have_missing, nice_to_have_missing,
                   job_hopping, job_count, scored_at
            FROM cv_results
            WHERE session_id = ?
            ORDER BY rank ASC
            """,
            (session_id,),
        ).fetchall()

    result = []
    for row in rows:
        result.append(
            {
                "id": row["id"],
                "rank": row["rank"],
                "filename": row["filename"],
                "score": row["score"],
                "semantic_score": row["semantic_score"],
                "skill_score": row["skill_score"],
                "skills_found": json.loads(row["skills_found"]),
                "skills_missing": json.loads(row["skills_missing"]),
                "summary": row["summary"],
                "candidate_name": row["candidate_name"] or "",
                "email": row["email"] or "",
                "phone": row["phone"] or "",
                "experience_years": row["experience_years"] or 0,
                "tags": json.loads(row["tags"] or "[]"),
                "red_flags": json.loads(row["red_flags"] or "[]"),
                "decision": row["decision"],
                "hr_notes": row["hr_notes"] or "",
                "interview_questions": json.loads(row["interview_questions"] or "[]"),
                "must_have_missing": json.loads(row["must_have_missing"] or "[]"),
                "nice_to_have_missing": json.loads(row["nice_to_have_missing"] or "[]"),
                "job_hopping": bool(row["job_hopping"]),
                "job_count": row["job_count"] or 0,
                "scored_at": row["scored_at"] or "",
            }
        )
    return result


def get_session_jd(session_id: int) -> str:
    """Return the full JD text for a session."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT jd_text FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return row["jd_text"] if row else ""


def delete_session(session_id: int) -> None:
    """Delete a session and all its CV results (cascade)."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
