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
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    rank            INTEGER NOT NULL,
    filename        TEXT    NOT NULL,
    score           REAL    NOT NULL,
    semantic_score  REAL    NOT NULL,
    skill_score     REAL    NOT NULL,
    skills_found    TEXT    NOT NULL,   -- JSON list
    skills_missing  TEXT    NOT NULL,   -- JSON list
    summary         TEXT    NOT NULL,
    full_text       TEXT    NOT NULL
);
"""


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
    """Create tables if they do not already exist."""
    with _get_conn() as conn:
        conn.executescript(_DDL)


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
                     skills_found, skills_missing, summary, full_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )

    return session_id


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
            SELECT rank, filename, score, semantic_score, skill_score,
                   skills_found, skills_missing, summary
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
                "rank": row["rank"],
                "filename": row["filename"],
                "score": row["score"],
                "semantic_score": row["semantic_score"],
                "skill_score": row["skill_score"],
                "skills_found": json.loads(row["skills_found"]),
                "skills_missing": json.loads(row["skills_missing"]),
                "summary": row["summary"],
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
