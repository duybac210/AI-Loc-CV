"""
modules/llm_analyzer.py

LLM-powered CV analysis with multi-provider support:
  OpenAI (GPT-4o/3.5) → Google Gemini → Groq → fallback rule-based

Provides:
  - generate_llm_summary(): generate a natural-language summary for a candidate
  - LLMConfig: settings passed from Streamlit session state

Usage
-----
    from modules.llm_analyzer import LLMConfig, generate_llm_summary

    config = LLMConfig(provider="openai", api_key="sk-...", model="gpt-4o-mini")
    summary, questions, decision = generate_llm_summary(config, cv_text, jd_text, cv_result)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.cv_analyzer import CVResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUPPORTED_PROVIDERS = ("openai", "gemini", "groq")


@dataclass
class LLMConfig:
    """LLM provider settings — never stored in DB."""
    provider: str = "openai"       # "openai" | "gemini" | "groq"
    api_key: str = ""
    model: str = ""                # leave empty to use provider default

    def is_configured(self) -> bool:
        return bool(self.api_key.strip())


# Default model names per provider
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "groq":   "llama3-8b-8192",
}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Bạn là chuyên gia HR / Talent Acquisition với hơn 10 năm kinh nghiệm.
Nhiệm vụ: phân tích CV ứng viên so với mô tả công việc (JD) và trả về JSON thuần tuý (không có markdown, không có ```json).

Output JSON schema (bắt buộc đúng format):
{
  "summary": "Nhận xét tự nhiên 2-3 câu về ứng viên, so sánh với JD",
  "strengths": ["điểm mạnh 1", "điểm mạnh 2"],
  "weaknesses": ["điểm yếu 1", "điểm yếu 2"],
  "decision": "Shortlist" | "Consider" | "Reject",
  "decision_reason": "Lý do ngắn gọn 1 câu",
  "interview_questions": ["câu hỏi phỏng vấn 1", "câu hỏi 2", "câu hỏi 3"],
  "potential_level": "High" | "Medium" | "Low"
}"""


def _build_user_prompt(
    cv_text: str,
    jd_text: str,
    result: "CVResult",
) -> str:
    """Build the user-facing prompt with CV/JD context and pre-computed scores."""
    skills_found_str = ", ".join(result.skills_found[:10]) or "Không có"
    skills_missing_str = ", ".join(result.skills_missing[:10]) or "Không có"
    must_missing_str = ", ".join(result.must_have_missing[:5]) or "Không có"

    return (
        f"=== MÔ TẢ CÔNG VIỆC (JD) ===\n{jd_text[:1500]}\n\n"
        f"=== CV ỨNG VIÊN ===\n{cv_text[:2000]}\n\n"
        f"=== KẾT QUẢ PHÂN TÍCH SƠ BỘ ===\n"
        f"- Điểm tổng: {round(result.score * 100, 1)}%\n"
        f"- Điểm ngữ nghĩa: {round(result.semantic_score * 100, 1)}%\n"
        f"- Điểm kỹ năng: {round(result.skill_score * 100, 1)}%\n"
        f"- Kinh nghiệm: {result.experience_years} năm\n"
        f"- Kỹ năng có: {skills_found_str}\n"
        f"- Kỹ năng thiếu: {skills_missing_str}\n"
        f"- Thiếu must-have: {must_missing_str}\n"
        f"- Job hopping: {'Có' if result.job_hopping else 'Không'}\n\n"
        f"Hãy phân tích và trả về JSON theo schema đã định."
    )


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------

def _call_openai(config: LLMConfig, user_prompt: str) -> str:
    """Call OpenAI Chat Completions API."""
    try:
        import openai
    except ImportError as e:
        raise RuntimeError("openai package not installed. Run: pip install openai") from e

    client = openai.OpenAI(api_key=config.api_key)
    model = config.model or _DEFAULT_MODELS["openai"]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _call_gemini(config: LLMConfig, user_prompt: str) -> str:
    """Call Google Gemini API via google-generativeai."""
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError(
            "google-generativeai package not installed. Run: pip install google-generativeai"
        ) from e

    genai.configure(api_key=config.api_key)
    model_name = config.model or _DEFAULT_MODELS["gemini"]
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=_SYSTEM_PROMPT,
    )
    full_prompt = user_prompt + "\n\nTrả về JSON thuần tuý, không có markdown."
    response = model.generate_content(full_prompt)
    return response.text or ""


def _call_groq(config: LLMConfig, user_prompt: str) -> str:
    """Call Groq API (OpenAI-compatible)."""
    try:
        from groq import Groq
    except ImportError as e:
        raise RuntimeError("groq package not installed. Run: pip install groq") from e

    client = Groq(api_key=config.api_key)
    model = config.model or _DEFAULT_MODELS["groq"]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# JSON extraction / parsing
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict:
    """
    Extract and parse JSON from an LLM response that may contain extra text.
    """
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_llm_summary(
    config: LLMConfig,
    cv_text: str,
    jd_text: str,
    result: "CVResult",
) -> tuple[str, list[str], str]:
    """
    Generate an LLM-powered summary for a candidate.

    Returns
    -------
    tuple[str, list[str], str]
        (summary_text, interview_questions, decision)
        Falls back to rule-based result.summary on any error.
    """
    if not config.is_configured():
        return result.summary, [], ""

    user_prompt = _build_user_prompt(cv_text, jd_text, result)

    raw = ""
    try:
        if config.provider == "openai":
            raw = _call_openai(config, user_prompt)
        elif config.provider == "gemini":
            raw = _call_gemini(config, user_prompt)
        elif config.provider == "groq":
            raw = _call_groq(config, user_prompt)
        else:
            return result.summary, [], ""
    except Exception as exc:
        # Surface the error message but fall back gracefully
        return f"[LLM error: {exc}] {result.summary}", [], ""

    data = _extract_json(raw)

    summary = data.get("summary") or result.summary
    questions = data.get("interview_questions") or []
    decision = data.get("decision") or ""
    reason = data.get("decision_reason") or ""

    if reason and decision:
        summary = f"{summary}\n\n💡 Quyết định gợi ý: **{decision}** — {reason}"

    return summary, questions, decision
