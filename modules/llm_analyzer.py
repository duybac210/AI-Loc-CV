"""
modules/llm_analyzer.py

LLM-powered CV analysis with multi-provider support:
  OpenAI (GPT-4o/3.5) → Google Gemini → Groq → fallback rule-based

Provides:
  - generate_llm_summary(): generate a natural-language summary for a candidate
  - LLMConfig: settings passed from Streamlit session state
  - LLMResult: structured result returned by generate_llm_summary

Usage
-----
    from modules.llm_analyzer import LLMConfig, LLMResult, generate_llm_summary

    config = LLMConfig(provider="openai", api_key="sk-...", model="gpt-4o-mini")
    llm_result = generate_llm_summary(config, cv_text, jd_text, cv_result)
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


@dataclass
class LLMResult:
    """Structured output from the LLM analysis step."""
    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    decision: str = ""
    decision_reason: str = ""
    interview_questions: list[str] = field(default_factory=list)
    potential_level: str = ""        # "High" | "Medium" | "Low"
    stability: str = ""              # e.g. "Ổn định" | "Job hopping"
    culture_fit: str = ""            # short culture-fit note
    score: float = 0.0               # 0-100 overall score from LLM
    skill_match: float = 0.0         # 0-100
    experience_match: float = 0.0    # 0-100
    ats_score: float = 0.0           # 0-100 ATS keyword score
    missing_skills: list[str] = field(default_factory=list)


# Default model names per provider
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "groq":   "llama3-8b-8192",
}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Bạn là một AI tuyển dụng cấp cao với 10+ năm kinh nghiệm như một HR Manager thực thụ.

Nhiệm vụ của bạn:
Phân tích CV ứng viên và so sánh với Job Description (JD) để đưa ra quyết định tuyển dụng chính xác, có lý do rõ ràng.

QUY TẮC:
- Không đoán bừa nếu CV thiếu thông tin
- Ưu tiên dữ liệu định lượng
- Giải thích logic như con người, không chung chung

Output JSON schema (bắt buộc đúng format, không có markdown, không có ```json):
{
  "score": <0-100>,
  "skill_match": <0-100>,
  "experience_match": <0-100>,
  "ats_score": <0-100>,
  "missing_skills": ["kỹ năng thiếu 1", "kỹ năng thiếu 2"],
  "strengths": ["điểm mạnh 1", "điểm mạnh 2"],
  "weaknesses": ["điểm yếu 1", "điểm yếu 2"],
  "potential_level": "High" | "Medium" | "Low",
  "stability": "mức độ ổn định ngắn gọn (ví dụ: Ổn định, Job hopping nhẹ, Job hopping nghiêm trọng)",
  "culture_fit": "nhận xét ngắn về độ phù hợp văn hóa (nếu suy luận được, không bắt buộc)",
  "summary": "Nhận xét tự nhiên 2-3 câu về ứng viên so với JD",
  "decision": "Shortlist" | "Consider" | "Reject",
  "decision_reason": "Lý do ngắn gọn 1 câu thuyết phục như HR thật",
  "interview_questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"]
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

    exp_source = result.experience_source
    if exp_source == "stated":
        exp_note = "tự khai trong CV"
    elif exp_source == "inferred_from_dates":
        exp_note = "suy ra từ khoảng ngày tháng trong CV (không tự khai)"
    else:
        exp_note = "không xác định được"

    exp_display = f"{result.experience_years} năm ({exp_note})" if result.experience_years else f"Không rõ ({exp_note})"

    return (
        f"=== MÔ TẢ CÔNG VIỆC (JD) ===\n{jd_text[:1500]}\n\n"
        f"=== CV ỨNG VIÊN ===\n{cv_text[:2000]}\n\n"
        f"=== KẾT QUẢ PHÂN TÍCH SƠ BỘ ===\n"
        f"- Điểm tổng (pre-AI): {round(result.score * 100, 1)}%\n"
        f"- Điểm ngữ nghĩa: {round(result.semantic_score * 100, 1)}%\n"
        f"- Điểm kỹ năng: {round(result.skill_score * 100, 1)}%\n"
        f"- Kinh nghiệm: {exp_display}\n"
        f"- Kỹ năng có: {skills_found_str}\n"
        f"- Kỹ năng thiếu: {skills_missing_str}\n"
        f"- Thiếu must-have: {must_missing_str}\n"
        f"- Job hopping: {'Có' if result.job_hopping else 'Không'}\n\n"
        f"Hãy phân tích toàn diện và trả về JSON theo schema đã định."
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
) -> "LLMResult":
    """
    Generate an LLM-powered analysis for a candidate.

    Returns
    -------
    LLMResult
        Structured analysis result. Falls back to rule-based values on any error.
        Also mutates *result* in-place to store ats_score, potential_level,
        culture_fit for later use in exports / DB.
    """
    fallback = LLMResult(
        summary=result.summary,
        potential_level="",
    )

    if not config.is_configured():
        return fallback

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
            return fallback
    except Exception as exc:
        fallback.summary = f"[LLM error: {exc}] {result.summary}"
        return fallback

    data = _extract_json(raw)

    def _safe_float(val: object, default: float = 0.0) -> float:
        try:
            return float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    llm_res = LLMResult(
        summary=data.get("summary") or result.summary,
        strengths=data.get("strengths") or [],
        weaknesses=data.get("weaknesses") or [],
        decision=data.get("decision") or "",
        decision_reason=data.get("decision_reason") or "",
        interview_questions=data.get("interview_questions") or [],
        potential_level=data.get("potential_level") or "",
        stability=data.get("stability") or "",
        culture_fit=data.get("culture_fit") or "",
        score=_safe_float(data.get("score")),
        skill_match=_safe_float(data.get("skill_match")),
        experience_match=_safe_float(data.get("experience_match")),
        ats_score=_safe_float(data.get("ats_score")),
        missing_skills=data.get("missing_skills") or [],
    )

    if llm_res.decision_reason and llm_res.decision:
        llm_res.summary = (
            f"{llm_res.summary}\n\n"
            f"💡 Quyết định gợi ý: **{llm_res.decision}** — {llm_res.decision_reason}"
        )

    # Propagate enriched scores back to the CVResult so exports/DB can use them
    result.ats_score = llm_res.ats_score
    result.potential_level = llm_res.potential_level
    result.culture_fit = llm_res.culture_fit

    return llm_res
