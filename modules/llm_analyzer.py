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


# ---------------------------------------------------------------------------
# Additional AI helpers: candidate comparison, JD skill extraction, CV sections
# ---------------------------------------------------------------------------

def generate_comparison_summary(
    config: LLMConfig,
    top_candidates: list[tuple["CVResult", float]],
    jd_text: str,
) -> str:
    """
    Generate a concise natural-language comparison of the top-N candidates.

    Parameters
    ----------
    config         : LLMConfig   provider configuration
    top_candidates : list of (CVResult, score) – up to 3 candidates
    jd_text        : str         raw job description text

    Returns
    -------
    str  – plain Vietnamese text comparing candidates. Empty string on error.
    """
    if not config.is_configured() or len(top_candidates) < 2:
        return ""

    lines = [f"=== MÔ TẢ CÔNG VIỆC ===\n{jd_text[:800]}\n\n=== ỨNG VIÊN (đã xếp hạng) ==="]
    for i, (r, score) in enumerate(top_candidates[:3], 1):
        lines.append(
            f"\n#{i} {r.candidate_name or r.filename} – Điểm tổng hợp: {round(score * 100, 1)}%\n"
            f"  Kỹ năng phù hợp: {', '.join(r.skills_found[:8]) or 'Không xác định'}\n"
            f"  Thiếu must-have: {', '.join(r.must_have_missing[:5]) or 'Không'}\n"
            f"  Kinh nghiệm: {r.experience_years} năm\n"
            f"  Job hopping: {'Có' if r.job_hopping else 'Không'}"
        )

    prompt = (
        "\n".join(lines)
        + "\n\nViết 3–5 câu tiếng Việt: so sánh các ứng viên và giải thích vì sao ứng viên #1 "
        "phù hợp nhất (hoặc nhận xét nếu điểm thấp). Nêu điểm khác biệt chính giữa họ. "
        "Văn phong tự nhiên như HR thật, không dùng JSON, không dùng bullet point."
    )

    try:
        if config.provider == "openai":
            import openai
            resp = openai.OpenAI(api_key=config.api_key).chat.completions.create(
                model=config.model or _DEFAULT_MODELS["openai"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=300,
            )
            return resp.choices[0].message.content or ""
        elif config.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=config.api_key)
            return (
                genai.GenerativeModel(config.model or _DEFAULT_MODELS["gemini"])
                .generate_content(prompt)
                .text or ""
            )
        elif config.provider == "groq":
            from groq import Groq
            resp = Groq(api_key=config.api_key).chat.completions.create(
                model=config.model or _DEFAULT_MODELS["groq"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=300,
            )
            return resp.choices[0].message.content or ""
    except Exception as exc:
        return f"[Lỗi so sánh: {exc}]"
    return ""


def extract_jd_skills_llm(
    config: LLMConfig,
    jd_text: str,
) -> tuple[list[str], list[str]]:
    """
    Use the LLM to extract must-have and nice-to-have skills from a JD.

    Returns
    -------
    tuple[list[str], list[str]]
        (must_have, nice_to_have) — both may be empty on failure or when not configured.
    """
    if not config.is_configured():
        return [], []

    prompt = (
        f"Phân tích Job Description sau và liệt kê kỹ năng kỹ thuật cần thiết.\n\n"
        f"JD:\n{jd_text[:2000]}\n\n"
        f'Trả về JSON với 2 key:\n'
        f'  "must_have": danh sách kỹ năng bắt buộc (tên ngắn, ví dụ "Python", "Docker")\n'
        f'  "nice_to_have": danh sách kỹ năng ưu tiên / bonus\n'
        f"Chỉ JSON thuần tuý, không markdown, không giải thích thêm."
    )

    raw = ""
    try:
        if config.provider == "openai":
            import openai
            resp = openai.OpenAI(api_key=config.api_key).chat.completions.create(
                model=config.model or _DEFAULT_MODELS["openai"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
        elif config.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=config.api_key)
            raw = (
                genai.GenerativeModel(config.model or _DEFAULT_MODELS["gemini"])
                .generate_content(prompt + "\nTrả về JSON thuần tuý.")
                .text or ""
            )
        elif config.provider == "groq":
            from groq import Groq
            resp = Groq(api_key=config.api_key).chat.completions.create(
                model=config.model or _DEFAULT_MODELS["groq"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content or ""
    except Exception:
        return [], []

    data = _extract_json(raw)
    must = [str(s).strip() for s in data.get("must_have", []) if s]
    nice = [str(s).strip() for s in data.get("nice_to_have", []) if s]
    return must, nice


def parse_cv_sections_llm(
    config: LLMConfig,
    cv_text: str,
) -> dict[str, str]:
    """
    Use the LLM to segment a CV into named sections.

    Useful as a fallback when regex-based :func:`parse_cv_sections` yields
    mostly empty results (e.g. non-standard CV layouts or Vietnamese-only CVs).

    Returns
    -------
    dict[str, str]
        Keys: ``"experience"``, ``"skills"``, ``"education"``,
        ``"projects"``, ``"summary"``.
        Values are the extracted section text (empty string if not found).
        Returns ``{}`` on any error or when not configured.
    """
    if not config.is_configured() or not cv_text.strip():
        return {}

    prompt = (
        f"Đây là nội dung CV (có thể tiếng Việt hoặc tiếng Anh):\n\n"
        f"{cv_text[:3000]}\n\n"
        f"Hãy trích xuất và phân loại theo các mục:\n"
        f'Trả về JSON với các key: "experience", "skills", "education", "projects", "summary"\n'
        f"Mỗi key chứa toàn bộ nội dung của mục đó dưới dạng chuỗi văn bản. "
        f"Nếu không tìm thấy một mục, để chuỗi rỗng. Chỉ JSON thuần tuý, không markdown."
    )

    raw = ""
    try:
        if config.provider == "openai":
            import openai
            resp = openai.OpenAI(api_key=config.api_key).chat.completions.create(
                model=config.model or _DEFAULT_MODELS["openai"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=900,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
        elif config.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=config.api_key)
            raw = (
                genai.GenerativeModel(config.model or _DEFAULT_MODELS["gemini"])
                .generate_content(prompt + "\nTrả về JSON thuần tuý.")
                .text or ""
            )
        elif config.provider == "groq":
            from groq import Groq
            resp = Groq(api_key=config.api_key).chat.completions.create(
                model=config.model or _DEFAULT_MODELS["groq"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=900,
            )
            raw = resp.choices[0].message.content or ""
    except Exception:
        return {}

    data = _extract_json(raw)
    return {
        k: str(data.get(k, ""))
        for k in ("experience", "skills", "education", "projects", "summary")
    }
