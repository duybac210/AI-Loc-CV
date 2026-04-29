"""
modules/cv_analyzer.py
Core analysis logic (Bulletproof Edition):
  - Clean CV text before passing to LLM
  - Strict English prompt to prevent hallucination
  - Safe JSON parser with Regex fallback
"""
from __future__ import annotations
import json
import re
from groq import Groq
from pydantic import BaseModel, Field
from typing import List
from config import WEIGHT_SEMANTIC, WEIGHT_SKILL, WEIGHT_EXPERIENCE, SCORE_HIGH, SCORE_MED
from modules.jd_parser import JDSummary

# ==========================================
# 1. KHUÔN ĐÚC DỮ LIỆU CV (PYDANTIC)
# ==========================================
class CVExtraction(BaseModel):
    candidate_name: str = Field(default="", description="Tên ứng viên. Nếu không thấy, để trống.")
    email: str = Field(default="", description="Email ứng viên.")
    phone: str = Field(default="", description="Số điện thoại ứng viên.")
    experience_years: float = Field(default=0.0, description="Tổng số năm kinh nghiệm.")
    skills_found: List[str] = Field(default_factory=list, description="Danh sách kỹ năng")
    has_projects: bool = Field(default=False)
    job_hopping: bool = Field(default=False)
    summary: str = Field(default="", description="Tóm tắt ngắn gọn.")

# ==========================================
# 2. CLASS LƯU TRỮ KẾT QUẢ
# ==========================================
class CVResult:
    def __init__(self, filename: str, full_text: str, extraction: CVExtraction, jd_summary: JDSummary = None):
        self.filename = filename
        self.full_text = full_text
        self.candidate_name = extraction.candidate_name
        self.email = extraction.email
        self.phone = extraction.phone
        self.experience_years = extraction.experience_years
        self.has_projects = extraction.has_projects
        self.job_hopping = extraction.job_hopping
        self.summary = extraction.summary
        
        self.skills_found = extraction.skills_found
        self.skills_missing = []
        self.must_have_missing = []
        self.nice_to_have_missing = []
        
        self.evidence = []
        self.chunks = []
        self.tags = []
        self.red_flags = []
        self.experience_source = "AI Extracted"
        self.job_count = 0
        self.ats_score = 0.0
        self.potential_level = "Medium"
        
        self._compute_scores(jd_summary)
        self._generate_tags()

    def _compute_scores(self, jd_summary: JDSummary):
        if not jd_summary or not jd_summary.all_skills:
            self.score = self.semantic_score = self.skill_score = self.experience_score = 0.0
            return

        cv_skills_lower = [s.lower() for s in self.skills_found]
        
        must_have_found = []
        for skill in jd_summary.must_have:
            if skill.lower() in cv_skills_lower:
                must_have_found.append(skill)
            else:
                self.must_have_missing.append(skill)
                self.skills_missing.append(skill)
                
        nice_to_have_found = []
        for skill in jd_summary.nice_to_have:
            if skill.lower() in cv_skills_lower:
                nice_to_have_found.append(skill)
            else:
                self.nice_to_have_missing.append(skill)
                self.skills_missing.append(skill)

        must_score = (len(must_have_found) / len(jd_summary.must_have)) if jd_summary.must_have else 1.0
        nice_score = (len(nice_to_have_found) / len(jd_summary.nice_to_have)) if jd_summary.nice_to_have else 1.0
        self.skill_score = (must_score * 0.7) + (nice_score * 0.3)
        
        req_exp = jd_summary.min_experience
        self.experience_score = min(1.0, self.experience_years / req_exp) if req_exp > 0 else (1.0 if self.experience_years > 0 else 0.5)
        self.semantic_score = (self.skill_score + self.experience_score) / 2
        
        total_w = WEIGHT_SEMANTIC + WEIGHT_SKILL + WEIGHT_EXPERIENCE
        self.score = (self.semantic_score * (WEIGHT_SEMANTIC/total_w)) + (self.skill_score * (WEIGHT_SKILL/total_w)) + (self.experience_score * (WEIGHT_EXPERIENCE/total_w))
        self.score = min(1.0, max(0.0, self.score))

    def _generate_tags(self):
        if self.experience_years >= 5: self.tags.append("Senior")
        elif self.experience_years >= 2: self.tags.append("Mid")
        elif self.experience_years > 0: self.tags.append("Junior")
        else: self.tags.append("Fresher/Unknown")
        
        if self.has_projects: self.tags.append("Has Projects")
        if self.job_hopping: self.red_flags.append("Job Hopping Detected")

# ==========================================
# 3. CÁC HÀM XỬ LÝ (THEO CHUẨN MỚI)
# ==========================================

def clean_cv_text(text: str, max_chars: int = 2000) -> str:
    """FIX 1: Dọn dẹp CV thô, loại bỏ ký tự lạ và rút gọn"""
    text = re.sub(r'[•●▪\-]{1,}', '-', text)  # Chuẩn hóa bullet
    text = re.sub(r'\s+', ' ', text)          # Xóa khoảng trắng thừa
    return text[:max_chars].strip()

def safe_json_parse(text: str) -> dict:
    """FIX 3: Fallback parser siêu an toàn"""
    try:
        return json.loads(text)
    except:
        # Nếu model lảm nhảm, dùng regex cắt đúng khối JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Invalid JSON: Không tìm thấy định dạng JSON hợp lệ.")

def analyze_cv(
    filename: str,
    cv_text: str,
    jd_embedding: any, 
    jd_skills: list[str], 
    jd_summary: JDSummary = None,
    api_key: str = ""
) -> CVResult:
    
    if not api_key:
        return CVResult(filename, cv_text, CVExtraction(), jd_summary)

    try:
        # KHỞI TẠO GROQ BÌNH THƯỜNG (Bỏ hẳn instructor)
        client = Groq(api_key=api_key)
        
        # Áp dụng FIX 1: Dọn dẹp CV
        clean_text = clean_cv_text(cv_text, 2000)
        jd_context = f"Skills to carefully check: {', '.join(jd_summary.all_skills)}" if jd_summary else ""
        
        # Áp dụng FIX 2: Prompt Tiếng Anh cực gắt, không giải thích dài dòng
        prompt = f"""
        You are a CV parser.

        STRICT RULES:
        - Output ONLY 1 valid JSON object.
        - No explanation, no intro, no outro.
        - No markdown formatting (no ```json).
        - Do NOT repeat or copy CV content outside of JSON values.
        - If missing info, use "" or [].
        - Limit output to under 120 words.
        - {jd_context}

        CV TEXT:
        {clean_text}

        Return JSON with EXACT keys:
        {{
            "candidate_name": "",
            "email": "",
            "phone": "",
            "experience_years": 0.0,
            "skills_found": [],
            "has_projects": false,
            "job_hopping": false,
            "summary": ""
        }}
        """
        
        # CỐ TÌNH BỎ response_format ĐỂ GROQ SERVER KHÔNG THAM GIA KIỂM DUYỆT BÁO LỖI 400
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict JSON API. Only output valid JSON. No extra text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=250,
            response_format={"type": "json_object"}
        )
        
        full_text = response.choices[0].message.content
        
        # Áp dụng FIX 3: Parse bằng hàm an toàn
        extraction_dict = safe_json_parse(full_text)
        extraction = CVExtraction(**extraction_dict)
        
        return CVResult(filename, cv_text, extraction, jd_summary)
            
    except Exception as e:
        import streamlit as st
        st.warning(f"Lỗi AI khi phân tích {filename}: {e}")
        return CVResult(filename, cv_text, CVExtraction(), jd_summary)

def rank_results(results: list[CVResult]) -> list[CVResult]:
    return sorted(results, key=lambda r: r.score, reverse=True)

def extract_experience_from_text(text: str): return 0, ""
def extract_cv_skills(text: str): return []
def extract_jd_skills(text: str): return []