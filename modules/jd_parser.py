"""
modules/jd_parser.py
Core logic for parsing Job Descriptions (Generative AI Edition - Clean Fallback)
"""
from __future__ import annotations
import json
import re
from groq import Groq
from pydantic import BaseModel, Field
from typing import List

# ==========================================
# 1. KHUÔN ĐÚC DỮ LIỆU JD (PYDANTIC)
# ==========================================
class JDSummary(BaseModel):
    position_title: str = Field(default="")
    level: str = Field(default="")
    min_experience: int = Field(default=0)
    must_have: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)
    all_skills: List[str] = Field(default_factory=list)

# ==========================================
# 2. HÀM FALLBACK AN TOÀN (CẮT JSON TỰ ĐỘNG)
# ==========================================
def safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("AI không trả về JSON hợp lệ.")

# ==========================================
# 3. HÀM PHÂN TÍCH CHÍNH (KHÔNG DÙNG INSTRUCTOR)
# ==========================================
def parse_jd(jd_text: str, api_key: str = "") -> JDSummary:
    # Nếu không có nội dung, trả về rỗng
    if not jd_text or not jd_text.strip():
        return JDSummary()
        
    # Nếu không có API Key, trả về title dự phòng
    if not api_key:
        title = jd_text.split("\n")[0][:80] if jd_text else ""
        return JDSummary(position_title=title)

    try:
        # Gọi thẳng Groq thuần, không bọc qua thư viện nào khác
        client = Groq(api_key=api_key)
        
        prompt = f"""
        You are an expert HR system parsing a Job Description.
        
        STRICT RULES:
        - Output ONLY 1 valid JSON object. No markdown. No text outside JSON.
        
        JD TEXT:
        {jd_text[:5000]}
        
        Return JSON EXACTLY matching this structure:
        {{
            "position_title": "Tên vị trí",
            "level": "Junior/Mid/Senior/...",
            "min_experience": 0,
            "must_have": ["skill 1", "skill 2"],
            "nice_to_have": ["skill 3"],
            "all_skills": ["skill 1", "skill 2", "skill 3"]
        }}
        """
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a JSON-only API. Never output normal text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1024
        )
        
        full_text = response.choices[0].message.content
        
        # Dùng hàm cắt Regex để nhặt JSON chuẩn xác
        extraction_dict = safe_json_parse(full_text)
        
        summary = JDSummary(**extraction_dict)
        # Gộp mảng để UI dễ dùng
        summary.all_skills = list(set(summary.must_have + summary.nice_to_have))
        
        return summary
        
    except Exception as e:
        import streamlit as st
        # Thông báo lỗi mới (Nếu bạn thấy dòng chữ Lỗi AI phân tích JD cũ nghĩa là code chưa lưu)
        st.error(f"Lỗi hệ thống Groq API khi đọc JD: {e}")
        return JDSummary()