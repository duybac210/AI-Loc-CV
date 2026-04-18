"""
app.py – Main Streamlit application for AI CV Screening Tool.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from config import (
    APP_DESCRIPTION,
    APP_ICON,
    APP_TITLE,
    SCORE_HIGH,
    SCORE_MED,
    WEIGHT_SEMANTIC,
    WEIGHT_SKILL,
    WEIGHT_EXPERIENCE,
    MIN_CANDIDATES_FOR_INTERVIEW,
    MIN_CANDIDATES_FOR_LOW_WARNING,
    LOW_MATCH_PCT_THRESHOLD,
    SKILL_GAP_WARNING_THRESHOLD,
)
from modules.cv_analyzer import CVResult, analyze_cv, extract_jd_skills, rank_results
from modules.database_manager import (
    delete_session,
    get_session_jd,
    get_session_results,
    init_db,
    list_sessions,
    save_session,
    update_decision,
)
from modules.embedding_manager import encode, get_model
from modules.export_manager import results_to_dataframe, to_csv_bytes, to_excel_bytes, to_pdf_bytes
from modules.jd_parser import JDSummary, parse_jd
from modules.llm_analyzer import LLMConfig, generate_llm_summary, SUPPORTED_PROVIDERS
from modules.pdf_processor import extract_text_from_pdf, extract_text_from_docx, OCR_PREFIX

# ---------------------------------------------------------------------------
# Page config & DB init
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
init_db()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_color(score: float) -> str:
    if score >= SCORE_HIGH:
        return "🟢"
    if score >= SCORE_MED:
        return "🟡"
    return "🔴"


def score_badge(score: float) -> str:
    pct = round(score * 100, 1)
    icon = score_color(score)
    return f"{icon} **{pct}%**"


@st.cache_resource(show_spinner="Loading multilingual AI model (first time ~60 s)…")
def load_model():
    return get_model()


def _compute_display_scores(
    results: list[CVResult],
    w_sem: float,
    w_sk: float,
    w_exp: float,
) -> tuple[list[CVResult], list[float]]:
    """
    Re-rank *results* using caller-supplied weights (normalised internally).

    Returns sorted (results, display_scores) pairs.
    """
    total = max(w_sem + w_sk + w_exp, 0.001)
    ws = w_sem / total
    wsk = w_sk / total
    we = w_exp / total

    display_scores: list[float] = []
    for r in results:
        if r.skills_found or r.skills_missing:  # JD had detected skills
            s = ws * r.semantic_score + wsk * r.skill_score + we * r.experience_score
            display_scores.append(float(np.clip(s, 0.0, 1.0)))
        else:
            display_scores.append(r.semantic_score)

    sorted_pairs = sorted(
        zip(results, display_scores), key=lambda x: x[1], reverse=True
    )
    return [r for r, _ in sorted_pairs], [s for _, s in sorted_pairs]


def _skill_badge(skill: str, color: str = "#4a90e2") -> str:
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;margin:2px;font-size:0.82em;display:inline-block">'
        f"{skill}</span>"
    )


def _tag_color(tag: str) -> str:
    if tag.startswith("Strong "):
        return "#27ae60"
    if tag.startswith("Missing "):
        return "#e74c3c"
    if tag in ("Has Projects",):
        return "#2980b9"
    if tag in ("No Projects",):
        return "#95a5a6"
    if tag == "Senior":
        return "#8e44ad"
    if tag == "Mid":
        return "#2471a3"
    if tag in ("Junior", "Fresher/Unknown"):
        return "#1a8a6e"
    return "#7f8c8d"


# ---------------------------------------------------------------------------
# UI – Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/artificial-intelligence.png",
            width=80,
        )
        st.title(APP_TITLE)
        st.caption(APP_DESCRIPTION)
        st.divider()

        # Weight sliders
        st.markdown("### ⚖️ Trọng số chấm điểm")
        st.caption("Điều chỉnh mức độ ưu tiên của từng tiêu chí (tự động chuẩn hóa)")
        w_sem = st.slider(
            "Ngữ nghĩa (Semantic)", 0, 100,
            int(WEIGHT_SEMANTIC * 100), 5, key="w_semantic",
        )
        w_sk = st.slider(
            "Kỹ năng (Skill coverage)", 0, 100,
            int(WEIGHT_SKILL * 100), 5, key="w_skill",
        )
        w_exp = st.slider(
            "Kinh nghiệm (Experience)", 0, 100,
            int(WEIGHT_EXPERIENCE * 100), 5, key="w_experience",
        )
        total_w = max(w_sem + w_sk + w_exp, 1)
        st.caption(
            f"**Phân bổ:** "
            f"Ngữ nghĩa {round(w_sem/total_w*100)}% · "
            f"Kỹ năng {round(w_sk/total_w*100)}% · "
            f"Kinh nghiệm {round(w_exp/total_w*100)}%"
        )
        st.divider()

        # LLM settings
        st.markdown("### ✨ AI nâng cao (LLM)")
        st.caption("Tạo nhận xét tự nhiên + gợi ý câu hỏi phỏng vấn")
        llm_enabled = st.toggle("Bật AI nâng cao", value=False, key="llm_enabled")
        if llm_enabled:
            llm_provider = st.selectbox(
                "Provider",
                options=list(SUPPORTED_PROVIDERS),
                key="llm_provider",
            )
            llm_api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="sk-... / AIza... / gsk_...",
                key="llm_api_key",
            )
            llm_model = st.text_input(
                "Model (để trống = mặc định)",
                placeholder="gpt-4o-mini / gemini-1.5-flash / llama3-8b-8192",
                key="llm_model",
            )
            if not llm_api_key:
                st.caption("ℹ️ Nhập API key để kích hoạt. Key không được lưu vào database.")
        st.divider()

        st.markdown("### Hướng dẫn sử dụng")
        st.markdown(
            """
1. Nhập **Mô tả công việc (JD)**
2. Xem **JD tóm tắt** & điều chỉnh **Bộ lọc**
3. Upload một hoặc nhiều **CV dạng PDF**
4. Nhấn **Phân tích CV**
5. Xem kết quả, shortlist ứng viên & tải xuống
6. Vào tab **Lịch sử** để xem lại
"""
        )
        st.divider()

        # Technical info in expander (hidden by default)
        with st.expander("⚙️ Thông tin kỹ thuật", expanded=False):
            st.markdown("**AI Engine:** `paraphrase-multilingual-MiniLM-L12-v2`")
            st.markdown("**Ngôn ngữ:** Tiếng Việt + Tiếng Anh (cross-lingual)")
            st.markdown("**Phương pháp:** Semantic embeddings + Skill coverage + Experience")
            st.markdown("**PDF hỗ trợ:** pdfplumber → PyMuPDF → PyPDF2 → OCR (Tesseract)")
            st.markdown("**DOCX hỗ trợ:** python-docx (paragraphs + tables)")


# ---------------------------------------------------------------------------
# JD Summary Card + Pre-filter
# ---------------------------------------------------------------------------

def _render_jd_summary_card(jd_summary: JDSummary) -> None:
    """Display the AI-parsed JD summary card."""
    with st.container():
        st.markdown("#### 🔍 JD đã phân tích (AI)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**📌 Vị trí:** {jd_summary.position_title or '—'}")
        with c2:
            level_label = jd_summary.level.capitalize() if jd_summary.level else "—"
            st.markdown(f"**🎓 Level:** {level_label}")
        with c3:
            exp_label = f"{jd_summary.min_experience}+ năm" if jd_summary.min_experience else "—"
            st.markdown(f"**📅 KN tối thiểu:** {exp_label}")

        if jd_summary.must_have:
            badges = " ".join(_skill_badge(s, "#c0392b") for s in jd_summary.must_have)
            st.markdown("**🔴 Must-have:**  " + badges, unsafe_allow_html=True)
        if jd_summary.nice_to_have:
            badges = " ".join(_skill_badge(s, "#d68910") for s in jd_summary.nice_to_have)
            st.markdown("**🟡 Nice-to-have:**  " + badges, unsafe_allow_html=True)
        if not jd_summary.must_have and not jd_summary.nice_to_have:
            st.caption("Không phát hiện kỹ năng cụ thể – AI sẽ dùng độ tương đồng ngữ nghĩa thuần tuý.")


def _render_pre_filter(jd_summary: JDSummary) -> None:
    """Show pre-filter controls below the JD summary card."""
    with st.expander("🔧 Bộ lọc ứng viên (pre-filter)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                "Kinh nghiệm tối thiểu (năm)",
                min_value=0, max_value=20,
                value=jd_summary.min_experience,
                step=1, key="pre_min_exp",
            )
        with col2:
            st.multiselect(
                "Kỹ năng bắt buộc phải có",
                options=jd_summary.all_skills,
                default=[s for s in jd_summary.must_have if s in jd_summary.all_skills],
                key="pre_req_skills",
            )
        st.checkbox("Ẩn CV có cảnh báo chất lượng (red flag)", value=False, key="pre_hide_red")
        st.caption(
            "💡 CV không qua bộ lọc vẫn được AI phân tích đầy đủ, "
            "nhưng được đánh dấu 🚫 và xếp cuối danh sách."
        )


# ---------------------------------------------------------------------------
# Insight Dashboard
# ---------------------------------------------------------------------------

def _render_insight_dashboard(
    results: list[CVResult],
    display_scores: list[float],
    jd_skills: list[str],
    jd_summary: JDSummary,
) -> None:
    st.subheader("📊 Insight Dashboard")

    total = len(results)
    high = sum(1 for s in display_scores if s >= SCORE_HIGH)
    med = sum(1 for s in display_scores if SCORE_MED <= s < SCORE_HIGH)
    low = sum(1 for s in display_scores if s < SCORE_MED)

    # ---- row 1: key metrics ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📄 Tổng CV", total)
    m2.metric("🟢 Phù hợp cao (≥70%)", high)
    m3.metric("🟡 Trung bình (50–70%)", med)
    m4.metric("🔴 Chưa phù hợp (<50%)", low)

    col_chart1, col_chart2 = st.columns(2)

    # ---- donut distribution ----
    with col_chart1:
        fig_donut = go.Figure(data=[
            go.Pie(
                labels=["🟢 Phù hợp cao", "🟡 Trung bình", "🔴 Chưa phù hợp"],
                values=[high, med, low],
                hole=0.55,
                marker_colors=["#2ecc71", "#f39c12", "#e74c3c"],
                textinfo="label+value",
            )
        ])
        fig_donut.update_layout(
            title="Phân bổ ứng viên",
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ---- skill presence bar chart ----
    with col_chart2:
        if jd_skills:
            skill_pcts = [
                round(
                    sum(1 for r in results if skill in r.skills_found) / total * 100, 1
                )
                for skill in jd_skills
            ]
            fig_skills = go.Figure(
                go.Bar(
                    x=skill_pcts,
                    y=jd_skills,
                    orientation="h",
                    marker_color=[
                        "#2ecc71" if p >= 50 else "#f39c12" if p >= 25 else "#e74c3c"
                        for p in skill_pcts
                    ],
                    text=[f"{p}%" for p in skill_pcts],
                    textposition="outside",
                )
            )
            fig_skills.update_layout(
                title="% ứng viên có kỹ năng JD",
                xaxis=dict(title="%", range=[0, 120]),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_skills, use_container_width=True)
        else:
            st.info("Không có kỹ năng JD để thống kê.")

    # ---- experience distribution histogram ----
    exp_years_list = [r.experience_years for r in results]
    if any(y > 0 for y in exp_years_list):
        col_hist, col_avg = st.columns(2)
        with col_hist:
            fig_hist = px.histogram(
                x=exp_years_list, nbins=8,
                labels={"x": "Số năm kinh nghiệm", "y": "Số ứng viên"},
                title="Phân bổ kinh nghiệm",
                color_discrete_sequence=["#4a90e2"],
            )
            fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_avg:
            avg_sem = round(np.mean([r.semantic_score for r in results]) * 100, 1)
            avg_sk = round(np.mean([r.skill_score for r in results]) * 100, 1)
            avg_exp = round(np.mean([y for y in exp_years_list if y > 0]), 1) if any(y > 0 for y in exp_years_list) else 0
            avg_total = round(np.mean(display_scores) * 100, 1)

            st.markdown("#### 📈 Điểm trung bình")
            st.metric("🎯 Tổng điểm TB", f"{avg_total}%")
            st.metric("🧠 Ngữ nghĩa TB", f"{avg_sem}%")
            st.metric("🔧 Kỹ năng TB", f"{avg_sk}%")
            st.metric("📅 Kinh nghiệm TB", f"{avg_exp} năm")

    # ---- next action suggestions ----
    st.markdown("#### 💡 Gợi ý hành động")
    suggestions: list[str] = []

    high_pct = high / total * 100 if total else 0
    if high >= MIN_CANDIDATES_FOR_INTERVIEW:
        suggestions.append(
            f"✅ Có **{high} ứng viên phù hợp cao** – nên mời phỏng vấn Top {min(high, MIN_CANDIDATES_FOR_INTERVIEW)}."
        )
    elif high > 0:
        suggestions.append(
            f"✅ Có **{high} ứng viên phù hợp cao** – mời phỏng vấn."
        )
    else:
        suggestions.append("⚠️ **Không có ứng viên phù hợp cao** – cân nhắc đăng tuyển thêm.")

    if high_pct < LOW_MATCH_PCT_THRESHOLD and total >= MIN_CANDIDATES_FOR_LOW_WARNING:
        suggestions.append(
            f"📉 Tỷ lệ phù hợp thấp (<{LOW_MATCH_PCT_THRESHOLD}%) – xem xét **hạ yêu cầu** hoặc **mở rộng phạm vi tuyển**."
        )

    if jd_skills:
        for skill in jd_skills:
            missing_count = sum(1 for r in results if skill in r.skills_missing)
            pct = round(missing_count / total * 100)
            if pct >= SKILL_GAP_WARNING_THRESHOLD:
                suggestions.append(
                    f"🔍 **{pct}% ứng viên thiếu {skill}** – xem xét giảm yêu cầu hoặc cung cấp đào tạo."
                )

    red_flag_count = sum(1 for r in results if r.red_flags)
    if red_flag_count > 0:
        suggestions.append(
            f"⚠️ **{red_flag_count} CV có cảnh báo chất lượng** – nên xem xét kỹ trước khi liên hệ."
        )

    for sug in suggestions:
        st.markdown(f"- {sug}")


# ---------------------------------------------------------------------------
# Candidate Cards
# ---------------------------------------------------------------------------

def _get_llm_config() -> LLMConfig:
    """Build LLMConfig from current session state."""
    if not st.session_state.get("llm_enabled", False):
        return LLMConfig()
    return LLMConfig(
        provider=st.session_state.get("llm_provider", "openai"),
        api_key=st.session_state.get("llm_api_key", ""),
        model=st.session_state.get("llm_model", ""),
    )


def _decision_color(decision: str) -> str:
    if decision == "Shortlist":
        return "#27ae60"
    if decision == "Consider":
        return "#f39c12"
    if decision == "Reject":
        return "#e74c3c"
    return "#95a5a6"


def _render_candidate_cards(
    display_list: list[tuple[CVResult, float]],
    pre_filtered_out: list[tuple[CVResult, float, list[str]]],
    jd_text: str = "",
) -> None:
    st.subheader("📋 Phân tích chi tiết ứng viên")

    llm_config = _get_llm_config()

    # Show passing candidates
    for rank, (result, score) in enumerate(display_list, start=1):
        pct = round(score * 100, 1)
        icon = score_color(score)

        tag_html = " ".join(_skill_badge(t, _tag_color(t)) for t in result.tags)
        red_flag_html = ""
        if result.red_flags:
            red_flag_html = " " + " ".join(
                _skill_badge(f"⚠️ {f}", "#c0392b") for f in result.red_flags
            )

        # Show decision badge in header if set
        decision_state_key = f"decision_{result.filename}_{rank}"
        current_decision = st.session_state.get(decision_state_key, "")
        decision_badge = ""
        if current_decision:
            dcolor = _decision_color(current_decision)
            decision_badge = f' <span style="background:{dcolor};color:white;padding:2px 8px;border-radius:10px;font-size:0.8em">{current_decision}</span>'

        # Contact info in header
        contact_str = ""
        if result.candidate_name:
            contact_str = f" — {result.candidate_name}"

        header_html = f"#{rank}{contact_str} · {result.filename} – {icon} {pct}%{decision_badge}"

        with st.expander(header_html, expanded=(rank == 1)):
            # Contact info row
            if result.candidate_name or result.email or result.phone:
                ci1, ci2, ci3 = st.columns(3)
                ci1.markdown(f"**👤 Tên:** {result.candidate_name or '—'}")
                ci2.markdown(f"**✉️ Email:** {result.email or '—'}")
                ci3.markdown(f"**📞 SĐT:** {result.phone or '—'}")
                st.markdown("")

            # Tags row
            if result.tags or result.red_flags:
                st.markdown(tag_html + red_flag_html, unsafe_allow_html=True)
                st.markdown("")

            # Score breakdown with progress bars
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Tổng điểm", f"{pct}%")
            sc1.progress(score)
            sc2.metric("Ngữ nghĩa", f"{round(result.semantic_score * 100, 1)}%")
            sc2.progress(result.semantic_score)
            sc3.metric("Kỹ năng", f"{round(result.skill_score * 100, 1)}%")
            sc3.progress(result.skill_score)
            sc4.metric(
                "Kinh nghiệm",
                f"{result.experience_years} năm" if result.experience_years else "—",
            )

            # Must-have / nice-to-have gap
            if result.must_have_missing or result.nice_to_have_missing:
                gap_col1, gap_col2 = st.columns(2)
                with gap_col1:
                    if result.must_have_missing:
                        badges = " ".join(_skill_badge(s, "#c0392b") for s in result.must_have_missing)
                        st.markdown("**🔴 Thiếu must-have:** " + badges, unsafe_allow_html=True)
                with gap_col2:
                    if result.nice_to_have_missing:
                        badges = " ".join(_skill_badge(s, "#d68910") for s in result.nice_to_have_missing)
                        st.markdown("**🟡 Thiếu nice-to-have:** " + badges, unsafe_allow_html=True)

            # Structured explanation
            st.markdown("**Phân tích chi tiết:**")
            exp_col, skill_col = st.columns(2)
            with exp_col:
                st.markdown("**✅ Điểm mạnh:**")
                if result.experience_years > 0:
                    st.markdown(f"- {result.experience_years} năm kinh nghiệm")
                if result.has_projects:
                    st.markdown("- Có dự án thực tế")
                if result.skills_found:
                    for skill in result.skills_found:
                        st.markdown(f"- ✔ {skill}")
                else:
                    st.markdown("*Chưa khớp kỹ năng nào trong JD*")
            with skill_col:
                st.markdown("**❌ Còn thiếu:**")
                if result.skills_missing:
                    for skill in result.skills_missing:
                        st.markdown(f"- ✗ {skill}")
                else:
                    st.markdown("*Đủ tất cả kỹ năng yêu cầu!* 🎉")

            # Red flags
            if result.red_flags:
                st.warning(
                    "⚠️ **Cảnh báo chất lượng CV:**\n"
                    + "\n".join(f"- {f}" for f in result.red_flags)
                )

            # LLM-enhanced summary
            summary_key = f"llm_summary_{result.filename}_{rank}"
            questions_key = f"llm_questions_{result.filename}_{rank}"

            if llm_config.is_configured():
                if summary_key not in st.session_state:
                    with st.spinner("✨ AI đang tạo nhận xét..."):
                        llm_sum, llm_questions, llm_decision = generate_llm_summary(
                            llm_config, result.full_text, jd_text, result
                        )
                    st.session_state[summary_key] = llm_sum
                    st.session_state[questions_key] = llm_questions
                    if llm_decision and not current_decision:
                        st.session_state[decision_state_key] = llm_decision
                st.info(f"✨ **Nhận xét AI nâng cao:** {st.session_state[summary_key]}")
                questions = st.session_state.get(questions_key, [])
                if questions:
                    with st.expander("💬 Gợi ý câu hỏi phỏng vấn (AI)", expanded=False):
                        for i, q in enumerate(questions, 1):
                            st.markdown(f"{i}. {q}")
            else:
                st.info(f"💡 **Nhận xét AI:** {result.summary}")

            # HR Decision buttons
            st.markdown("**Quyết định HR:**")
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
            with btn_col1:
                if st.button("✅ Shortlist", key=f"sl_{result.filename}_{rank}"):
                    st.session_state[decision_state_key] = "Shortlist"
                    st.rerun()
            with btn_col2:
                if st.button("⏸ Cân nhắc", key=f"cs_{result.filename}_{rank}"):
                    st.session_state[decision_state_key] = "Consider"
                    st.rerun()
            with btn_col3:
                if st.button("❌ Loại", key=f"rj_{result.filename}_{rank}"):
                    st.session_state[decision_state_key] = "Reject"
                    st.rerun()
            with btn_col4:
                if current_decision:
                    dcolor = _decision_color(current_decision)
                    st.markdown(
                        f'<span style="background:{dcolor};color:white;padding:4px 12px;'
                        f'border-radius:12px;font-weight:bold">{current_decision}</span>',
                        unsafe_allow_html=True,
                    )

            # Evidence
            st.markdown("**Đoạn text liên quan nhất (AI):**")
            for i, (chunk, sim) in enumerate(result.evidence, start=1):
                sim_pct = round(sim * 100, 1)
                st.markdown(
                    f"<div style='background:#f0f4ff;border-left:4px solid #4a90e2;"
                    f"padding:8px 12px;margin:4px 0;border-radius:4px;color:#1a1a1a;'>"
                    f"<small>Bằng chứng #{i} – tương đồng {sim_pct}%</small><br>{chunk}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Show filtered-out candidates at bottom
    if pre_filtered_out:
        with st.expander(
            f"🚫 {len(pre_filtered_out)} ứng viên bị lọc ra (pre-filter)",
            expanded=False,
        ):
            st.caption(
                "Các ứng viên này không đáp ứng bộ lọc bạn đặt. "
                "AI vẫn phân tích đầy đủ – bạn có thể xem chi tiết bên dưới."
            )
            for result, score, reasons in pre_filtered_out:
                pct = round(score * 100, 1)
                with st.expander(
                    f"🚫 {result.filename} – {round(score * 100, 1)}% | "
                    f"Lý do lọc: {'; '.join(reasons)}",
                    expanded=False,
                ):
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Điểm", f"{pct}%")
                    sc2.metric("KN", f"{result.experience_years} năm" if result.experience_years else "—")
                    sc3.metric("Kỹ năng có", str(len(result.skills_found)))
                    if result.skills_found:
                        st.markdown("**Kỹ năng có:** " + ", ".join(result.skills_found))
                    if result.skills_missing:
                        st.markdown("**Thiếu:** " + ", ".join(result.skills_missing))


# ---------------------------------------------------------------------------
# Comparison View
# ---------------------------------------------------------------------------

def _render_comparison_view(
    sorted_results: list[CVResult],
    display_scores: list[float],
) -> None:
    st.subheader("⚖️ So sánh ứng viên")
    filenames = [r.filename for r in sorted_results]
    selected = st.multiselect(
        "Chọn 2–4 ứng viên để so sánh side-by-side",
        options=filenames,
        default=filenames[:min(3, len(filenames))],
        key="compare_select",
    )
    if len(selected) < 2:
        st.caption("Chọn ít nhất 2 ứng viên để so sánh.")
        return

    chosen_pairs = [
        (r, s) for r, s in zip(sorted_results, display_scores)
        if r.filename in selected
    ]

    rows = []
    for r, s in chosen_pairs:
        rows.append({
            "Ứng viên": r.filename,
            "Tổng điểm (%)": round(s * 100, 1),
            "Ngữ nghĩa (%)": round(r.semantic_score * 100, 1),
            "Kỹ năng (%)": round(r.skill_score * 100, 1),
            "Kinh nghiệm (năm)": r.experience_years if r.experience_years else "—",
            "Có dự án": "✔" if r.has_projects else "✗",
            "Kỹ năng có": ", ".join(r.skills_found) or "—",
            "Thiếu kỹ năng": ", ".join(r.skills_missing) or "—",
            "Red Flags": len(r.red_flags),
            "Tags": " | ".join(r.tags),
        })

    df_cmp = pd.DataFrame(rows).set_index("Ứng viên").T
    st.dataframe(df_cmp, use_container_width=True)

    # ---- Radar / Spider chart ----
    _render_radar_chart(chosen_pairs)


def _render_radar_chart(
    chosen_pairs: list[tuple[CVResult, float]],
) -> None:
    """
    Radar chart comparing multiple candidates across 5 dimensions:
    Total Score, Semantic, Skill Coverage, Experience, Projects.
    """
    if len(chosen_pairs) < 2:
        return

    categories = ["Tổng điểm", "Ngữ nghĩa", "Kỹ năng", "Kinh nghiệm", "Dự án"]

    fig_radar = go.Figure()
    for r, s in chosen_pairs:
        exp_norm = min(100.0, (r.experience_years or 0) / 5.0 * 100)
        project_score = 100.0 if r.has_projects else 0.0
        values = [
            round(s * 100, 1),
            round(r.semantic_score * 100, 1),
            round(r.skill_score * 100, 1),
            round(exp_norm, 1),
            project_score,
        ]
        # Close the polygon
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        label = r.candidate_name or r.filename
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill="toself",
            opacity=0.5,
            name=label[:40],
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        title="📡 Radar Chart – so sánh đa chiều ứng viên",
        showlegend=True,
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ---------------------------------------------------------------------------
# UI – Analysis Tab
# ---------------------------------------------------------------------------

def render_analysis_tab():
    st.markdown(APP_DESCRIPTION)
    st.divider()

    # Warm-up model
    load_model()

    col_jd, col_cv = st.columns([1, 1], gap="large")

    with col_jd:
        st.subheader("📋 Mô tả công việc (JD)")
        jd_text = st.text_area(
            label="Nhập mô tả công việc (tiếng Việt hoặc tiếng Anh)",
            height=300,
            placeholder=(
                "Ví dụ: Chúng tôi tìm kiếm Senior Backend Developer với 3+ năm kinh nghiệm "
                "Python, FastAPI, PostgreSQL và Docker. Yêu cầu tư duy phân tích tốt…"
            ),
            key="jd_input",
        )

    with col_cv:
        st.subheader("📄 Upload CV (PDF / DOCX)")
        uploaded_files = st.file_uploader(
            label="Upload một hoặc nhiều file CV dạng PDF hoặc DOCX",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="cv_upload",
        )
        if uploaded_files:
            st.success(f"✅ Đã tải {len(uploaded_files)} file(s)")

    # JD Summary Card + Pre-filter (reactive: shows as user types)
    jd_summary = JDSummary()
    if jd_text and jd_text.strip():
        jd_summary = parse_jd(jd_text)
        st.divider()
        _render_jd_summary_card(jd_summary)
        _render_pre_filter(jd_summary)

    st.divider()

    run = st.button(
        "🚀 Phân tích CV",
        type="primary",
        disabled=not (jd_text and uploaded_files),
    )

    if run:
        # -------------------------------------------------------------------
        # Processing
        # -------------------------------------------------------------------
        with st.spinner("🔍 AI đang phân tích CV…"):
            jd_embedding = encode([jd_text])[0]
            jd_skills = extract_jd_skills(jd_text)

            raw_results: list[CVResult] = []
            progress = st.progress(0, text="Đang xử lý CV…")
            errors: list[str] = []
            ocr_files: list[str] = []

            for i, uploaded_file in enumerate(uploaded_files):
                cv_bytes = uploaded_file.read()
                fname_lower = uploaded_file.name.lower()

                if fname_lower.endswith(".docx"):
                    cv_text = extract_text_from_docx(cv_bytes)
                else:
                    cv_text = extract_text_from_pdf(cv_bytes)

                if cv_text.startswith("[PDF extraction failed") or cv_text.startswith("[DOCX extraction failed"):
                    errors.append(uploaded_file.name)
                elif cv_text.startswith(OCR_PREFIX):
                    ocr_files.append(uploaded_file.name)
                    cv_text = cv_text[len(OCR_PREFIX):]

                result = analyze_cv(
                    filename=uploaded_file.name,
                    cv_text=cv_text,
                    jd_embedding=jd_embedding,
                    jd_skills=jd_skills,
                    jd_summary=jd_summary,
                )
                raw_results.append(result)
                progress.progress(
                    (i + 1) / len(uploaded_files),
                    text=f"Đã xử lý: {uploaded_file.name}",
                )

            progress.empty()
            ranked = rank_results(raw_results)

            # Save to database
            session_id = save_session(jd_text, jd_skills, ranked)

        if errors:
            st.warning(
                f"⚠️ Không thể trích xuất text từ {len(errors)} file(s): "
                f"{', '.join(errors)}. "
                "File có thể là PDF scan, bị mã hoá, hoặc DOCX bị lỗi – điểm sẽ thấp bất thường."
            )
        if ocr_files:
            st.info(
                f"🔍 **OCR được dùng cho {len(ocr_files)} file(s):** {', '.join(ocr_files)}. "
                "Đây là CV dạng ảnh/scan – text được nhận dạng tự động bằng Tesseract OCR."
            )

        st.success(f"✅ Đã lưu kết quả vào lịch sử (Session #{session_id})")

        # Store in session_state for reactive post-filter / weight adjustment
        st.session_state["results_data"] = {
            "raw_results": raw_results,
            "jd_skills": jd_skills,
            "jd_summary": jd_summary,
            "jd_text": jd_text,
        }

    # Show results if available (from this run or previous interaction)
    if "results_data" in st.session_state:
        data = st.session_state["results_data"]
        _render_results(
            data["raw_results"],
            data["jd_skills"],
            data["jd_summary"],
            data.get("jd_text", ""),
        )
    elif not run:
        if not jd_text:
            st.info("👆 Vui lòng nhập Mô tả công việc để bắt đầu.")
        elif not uploaded_files:
            st.info("👆 Vui lòng upload ít nhất một CV PDF.")


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def _render_results(
    raw_results: list[CVResult],
    jd_skills: list[str],
    jd_summary: JDSummary,
    jd_text: str = "",
) -> None:
    """Full results section with dashboard, cards, comparison, export."""

    # Read current weights from sidebar sliders
    w_sem = st.session_state.get("w_semantic", int(WEIGHT_SEMANTIC * 100))
    w_sk = st.session_state.get("w_skill", int(WEIGHT_SKILL * 100))
    w_exp = st.session_state.get("w_experience", 0)

    # Re-rank with current weights
    sorted_results, display_scores = _compute_display_scores(
        raw_results, w_sem, w_sk, w_exp
    )

    st.divider()
    st.subheader("🏆 Kết quả xếp hạng")

    if jd_skills:
        st.markdown(f"**Kỹ năng phát hiện trong JD:** {', '.join(jd_skills)}")
    else:
        st.warning(
            "Không phát hiện kỹ năng cụ thể trong JD – điểm dựa hoàn toàn vào "
            "độ tương đồng ngữ nghĩa."
        )

    # Post-filter controls
    with st.expander("🔧 Tinh chỉnh & Lọc kết quả", expanded=False):
        pf1, pf2, pf3 = st.columns(3)
        with pf1:
            post_min_score = st.slider(
                "Điểm tối thiểu (%)", 0, 100, 0, 5, key="post_min_score"
            )
            top_n = st.number_input(
                "Hiển thị tối đa N ứng viên",
                min_value=1,
                max_value=len(sorted_results),
                value=min(10, len(sorted_results)),
                step=1,
                key="post_top_n",
            )
        with pf2:
            post_req_skills = st.multiselect(
                "Phải có kỹ năng",
                options=jd_skills,
                key="post_req_skills",
            )
        with pf3:
            post_hide_red = st.checkbox(
                "Ẩn CV có cảnh báo (red flag)", key="post_hide_red"
            )
        if st.button("🔄 Reset bộ lọc", key="reset_post_filter"):
            for k in ("post_min_score", "post_req_skills", "post_hide_red", "post_top_n"):
                st.session_state.pop(k, None)
            st.rerun()

    # Read pre-filter values
    pre_min_exp = int(st.session_state.get("pre_min_exp", 0))
    pre_req_skills = list(st.session_state.get("pre_req_skills", []))
    pre_hide_red = bool(st.session_state.get("pre_hide_red", False))

    # Separate passing and filtered-out candidates
    display_list: list[tuple[CVResult, float]] = []
    pre_filtered_out: list[tuple[CVResult, float, list[str]]] = []

    for result, score in zip(sorted_results, display_scores):
        # Pre-filter checks
        pre_reasons: list[str] = []
        if pre_min_exp > 0 and result.experience_years < pre_min_exp:
            pre_reasons.append(f"KN < {pre_min_exp} năm")
        for skill in pre_req_skills:
            if skill not in result.skills_found:
                pre_reasons.append(f"Thiếu {skill}")
        if pre_hide_red and result.red_flags:
            pre_reasons.append("Có red flag")

        # Post-filter checks
        post_pass = (
            score * 100 >= post_min_score
            and all(s in result.skills_found for s in post_req_skills)
            and not (post_hide_red and result.red_flags)
        )

        if pre_reasons or not post_pass:
            pre_filtered_out.append((result, score, pre_reasons))
        else:
            display_list.append((result, score))

    # Apply top_n
    display_list = display_list[:int(top_n)]

    # Show filter status
    if pre_filtered_out:
        total_filtered = len(pre_filtered_out)
        st.info(
            f"🔧 **Bộ lọc:** Hiển thị {len(display_list)} ứng viên | "
            f"{total_filtered} ứng viên bị lọc ra"
        )

    # -------------------------------------------------------------------
    # Bar chart (display score)
    # -------------------------------------------------------------------
    names = [r.filename for r, _ in display_list]
    scores_pct = [round(s * 100, 1) for _, s in display_list]
    semantic_pcts = [round(r.semantic_score * 100, 1) for r, _ in display_list]
    skill_pcts = [round(r.skill_score * 100, 1) for r, _ in display_list]
    colors = [
        "#2ecc71" if s >= SCORE_HIGH * 100 else "#f39c12" if s >= SCORE_MED * 100 else "#e74c3c"
        for s in scores_pct
    ]

    if names:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=scores_pct, y=names, orientation="h",
                marker_color=colors,
                text=[f"{s}%" for s in scores_pct],
                textposition="outside",
                name="Tổng điểm",
            )
        )
        fig.update_layout(
            xaxis=dict(title="Điểm phù hợp (%)", range=[0, 115]),
            yaxis=dict(autorange="reversed"),
            height=max(200, 60 * len(names)),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Score breakdown
        with st.expander("📊 Xem chi tiết điểm thành phần", expanded=False):
            total_w = max(w_sem + w_sk + w_exp, 1)
            fig2 = go.Figure()
            fig2.add_trace(
                go.Bar(
                    x=semantic_pcts, y=names, orientation="h",
                    name=f"Ngữ nghĩa ({round(w_sem/total_w*100)}%)",
                    marker_color="#4a90e2", opacity=0.8,
                )
            )
            fig2.add_trace(
                go.Bar(
                    x=skill_pcts, y=names, orientation="h",
                    name=f"Kỹ năng ({round(w_sk/total_w*100)}%)",
                    marker_color="#9b59b6", opacity=0.8,
                )
            )
            if w_exp > 0:
                exp_pcts = [round(r.experience_score * 100, 1) for r, _ in display_list]
                fig2.add_trace(
                    go.Bar(
                        x=exp_pcts, y=names, orientation="h",
                        name=f"Kinh nghiệm ({round(w_exp/total_w*100)}%)",
                        marker_color="#e67e22", opacity=0.8,
                    )
                )
            fig2.update_layout(
                barmode="group",
                xaxis=dict(title="Điểm (%)", range=[0, 110]),
                yaxis=dict(autorange="reversed"),
                height=max(200, 70 * len(names)),
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------------
    # Insight Dashboard
    # -------------------------------------------------------------------
    _render_insight_dashboard(
        [r for r, _ in display_list] + [r for r, _, _ in pre_filtered_out],
        [s for _, s in display_list] + [s for _, s, _ in pre_filtered_out],
        jd_skills,
        jd_summary,
    )

    # -------------------------------------------------------------------
    # Tabs: Detailed Cards | Shortlist | Real-time Filter Table
    # -------------------------------------------------------------------
    tab_cards, tab_shortlist, tab_table = st.tabs([
        "📋 Chi tiết ứng viên",
        "⭐ Danh sách Shortlist",
        "🔍 Bảng lọc real-time",
    ])

    with tab_cards:
        _render_candidate_cards(display_list, pre_filtered_out, jd_text=jd_text)

    with tab_shortlist:
        _render_shortlist_tab(display_list, sorted_results, display_scores)

    with tab_table:
        _render_realtime_filter_table(sorted_results, display_scores, jd_skills)

    # -------------------------------------------------------------------
    # Comparison view
    # -------------------------------------------------------------------
    if len(sorted_results) >= 2:
        st.divider()
        _render_comparison_view(sorted_results, display_scores)

    # -------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------
    st.divider()
    st.subheader("📥 Xuất kết quả")

    # Build export dataframe from display_list (respects current filters)
    all_for_export = [r for r, _ in display_list] + [r for r, _, _ in pre_filtered_out]
    df = results_to_dataframe(all_for_export)
    st.dataframe(df, use_container_width=True)

    col_csv, col_xlsx, col_pdf = st.columns(3)
    with col_csv:
        st.download_button(
            label="⬇️ Tải CSV",
            data=to_csv_bytes(df),
            file_name="cv_screening_results.csv",
            mime="text/csv",
        )
    with col_xlsx:
        st.download_button(
            label="⬇️ Tải Excel (Multi-sheet)",
            data=to_excel_bytes(df, jd_skills=jd_skills),
            file_name="cv_screening_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col_pdf:
        jd_snippet = jd_text[:120].replace("\n", " ") if jd_text else ""
        st.download_button(
            label="⬇️ Tải báo cáo PDF",
            data=to_pdf_bytes(all_for_export, jd_snippet=jd_snippet),
            file_name="cv_screening_report.pdf",
            mime="application/pdf",
        )


# ---------------------------------------------------------------------------
# Shortlist Tab
# ---------------------------------------------------------------------------

def _render_shortlist_tab(
    display_list: list[tuple[CVResult, float]],
    sorted_results: list[CVResult],
    display_scores: list[float],
) -> None:
    """Display only candidates that the HR has marked as Shortlist."""
    st.markdown("#### ⭐ Ứng viên đã shortlist")

    shortlisted = []
    for rank, (result, score) in enumerate(display_list, start=1):
        decision_key = f"decision_{result.filename}_{rank}"
        if st.session_state.get(decision_key) == "Shortlist":
            shortlisted.append((rank, result, score))

    # Also check full sorted list
    if not shortlisted:
        for rank, (result, score) in enumerate(
            zip(sorted_results, display_scores), start=1
        ):
            decision_key = f"decision_{result.filename}_{rank}"
            if st.session_state.get(decision_key) == "Shortlist":
                shortlisted.append((rank, result, score))

    if not shortlisted:
        st.info(
            "Chưa có ứng viên nào được shortlist. "
            "Nhấn nút **✅ Shortlist** trên card ứng viên để thêm vào đây."
        )
        return

    st.success(f"✅ {len(shortlisted)} ứng viên được shortlist")

    rows = []
    for rank, result, score in shortlisted:
        rows.append({
            "Rank": rank,
            "Tên ứng viên": result.candidate_name or result.filename,
            "Email": result.email or "—",
            "SĐT": result.phone or "—",
            "Điểm (%)": round(score * 100, 1),
            "Kinh nghiệm": f"{result.experience_years} năm" if result.experience_years else "—",
            "Kỹ năng có": ", ".join(result.skills_found[:5]) or "—",
            "Thiếu must-have": ", ".join(result.must_have_missing) or "—",
        })

    sl_df = pd.DataFrame(rows)
    st.dataframe(sl_df, use_container_width=True)

    # Export shortlist
    if not sl_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Tải Shortlist CSV",
                data=sl_df.to_csv(index=False).encode("utf-8"),
                file_name="shortlist.csv",
                mime="text/csv",
                key="shortlist_csv_dl",
            )
        with col2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                sl_df.to_excel(w, index=False, sheet_name="Shortlist")
            st.download_button(
                "⬇️ Tải Shortlist Excel",
                data=buf.getvalue(),
                file_name="shortlist.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="shortlist_xlsx_dl",
            )


# ---------------------------------------------------------------------------
# Real-time Filter Table
# ---------------------------------------------------------------------------

def _render_realtime_filter_table(
    sorted_results: list[CVResult],
    display_scores: list[float],
    jd_skills: list[str],
) -> None:
    """Interactive st.dataframe with real-time search/filter/sort."""
    st.markdown("#### 🔍 Lọc & sắp xếp real-time")
    st.caption(
        "Lọc theo tên, kỹ năng, điểm số. "
        "Nhấp vào cột header để sắp xếp. "
        "Dùng thanh search để tìm ứng viên theo tên."
    )

    # Build dataframe
    rows = []
    for rank, (result, score) in enumerate(zip(sorted_results, display_scores), start=1):
        decision_key = f"decision_{result.filename}_{rank}"
        decision = st.session_state.get(decision_key, "—")
        rows.append({
            "Rank": rank,
            "Tên ứng viên": result.candidate_name or "—",
            "File": result.filename,
            "Email": result.email or "—",
            "SĐT": result.phone or "—",
            "Điểm (%)": round(score * 100, 1),
            "Ngữ nghĩa (%)": round(result.semantic_score * 100, 1),
            "Kỹ năng (%)": round(result.skill_score * 100, 1),
            "Kinh nghiệm (năm)": result.experience_years if result.experience_years else 0,
            "Có dự án": "✔" if result.has_projects else "✗",
            "Job Hopping": "⚠️" if result.job_hopping else "—",
            "Kỹ năng có": ", ".join(result.skills_found[:6]) or "—",
            "Thiếu must-have": ", ".join(result.must_have_missing[:4]) or "—",
            "Red Flags": len(result.red_flags),
            "Quyết định HR": decision,
        })

    df_rt = pd.DataFrame(rows)

    # Filter controls
    f1, f2, f3 = st.columns(3)
    with f1:
        name_search = st.text_input(
            "🔎 Tìm theo tên ứng viên / tên file",
            key="rt_name_search",
        )
    with f2:
        min_score_rt = st.slider(
            "Điểm tối thiểu (%)", 0, 100, 0, 5, key="rt_min_score"
        )
    with f3:
        skill_filter = st.multiselect(
            "Phải có kỹ năng",
            options=jd_skills,
            key="rt_skill_filter",
        )

    # Apply filters
    filtered_df = df_rt.copy()
    if name_search:
        mask = (
            filtered_df["Tên ứng viên"].str.contains(name_search, case=False, na=False)
            | filtered_df["File"].str.contains(name_search, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    if min_score_rt > 0:
        filtered_df = filtered_df[filtered_df["Điểm (%)"] >= min_score_rt]
    if skill_filter:
        for skill in skill_filter:
            filtered_df = filtered_df[
                filtered_df["Kỹ năng có"].str.contains(skill, case=False, na=False)
            ]

    st.caption(f"Hiển thị **{len(filtered_df)}** / {len(df_rt)} ứng viên")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "Điểm (%)": st.column_config.ProgressColumn(
                "Điểm (%)", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Ngữ nghĩa (%)": st.column_config.ProgressColumn(
                "Ngữ nghĩa (%)", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Kỹ năng (%)": st.column_config.ProgressColumn(
                "Kỹ năng (%)", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
        hide_index=True,
    )




# ---------------------------------------------------------------------------
# UI – History Tab
# ---------------------------------------------------------------------------

def render_history_tab():
    st.subheader("📚 Lịch sử phân tích")
    sessions = list_sessions(limit=50)

    if not sessions:
        st.info("Chưa có phiên phân tích nào được lưu. Hãy phân tích CV trước!")
        return

    st.markdown(f"Tìm thấy **{len(sessions)}** phiên phân tích gần nhất.")

    for session in sessions:
        skills_str = ", ".join(session["jd_skills"][:5]) if session["jd_skills"] else "—"
        label = (
            f"🗓 {session['created_at']} | "
            f"{session['cv_count']} CV | "
            f"JD: {session['jd_snippet'][:80]}…"
        )
        with st.expander(label, expanded=False):
            col_meta, col_action = st.columns([4, 1])
            with col_meta:
                st.markdown(f"**Kỹ năng JD:** {skills_str}")
                st.markdown(f"**Số CV:** {session['cv_count']}")
                st.markdown(f"**Thời gian:** {session['created_at']}")
            with col_action:
                if st.button("🗑 Xoá", key=f"del_{session['id']}"):
                    delete_session(session["id"])
                    st.success("Đã xoá phiên!")
                    st.rerun()

            # Show results table
            results = get_session_results(session["id"])
            if results:
                df = pd.DataFrame(
                    [
                        {
                            "Rank": r["rank"],
                            "Tên ứng viên": r["candidate_name"] or "—",
                            "Email": r["email"] or "—",
                            "SĐT": r["phone"] or "—",
                            "Tên file": r["filename"],
                            "Tổng điểm (%)": round(r["score"] * 100, 1),
                            "Ngữ nghĩa (%)": round(r["semantic_score"] * 100, 1),
                            "Kỹ năng (%)": round(r["skill_score"] * 100, 1),
                            "Kinh nghiệm (năm)": r.get("experience_years", "—"),
                            "Quyết định HR": r.get("decision") or "—",
                            "Kỹ năng có": ", ".join(r["skills_found"]) or "—",
                            "Thiếu kỹ năng": ", ".join(r["skills_missing"]) or "—",
                            "Nhận xét": r["summary"],
                        }
                        for r in results
                    ]
                )
                st.dataframe(df, use_container_width=True)

                col_csv2, col_xlsx2 = st.columns(2)
                with col_csv2:
                    st.download_button(
                        label="⬇️ CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name=f"session_{session['id']}_results.csv",
                        mime="text/csv",
                        key=f"csv_{session['id']}",
                    )
                with col_xlsx2:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as w:
                        df.to_excel(w, index=False, sheet_name="Results")
                    st.download_button(
                        label="⬇️ Excel",
                        data=buf.getvalue(),
                        file_name=f"session_{session['id']}_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"xlsx_{session['id']}",
                    )


# ---------------------------------------------------------------------------
# UI – About Tab
# ---------------------------------------------------------------------------

def render_about_tab():
    st.subheader("ℹ️ Về hệ thống")
    st.markdown(
        """
### Kiến trúc hệ thống

| Thành phần | Chi tiết |
|---|---|
| **AI Model** | `paraphrase-multilingual-MiniLM-L12-v2` (hỗ trợ 50+ ngôn ngữ) |
| **Ngôn ngữ** | Tiếng Việt + Tiếng Anh (cross-lingual matching) |
| **PDF Extraction** | pdfplumber → PyMuPDF → PyPDF2 → OCR (Tesseract vie+eng) |
| **DOCX Extraction** | python-docx (paragraphs + tables) |
| **Cơ sở dữ liệu** | SQLite (lưu lịch sử tất cả phiên phân tích) |
| **JD Parser** | Rule-based: trích xuất must-have / nice-to-have / level / kinh nghiệm |
| **Scoring** | Composite = Ngữ nghĩa + Kỹ năng + Kinh nghiệm (trọng số tuỳ chỉnh) |

### Công thức chấm điểm

```
Semantic Score    = mean(top-3 cosine similarity giữa chunks CV và JD)
Skill Score       = weighted coverage (must-have penalty -15pt/skill, nice-to-have -5pt/skill)
Experience Score  = min(1.0, số_năm / 5.0)
Composite Score   = w_sem × Semantic + w_skill × Skill + w_exp × Experience
                    (w_sem + w_skill + w_exp được chuẩn hóa = 100%)
```

### Tính năng mới (v3)

| Tính năng | Mô tả |
|---|---|
| **Hỗ trợ DOCX** | Upload file Word (.docx) — python-docx đọc paragraphs + tables |
| **Radar chart** | So sánh đa chiều ứng viên (Tổng điểm, Ngữ nghĩa, Kỹ năng, Kinh nghiệm, Dự án) |
| **Báo cáo PDF** | Xuất báo cáo PDF nhiều trang: trang tổng hợp + 1 trang chi tiết/ứng viên |
| **Mở rộng SKILL_KEYWORDS** | +20 kỹ năng mới: LangChain, Kafka, Elasticsearch, Spark, Airflow, Terraform, Flutter, React Native, v.v. |

### Tính năng (v2)

| Tính năng | Mô tả |
|---|---|
| **Must-have / Nice-to-have scoring** | Thiếu must-have → trừ 15đ/skill; thiếu nice-to-have → trừ 5đ/skill |
| **Contact extraction** | Tự động trích tên, email, SĐT từ CV để HR liên hệ trực tiếp |
| **CV Section Parser** | Phân tích Work Experience / Skills / Education riêng biệt |
| **Job-hopping detection** | Phát hiện ứng viên nhảy việc từ section Work Experience |
| **LLM summary** | OpenAI/Gemini/Groq tạo nhận xét tự nhiên, gợi ý câu hỏi phỏng vấn |
| **Decision buttons** | ✅ Shortlist / ⏸ Cân nhắc / ❌ Loại ngay trên card ứng viên |
| **Shortlist tab** | Tab riêng tổng hợp ứng viên được shortlist, kèm export |
| **Real-time filter table** | Bảng lọc/sort theo tên, kỹ năng, điểm — progress bar trực quan |
| **Multi-sheet Excel** | "Tất cả ứng viên" + "Shortlist" + "Thống kê JD" + conditional formatting |
| **JD Intelligence** | Tự động phân loại must-have / nice-to-have skills từ JD |
| **Pre-filter** | Lọc ứng viên trước khi hiển thị (KN tối thiểu, kỹ năng bắt buộc) |
| **Configurable Weights** | Sidebar sliders điều chỉnh trọng số chấm điểm |
| **Quick Tags** | Badge tóm tắt: Senior, Strong Python, Missing Docker, Has Projects… |
| **Red Flag Detection** | Phát hiện CV đáng ngờ: quá ngắn, nhiều skill không có project… |
| **Insight Dashboard** | Tổng quan: phân bổ match, skill % ứng viên, lịch sử kinh nghiệm |

### Pipeline trích xuất file CV

| Định dạng | Bước | Thư viện | Xử lý tốt |
|---|---|---|---|
| **PDF** | 1 | **pdfplumber** | CV text thông thường, bảng biểu |
| **PDF** | 2 | **PyMuPDF** | CV thiết kế (Canva, template đẹp), font đặc biệt |
| **PDF** | 3 | **PyPDF2** | Fallback nhẹ |
| **PDF** | 4 | **Tesseract OCR** | CV scan, ảnh chụp, PDF hình ảnh |
| **DOCX** | — | **python-docx** | File Word (.docx), đọc paragraphs + tables |

> **Lưu ý:** Bước OCR cần cài `tesseract-ocr` và `poppler-utils` trên hệ thống.
> Trên Ubuntu/Debian: `sudo apt install tesseract-ocr tesseract-ocr-vie poppler-utils`
"""
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    render_sidebar()
    st.title(f"{APP_ICON} {APP_TITLE}")

    tab_analysis, tab_history, tab_about = st.tabs(
        ["🚀 Phân tích CV", "📚 Lịch sử", "ℹ️ Thông tin"]
    )

    with tab_analysis:
        render_analysis_tab()

    with tab_history:
        render_history_tab()

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()

