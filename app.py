"""
app.py – Main Streamlit application for AI CV Screening Tool.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from config import (
    APP_DESCRIPTION,
    APP_ICON,
    APP_TITLE,
    SCORE_HIGH,
    SCORE_MED,
    WEIGHT_SEMANTIC,
    WEIGHT_SKILL,
)
from modules.cv_analyzer import analyze_cv, extract_jd_skills, rank_results
from modules.database_manager import (
    delete_session,
    get_session_jd,
    get_session_results,
    init_db,
    list_sessions,
    save_session,
)
from modules.embedding_manager import encode, get_model
from modules.export_manager import results_to_dataframe, to_csv_bytes, to_excel_bytes
from modules.pdf_processor import extract_text_from_pdf, OCR_PREFIX

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
        st.markdown("### Hướng dẫn sử dụng")
        st.markdown(
            """
1. Nhập **Mô tả công việc (JD)** – tiếng Việt hoặc tiếng Anh
2. Upload một hoặc nhiều **CV dạng PDF**
3. Nhấn **Phân tích CV**
4. Xem kết quả xếp hạng & tải xuống CSV/Excel
5. Vào tab **Lịch sử** để xem lại các lần chấm trước
"""
        )
        st.divider()
        st.markdown("**🤖 AI Engine:** `paraphrase-multilingual-MiniLM-L12-v2`")
        st.markdown("**🌐 Ngôn ngữ:** Tiếng Việt + Tiếng Anh (cross-lingual)")
        st.markdown("**📊 Phương pháp:** Semantic embeddings + Skill coverage")
        st.markdown(
            f"**⚖️ Trọng số:** Semantic {round(WEIGHT_SEMANTIC*100)}% + "
            f"Kỹ năng {round(WEIGHT_SKILL*100)}%"
        )
        st.divider()
        st.markdown("**📄 PDF hỗ trợ:** pdfplumber → PyMuPDF → PyPDF2 → OCR (Tesseract)")


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
        st.subheader("📄 Upload CV (PDF)")
        uploaded_files = st.file_uploader(
            label="Upload một hoặc nhiều file CV dạng PDF",
            type=["pdf"],
            accept_multiple_files=True,
            key="cv_upload",
        )
        if uploaded_files:
            st.success(f"✅ Đã tải {len(uploaded_files)} file(s)")

    st.divider()

    run = st.button(
        "🚀 Phân tích CV",
        type="primary",
        disabled=not (jd_text and uploaded_files),
    )

    if not run:
        if not jd_text:
            st.info("👆 Vui lòng nhập Mô tả công việc để bắt đầu.")
        elif not uploaded_files:
            st.info("👆 Vui lòng upload ít nhất một CV PDF.")
        return

    # -----------------------------------------------------------------------
    # Processing
    # -----------------------------------------------------------------------
    with st.spinner("🔍 AI đang phân tích CV…"):
        jd_embedding = encode([jd_text])[0]
        jd_skills = extract_jd_skills(jd_text)

        results = []
        progress = st.progress(0, text="Đang xử lý CV…")
        errors = []
        ocr_files: list[str] = []

        for i, uploaded_file in enumerate(uploaded_files):
            cv_bytes = uploaded_file.read()
            cv_text = extract_text_from_pdf(cv_bytes)

            if cv_text.startswith("[PDF extraction failed"):
                errors.append(uploaded_file.name)
            elif cv_text.startswith(OCR_PREFIX):
                ocr_files.append(uploaded_file.name)
                cv_text = cv_text[len(OCR_PREFIX):]  # strip marker before analysis

            result = analyze_cv(
                filename=uploaded_file.name,
                cv_text=cv_text,
                jd_embedding=jd_embedding,
                jd_skills=jd_skills,
            )
            results.append(result)
            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"Đã xử lý: {uploaded_file.name}",
            )

        progress.empty()
        ranked = rank_results(results)

        # Save to database
        session_id = save_session(jd_text, jd_skills, ranked)

    if errors:
        st.warning(
            f"⚠️ Không thể trích xuất text từ {len(errors)} file(s): "
            f"{', '.join(errors)}. "
            "File có thể là PDF scan hoặc bị mã hoá – điểm sẽ thấp bất thường."
        )

    if ocr_files:
        st.info(
            f"🔍 **OCR được dùng cho {len(ocr_files)} file(s):** {', '.join(ocr_files)}. "
            "Đây là CV dạng ảnh/scan – text được nhận dạng tự động bằng Tesseract OCR. "
            "Độ chính xác có thể thấp hơn CV dạng text thông thường."
        )

    st.success(f"✅ Đã lưu kết quả vào lịch sử (Session #{session_id})")

    _render_results(ranked, jd_skills)


def _render_results(ranked, jd_skills):
    """Render ranked results, chart, cards, and export."""
    # -----------------------------------------------------------------------
    # Results header
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("🏆 Kết quả xếp hạng")

    if jd_skills:
        st.markdown(f"**Kỹ năng phát hiện trong JD:** {', '.join(jd_skills)}")
    else:
        st.warning(
            "Không phát hiện kỹ năng cụ thể trong JD – điểm dựa hoàn toàn vào "
            "độ tương đồng ngữ nghĩa."
        )

    # -----------------------------------------------------------------------
    # Bar chart
    # -----------------------------------------------------------------------
    names = [r.filename for r in ranked]
    scores = [round(r.score * 100, 1) for r in ranked]
    semantic_scores = [round(r.semantic_score * 100, 1) for r in ranked]
    skill_scores = [round(r.skill_score * 100, 1) for r in ranked]
    colors = [
        "#2ecc71" if s >= SCORE_HIGH * 100 else "#f39c12" if s >= SCORE_MED * 100 else "#e74c3c"
        for s in scores
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{s}%" for s in scores],
            textposition="outside",
            name="Tổng điểm",
        )
    )
    fig.update_layout(
        xaxis=dict(title="Điểm phù hợp (%)", range=[0, 115]),
        yaxis=dict(autorange="reversed"),
        height=max(200, 60 * len(ranked)),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Score breakdown chart
    with st.expander("📊 Xem chi tiết điểm thành phần", expanded=False):
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=semantic_scores,
                y=names,
                orientation="h",
                name=f"Ngữ nghĩa ({round(WEIGHT_SEMANTIC*100)}%)",
                marker_color="#4a90e2",
                opacity=0.8,
            )
        )
        fig2.add_trace(
            go.Bar(
                x=skill_scores,
                y=names,
                orientation="h",
                name=f"Kỹ năng ({round(WEIGHT_SKILL*100)}%)",
                marker_color="#9b59b6",
                opacity=0.8,
            )
        )
        fig2.update_layout(
            barmode="group",
            xaxis=dict(title="Điểm (%)", range=[0, 110]),
            yaxis=dict(autorange="reversed"),
            height=max(200, 70 * len(ranked)),
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------------------------
    # Detailed cards
    # -----------------------------------------------------------------------
    st.subheader("📋 Phân tích chi tiết")
    for rank, result in enumerate(ranked, start=1):
        pct = round(result.score * 100, 1)
        icon = score_color(result.score)
        with st.expander(f"#{rank} {result.filename} – {icon} {pct}%", expanded=(rank == 1)):
            # Summary
            st.info(f"💡 **Nhận xét AI:** {result.summary}")

            # Score breakdown
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("🎯 Tổng điểm", f"{pct}%")
            sc2.metric(
                f"🧠 Ngữ nghĩa ({round(WEIGHT_SEMANTIC*100)}%)",
                f"{round(result.semantic_score * 100, 1)}%",
            )
            sc3.metric(
                f"🔧 Kỹ năng ({round(WEIGHT_SKILL*100)}%)",
                f"{round(result.skill_score * 100, 1)}%",
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ Kỹ năng có**")
                if result.skills_found:
                    for skill in result.skills_found:
                        st.markdown(f"- {skill}")
                else:
                    st.markdown("*Không có kỹ năng nào trong JD*")

            with c2:
                st.markdown("**❌ Kỹ năng còn thiếu**")
                if result.skills_missing:
                    for skill in result.skills_missing:
                        st.markdown(f"- {skill}")
                else:
                    st.markdown("*Đủ tất cả kỹ năng yêu cầu!* 🎉")

            st.markdown("**🔍 Đoạn text liên quan nhất (AI)**")
            for i, (chunk, sim) in enumerate(result.evidence, start=1):
                sim_pct = round(sim * 100, 1)
                st.markdown(
                    f"<div style='background:#f0f4ff;border-left:4px solid #4a90e2;"
                    f"padding:8px 12px;margin:4px 0;border-radius:4px;color:#1a1a1a;'>"
                    f"<small>Bằng chứng #{i} – tương đồng {sim_pct}%</small><br>{chunk}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("📥 Xuất kết quả")
    df = results_to_dataframe(ranked)
    st.dataframe(df, use_container_width=True)

    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            label="⬇️ Tải CSV",
            data=to_csv_bytes(df),
            file_name="cv_screening_results.csv",
            mime="text/csv",
        )
    with col_xlsx:
        st.download_button(
            label="⬇️ Tải Excel",
            data=to_excel_bytes(df),
            file_name="cv_screening_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
                            "Tên file": r["filename"],
                            "Tổng điểm (%)": round(r["score"] * 100, 1),
                            "Ngữ nghĩa (%)": round(r["semantic_score"] * 100, 1),
                            "Kỹ năng (%)": round(r["skill_score"] * 100, 1),
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
| **Cơ sở dữ liệu** | SQLite (lưu lịch sử tất cả phiên phân tích) |
| **Scoring** | Composite = 65% Semantic + 35% Skill Coverage |

### Công thức chấm điểm

```
Semantic Score  = mean(top-3 cosine similarity giữa chunks CV và JD)
Skill Score     = số kỹ năng CV có / tổng kỹ năng JD yêu cầu
Composite Score = 0.65 × Semantic + 0.35 × Skill Coverage
```

### Tại sao multilingual model?

Model `paraphrase-multilingual-MiniLM-L12-v2` được train trên **50+ ngôn ngữ**, 
cho phép so sánh cross-lingual: JD tiếng Việt ↔ CV tiếng Anh và ngược lại.

### Pipeline trích xuất PDF (4 tầng)

| Bước | Thư viện | Xử lý tốt |
|---|---|---|
| 1 | **pdfplumber** | CV text thông thường, bảng biểu |
| 2 | **PyMuPDF** | CV thiết kế (Canva, template đẹp), font đặc biệt |
| 3 | **PyPDF2** | Fallback nhẹ |
| 4 | **Tesseract OCR** | CV scan, ảnh chụp, PDF hình ảnh |

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

