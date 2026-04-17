"""
app.py – Main Streamlit application for AI CV Screening Tool.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from config import (
    APP_DESCRIPTION,
    APP_ICON,
    APP_TITLE,
    SCORE_HIGH,
    SCORE_MED,
)
from modules.cv_analyzer import analyze_cv, extract_jd_skills, rank_results
from modules.embedding_manager import encode, get_model
from modules.export_manager import results_to_dataframe, to_csv_bytes, to_excel_bytes
from modules.pdf_processor import extract_text_from_pdf

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


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


@st.cache_resource(show_spinner="Loading AI model (first time may take ~30 s)…")
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
        st.markdown("### How it works")
        st.markdown(
            """
1. Paste or type the **Job Description**
2. Upload one or more **CV PDFs**
3. Click **Analyse CVs**
4. Browse ranked results & download CSV/Excel
"""
        )
        st.divider()
        st.markdown("**AI Engine:** `all-MiniLM-L6-v2`")
        st.markdown("**Method:** Semantic embeddings + cosine similarity")


# ---------------------------------------------------------------------------
# UI – Main
# ---------------------------------------------------------------------------

def render_main():
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown(APP_DESCRIPTION)
    st.divider()

    # Warm-up model
    load_model()

    col_jd, col_cv = st.columns([1, 1], gap="large")

    # --- Job Description input ---
    with col_jd:
        st.subheader("📋 Job Description")
        jd_text = st.text_area(
            label="Paste job description here",
            height=300,
            placeholder="e.g.  We are looking for a Senior Full-Stack Developer with 3+ years of "
                        "experience in Python, React, PostgreSQL and Docker…",
            key="jd_input",
        )

    # --- CV upload ---
    with col_cv:
        st.subheader("📄 Upload CVs (PDF)")
        uploaded_files = st.file_uploader(
            label="Upload one or more PDF CVs",
            type=["pdf"],
            accept_multiple_files=True,
            key="cv_upload",
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) loaded")

    st.divider()

    # --- Analyse button ---
    run = st.button("🚀 Analyse CVs", type="primary", disabled=not (jd_text and uploaded_files))

    if not run:
        if not jd_text:
            st.info("👆 Please enter a Job Description to get started.")
        elif not uploaded_files:
            st.info("👆 Please upload at least one CV PDF.")
        return

    # -----------------------------------------------------------------------
    # Processing
    # -----------------------------------------------------------------------
    with st.spinner("Analysing CVs with AI…"):
        # Embed JD
        jd_embedding = encode([jd_text])[0]
        jd_skills = extract_jd_skills(jd_text)

        results = []
        progress = st.progress(0, text="Processing CVs…")
        for i, uploaded_file in enumerate(uploaded_files):
            cv_bytes = uploaded_file.read()
            cv_text = extract_text_from_pdf(cv_bytes)
            result = analyze_cv(
                filename=uploaded_file.name,
                cv_text=cv_text,
                jd_embedding=jd_embedding,
                jd_skills=jd_skills,
            )
            results.append(result)
            progress.progress((i + 1) / len(uploaded_files), text=f"Processed {uploaded_file.name}")

        progress.empty()
        ranked = rank_results(results)

    # -----------------------------------------------------------------------
    # Results header
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("🏆 Ranked Results")

    if jd_skills:
        st.markdown(f"**Skills detected in JD:** {', '.join(jd_skills)}")
    else:
        st.warning("No recognised skills found in the Job Description – results are based on pure semantic similarity.")

    # -----------------------------------------------------------------------
    # Bar chart
    # -----------------------------------------------------------------------
    names = [r.filename for r in ranked]
    scores = [round(r.score * 100, 1) for r in ranked]
    colors = [
        "#2ecc71" if s >= SCORE_HIGH * 100 else "#f39c12" if s >= SCORE_MED * 100 else "#e74c3c"
        for s in scores
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{s}%" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(title="Match Score (%)", range=[0, 110]),
        yaxis=dict(autorange="reversed"),
        height=max(200, 60 * len(ranked)),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # Detailed cards
    # -----------------------------------------------------------------------
    st.subheader("📊 Detailed Analysis")
    for rank, result in enumerate(ranked, start=1):
        pct = round(result.score * 100, 1)
        icon = score_color(result.score)
        with st.expander(f"#{rank} {result.filename} – {icon} {pct}%", expanded=(rank == 1)):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**✅ Skills Found**")
                if result.skills_found:
                    for skill in result.skills_found:
                        st.markdown(f"- {skill}")
                else:
                    st.markdown("*None of the JD skills detected*")

            with c2:
                st.markdown("**❌ Skills Missing**")
                if result.skills_missing:
                    for skill in result.skills_missing:
                        st.markdown(f"- {skill}")
                else:
                    st.markdown("*All required skills found!* 🎉")

            st.markdown("**🔍 Top Matching Evidence (AI)**")
            for i, (chunk, sim) in enumerate(result.evidence, start=1):
                sim_pct = round(sim * 100, 1)
                st.markdown(
                    f"<div style='background:#f0f4ff;border-left:4px solid #4a90e2;"
                    f"padding:8px 12px;margin:4px 0;border-radius:4px;'>"
                    f"<small>Evidence #{i} – similarity {sim_pct}%</small><br>{chunk}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("📥 Export Results")
    df = results_to_dataframe(ranked)
    st.dataframe(df, use_container_width=True)

    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            label="⬇️ Download CSV",
            data=to_csv_bytes(df),
            file_name="cv_screening_results.csv",
            mime="text/csv",
        )
    with col_xlsx:
        st.download_button(
            label="⬇️ Download Excel",
            data=to_excel_bytes(df),
            file_name="cv_screening_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()

