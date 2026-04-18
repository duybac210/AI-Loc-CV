# 🤖 AI CV Screening Tool

> **AI-for-Business End-of-Term Project**  
> Automatically rank, analyse, and export results for multiple CV PDFs against a Job Description — powered by **semantic AI** (sentence-transformers) + **LLM analysis** (Groq / OpenAI / Gemini).

---

## 🚀 Quick Start with AI (Free!)

1. Get a **free Groq API key** at [console.groq.com](https://console.groq.com) (no credit card needed)
2. Open the app → Sidebar → **"Bật AI nâng cao"** toggle
3. Select **Groq**, paste your key → done!

The LLM layer adds: automatic Shortlist/Reject decisions, interview questions, per-candidate explanations, and top-3 candidate comparison.

---

## 📋 Features

| Feature | AI? | Detail |
|---|---|---|
| **Semantic Ranking** | ✅ Embedding AI | `paraphrase-multilingual-MiniLM-L12-v2` + cosine similarity |
| **Evidence Highlighting** | ✅ Embedding AI | Top matching CV chunks via semantic similarity |
| **AI Candidate Comparison** | ✅ LLM | Compare top-3, explain why #1 is best |
| **AI JD Skill Extraction** | ✅ LLM | Extract must-have / nice-to-have more accurately |
| **AI CV Section Parsing** | ✅ LLM | Parse non-standard CV layouts (Vietnamese-only, etc.) |
| **Interview Questions** | ✅ LLM | Candidate-specific questions based on JD + CV |
| **AI Insights Tab** | ✅ LLM | Decision, ATS Score, potential, strengths/weaknesses |
| **Per-candidate Radar** | ✅ LLM + Embedding | Ngữ nghĩa / Kỹ năng / Kinh nghiệm / ATS / Tiềm năng |
| **Score Transparency** | ✅ | Formula explainer per candidate |
| **Skill Gap Detection** | Regex + AI | 40+ skill keywords + LLM supplement |
| **PDF Extraction** | Library chain | pdfplumber → PyMuPDF → PyPDF2 → OCR |
| **CSV / Excel / PDF Export** | Library | pandas + openpyxl + fpdf2 |
| **Interactive UI** | Framework | Streamlit |

---

## 🗂️ Project Structure

```
AI-Loc-CV/
├── app.py                       # Main Streamlit app
├── config.py                    # Constants & skills catalogue
├── requirements.txt             # Python dependencies
├── modules/
│   ├── pdf_processor.py         # PDF text extraction + chunking
│   ├── embedding_manager.py     # Sentence-transformer encoding
│   ├── cv_analyzer.py           # Core ranking + skill gap logic
│   ├── cv_section_parser.py     # CV section detection (regex + AI fallback)
│   ├── jd_parser.py             # JD intelligence parser
│   ├── llm_analyzer.py          # LLM integration (Groq/OpenAI/Gemini)
│   ├── database_manager.py      # SQLite history
│   └── export_manager.py        # DataFrame + CSV/Excel/PDF export
├── sample_data/
│   ├── job_description.txt      # Sample JD (Senior Full-Stack Dev)
│   ├── sample_cv_1.pdf          # Strong match (~90 %)
│   └── sample_cv_2.pdf          # Partial match (~60 %)
└── demo_script.md               # 2-minute demo script
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/duybac210/AI-Loc-CV.git
cd AI-Loc-CV
pip install -r requirements.txt
```

> **Note:** The first run downloads the `paraphrase-multilingual-MiniLM-L12-v2` model (~120 MB). Subsequent runs use the cached version.

### 2. Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

### 3. (Optional) Enable LLM AI

1. Get a free key at [console.groq.com](https://console.groq.com)
2. Sidebar → **"Bật AI nâng cao"** → Provider: Groq → paste key
3. Re-run analysis to get AI Insights, interview questions, and candidate comparison

---

## 🧠 How the AI Works

### Layer 1 – Embedding (always on, offline, free)

```
Job Description text
        │
        ▼
  sentence-transformers
  (paraphrase-multilingual-MiniLM-L12-v2)  ← ~120 MB, runs on CPU
        │
        ▼
  multilingual embedding  ─────────────────────────┐
                                                    │
CV text → chunked (400 chars)                      │
        │                                           │
        ▼                                           │
  chunk embeddings (N × 384)                       │
        │                                           │
        ▼                                           ▼
  cosine similarity  →  mean of top-3 scores  →  semantic match %
```

### Layer 2 – LLM (optional, requires API key)

```
CV text + JD + embedding scores
        │
        ▼
  Groq / OpenAI / Gemini
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │  • Shortlist / Consider / Reject decision   │
  │  • Strengths & weaknesses                  │
  │  • ATS Score, Tiềm năng, Văn hóa           │
  │  • Interview questions (candidate-specific) │
  │  • JD skill extraction (must/nice-to-have)  │
  │  • Top-3 candidate comparison paragraph     │
  └─────────────────────────────────────────────┘
```

**Why Groq first?**
- **Free tier** – no credit card, generous rate limits
- Fast (Llama 3 / Llama 3.1 inference at ~500 tok/s)
- OpenAI and Gemini also supported as alternatives

---

## 📊 Output Example

| Rank | Filename | Match Score | Skills Found | Skills Missing |
|------|----------|-------------|--------------|----------------|
| 1 | sample_cv_1.pdf | 91.3 % | Python, React, PostgreSQL, Docker, … | — |
| 2 | sample_cv_2.pdf | 62.7 % | React, JavaScript, Git | PostgreSQL, Docker, AWS, … |

---

## 🛠️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB+ |
| CPU | Any x86-64 | Modern multi-core |
| GPU | Not required | Optional (speeds up encoding) |
| Disk | 500 MB | 1 GB |

---

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
PyPDF2>=3.0.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.2
torch>=2.0.0
fpdf2>=2.7.0
plotly>=5.17.0
openpyxl>=3.1.0
openai>=1.0.0        # optional LLM provider
google-generativeai  # optional LLM provider
groq                 # optional LLM provider (free tier available)
```

---

## 👤 Author

**Duybac210** – AI-for-Business, End-of-Term Project
