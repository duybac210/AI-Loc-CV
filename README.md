# 🤖 AI CV Screening Tool

> **AI-for-Business End-of-Term Project**  
> Automatically rank, analyse, and export results for multiple CV PDFs against a Job Description — powered by **semantic AI** (sentence-transformers).

---

## 📋 Features

| Feature | AI? | Detail |
|---|---|---|
| **Semantic Ranking** | ✅ YES | sentence-transformers `all-MiniLM-L6-v2` + cosine similarity |
| **Evidence Highlighting** | ✅ YES | Top matching CV chunks found via semantic similarity |
| **Skill Gap Detection** | ❌ (regex) | Keyword scan against 40+ skills catalogue |
| **PDF Extraction** | ❌ (library) | PyPDF2 |
| **CSV / Excel Export** | ❌ (library) | pandas + openpyxl |
| **Interactive UI** | ❌ (framework) | Streamlit |

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
│   └── export_manager.py        # DataFrame + CSV/Excel export
├── sample_data/
│   ├── job_description.txt      # Sample JD (Senior Full-Stack Dev)
│   ├── sample_cv_1.pdf          # Strong match (~90 %)
│   └── sample_cv_2.pdf          # Partial match (~60 %)
├── demo_script.md               # 2-minute demo script
└── REPORT.md                    # Report template
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/duybac210/AI-Loc-CV.git
cd AI-Loc-CV
pip install -r requirements.txt
```

> **Note:** The first run downloads the `all-MiniLM-L6-v2` model (~33 MB). Subsequent runs use the cached version.

### 2. Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

### 3. Try with sample data

1. Copy the content of `sample_data/job_description.txt` into the **Job Description** text area.
2. Upload `sample_data/sample_cv_1.pdf` and `sample_cv_2.pdf`.
3. Click **🚀 Analyse CVs**.

---

## 🧠 How the AI Works

```
Job Description text
        │
        ▼
  sentence-transformers
  (all-MiniLM-L6-v2)        ← 33 MB model, runs on CPU
        │
        ▼
  384-dimensional embedding  ─────────────────────────┐
                                                       │
CV text → chunked (400 chars)                         │
        │                                              │
        ▼                                              │
  chunk embeddings (N × 384)                          │
        │                                              │
        ▼                                              ▼
  cosine similarity  →  mean of top-3 scores  →  overall match %
```

**Why `all-MiniLM-L6-v2`?**
- 33 MB – fits in RAM, no GPU required
- Free & offline – no API key
- Fast – encodes a full CV in < 1 second on CPU
- Strong semantic understanding (e.g. "built UIs with React" ≈ "React developer")

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
```

---

## 👤 Author

**Duybac210** – AI-for-Business, End-of-Term Project
