# REPORT – AI CV Screening Tool

> **Course:** AI for Business  
> **Student:** [Your Name] – [Student ID]  
> **Instructor:** [Instructor Name]  
> **Date:** [Date]  
> **GitHub:** https://github.com/duybac210/AI-Loc-CV

---

## 1. Problem Statement

Manually screening large volumes of CVs is time-consuming and inconsistent. This project builds an
automated **AI CV Screening Tool** that:

- Accepts a Job Description (JD) and multiple CV PDFs as input.
- Ranks candidates by **semantic similarity** to the JD.
- Highlights the most relevant CV passages (evidence).
- Detects skill gaps between the JD requirements and each candidate.
- Exports results to CSV / Excel.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────┐
│              Streamlit Web UI               │
│  (app.py – input, results, export buttons)  │
└───────────────────┬─────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │    PDF Processor      │  PyPDF2 extraction + chunking
        └───────────┬───────────┘
                    │ text chunks
        ┌───────────▼───────────┐
        │  Embedding Manager    │  sentence-transformers (all-MiniLM-L6-v2)
        └───────────┬───────────┘
                    │ 384-d vectors
        ┌───────────▼───────────┐
        │    CV Analyzer        │  cosine similarity + skill keyword scan
        └───────────┬───────────┘
                    │ CVResult objects
        ┌───────────▼───────────┐
        │   Export Manager      │  pandas → CSV / Excel
        └───────────────────────┘
```

---

## 3. AI Component Detail

### 3.1 Model Choice

| Model | Size | Speed (CPU) | Accuracy | Cost |
|-------|------|-------------|----------|------|
| **all-MiniLM-L6-v2** | 33 MB | ~50 ms/CV | Good | Free |
| all-mpnet-base-v2 | 420 MB | ~200 ms/CV | Better | Free |
| OpenAI text-embedding-3-small | N/A | Network | Best | API cost |

`all-MiniLM-L6-v2` was chosen for its balance of speed, size, and quality — suitable for 16 GB
RAM without a GPU.

### 3.2 Ranking Algorithm

1. Encode the full JD text → vector **q** ∈ ℝ³⁸⁴  
2. Split each CV into overlapping 400-character chunks  
3. Encode all chunks → matrix **C** ∈ ℝ^(N×384)  
4. Compute similarity scores: **s** = **C** · **q**  
5. Overall score = mean of top-3 scores (handles long CVs fairly)

### 3.3 Evidence Extraction

The top-K chunks by cosine similarity are returned as "evidence" — the exact passages that most
closely match the JD semantically.

### 3.4 Skill Detection

Skill detection uses **regex keyword matching** (not AI) against a catalogue of 40+ skills. Skills
found in the JD are used as the required set; the CV is scanned for the same skills.

---

## 4. Demo Results

*(Fill in after running the demo with screenshots)*

| Candidate | Match Score | Skills Found | Skills Missing |
|-----------|-------------|--------------|----------------|
| sample_cv_1.pdf | ~ % | | |
| sample_cv_2.pdf | ~ % | | |

**Screenshot 1:** Bar chart ranking  
*(insert image)*

**Screenshot 2:** Evidence highlights for top candidate  
*(insert image)*

**Screenshot 3:** Skill gap for CV 2  
*(insert image)*

**Screenshot 4:** CSV export  
*(insert image)*

---

## 5. Key Observations

- Semantic matching correctly separates a strong full-stack candidate from a front-end only candidate.
- The tool successfully identifies missing PostgreSQL, Docker, and Kubernetes skills in CV 2.
- Evidence chunks show *why* a score was assigned, making the ranking explainable.

---

## 6. Limitations & Future Work

| Limitation | Possible Improvement |
|------------|----------------------|
| English-only model | Use `paraphrase-multilingual-MiniLM-L12-v2` for Vietnamese |
| Keyword-based skill detection | Replace with NER model or LLM extraction |
| No authentication | Add login / API key gate |
| Single-page PDF assumption | Add multi-page layout handling |
| No database persistence | Store results in SQLite / PostgreSQL |

---

## 7. Conclusion

This project demonstrates that a lightweight, fully offline AI pipeline (sentence-transformers +
Streamlit) can provide meaningful CV screening results without any cloud API cost. The semantic
ranking successfully distinguishes candidates based on contextual understanding, going beyond
simple keyword matching.

---

## 8. References

1. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
2. Streamlit documentation – https://docs.streamlit.io
3. sentence-transformers documentation – https://www.sbert.net
4. PyPDF2 documentation – https://pypdf2.readthedocs.io
