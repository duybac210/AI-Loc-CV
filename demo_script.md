# 🎬 Demo Script – AI CV Screening Tool (2 minutes)

> **Presenter:** Duybac210  
> **Time budget:** ~2 minutes  
> **Slide / tab order:** Terminal → Browser (Streamlit app)

---

## ⏱️ Minute 0:00 – 0:15 | Introduction (spoken)

> "Xin chào. Hôm nay tôi sẽ demo **AI CV Screening Tool** – một ứng dụng sử dụng AI ngữ nghĩa để tự động xếp hạng và phân tích CV ứng viên so với mô tả công việc."

**Key points to mention:**
- Stack: Streamlit + sentence-transformers (all-MiniLM-L6-v2, 33 MB, offline)
- No API key needed, runs locally on CPU

---

## ⏱️ Minute 0:15 – 0:30 | Start the App

```bash
streamlit run app.py
```

> "Ứng dụng khởi động tại `localhost:8501`. Model AI được load một lần và cache lại."

**Show:** The sidebar with the AI engine description.

---

## ⏱️ Minute 0:30 – 0:55 | Input Data

**Step 1 – Paste Job Description**

Open `sample_data/job_description.txt`, copy all text, paste into the **Job Description** text area.

> "Đây là JD cho vị trí **Senior Full-Stack Developer** yêu cầu Python, React, PostgreSQL, Docker, Kubernetes và AWS."

**Step 2 – Upload CVs**

Upload both `sample_data/sample_cv_1.pdf` and `sample_data/sample_cv_2.pdf`.

> "Tôi upload 2 CV – một ứng viên senior có đủ kỹ năng, một ứng viên junior thiếu một số kỹ năng quan trọng."

---

## ⏱️ Minute 0:55 – 1:20 | Run Analysis

Click **🚀 Analyse CVs**.

> "AI encode cả JD và từng đoạn văn bản trong CV thành vector 384 chiều, sau đó tính cosine similarity để xếp hạng."

**Show the bar chart:**
> "CV 1 đạt ~90%, CV 2 đạt ~60% – AI phân biệt được sự khác biệt ngữ nghĩa."

---

## ⏱️ Minute 1:20 – 1:45 | Explore Results

**Click on CV 1 card (expanded):**
- Point to ✅ Skills Found: Python, React, PostgreSQL, Docker, AWS…
- Point to ❌ Skills Missing: none
- Point to 🔍 Evidence chunks: show one chunk with similarity score

**Click on CV 2 card:**
- Point to ❌ Skills Missing: PostgreSQL, Docker, Kubernetes, AWS…
- Mention: > "AI không chỉ so từ khóa – nó hiểu nghĩa. 'I built UIs with React' được match với 'React developer needed'."

---

## ⏱️ Minute 1:45 – 2:00 | Export & Close

Scroll to the **Export** section:

> "Kết quả có thể export ra CSV hoặc Excel để lưu trữ và chia sẻ."

Click **⬇️ Download CSV** to demo the download.

**Closing (spoken):**
> "Tóm lại, tool này giải quyết bài toán sàng lọc CV tự động bằng AI ngữ nghĩa – nhanh, offline, không cần API key. Cảm ơn mọi người!"

---

## 🎯 Key Talking Points (if asked)

| Question | Answer |
|----------|--------|
| Why sentence-transformers? | Free, offline, 33 MB, runs on CPU, state-of-the-art semantic understanding |
| How is the score calculated? | Mean cosine similarity of top-3 CV chunks vs JD embedding |
| What is cosine similarity? | Angle between two embedding vectors – 1.0 = identical meaning, 0 = unrelated |
| Why chunk the CV? | Long texts lose detail; chunking captures local context |
| Can it handle Vietnamese CVs? | Partially – the model understands some multilingual text but is optimised for English |
