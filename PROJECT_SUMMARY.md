# Project Summary: AI Resume Analyzer & Interview Coach 📊

## 1. Business Problem
Modern hiring processes rely heavily on automated screening tools, primarily Applicant Tracking Systems (ATS), to filter through hundreds of resumes for a single opening. This presents two distinct challenges:
*   **For Candidates:** Job seekers struggle to understand why their profiles are rejected or how to align their resumes with complex job descriptions. They lack visibility into skill gaps, keyword requirements, and ATS scoring dimensions.
*   **For Coaches/Educators:** Career counselors and internship mentors require a scalable, data-backed method to provide high-impact, personalized feedback on resumes, along with realistic mock interview preparation.

---

## 2. The Solution
**AI Resume Analyzer & Interview Coach** is a comprehensive, production-grade full-stack platform that demystifies the black box of ATS screening and provides candidates with automated career coaching.

The application parses any PDF resume, extracts key metadata, and compares it directly against a job description. The platform delivers:
1.  **A Weighted ATS Score:** A 4-dimensional breakdown (Skills, Keywords, Experience, Education) to isolate specific weaknesses.
2.  **Semantic Match Score:** Measuring conceptual alignment using dense vector embeddings, capturing contexts that simple keyword matchers miss.
3.  **Actionable Optimizations:** Real-time feedback, including Google-style **X-Y-Z formula** bullet point rewrites and strong action verb recommendations.
4.  **Tailored Learning Paths:** Custom educational plans categorized by difficulty to close technical skill gaps.
5.  **AI Mock Interview Simulator:** Interactive question generation and a real-time answer sandbox giving constructive, score-based coaching feedback.

---

## 3. System Architecture & Technical Decisions

### Service-Layer Architecture
The application uses a highly decoupled, modular service architecture. The backend is built on **FastAPI** for high-speed asynchronous request handling and uses **SQLAlchemy** for database operations, making it ready to scale from **SQLite** to **PostgreSQL**.

### The AI Layer (Hybrid Approach)
*   **LLM Parsing & Coaching:** We utilize the **Google Gemini API** (using `gemini-1.5-flash`) for structural data extraction and advanced coaching logic. To enforce reliability, we defined strict JSON schemas for the API responses.
*   **Local Heuristics Fallback:** Recognizing the need for offline capability and resilience, the parser falls back to custom regular expressions, token boundary checks, and a pre-defined technical skills dictionary if the API key is missing or calls fail.
*   **Semantic Vector Space:** We implement a lazy-loaded **Sentence Transformers** model (`all-MiniLM-L6-v2`) to compute 384-dimensional dense embeddings for the resume and job description, calculating their Cosine Similarity. If system resources are constrained, the service automatically falls back to a custom **TF-IDF Vectorizer + Cosine Similarity** engine.

---

## 4. Tech Stack Summary

*   **Backend Framework:** FastAPI (Uvicorn, Pydantic v2, Pydantic-Settings)
*   **ORM & Database:** SQLAlchemy 2.0, SQLite (Development), PostgreSQL-compatible
*   **Document Utility:** PyMuPDF (`fitz` for fast PDF text extraction)
*   **AI & NLP Libraries:** Google Generative AI SDK, Sentence-Transformers (PyTorch-backed), Scikit-Learn, NumPy, Pandas
*   **Frontend Dashboard:** Streamlit, Plotly (Gauge, Radar, Bar, and Pie charts)
*   **DevOps & Deployment:** Docker, Docker Compose (multi-stage containerization)
*   **Testing Suite:** PyTest (FastAPI TestClient, in-memory DB with `StaticPool`, Mocking/Patching)

---

## 5. Future Improvements & Scaling Roadmap

1.  **Multi-Resume Comparison:** Allow candidates to upload multiple versions of their resumes and automatically rank which version is the best match for a specific job posting.
2.  **Real-Time Job Scraping:** Integrate a web-scraping module to pull job descriptions directly from links (LinkedIn, Indeed, Wellfound) instead of requiring manual copy-pasting.
3.  **Fine-Tuned Similarity Models:** Fine-tune a sentence-transformer model on a specialized dataset of resumes and job descriptions to capture industry-specific semantic nuances (e.g., recognizing that "MERN stack" and "React/Node" represent the same concept).
4.  **Audio Interview Coach:** Utilize WebRTC or speech-to-text integration in the Streamlit frontend to allow candidates to speak their answers out loud, providing vocal pacing, tone, and content coaching.
