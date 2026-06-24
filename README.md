# AI Resume Analyzer & Interview Coach 💼

An AI-powered, production-grade full-stack platform designed to optimize resumes for Applicant Tracking Systems (ATS), evaluate semantic alignment, and prepare candidates for technical and behavioral interviews through interactive AI simulation.

---
## 🎯 Business Value

The platform helps job seekers:

- Improve ATS compatibility
- Identify missing skills
- Understand resume-job alignment
- Practice technical interviews
- Generate targeted learning roadmaps

The system reduces manual resume review effort and provides actionable feedback within seconds.

---


## 📁 Project Structure

```text
├── assets/                  # Documentation assets, images, and logos
├── data/                    # Sample resumes, job descriptions, and raw data
├── src/                     # Core application source code
│   ├── api/                 # API routes, endpoints, and request controllers
│   ├── database/            # Database connections, migrations, and seeds
│   ├── models/              # Schema definitions and data models
│   ├── services/            # Core business logic (AI integration, parsing, ATS engine)
│   ├── utils/               # Helper functions, constants, and logging utilities
│   └── web_app/             # Frontend UI components or web application logic
├── tests/                   # Unit, integration, and system tests
├── .gitignore               # Specified untracked files to ignore
├── Dockerfile               # Docker build instructions for the application
├── PROJECT_SUMMARY.md       # High-level overview of project milestones
├── README.md                # Project documentation (this file)
├── docker-compose.yml       # Multi-container Docker application configuration
└── requirements.txt         # Python dependencies and package versions
---

## 🌟 Key Features

1. **Structured Resume Parser (PDF):** Extracts raw text from PDF files using `PyMuPDF` and structures the profile (Contact, Skills, Education, Experience) using Google Gemini API JSON Schema (with a robust local regex/keyword fallback).
2. **Deterministic ATS Engine:** Calculates compatibility based on a weighted 4-factor algorithm:
   *   **Skill Match (40%):** Compares candidate's skills against job description expectations.
   *   **Keyword Match (30%):** Measures noun and technical keyword frequencies.
   *   **Experience Match (20%):** Matches candidate's parsed years of experience against job requirements.
   *   **Education Match (10%):** Evaluates educational degree compatibility.
3. **Semantic Similarity Scorer:** Computes conceptual alignment using `sentence-transformers` (`all-MiniLM-L6-v2`) with a dual-mode fallback (Gemini or TF-IDF Cosine Similarity) to ensure zero-downtime startups.
4. **AI Coach & Optimizer:** Prompt-engineered Google Gemini integrations that suggest:
   *   Weak sections to fix.
   *   Google-style **X-Y-Z formula** bullet point rewrites.
   *   Stronger action verbs.
   *   Custom **Beginner ➡️ Intermediate ➡️ Advanced** learning paths.
5. **Interactive Interview Simulator:** Generates custom technical (by skill) and behavioral (STAR method) questions across Easy/Medium/Hard difficulties, featuring an **Interactive Practice Sandbox** that gives instant feedback on user-typed answers.
6. **Full-Featured Portfolio Dashboard:** Beautiful multi-tab dark-mode Streamlit dashboard featuring Plotly indicators (Radar, Bar, Gauge, and Pie charts).

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Streamlit Frontend] -->|API Requests| B[FastAPI Backend]
    B -->|SQLAlchemy ORM| C[SQLite/PostgreSQL DB]
    B -->|PyMuPDF| D[Parser Service]
    B -->|Sentence Transformers| E[Semantic Match Service]
    B -->|Google Gemini API| F[Gemini Coach Service]
```

---

## 🚀 Quick Start Guide

### Prerequisites
*   Python 3.12 (Recommended for pre-built package wheels)
*   Google Gemini API Key (Get one from Google AI Studio)

### Local Setup & Environment Configuration

1.  **Navigate to the project directory:**
    ```bash
    cd /Users/pranav/Desktop/ai_resume_analyzer
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    The application uses a `.env` file at the root level to manage configuration keys and database connections.
    
    Copy the example environment template:
    ```bash
    cp .env.example .env
    ```
    
    Open `.env` in your preferred editor and populate the variables:
    ```env
    # Google Gemini API Key (Required for AI suggestions and interactive coaching)
    GEMINI_API_KEY=your_actual_api_key_here

    # Database connection string (Defaults to local SQLite)
    DATABASE_URL=sqlite:///data/analysis/resume_analyzer.db
    ```
    
    *Note: If `GEMINI_API_KEY` is not provided, the application will automatically start up in **Demo Mode**, utilizing mock NLP responses so that all interface features remain fully interactive and testable.*

5.  **Start the FastAPI Backend Service:**
    ```bash
    python -m src.api.main
    ```
    The backend service will initialize the database schemas and start running at `http://localhost:8000`. You can explore the interactive API Swagger documentation at `http://localhost:8000/docs`.

6.  **Start the Streamlit Dashboard Frontend:**
    Open a new terminal session, activate the virtual environment, and run:
    ```bash
    streamlit run src/web_app/app.py
    ```
    The gorgeous visual dashboard will launch in your default web browser at `http://localhost:8501`.

---

## 🛠️ Troubleshooting

If you encounter any issues during startup or execution, review these common troubleshooting pathways:

### 1. ⚠️ Startup Diagnostics Showing Red/Yellow Badges
The Streamlit frontend runs a high-speed system health check at launch. If any badge is red:
*   **Database: Connection Failed:**
    *   Ensure that you have run the backend FastAPI service (`python -m src.api.main`) at least once. This initializes the SQLite database file and builds the required tables.
    *   Check your write permissions on the project directory. The SQLite database file is created at `data/analysis/resume_analyzer.db`.
*   **Storage: Write Blocked:**
    *   The application requires folder write permissions to save uploaded PDF resumes in the `data/uploads` directory. Ensure the user running the script has adequate read/write permissions for the `data/` folder.
*   **AI Engine: Demo Mode (Yellow):**
    *   This is a warning, not an error. It indicates that the `GEMINI_API_KEY` is missing or is using the placeholder from `.env.example`.
    *   To unlock full AI coaching feedback, bullet point rewrites, and custom learning paths, ensure you obtain a key from [Google AI Studio](https://aistudio.google.com/) and paste it correctly into your `.env` file.

### 2. 🗄️ Database Lock Errors
If you see database lock errors or SQLite exceptions under heavy usage:
*   By default, SQLite handles concurrent database connections sequentially. If you are running multiple clients, ensure the backend FastAPI service is running. It handles connection pooling automatically through SQLAlchemy.
*   In production environments, you can easily transition to **PostgreSQL** by changing the `DATABASE_URL` in your `.env` file (e.g., `DATABASE_URL=postgresql://user:password@localhost:5432/dbname`).

---

## 🐳 Docker Deployment

The application is fully containerized and deployment-ready.

1.  **Build and run both containers (FastAPI + Streamlit):**
    ```bash
    GEMINI_API_KEY=your_api_key_here docker-compose up --build
    ```
2.  Access the web application at `http://localhost:8501` and the API at `http://localhost:8000`.

---

## 📖 API Documentation

### `POST /api/upload-resume`
Uploads a PDF resume, parses its details, registers its skills, and saves it in the database.
*   **Request:** Multipart Form (`file`: PDF file)
*   **Response:** JSON structured profile including `resume_id`.

### `POST /api/analyze`
Performs comprehensive ATS calculations, semantic matching, and generates suggestions, learning paths, and interview questions.
*   **Request Body:**
    ```json
    {
      "resume_id": 1,
      "job_title": "Software Engineer",
      "job_description": "We need a Python developer who knows SQL and Docker..."
    }
    ```
*   **Response:** Complete analysis payload containing scores, missing skills, suggestions, and interview questions.

### `POST /api/generate-interview-questions`
Directly generates interview questions for a list of skills.
*   **Request Body:** `{"skills": ["fastapi", "docker"]}`
*   **Response:** Technical and behavioral questions.

### `GET /api/history`
Returns history of all uploaded resumes and past analysis summaries.

### `GET /api/analysis/{id}`
Retrieves a completed analysis record by ID.

---
## 🛠️ Tech Stack

### Frontend
- Streamlit
- Plotly

### Backend
- FastAPI
- Pydantic
- SQLAlchemy

### AI / NLP
- Google Gemini
- Sentence Transformers
- TF-IDF
- Scikit-Learn

### Database
- SQLite
- PostgreSQL (Production Ready)

### DevOps
- Docker
- Docker Compose
- Pytest


---
##  Future Roadmap

- JWT Authentication
- Multi-user Accounts
- Resume Version Comparison
- Audio Mock Interviews
- Real-time Gemini Coaching
- PostgreSQL Cloud Deployment
- AWS Deployment Pipeline

---

