import os
import sys
import json
import httpx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from src.utils.date_utils import format_date

# Add project root to path for direct imports (fallback mode)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Fallback imports
try:
    from src.utils.config import settings
    from src.database.session import SessionLocal
    from src.models.models import Resume, Analysis, Skill
    from src.services.parser_service import ParserService
    from src.services.ats_service import ATSService
    from src.services.semantic_service import SemanticService
    from src.services.gemini_service import GeminiService
    DIRECT_SERVICES_AVAILABLE = True
except Exception as e:
    DIRECT_SERVICES_AVAILABLE = False

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Resume Analyzer & Interview Coach",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme & CSS Styling ---
st.markdown("""
<style>
    /* Premium styling and dark-mode optimization */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .metric-val {
        font-size: 32px;
        font-weight: 700;
        color: #38bdf8;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 4px;
    }
    .badge-present {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
    }
    .badge-missing {
        background-color: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #dc2626;
    }
    .badge-keyword {
        background-color: #311042;
        color: #e879f9;
        border: 1px solid #c084fc;
    }
    .badge-skill {
        background-color: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #2563eb;
    }
    .custom-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .custom-title {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 15px;
        border-bottom: 1px solid #374151;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions for API/Direct Service Calls ---

def is_backend_online() -> bool:
    """Checks if the FastAPI backend is online."""
    try:
        response = httpx.get(API_URL.replace("/api", "/"), timeout=1.0)
        return response.status_code == 200
    except Exception:
        return False

# Initialize Session State
if "use_direct_mode" not in st.session_state:
    st.session_state.use_direct_mode = not is_backend_online()

# API Client
class APIClient:
    @staticmethod
    def get_history():
        if st.session_state.use_direct_mode and DIRECT_SERVICES_AVAILABLE:
            db = SessionLocal()
            try:
                resumes = db.query(Resume).order_by(Resume.upload_date.desc()).all()
                analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).all()
                return {
                    "resumes": [{"resume_id": r.resume_id, "file_name": r.file_name, "candidate_name": r.candidate_name, "upload_date": r.upload_date} for r in resumes],
                    "analyses": [{"analysis_id": a.analysis_id, "resume_id": a.resume_id, "job_title": a.job_title, "ats_score": a.ats_score, "semantic_score": a.semantic_score, "created_at": a.created_at} for a in analyses]
                }
            finally:
                db.close()
        else:
            try:
                response = httpx.get(f"{API_URL}/history")
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                st.error(f"Error fetching history: {e}")
        return {"resumes": [], "analyses": []}

    @staticmethod
    def upload_resume(file):
        if st.session_state.use_direct_mode and DIRECT_SERVICES_AVAILABLE:
            db = SessionLocal()
            try:
                # Save file locally
                upload_dir = Path(settings.UPLOAD_DIR)
                filename = f"{int(pd.Timestamp.now().timestamp())}_{file.name}"
                file_path = upload_dir / filename
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                
                # Run Parser
                parser = ParserService()
                parsed_data = parser.parse_resume(str(file_path))
                
                # Save to DB
                db_resume = Resume(
                    file_name=file.name,
                    candidate_name=parsed_data.get("candidate_name"),
                    email=parsed_data.get("email"),
                    phone=parsed_data.get("phone"),
                    education=json.dumps(parsed_data.get("education")),
                    experience=json.dumps(parsed_data.get("experience")),
                    raw_text=parsed_data.get("raw_text")
                )
                
                # Add skills
                for sname in parsed_data.get("skills", []):
                    sname_clean = sname.strip().lower()
                    if not sname_clean:
                        continue
                    db_skill = db.query(Skill).filter(Skill.skill_name == sname_clean).first()
                    if not db_skill:
                        db_skill = Skill(skill_name=sname_clean)
                        db.add(db_skill)
                        db.commit()
                        db.refresh(db_skill)
                    db_resume.skills.append(db_skill)
                
                db.add(db_resume)
                db.commit()
                db.refresh(db_resume)
                return {
                    "resume_id": db_resume.resume_id,
                    "file_name": db_resume.file_name,
                    "candidate_name": db_resume.candidate_name,
                    "email": db_resume.email,
                    "phone": db_resume.phone,
                    "skills": [s.skill_name for s in db_resume.skills],
                    "education": parsed_data.get("education", []),
                    "experience": parsed_data.get("experience", []),
                    "upload_date": db_resume.upload_date.isoformat()
                }
            finally:
                db.close()
        else:
            try:
                files = {"file": (file.name, file.getvalue(), "application/pdf")}
                response = httpx.post(f"{API_URL}/upload-resume", files=files, timeout=60.0)
                if response.status_code == 200:
                    return response.json()
                st.error(f"Upload failed: {response.text}")
            except Exception as e:
                st.error(f"Error calling upload API: {e}")
        return None

    @staticmethod
    def analyze_resume(resume_id, job_title, job_description):
        if st.session_state.use_direct_mode and DIRECT_SERVICES_AVAILABLE:
            db = SessionLocal()
            try:
                db_resume = db.query(Resume).filter(Resume.resume_id == resume_id).first()
                if not db_resume:
                    st.error("Resume not found.")
                    return None
                
                # Setup models
                ats = ATSService()
                sem = SemanticService()
                gem = GeminiService()
                
                resume_data = {
                    "skills": [s.skill_name for s in db_resume.skills],
                    "raw_text": db_resume.raw_text,
                    "education": json.loads(db_resume.education) if db_resume.education else [],
                    "experience": json.loads(db_resume.experience) if db_resume.experience else []
                }
                
                ats_result = ats.calculate_ats_score(resume_data, job_description)
                sem_score = sem.calculate_similarity(db_resume.raw_text, job_description)
                sem_percentage = round(sem_score * 100.0, 1)
                
                suggestions = gem.generate_suggestions(db_resume.raw_text, job_description)
                learning_path = gem.generate_learning_path(ats_result["missing_skills"])
                interview_questions = gem.generate_interview_questions(resume_data["skills"])
                
                db_analysis = Analysis(
                    resume_id=db_resume.resume_id,
                    job_title=job_title,
                    job_description=job_description,
                    ats_score=ats_result["ats_score"],
                    skill_match_score=ats_result["sub_scores"]["skill_match"],
                    keyword_match_score=ats_result["sub_scores"]["keyword_match"],
                    experience_match_score=ats_result["sub_scores"]["experience_match"],
                    education_match_score=ats_result["sub_scores"]["education_match"],
                    semantic_score=sem_percentage,
                    skills_present=json.dumps(ats_result["skills_present"]),
                    missing_skills=json.dumps(ats_result["missing_skills"]),
                    learning_path=json.dumps(learning_path),
                    suggestions=json.dumps(suggestions),
                    interview_questions=json.dumps(interview_questions)
                )
                
                db.add(db_analysis)
                db.commit()
                db.refresh(db_analysis)
                
                return {
                    "analysis_id": db_analysis.analysis_id,
                    "resume_id": db_analysis.resume_id,
                    "job_title": db_analysis.job_title,
                    "job_description": db_analysis.job_description,
                    "ats_score": db_analysis.ats_score,
                    "sub_scores": {
                        "skill_match": db_analysis.skill_match_score,
                        "keyword_match": db_analysis.keyword_match_score,
                        "experience_match": db_analysis.experience_match_score,
                        "education_match": db_analysis.education_match_score
                    },
                    "semantic_score": db_analysis.semantic_score,
                    "skills_present": ats_result["skills_present"],
                    "missing_skills": ats_result["missing_skills"],
                    "learning_path": learning_path,
                    "suggestions": suggestions,
                    "interview_questions": interview_questions,
                    "created_at": db_analysis.created_at.isoformat()
                }
            finally:
                db.close()
        else:
            try:
                payload = {
                    "resume_id": resume_id,
                    "job_title": job_title,
                    "job_description": job_description
                }
                response = httpx.post(f"{API_URL}/analyze", json=payload, timeout=90.0)
                if response.status_code == 200:
                    return response.json()
                st.error(f"Analysis failed: {response.text}")
            except Exception as e:
                st.error(f"Error calling analyze API: {e}")
        return None

    @staticmethod
    def get_analysis(analysis_id):
        if st.session_state.use_direct_mode and DIRECT_SERVICES_AVAILABLE:
            db = SessionLocal()
            try:
                db_analysis = db.query(Analysis).filter(Analysis.analysis_id == analysis_id).first()
                if not db_analysis:
                    return None
                return {
                    "analysis_id": db_analysis.analysis_id,
                    "resume_id": db_analysis.resume_id,
                    "job_title": db_analysis.job_title,
                    "job_description": db_analysis.job_description,
                    "ats_score": db_analysis.ats_score,
                    "sub_scores": {
                        "skill_match": db_analysis.skill_match_score,
                        "keyword_match": db_analysis.keyword_match_score,
                        "experience_match": db_analysis.experience_match_score,
                        "education_match": db_analysis.education_match_score
                    },
                    "semantic_score": db_analysis.semantic_score,
                    "skills_present": json.loads(db_analysis.skills_present) if db_analysis.skills_present else [],
                    "missing_skills": json.loads(db_analysis.missing_skills) if db_analysis.missing_skills else [],
                    "learning_path": json.loads(db_analysis.learning_path) if db_analysis.learning_path else {},
                    "suggestions": json.loads(db_analysis.suggestions) if db_analysis.suggestions else {},
                    "interview_questions": json.loads(db_analysis.interview_questions) if db_analysis.interview_questions else {},
                    "created_at": db_analysis.created_at.isoformat()
                }
            finally:
                db.close()
        else:
            try:
                response = httpx.get(f"{API_URL}/analysis/{analysis_id}")
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                st.error(f"Error fetching analysis: {e}")
        return None


# --- Sidebar Section ---
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)

# Connection Status
backend_ok = is_backend_online()
if backend_ok:
    st.sidebar.success("🟢 API Server: Connected")
else:
    st.sidebar.warning("🟡 API Server: Offline (Direct Mode)")
    st.session_state.use_direct_mode = True

# API Key Status
api_key = settings.GEMINI_API_KEY if DIRECT_SERVICES_AVAILABLE else os.getenv("GEMINI_API_KEY", "")
if api_key:
    st.sidebar.success("🔑 Gemini API: Configured")
else:
    st.sidebar.error("❌ Gemini API: Missing (Using Mock)")

# Mode Toggle (If backend is online, user can toggle between API and Direct for demonstration)
if backend_ok and DIRECT_SERVICES_AVAILABLE:
    st.session_state.use_direct_mode = st.sidebar.checkbox(
        "Use Direct Python Fallback",
        value=st.session_state.use_direct_mode,
        help="Bypass FastAPI backend and run services directly in the Streamlit process."
    )

# Fetch History
history = APIClient.get_history()
resumes = history.get("resumes", [])
analyses = history.get("analyses", [])

# 1. Resume Selection / Upload
st.sidebar.markdown("### 📄 Select Resume")
upload_mode = st.sidebar.radio("Resume Source:", ["Upload New PDF", "Select from History"], label_visibility="collapsed")

selected_resume_id = None
resume_details = None

if upload_mode == "Upload New PDF":
    uploaded_file = st.sidebar.file_uploader("Upload PDF Resume:", type=["pdf"])
    if uploaded_file is not None:
        # Check if we already uploaded this file in this session
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            with st.spinner("Parsing and extracting resume..."):
                result = APIClient.upload_resume(uploaded_file)
                if result:
                    st.session_state.last_uploaded = uploaded_file.name
                    st.session_state.active_resume = result
                    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
                    # Force reload history
                    history = APIClient.get_history()
                    resumes = history.get("resumes", [])
        
        if "active_resume" in st.session_state:
            resume_details = st.session_state.active_resume
            selected_resume_id = resume_details["resume_id"]
else:
    if resumes:
        resume_options = {f"{r['candidate_name'] or 'Unknown'} ({r['file_name']})": r['resume_id'] for r in resumes}
        selected_option = st.sidebar.selectbox("Choose resume:", list(resume_options.keys()))
        selected_resume_id = resume_options[selected_option]
        
        # Load resume details (simulate or query. Here we look up in local state or DB)
        if DIRECT_SERVICES_AVAILABLE:
            db = SessionLocal()
            try:
                r = db.query(Resume).filter(Resume.resume_id == selected_resume_id).first()
                if r:
                    resume_details = {
                        "resume_id": r.resume_id,
                        "file_name": r.file_name,
                        "candidate_name": r.candidate_name,
                        "email": r.email,
                        "phone": r.phone,
                        "skills": [s.skill_name for s in r.skills],
                        "education": json.loads(r.education) if r.education else [],
                        "experience": json.loads(r.experience) if r.experience else []
                    }
            finally:
                db.close()
    else:
        st.sidebar.info("No resumes in history. Please upload a PDF.")

# 2. Job Details Input
st.sidebar.markdown("### 💼 Job Details")
job_title = st.sidebar.text_input("Job Title:", value="Senior Backend Engineer")
job_description = st.sidebar.text_area(
    "Job Description:",
    height=200,
    value="""We are looking for a Senior Backend Engineer to design and build our core platform APIs.
Required Skills: Python, SQL, PostgreSQL, FastAPI, Docker, CI/CD, Microservices.
Experience: 4+ years of experience in software engineering.
Education: Bachelor's degree in Computer Science or related field.
Responsibilities:
- Write clean, maintainable FastAPI endpoints.
- Optimize complex database queries in PostgreSQL.
- Package services in Docker containers and deploy using CI/CD pipelines."""
)

# 3. Analyze Trigger
if st.sidebar.button("🚀 Run Full AI Analysis", use_container_width=True):
    if selected_resume_id is None:
        st.sidebar.error("Please upload or select a resume first.")
    elif not job_description.strip():
        st.sidebar.error("Please paste a job description.")
    else:
        with st.spinner("Running comprehensive ATS, semantic, and AI coaching analysis..."):
            analysis_result = APIClient.analyze_resume(selected_resume_id, job_title, job_description)
            if analysis_result:
                st.session_state.active_analysis = analysis_result
                st.sidebar.success("Analysis completed successfully!")
                # Force reload history
                history = APIClient.get_history()
                analyses = history.get("analyses", [])

# Select from Past Analyses if available
if analyses:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📜 Past Analyses")
    analysis_options = {f"{a['job_title']} - {a['ats_score']}% ATS ({format_date(a['created_at'])})": a['analysis_id'] for a in analyses}
    selected_anal_option = st.sidebar.selectbox("Load past analysis:", ["-- Select --"] + list(analysis_options.keys()))
    if selected_anal_option != "-- Select --":
        anal_id = analysis_options[selected_anal_option]
        if "active_analysis" not in st.session_state or st.session_state.active_analysis.get("analysis_id") != anal_id:
            with st.spinner("Loading analysis..."):
                st.session_state.active_analysis = APIClient.get_analysis(anal_id)
                # Load the linked resume as well
                if DIRECT_SERVICES_AVAILABLE:
                    db = SessionLocal()
                    try:
                        r = db.query(Resume).filter(Resume.resume_id == st.session_state.active_analysis["resume_id"]).first()
                        if r:
                            resume_details = {
                                "resume_id": r.resume_id,
                                "file_name": r.file_name,
                                "candidate_name": r.candidate_name,
                                "email": r.email,
                                "phone": r.phone,
                                "skills": [s.skill_name for s in r.skills],
                                "education": json.loads(r.education) if r.education else [],
                                "experience": json.loads(r.experience) if r.experience else []
                            }
                    finally:
                        db.close()


# --- Main Dashboard Content ---

st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>💼 AI Resume Analyzer & Interview Coach</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af; margin-bottom: 30px;'>Production-Grade ATS Optimizer, Semantic Scorer, and AI Interview Simulator</p>", unsafe_allow_html=True)

# Active state verification
analysis = st.session_state.get("active_analysis")

if not analysis:
    # Jumbotron/Onboarding screen when no analysis is active
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown(f"""
        <div class="custom-card" style="text-align: center; padding: 40px 30px;">
            <h2 style="margin-bottom: 20px;">Welcome to your AI Career Dashboard</h2>
            <p style="font-size: 16px; color: #9ca3af; line-height: 1.6;">
                This platform utilizes cutting-edge NLP and Generative AI to analyze your resume, optimize it for Applicant Tracking Systems (ATS), and prepare you for technical and behavioral interviews.
            </p>
            <div style="margin: 30px 0; border-top: 1px solid #374151; border-bottom: 1px solid #374151; padding: 20px 0; text-align: left;">
                <h4 style="color: #ffffff; margin-bottom: 15px;">👉 To get started:</h4>
                <ol style="color: #c9d1d9; font-size: 15px; padding-left: 20px; line-height: 1.8;">
                    <li>Upload your <b>PDF Resume</b> in the left sidebar (or select one from history).</li>
                    <li>Verify the candidate information and skills parsed from your file.</li>
                    <li>Provide a <b>Job Title</b> and paste the <b>Job Description</b> you are targeting.</li>
                    <li>Click <b>Run Full AI Analysis</b> to generate your score, skill gaps, and interview plan.</li>
                </ol>
            </div>
            <p style="color: #38bdf8; font-size: 14px; font-weight: 500;">
                Ready to accelerate your career? Use the Control Panel on the left!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display resume details if a resume is uploaded but not analyzed yet
        if resume_details:
            st.markdown("### 📄 Parsed Resume Preview")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="custom-card">
                    <div class="custom-title">Contact Information</div>
                    <b>Name:</b> {resume_details['candidate_name'] or 'Not detected'}<br>
                    <b>Email:</b> {resume_details['email'] or 'Not detected'}<br>
                    <b>Phone:</b> {resume_details['phone'] or 'Not detected'}
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='custom-card'><div class='custom-title'>Detected Skills</div>" + 
                            " ".join([f'<span class="badge badge-skill">{s}</span>' for s in resume_details['skills']]) + 
                            "</div>", unsafe_allow_html=True)
else:
    # Setup Tabs
    tab_overview, tab_gap, tab_coach, tab_suggest = st.tabs([
        "📊 Overview Dashboard",
        "⚖️ Skill Gap Analysis",
        "🗣️ Interview Preparation",
        "💡 Improvement Suggestions"
    ])

    # ---------------------------------------------------------
    # TAB 1: OVERVIEW DASHBOARD
    # ---------------------------------------------------------
    with tab_overview:
        st.markdown("### 📈 Analysis Overview")
        
        # High-level Metrics Row
        col_gauge, col_metrics = st.columns([2, 3])
        
        with col_gauge:
            # Gauge chart for ATS Score using Plotly
            ats_val = analysis["ats_score"]
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = ats_val,
                title = {'text': "Overall ATS Compatibility", 'font': {'size': 20, 'color': '#ffffff'}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4b5563"},
                    'bar': {'color': "#0284c7"}, # Blue bar
                    'bgcolor': "#1f2937",
                    'borderwidth': 2,
                    'bordercolor': "#374151",
                    'steps': [
                        {'range': [0, 50], 'color': '#7f1d1d'},    # Red
                        {'range': [50, 75], 'color': '#854d0e'},   # Orange/Yellow
                        {'range': [75, 100], 'color': '#064e3b'}   # Green
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#ffffff"},
                height=320,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_metrics:
            # Multi-metric display
            st.write("")
            st.write("")
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                # Skill Match Sub-Score
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🎯 Skill Match</div>
                    <div class="metric-val">{analysis['sub_scores']['skill_match']}%</div>
                    <div style="font-size: 12px; color: #9ca3af;">40% ATS Weight</div>
                </div>
                """, unsafe_allow_html=True)
                # Experience Match Sub-Score
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">⏳ Experience Match</div>
                    <div class="metric-val">{analysis['sub_scores']['experience_match']}%</div>
                    <div style="font-size: 12px; color: #9ca3af;">20% ATS Weight</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col2:
                # Keyword Match Sub-Score
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🔑 Keyword Match</div>
                    <div class="metric-val">{analysis['sub_scores']['keyword_match']}%</div>
                    <div style="font-size: 12px; color: #9ca3af;">30% ATS Weight</div>
                </div>
                """, unsafe_allow_html=True)
                # Education Match Sub-Score
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🎓 Education Match</div>
                    <div class="metric-val">{analysis['sub_scores']['education_match']}%</div>
                    <div style="font-size: 12px; color: #9ca3af;">10% ATS Weight</div>
                </div>
                """, unsafe_allow_html=True)

        # Semantic Similarity Banner
        sem_score = analysis.get("semantic_score", 0.0)
        sem_color = "#34d399" if sem_score >= 80 else "#fbbf24" if sem_score >= 60 else "#f87171"
        st.markdown(f"""
        <div class="custom-card" style="border-left: 5px solid {sem_color}; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 18px; font-weight: 600; color: #ffffff;">🧠 Semantic Alignment Score:</span>
                <span style="font-size: 22px; font-weight: 700; color: {sem_color}; margin-left: 10px;">{sem_score}%</span>
            </div>
            <div style="font-size: 13px; color: #9ca3af; max-width: 60%; text-align: right;">
                Calculated via <b>sentence-transformers (all-MiniLM-L6-v2)</b>. Represents the conceptual overlap between your entire professional background and the job requirements.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Summary / Candidate Info Block
        c_left, c_right = st.columns([1, 2])
        with c_left:
            if resume_details:
                st.markdown(f"""
                <div class="custom-card" style="height: 100%;">
                    <div class="custom-title">👤 Candidate Profile</div>
                    <table style="width:100%; border-collapse: collapse; color: #c9d1d9;">
                        <tr style="border-bottom: 1px solid #374151;"><td style="padding: 8px 0; font-weight: 600; color: #ffffff;">Name</td><td>{resume_details.get('candidate_name') or 'N/A'}</td></tr>
                        <tr style="border-bottom: 1px solid #374151;"><td style="padding: 8px 0; font-weight: 600; color: #ffffff;">Email</td><td>{resume_details.get('email') or 'N/A'}</td></tr>
                        <tr style="border-bottom: 1px solid #374151;"><td style="padding: 8px 0; font-weight: 600; color: #ffffff;">Phone</td><td>{resume_details.get('phone') or 'N/A'}</td></tr>
                        <tr style="border-bottom: 1px solid #374151;"><td style="padding: 8px 0; font-weight: 600; color: #ffffff;">Parsed Skills</td><td>{analysis.get('candidate_metrics', {}).get('parsed_skills_count', 0)} skills</td></tr>
                        <tr><td style="padding: 8px 0; font-weight: 600; color: #ffffff;">Experience</td><td>{analysis.get('candidate_metrics', {}).get('parsed_experience_years', 0)} years</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
        with c_right:
            # Resume Executive Summary (Synthesized or generated)
            st.markdown(f"""
            <div class="custom-card" style="height: 100%;">
                <div class="custom-title">📝 Analysis Executive Summary</div>
                <p style="line-height: 1.6; color: #c9d1d9;">
                    The candidate's profile is a <b>{ats_val}% match</b> for the <b>{analysis['job_title']}</b> position. 
                    The resume demonstrates strong compatibility in <b>{analysis['sub_scores']['experience_match']}% experience matching</b> with {analysis.get('candidate_metrics', {}).get('parsed_experience_years', 0)} years of parsed experience versus {analysis.get('job_metrics', {}).get('required_experience_years', 0)} years required.
                    However, there is a gap of <b>{len(analysis['missing_skills'])} missing skills</b> (e.g., {', '.join(analysis['missing_skills'][:4]) if analysis['missing_skills'] else 'None'}) which, if addressed, could significantly boost the ATS compatibility and improve interview chances.
                </p>
                <div style="margin-top: 15px; color: #38bdf8; font-size: 14px; font-weight: 600;">
                    💡 Recommended next step: Review the 'Skill Gap Analysis' page to map out a learning plan.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 2: SKILL GAP ANALYSIS
    # ---------------------------------------------------------
    with tab_gap:
        st.markdown("### ⚖️ Technical Skill Gap Analysis")
        
        # Skill comparison cards
        col_present, col_missing = st.columns(2)
        
        with col_present:
            st.markdown("<div class='custom-card' style='min-height: 250px;'>", unsafe_allow_html=True)
            st.markdown("<div class='custom-title'>✅ Skills Present (Matching)</div>", unsafe_allow_html=True)
            present_skills = analysis.get("skills_present", [])
            if present_skills:
                st.markdown(" ".join([f'<span class="badge badge-present">{s}</span>' for s in present_skills]), unsafe_allow_html=True)
            else:
                st.write("No matching skills detected between the resume and the job description.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_missing:
            st.markdown("<div class='custom-card' style='min-height: 250px;'>", unsafe_allow_html=True)
            st.markdown("<div class='custom-title'>❌ Missing Skills</div>", unsafe_allow_html=True)
            missing_skills = analysis.get("missing_skills", [])
            if missing_skills:
                st.markdown(" ".join([f'<span class="badge badge-missing">{s}</span>' for s in missing_skills]), unsafe_allow_html=True)
            else:
                st.success("Excellent! You have all the skills explicitly required in the job description.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Plotly Bar Chart: Present vs Missing Skills
        st.markdown("### 📊 Skill Distribution Breakdown")
        categories = ['Matching Skills', 'Missing Skills']
        counts = [len(present_skills), len(missing_skills)]
        
        fig_bar = go.Figure([go.Bar(
            x=categories, 
            y=counts,
            marker_color=['#059669', '#dc2626'],
            width=0.4
        )])
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#ffffff"},
            height=300,
            yaxis_title="Count of Skills",
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Recommended Learning Path Phase (Beginner, Intermediate, Advanced)
        st.markdown("### 🗺️ Custom Learning Path")
        st.write("A tailored educational roadmap to bridge your technical gaps and acquire missing capabilities.")
        
        learning_data = analysis.get("learning_path", {})
        if learning_data:
            tab_beg, tab_int, tab_adv = st.tabs(["🌱 Beginner Phase", "🚀 Intermediate Phase", "🏆 Advanced Phase"])
            
            with tab_beg:
                beg_list = learning_data.get("beginner", [])
                if beg_list:
                    for idx, item in enumerate(beg_list):
                        st.markdown(f"""
                        <div class="custom-card" style="margin-bottom: 10px;">
                            <div style="font-weight: 600; font-size: 16px; color: #38bdf8;">Topic {idx+1}: {item.get('topic')}</div>
                            <p style="margin: 8px 0; font-size: 14px; color: #c9d1d9;">{item.get('description')}</p>
                            <div style="font-size: 12px; color: #9ca3af;"><b>Recommended Resource:</b> {item.get('resource_suggestion')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No beginner topics required.")
                    
            with tab_int:
                int_list = learning_data.get("intermediate", [])
                if int_list:
                    for idx, item in enumerate(int_list):
                        st.markdown(f"""
                        <div class="custom-card" style="margin-bottom: 10px;">
                            <div style="font-weight: 600; font-size: 16px; color: #a855f7;">Topic {idx+1}: {item.get('topic')}</div>
                            <p style="margin: 8px 0; font-size: 14px; color: #c9d1d9;">{item.get('description')}</p>
                            <div style="font-size: 12px; color: #9ca3af;"><b>Recommended Hands-on Project:</b> {item.get('resource_suggestion')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No intermediate topics required.")
                    
            with tab_adv:
                adv_list = learning_data.get("advanced", [])
                if adv_list:
                    for idx, item in enumerate(adv_list):
                        st.markdown(f"""
                        <div class="custom-card" style="margin-bottom: 10px;">
                            <div style="font-weight: 600; font-size: 16px; color: #f43f5e;">Topic {idx+1}: {item.get('topic')}</div>
                            <p style="margin: 8px 0; font-size: 14px; color: #c9d1d9;">{item.get('description')}</p>
                            <div style="font-size: 12px; color: #9ca3af;"><b>Advanced System Challenge:</b> {item.get('resource_suggestion')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No advanced topics required.")
        else:
            st.info("No missing skills detected, no learning path necessary.")

    # ---------------------------------------------------------
    # TAB 3: INTERVIEW PREPARATION
    # ---------------------------------------------------------
    with tab_coach:
        st.markdown("### 🗣️ AI Interview Coach & Q&A")
        
        # Display questions from analysis
        questions_data = analysis.get("interview_questions", {})
        
        col_qs_list, col_sandbox = st.columns([3, 2])
        
        with col_qs_list:
            st.markdown("#### 🎯 Practice Interview Questions")
            st.write("Prepare with these questions tailored to your parsed skills and experience.")
            
            # Difficulty selector
            diff_filter = st.radio("Filter by Difficulty:", ["All", "Easy", "Medium", "Hard"], horizontal=True)
            
            # Technical Questions
            st.markdown("##### 💻 Technical Questions")
            tech_qs = questions_data.get("technical", [])
            visible_tech = 0
            for idx, q in enumerate(tech_qs):
                if diff_filter != "All" and q.get("difficulty") != diff_filter:
                    continue
                visible_tech += 1
                diff_color = "#34d399" if q.get("difficulty") == "Easy" else "#fbbf24" if q.get("difficulty") == "Medium" else "#f87171"
                
                with st.expander(f"Q: {q.get('question')} ({q.get('skill')})"):
                    st.markdown(f"<span style='color: {diff_color}; font-weight: 600;'>Difficulty: {q.get('difficulty')}</span>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-top: 10px;'><b>Ideal Answer:</b><br>{q.get('ideal_answer')}</p>", unsafe_allow_html=True)
                    
            if visible_tech == 0:
                st.info("No technical questions match this difficulty filter.")
                
            # Behavioral Questions
            st.markdown("##### 👥 Behavioral Questions")
            beh_qs = questions_data.get("behavioral", [])
            visible_beh = 0
            for idx, q in enumerate(beh_qs):
                if diff_filter != "All" and q.get("difficulty") != diff_filter:
                    continue
                visible_beh += 1
                diff_color = "#34d399" if q.get("difficulty") == "Easy" else "#fbbf24" if q.get("difficulty") == "Medium" else "#f87171"
                
                with st.expander(f"Q: {q.get('question')} ({q.get('scenario')})"):
                    st.markdown(f"<span style='color: {diff_color}; font-weight: 600;'>Difficulty: {q.get('difficulty')}</span>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-top: 10px;'><b>Ideal Answer (STAR Method):</b><br>{q.get('ideal_answer')}</p>", unsafe_allow_html=True)
                    
            if visible_beh == 0:
                st.info("No behavioral questions match this difficulty filter.")
                
        with col_sandbox:
            st.markdown("#### 💬 AI Practice Sandbox (WOW Feature)")
            st.write("Test your readiness! Type your answer to any question above and receive instant coaching feedback.")
            
            # Select a question to answer
            all_questions = []
            for q in tech_qs:
                all_questions.append(f"Technical: {q.get('question')}")
            for q in beh_qs:
                all_questions.append(f"Behavioral: {q.get('question')}")
                
            if all_questions:
                selected_q = st.selectbox("Select question to practice:", all_questions)
                user_answer = st.text_area("Your Response:", height=200, placeholder="Type your response here. For behavioral questions, try to use the STAR method...")
                
                if st.button("Evaluate My Answer", use_container_width=True):
                    if not user_answer.strip():
                        st.error("Please enter your answer first.")
                    elif not api_key:
                        st.warning("Running in Demo Mode without Gemini API. Simulating feedback...")
                        st.success("""
                        ### 📝 Mock AI Coaching Feedback
                        
                        **Analysis of your response:**
                        *   **Structure:** You structure your answer well, presenting the core technical concepts.
                        *   **Strengths:** You mentioned the exact keywords matching the technology stack (e.g. FastAPI, Docker).
                        *   **Areas of Improvement:** You need to quantify the results of your actions. Instead of saying 'helped with database speed', try saying 'improved query latency by 35%'.
                        
                        **Estimated Answer Score: 78/100 (Strong Effort)**
                        """)
                    else:
                        with st.spinner("Analyzing your response..."):
                            # Call Gemini to evaluate the answer in the sandbox
                            try:
                                model = genai.GenerativeModel("gemini-1.5-flash")
                                eval_prompt = f"""
                                You are a technical hiring coach. Evaluate the candidate's response to the following interview question.
                                Provide constructive, actionable feedback and score their response out of 100.
                                
                                Question:
                                {selected_q}
                                
                                Candidate's Response:
                                {user_answer}
                                
                                Format your response beautifully using markdown:
                                ### 📝 AI Coaching Feedback
                                **Estimated Answer Score: XX/100**
                                - **Strengths:** [What they did well]
                                - **Key Gaps:** [What was missing]
                                - **Suggested Improvement:** [How to rewrite or expand it]
                                """
                                eval_response = model.generate_content(eval_prompt)
                                st.markdown(eval_response.text)
                            except Exception as e:
                                st.error(f"Failed to analyze response: {e}")
            else:
                st.info("Run an analysis first to populate practice questions.")

    # ---------------------------------------------------------
    # TAB 4: IMPROVEMENT SUGGESTIONS
    # ---------------------------------------------------------
    with tab_suggest:
        st.markdown("### 💡 Resume Optimization & Suggestions")
        
        suggestions_data = analysis.get("suggestions", {})
        
        col_charts, col_text = st.columns([2, 3])
        
        with col_charts:
            st.markdown("#### 📊 ATS Category Comparison")
            # Radar chart showing ATS breakdown (Plotly)
            categories = ['Skill Match', 'Keyword Match', 'Experience Match', 'Education Match']
            scores = [
                analysis['sub_scores']['skill_match'],
                analysis['sub_scores']['keyword_match'],
                analysis['sub_scores']['experience_match'],
                analysis['sub_scores']['education_match']
            ]
            
            # Close the radar loop
            categories_radar = [*categories, categories[0]]
            scores_radar = [*scores, scores[0]]
            
            fig_radar = go.Figure(
                data=[
                    go.Scatterpolar(r=scores_radar, theta=categories_radar, fill='toself', name='Sub Scores', line_color='#38bdf8')
                ],
                layout=go.Layout(
                    title=dict(text='ATS Dimension Analysis', font=dict(color='#ffffff')),
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor='#374151', linecolor='#374151'),
                        angularaxis=dict(gridcolor='#374151', linecolor='#374151'),
                        bgcolor='rgba(31, 41, 55, 0.5)'
                    ),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff'),
                    height=320,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Pie Chart for Skill breakdown if available
            st.markdown("#### ⚙️ Skill Category Breakdown")
            all_skills = (analysis.get("skills_present", []) + analysis.get("missing_skills", []))
            
            # Let's categorize skills heuristically
            languages = ["python", "sql", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "html", "css"]
            frameworks = ["react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring boot"]
            db_cloud = ["postgresql", "mysql", "mongodb", "redis", "aws", "azure", "gcp", "docker", "kubernetes", "snowflake", "sqlite"]
            
            lang_count = sum(1 for s in all_skills if s in languages)
            frame_count = sum(1 for s in all_skills if s in frameworks)
            db_count = sum(1 for s in all_skills if s in db_cloud)
            other_count = max(0, len(all_skills) - (lang_count + frame_count + db_count))
            
            pie_labels = ['Languages', 'Frameworks/APIs', 'Databases/Cloud/DevOps', 'Other Tech']
            pie_values = [lang_count, frame_count, db_count, other_count]
            
            # Only draw if we have items
            if sum(pie_values) > 0:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=pie_labels, 
                    values=pie_values,
                    hole=.3,
                    marker_colors=['#38bdf8', '#a855f7', '#059669', '#6b7280']
                )])
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff'),
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.write("No skills to display category breakdown.")
                
        with col_text:
            st.markdown("#### 💡 Resume Fixes & Optimizations")
            st.write("Implement these changes to significantly improve your resume readability and compatibility.")
            
            # 1. Weak Sections
            st.markdown("##### ⚠️ Weak Sections Identified")
            for ws in suggestions_data.get("weak_sections", []):
                st.markdown(f"""
                <div style="background-color: rgba(220, 38, 38, 0.1); border-left: 4px solid #dc2626; padding: 12px; border-radius: 4px; margin-bottom: 10px;">
                    <div style="font-weight: 600; color: #fca5a5;">{ws.get('section')}</div>
                    <div style="font-size: 13px; margin: 4px 0;"><b>Issue:</b> {ws.get('issue')}</div>
                    <div style="font-size: 13px; color: #34d399;"><b>Fix:</b> {ws.get('improvement')}</div>
                </div>
                """, unsafe_allow_html=True)
                
            # 2. Action Verbs
            st.markdown("##### ⚡ Stronger Action Verbs")
            for av in suggestions_data.get("action_verbs", []):
                st.markdown(f"""
                <div style="background-color: rgba(56, 189, 248, 0.1); border-left: 4px solid #38bdf8; padding: 12px; border-radius: 4px; margin-bottom: 10px; display: flex; justify-content: space-between;">
                    <div>
                        <span style="text-decoration: line-through; color: #9ca3af; font-size: 13px; margin-right: 10px;">{av.get('original')}</span>
                        <span style="font-weight: 600; color: #38bdf8; font-size: 15px;">➡️ {av.get('suggested')}</span>
                    </div>
                    <div style="font-size: 13px; color: #d1d5db; max-width: 60%; text-align: right;">
                        {av.get('context')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # 3. Bullet Point Enhancements (X-Y-Z formula)
            st.markdown("##### ✍️ Bullet Point Re-writes (Google X-Y-Z Formula)")
            for bp in suggestions_data.get("bullet_points", []):
                st.markdown(f"""
                <div class="custom-card" style="padding: 15px; margin-bottom: 10px; border: 1px dashed #4b5563;">
                    <div style="font-size: 13px; color: #f87171; margin-bottom: 5px;"><b>Original:</b> <i>"{bp.get('original')}"</i></div>
                    <div style="font-size: 13px; color: #34d399; margin-bottom: 5px;"><b>Optimized:</b> <b>"{bp.get('improved')}"</b></div>
                    <div style="font-size: 12px; color: #9ca3af;"><b>Rationale:</b> {bp.get('rationale')}</div>
                </div>
                """, unsafe_allow_html=True)

        # PDF/Markdown Report Export Button
        st.markdown("---")
        st.markdown("#### 📁 Export Portfolio-Ready Analysis Report")
        
        # Build Markdown content for download
        report_md = f"""# AI Resume Analysis & Coaching Report
Job Title Target: {analysis['job_title']}
Overall ATS Compatibility Score: {analysis['ats_score']}/100
Semantic Alignment Score: {analysis['semantic_score']}%

## ATS Sub-Score Breakdown:
- Skill Match: {analysis['sub_scores']['skill_match']}%
- Keyword Match: {analysis['sub_scores']['keyword_match']}%
- Experience Match: {analysis['sub_scores']['experience_match']}%
- Education Match: {analysis['sub_scores']['education_match']}%

## Skill Gap Analysis:
### Matching Skills:
{", ".join(analysis['skills_present']) if analysis['skills_present'] else 'None'}

### Missing Skills:
{", ".join(analysis['missing_skills']) if analysis['missing_skills'] else 'None'}

## Key Resume Suggestions:
"""
        for idx, ws in enumerate(suggestions_data.get("weak_sections", [])):
            report_md += f"{idx+1}. **Section: {ws.get('section')}**\n   - *Issue:* {ws.get('issue')}\n   - *Fix:* {ws.get('improvement')}\n\n"
            
        report_md += "\n## Practice Questions Highlights:\n"
        for idx, q in enumerate(questions_data.get("technical", [])[:2]):
            report_md += f"- **Q ({q.get('skill')}):** {q.get('question')}\n  - *Ideal Answer:* {q.get('ideal_answer')}\n\n"
            
        st.download_button(
            label="📥 Download Polish Markdown Report",
            data=report_md,
            file_name=f"resume_coaching_report_{analysis['job_title'].replace(' ', '_').lower()}.md",
            mime="text/markdown",
            use_container_width=True
        )
