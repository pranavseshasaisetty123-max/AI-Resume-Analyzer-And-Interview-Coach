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
from dotenv import load_dotenv

# --- Load Environment Variables (Startup) ---
# Call load_dotenv() immediately at startup to populate environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Add project root to path for direct imports (fallback mode)
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
    from src.utils.date_utils import format_date
    DIRECT_SERVICES_AVAILABLE = True
except Exception as e:
    DIRECT_SERVICES_AVAILABLE = False
    # Local fallback format_date if import fails
    def format_date(dt_val):
        if not dt_val:
            return "N/A"
        try:
            return str(dt_val)[:10]
        except Exception:
            return "N/A"

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api")

# --- Global Gemini Configuration (Single Source of Truth) ---
GEMINI_API_KEY = settings.GEMINI_API_KEY if DIRECT_SERVICES_AVAILABLE else os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        # Filter mock keys or empty strings
        if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_") and len(GEMINI_API_KEY) > 8:
            genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        pass

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Resume Analyzer & Interview Coach",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme & CSS Styling (Stripe, Notion, Linear & ChatGPT Inspired) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* Main App Wrapper and Background styling */
    .stApp {
        background-color: #080c16 !important;
        background-image: radial-gradient(at 0% 0%, rgba(17, 24, 39, 0.8) 0, transparent 50%), 
                          radial-gradient(at 50% 0%, rgba(99, 102, 241, 0.05) 0, transparent 50%),
                          radial-gradient(at 100% 0%, rgba(56, 189, 248, 0.05) 0, transparent 50%);
        color: #e2e8f0 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    /* Typography Hierarchy & Contrast */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', -apple-system, sans-serif !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.025em !important;
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif !important;
        background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 800 !important;
        font-size: 38px !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 8px !important;
        text-align: center !important;
    }
    
    .main-subheader {
        color: #94a3b8 !important;
        font-size: 16px !important;
        text-align: center !important;
        max-width: 750px !important;
        margin: 0 auto 35px auto !important;
        line-height: 1.6 !important;
    }

    /* Premium Glassmorphic Cards */
    .custom-card {
        background: rgba(19, 27, 46, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        padding: 24px !important;
        border-radius: 16px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .custom-card:hover {
        border-color: rgba(99, 102, 241, 0.25) !important;
        box-shadow: 0 12px 35px -8px rgba(99, 102, 241, 0.12) !important;
    }
    
    /* Card Border Accents */
    .accent-card-blue { border-left: 4px solid #38bdf8 !important; }
    .accent-card-indigo { border-left: 4px solid #6366f1 !important; }
    .accent-card-emerald { border-left: 4px solid #10b981 !important; }
    .accent-card-rose { border-left: 4px solid #f43f5e !important; }
    .accent-card-amber { border-left: 4px solid #fbbf24 !important; }

    /* Custom Title for Cards */
    .custom-title {
        font-family: 'Outfit', sans-serif !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        margin-bottom: 16px !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding-bottom: 10px !important;
    }

    /* Elegant Pill Badges */
    .pill-badge {
        display: inline-flex !important;
        align-items: center !important;
        padding: 6px 12px !important;
        border-radius: 9999px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        margin: 4px !important;
        transition: all 0.2s ease !important;
    }
    .badge-present {
        background-color: rgba(16, 185, 129, 0.1) !important;
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
    }
    .badge-present:hover {
        background-color: rgba(16, 185, 129, 0.18) !important;
    }
    .badge-missing {
        background-color: rgba(244, 63, 94, 0.1) !important;
        color: #f87171 !important;
        border: 1px solid rgba(244, 63, 94, 0.25) !important;
    }
    .badge-missing:hover {
        background-color: rgba(244, 63, 94, 0.18) !important;
    }
    .badge-keyword {
        background-color: rgba(168, 85, 247, 0.1) !important;
        color: #c084fc !important;
        border: 1px solid rgba(168, 85, 247, 0.25) !important;
    }
    .badge-skill {
        background-color: rgba(56, 189, 248, 0.1) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
    }

    /* Custom Responsive CSS Progress Bars */
    .progress-wrapper {
        margin-bottom: 15px !important;
        width: 100% !important;
    }
    .progress-bg {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border-radius: 9999px !important;
        height: 8px !important;
        width: 100% !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
    }
    .progress-fill {
        height: 100% !important;
        border-radius: 9999px !important;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .fill-primary { background: linear-gradient(90deg, #6366f1, #38bdf8) !important; }
    .fill-success { background: linear-gradient(90deg, #059669, #10b981) !important; }
    .fill-warning { background: linear-gradient(90deg, #d97706, #fbbf24) !important; }
    .fill-danger { background: linear-gradient(90deg, #dc2626, #f43f5e) !important; }

    /* Hero Metrics & Score styling */
    .hero-score-val {
        font-family: 'Outfit', sans-serif !important;
        font-size: 64px !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
        background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin: 12px 0 !important;
    }
    
    .kpi-card {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 18px !important;
        border-radius: 14px !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
    }
    .kpi-card:hover {
        background: rgba(30, 41, 59, 0.45) !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
    }
    .kpi-val {
        font-family: 'Outfit', sans-serif !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-top: 6px !important;
    }
    .kpi-label {
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    /* Structured Profile Tables */
    .profile-table {
        width: 100% !important;
        border-collapse: collapse !important;
    }
    .profile-table tr {
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    .profile-table tr:last-child {
        border-bottom: none !important;
    }
    .profile-table td {
        padding: 12px 6px !important;
        font-size: 14px !important;
        color: #e2e8f0 !important;
    }
    .profile-table td.label {
        font-weight: 600 !important;
        color: #94a3b8 !important;
        width: 110px !important;
    }

    /* Styled Accordions / Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.25) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        padding: 12px 18px !important;
        transition: background-color 0.2s ease !important;
    }
    .streamlit-expanderHeader:hover {
        background-color: rgba(30, 41, 59, 0.45) !important;
    }
    .streamlit-expanderContent {
        background-color: rgba(19, 27, 46, 0.2) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-bottom-left-radius: 10px !important;
        border-bottom-right-radius: 10px !important;
        padding: 18px !important;
        color: #cbd5e1 !important;
    }

    /* Before/After Resume Fix Layouts */
    .rewrite-card {
        background: rgba(19, 27, 46, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 15px !important;
    }
    .original-box {
        background: rgba(244, 63, 94, 0.04) !important;
        border-left: 3px solid #f43f5e !important;
        padding: 10px 14px !important;
        border-radius: 6px !important;
        font-size: 13.5px !important;
        color: #cbd5e1 !important;
        margin-bottom: 10px !important;
        font-style: italic !important;
    }
    .improved-box {
        background: rgba(16, 185, 129, 0.04) !important;
        border-left: 3px solid #10b981 !important;
        padding: 10px 14px !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        margin-bottom: 10px !important;
    }
    .rationale-box {
        font-size: 12px !important;
        color: #94a3b8 !important;
        padding-left: 4px !important;
    }

    /* Question Scanning Elements */
    .question-card-header {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        margin-bottom: 8px !important;
        flex-wrap: wrap !important;
        gap: 6px !important;
    }
    .question-text {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        line-height: 1.5 !important;
    }

    /* Sleek Tab Customization (Notion-like) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(19, 27, 46, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        padding: 6px !important;
        border-radius: 14px !important;
        gap: 8px !important;
        margin-bottom: 25px !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.1) !important;
    }

    /* Input Fields Form Overrides */
    .stTextInput input, .stTextArea textarea, .stSelectbox [role="combobox"] {
        background-color: #0e1322 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox [role="combobox"]:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }

    /* Primary CTA Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%) !important;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-size: 15px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 18px rgba(99, 102, 241, 0.35) !important;
        color: #ffffff !important;
    }
    .stButton>button:active {
        transform: translateY(1px) !important;
    }

    /* ==============================================================================
       SIDEBAR PREMIUM REDESIGN (Linear & Vercel Inspired)
       ============================================================================== */
    div[data-testid="stSidebar"] {
        background-color: #111827 !important; /* Rich obsidian charcoal background */
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e5e7eb !important; /* High-contrast text color */
    }
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        font-size: 20px !important;
        font-weight: 700 !important;
        margin-top: 15px !important;
        margin-bottom: 20px !important;
        letter-spacing: -0.02em !important;
    }
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-top: 25px !important;
        margin-bottom: 12px !important;
    }

    /* Sidebar Button Cards (Past Analyses items) */
    div[data-testid="stSidebar"] button {
        background-color: rgba(31, 41, 55, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        color: #cbd5e1 !important;
        text-align: left !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        width: 100% !important;
        display: block !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stSidebar"] button:hover {
        background-color: rgba(55, 65, 81, 0.8) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1) !important;
    }
    div[data-testid="stSidebar"] button:active {
        transform: translateY(1px) !important;
    }

    /* Active Report Highlight inside History list */
    .active-history-card {
        border-color: rgba(56, 189, 248, 0.5) !important;
        background-color: rgba(56, 189, 248, 0.06) !important;
        color: #38bdf8 !important;
    }

    /* Reset Button - Vibrant Accent Highlight */
    div[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        text-align: center !important;
        padding: 12px 20px !important;
        margin-top: 10px !important;
        margin-bottom: 15px !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
    }
    div[data-testid="stSidebar"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5a52e6 0%, #0cc0dd 100%) !important;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Sidebar Summary Stats Grid */
    .sidebar-stats-container {
        display: grid !important;
        grid-template-columns: 1fr 1fr 1fr !important;
        gap: 8px !important;
        margin-bottom: 20px !important;
    }
    .sidebar-stat-card {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 10px 4px !important;
        border-radius: 10px !important;
        text-align: center !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
    }
    .sidebar-stat-label {
        font-size: 9px !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        margin-bottom: 2px !important;
    }
    .sidebar-stat-val {
        font-size: 13px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    /* Sidebar Expander Overrides */
    div[data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: rgba(31, 41, 55, 0.35) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        font-size: 13px !important;
    }
    div[data-testid="stSidebar"] .streamlit-expanderContent {
        background-color: rgba(17, 24, 39, 0.3) !important;
        border-radius: 0 0 10px 10px !important;
        padding: 14px !important;
        border-left: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
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
                pass
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
                    "job_title": job_title,
                    "job_description": job_description,
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

# --- Startup Validation & Diagnostics Helper (Robust, Friendly, Cached) ---

def run_startup_validation() -> dict:
    """Verifies write permissions, database access, and API credentials gracefully."""
    results = {
        "storage_ok": False,
        "storage_msg": "🟢 Ready",
        "db_ok": False,
        "db_msg": "🟢 Connected",
        "gemini_ok": False,
        "gemini_msg": "🟢 Gemini Enabled"
    }
    
    # 1. Check Storage Paths (Uploads)
    try:
        upload_dir = settings.UPLOAD_DIR if DIRECT_SERVICES_AVAILABLE else str(PROJECT_ROOT / "data" / "uploads")
        upload_path = Path(upload_dir)
        upload_path.mkdir(parents=True, exist_ok=True)
        # Test write/delete permission
        temp_file = upload_path / ".startup_write_test"
        temp_file.write_text("diagnostics_ok")
        temp_file.unlink()
        results["storage_ok"] = True
    except Exception as e:
        results["storage_ok"] = False
        results["storage_msg"] = f"🔴 Storage Write Blocked: {str(e)}"
        
    # 2. Check Database Connectivity
    try:
        if DIRECT_SERVICES_AVAILABLE:
            db = SessionLocal()
            try:
                # Query schema checks
                db.query(Resume).first()
                results["db_ok"] = True
            finally:
                db.close()
        else:
            # Check backend URL directly
            response = httpx.get(f"{API_URL}/history", timeout=1.5)
            if response.status_code == 200:
                results["db_ok"] = True
            else:
                results["db_ok"] = False
                results["db_msg"] = f"🔴 Database Service Error (Status {response.status_code})"
    except Exception as e:
        results["db_ok"] = False
        results["db_msg"] = f"🔴 Connection Failed: {str(e)}"
        
    # 3. Check Gemini API Configuration Status
    try:
        # Filter mock keys or empty strings
        if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_") and len(GEMINI_API_KEY) > 8:
            results["gemini_ok"] = True
        else:
            results["gemini_ok"] = False
            results["gemini_msg"] = "🟡 Running in Demo Mode (Mock Coaching)"
    except Exception as e:
        results["gemini_ok"] = False
        results["gemini_msg"] = f"🔴 Config Error: {str(e)}"
        
    return results

# Execute Diagnostics Check once at application start
if "startup_diagnostics" not in st.session_state:
    st.session_state.startup_diagnostics = run_startup_validation()

diag = st.session_state.startup_diagnostics

# --- Resolve Active Resume & Details ---
if "active_analysis" in st.session_state and "active_resume" not in st.session_state:
    active_analysis = st.session_state.active_analysis
    resume_id = active_analysis.get("resume_id")
    if resume_id and DIRECT_SERVICES_AVAILABLE:
        db = SessionLocal()
        try:
            r = db.query(Resume).filter(Resume.resume_id == resume_id).first()
            if r:
                st.session_state.active_resume = {
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

resume_details = st.session_state.get("active_resume")
selected_resume_id = resume_details["resume_id"] if resume_details else None

# Fetch History
history = APIClient.get_history()
resumes = history.get("resumes", [])
analyses = history.get("analyses", [])

# --- Sidebar Section (Premium Redesigned Control Panel) ---

# Check backend status once to avoid multiple network calls and resolve NameError
backend_ok = is_backend_online()

st.sidebar.markdown("<h2 style='text-align: center;'>💼 Career Dashboard</h2>", unsafe_allow_html=True)

# 1. System Diagnostics Card
api_status_text = "🟢 Connected" if backend_ok else "🟡 Direct Mode"
ai_status_text = "🟢 Gemini Enabled" if diag["gemini_ok"] else "🟡 Demo Mode"
db_status_text = "🟢 Connected" if diag["db_ok"] else "🔴 Connection Error"
storage_status_text = "🟢 Ready" if diag["storage_ok"] else "🔴 Storage Error"

st.sidebar.markdown(f"""
<div class="custom-card" style="padding: 16px !important; margin-bottom: 20px; background: rgba(30, 41, 59, 0.25) !important;">
    <div style="font-size: 11px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px;">System Diagnostics</div>
    <div style="display: flex; flex-direction: column; gap: 8px; font-size: 13px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #94a3b8; font-weight: 500;">API Status:</span>
            <span style="font-weight: 600; color: {'#34d399' if backend_ok else '#fbbf24'};">{api_status_text}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #94a3b8; font-weight: 500;">AI Status:</span>
            <span style="font-weight: 600; color: {'#34d399' if diag['gemini_ok'] else '#fbbf24'};">{ai_status_text}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #94a3b8; font-weight: 500;">Database:</span>
            <span style="font-weight: 600; color: {'#34d399' if diag['db_ok'] else '#f87171'};">{db_status_text}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #94a3b8; font-weight: 500;">Storage:</span>
            <span style="font-weight: 600; color: {'#34d399' if diag['storage_ok'] else '#f87171'};">{storage_status_text}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Quick Summary Statistics
total_analyses = len(analyses)
highest_ats = max([float(a['ats_score']) for a in analyses]) if analyses else 0.0
average_ats = sum([float(a['ats_score']) for a in analyses]) / len(analyses) if analyses else 0.0

st.sidebar.markdown(f"""
<div class="sidebar-stats-container">
    <div class="sidebar-stat-card">
        <div class="sidebar-stat-label">Analyses</div>
        <div class="sidebar-stat-val">{total_analyses}</div>
    </div>
    <div class="sidebar-stat-card">
        <div class="sidebar-stat-label">Highest</div>
        <div class="sidebar-stat-val">{highest_ats}%</div>
    </div>
    <div class="sidebar-stat-card">
        <div class="sidebar-stat-label">Average</div>
        <div class="sidebar-stat-val">{average_ats:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 3. AI Configuration Settings UI
gemini_status_label = "Configured" if diag["gemini_ok"] else "Not Configured"
st.sidebar.markdown("### 🔑 AI Configuration")
st.sidebar.markdown(f"""
<div class="custom-card" style="padding: 14px !important; margin-bottom: 15px; background: rgba(30, 41, 59, 0.15) !important;">
    <div style="font-size: 13px; line-height: 1.5;">
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94a3b8;">Gemini Key:</span>
            <span style="font-weight: 600; color: {'#34d399' if diag['gemini_ok'] else '#f87171'};">{gemini_status_label}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 4px;">
            <span style="color: #94a3b8;">Active Mode:</span>
            <span style="font-weight: 600; color: #ffffff;">{"Full AI Coaching" if diag["gemini_ok"] else "Mock Demo Mode"}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Display friendly instructions if key is missing
if not diag["gemini_ok"]:
    st.sidebar.markdown("""
    <div style="font-size: 12px; color: #cbd5e1; background: rgba(251, 191, 36, 0.04); border: 1px solid rgba(251, 191, 36, 0.15); padding: 12px; border-radius: 10px; line-height: 1.5; margin-bottom: 15px;">
        <span style="font-weight: 700; color: #fbbf24;">🔑 Unlock Full AI Capabilities:</span>
        <ol style="margin: 6px 0 0 16px; padding: 0; line-height: 1.6;">
            <li>Create a <code>.env</code> file in the project root directory.</li>
            <li>Add your Gemini API Key:
                <pre style="background: #090d16; padding: 6px; border-radius: 4px; margin-top: 4px; border: 1px solid rgba(255,255,255,0.08); font-size: 10px; color: #f43f5e; overflow-x: auto;">GEMINI_API_KEY=your_key</pre>
            </li>
            <li>Restart the application.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# 4. Active Analysis & Onboarding Switcher
analysis = st.session_state.get("active_analysis")

if analysis:
    # Render active candidate card context
    if resume_details:
        st.sidebar.markdown(f"""
        <div class="custom-card accent-card-blue" style="padding: 15px !important; margin-bottom: 10px;">
            <div style="font-size: 11px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Active Profile</div>
            <div style="font-size: 14px; font-weight: 700; color: #ffffff; line-height: 1.3;">{resume_details.get('candidate_name') or 'Unknown Candidate'}</div>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{resume_details.get('email') or 'No email parsed'}</div>
        </div>
        """, unsafe_allow_html=True)

    # Reset button (styled primary to stand out)
    if st.sidebar.button("🔄 Analyze New Resume", type="primary", use_container_width=True, help="Clear active report and return to the input screen."):
        st.session_state.active_analysis = None
        st.session_state.active_resume = None
        st.toast("Reset dashboard state", icon="🔄")
        st.rerun()

# 5. Analysis History List (Linear-Inspired Clickable Cards)
st.sidebar.markdown("### 📜 Analysis History")
if analyses:
    # Display the top 5 recent analyses as cards
    recent_analyses = analyses[:5]
    for idx, a in enumerate(recent_analyses):
        # Determine if this card represents the currently active report
        is_active = analysis and analysis.get("analysis_id") == a["analysis_id"]
        card_class = "active-history-card" if is_active else ""
        
        # Streamlit button for history selection
        btn_label = f"💼 {a['job_title']}\nATS Score: {a['ats_score']}%"
        if st.sidebar.button(btn_label, key=f"hist_btn_{a['analysis_id']}", use_container_width=True):
            with st.spinner("Loading analysis..."):
                st.session_state.active_analysis = APIClient.get_analysis(a['analysis_id'])
                if DIRECT_SERVICES_AVAILABLE:
                    db = SessionLocal()
                    try:
                        r = db.query(Resume).filter(Resume.resume_id == st.session_state.active_analysis["resume_id"]).first()
                        if r:
                            st.session_state.active_resume = {
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
                st.toast(f"Loaded: {a['job_title']}", icon="💼")
                st.rerun()
                
    # Selectbox fallback for older analyses
    if len(analyses) > 5:
        older_options = {f"💼 {a['job_title']} ({a['ats_score']}% ATS)": a['analysis_id'] for a in analyses[5:]}
        selected_older = st.sidebar.selectbox("View Older Reports:", ["-- Select Report --"] + list(older_options.keys()), key="older_reports_dropdown")
        if selected_older != "-- Select Report --":
            older_id = older_options[selected_older]
            with st.spinner("Loading analysis..."):
                st.session_state.active_analysis = APIClient.get_analysis(older_id)
                if DIRECT_SERVICES_AVAILABLE:
                    db = SessionLocal()
                    try:
                        r = db.query(Resume).filter(Resume.resume_id == st.session_state.active_analysis["resume_id"]).first()
                        if r:
                            st.session_state.active_resume = {
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
                st.toast("Loaded past report", icon="💼")
                st.rerun()
else:
    st.sidebar.markdown("<p style='font-size: 13px; color: #94a3b8; font-style: italic; text-align: center; margin: 10px 0;'>No past reports found.</p>", unsafe_allow_html=True)

# 6. Collapsible Advanced Settings (collapsible expander)
st.sidebar.write("")
with st.sidebar.expander("⚙️ Advanced Settings"):
    # Direct Fallback
    if backend_ok and DIRECT_SERVICES_AVAILABLE:
        st.session_state.use_direct_mode = st.checkbox(
            "Direct Python Fallback",
            value=st.session_state.use_direct_mode,
            help="Bypass FastAPI backend and run services directly in the Streamlit process."
        )
    
    # Debug Mode
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode, help="Outputs system path configuration and logging details.")
    st.session_state.debug_mode = debug_mode
    
    # Developer Options
    if "dev_options" not in st.session_state:
        st.session_state.dev_options = False
    dev_options = st.checkbox("Developer Options", value=st.session_state.dev_options, help="Access raw JSON payloads for analysis debugging.")
    st.session_state.dev_options = dev_options

# --- Main Page Dashboard Content ---

# Friendly Startup Diagnostics Banner (Top-level warning block for DB/Storage failures)
if not diag["db_ok"] or not diag["storage_ok"]:
    st.error("🚨 System Initialization Gaps Detected")
    if not diag["db_ok"]:
        st.markdown(f"""
        <div style="background: rgba(244,63,94,0.05); border-left: 4px solid #f43f5e; padding: 14px; border-radius: 8px; margin-bottom: 12px; font-size: 13.5px; border-top: 1px solid rgba(255,255,255,0.02); border-right: 1px solid rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.02);">
            <div style="font-weight: 700; color: #ffffff; margin-bottom: 5px; font-family: 'Outfit', sans-serif;">Database Connection Error</div>
            <div style="color: #cbd5e1; margin-bottom: 8px;">{diag['db_msg']}</div>
            <div style="color: #94a3b8; font-size: 12px;"><b>How to fix:</b> Verify your database configuration. If running for the first time, check your file permissions and execute <code>python -m src.api.main</code> to initialize the database schemas.</div>
        </div>
        """, unsafe_allow_html=True)
    if not diag["storage_ok"]:
        st.markdown(f"""
        <div style="background: rgba(244,63,94,0.05); border-left: 4px solid #f43f5e; padding: 14px; border-radius: 8px; margin-bottom: 12px; font-size: 13.5px; border-top: 1px solid rgba(255,255,255,0.02); border-right: 1px solid rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.02);">
            <div style="font-weight: 700; color: #ffffff; margin-bottom: 5px; font-family: 'Outfit', sans-serif;">Storage Permissions Error</div>
            <div style="color: #cbd5e1; margin-bottom: 8px;">{diag['storage_msg']}</div>
            <div style="color: #94a3b8; font-size: 12px;"><b>How to fix:</b> Verify that the application directory has write permissions for path <code>{settings.UPLOAD_DIR}</code>.</div>
        </div>
        """, unsafe_allow_html=True)

if not analysis:
    # 1. Main Page Onboarding Wizard (Spacious & Stepped)
    st.markdown("<div class='main-header'>💼 AI Resume Analyzer & Interview Coach</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subheader'>Optimize your resume for ATS algorithms, map technical skill gaps, and master technical & behavioral interview preparation using personalized AI coaching.</div>", unsafe_allow_html=True)
    
    col_input_left, col_input_right = st.columns([1, 1], gap="large")
    
    with col_input_left:
        st.markdown("<div class='custom-card accent-card-blue'><div class='custom-title'>📄 Step 1: Provide Your Resume</div>", unsafe_allow_html=True)
        
        upload_mode = st.radio("Resume Source:", ["Upload New PDF", "Select from History"], horizontal=True, label_visibility="collapsed")
        
        selected_resume_id = None
        resume_details = None
        
        if upload_mode == "Upload New PDF":
            uploaded_file = st.file_uploader("Upload PDF Resume:", type=["pdf"], label_visibility="collapsed")
            if uploaded_file is not None:
                if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
                    with st.spinner("Parsing and extracting resume..."):
                        result = APIClient.upload_resume(uploaded_file)
                        if result:
                            st.session_state.last_uploaded = uploaded_file.name
                            st.session_state.active_resume = result
                            st.toast(f"Parsed resume: {uploaded_file.name}", icon="✅")
                            history = APIClient.get_history()
                            resumes = history.get("resumes", [])
                
                if "active_resume" in st.session_state:
                    resume_details = st.session_state.active_resume
                    selected_resume_id = resume_details["resume_id"]
        else:
            if resumes:
                resume_options = {f"{r['candidate_name'] or 'Unknown'} ({r['file_name']})": r['resume_id'] for r in resumes}
                selected_option = st.selectbox("Choose a resume from your history:", list(resume_options.keys()))
                selected_resume_id = resume_options[selected_option]
                
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
                            st.session_state.active_resume = resume_details
                    finally:
                        db.close()
            else:
                st.info("No resumes found in history. Please upload a new PDF.")
        
        # Display parsed preview if resume is active
        if resume_details:
            st.markdown("<div style='margin-top: 20px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 15px;'>", unsafe_allow_html=True)
            st.markdown(f"**Candidate:** `{resume_details['candidate_name'] or 'Not detected'}`")
            st.markdown(f"**Email:** `{resume_details['email'] or 'Not detected'}`")
            st.markdown("**Extracted Skills Preview:**")
            skills_html = " ".join([f'<span class="pill-badge badge-skill">{s}</span>' for s in resume_details['skills'][:10]])
            if len(resume_details['skills']) > 10:
                skills_html += f" <span style='font-size: 12px; color: #94a3b8;'>+{len(resume_details['skills']) - 10} more</span>"
            st.markdown(skills_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_input_right:
        st.markdown("<div class='custom-card accent-card-indigo'><div class='custom-title'>🎯 Step 2: Target Job Details</div>", unsafe_allow_html=True)
        
        if "job_title_input" not in st.session_state:
            st.session_state.job_title_input = "Senior Backend Engineer"
        if "job_desc_input" not in st.session_state:
            st.session_state.job_desc_input = """We are looking for a Senior Backend Engineer to design and build our core platform APIs.
Required Skills: Python, SQL, PostgreSQL, FastAPI, Docker, CI/CD, Microservices.
Experience: 4+ years of experience in software engineering.
Education: Bachelor's degree in Computer Science or related field.
Responsibilities:
- Write clean, maintainable FastAPI endpoints.
- Optimize complex database queries in PostgreSQL.
- Package services in Docker containers and deploy using CI/CD pipelines."""

        job_title = st.text_input("Target Job Title:", value=st.session_state.job_title_input, key="job_title_field")
        st.session_state.job_title_input = job_title
        
        job_description = st.text_area("Paste Target Job Description:", value=st.session_state.job_desc_input, height=230, key="job_desc_field")
        st.session_state.job_desc_input = job_description
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("<div style='text-align: center; margin-top: 15px;'>", unsafe_allow_html=True)
    if st.button("🚀 Run Comprehensive AI Analysis", use_container_width=True):
        if selected_resume_id is None:
            st.error("Please upload or select a resume first.")
        elif not job_description.strip():
            st.error("Please paste a job description.")
        else:
            with st.spinner("Running comprehensive ATS, semantic, and AI coaching analysis..."):
                analysis_result = APIClient.analyze_resume(selected_resume_id, job_title, job_description)
                if analysis_result:
                    st.session_state.active_analysis = analysis_result
                    st.toast("Analysis completed successfully!", icon="🚀")
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # 2. Main Page Active Dashboard (Tabs)
    st.markdown("<div style='text-align: center; margin-top: 10px; margin-bottom: 20px;'><h1 class='main-header'>💼 Career Analytics & Coaching Dashboard</h1></div>", unsafe_allow_html=True)
    
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
        st.markdown("<div style='margin-bottom: 20px;'><h3>📈 Performance Overview</h3></div>", unsafe_allow_html=True)
        
        # Hero KPI Row
        col_hero_score, col_metrics_grid = st.columns([2, 3], gap="medium")
        
        with col_hero_score:
            ats_val = int(analysis["ats_score"])
            
            if ats_val >= 75:
                score_color_class = "fill-success"
                rating_text = "Strong Compatibility"
                rating_badge_style = "badge-present"
            elif ats_val >= 50:
                score_color_class = "fill-warning"
                rating_text = "Moderate Gaps"
                rating_badge_style = "badge-keyword"
            else:
                score_color_class = "fill-danger"
                rating_text = "Critical Optimization Required"
                rating_badge_style = "badge-missing"
                
            st.markdown(f"""
            <div class="custom-card accent-card-blue" style="height: 100%; display: flex; flex-direction: column; justify-content: center; text-align: center; padding: 30px !important;">
                <div style="font-size: 14px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Overall ATS Compatibility</div>
                <div class="hero-score-val">{ats_val}%</div>
                <div style="margin-top: 5px; margin-bottom: 20px;">
                    <span class="pill-badge {rating_badge_style}" style="font-size: 13px; font-weight: 600; padding: 6px 16px;">{rating_text}</span>
                </div>
                <div class="progress-wrapper" style="margin: 0 10px; width: auto !important;">
                    <div class="progress-bg">
                        <div class="progress-fill {score_color_class}" style="width: {ats_val}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_metrics_grid:
            st.markdown("<div style='display: flex; flex-direction: column; gap: 15px; height: 100%; justify-content: space-between;'>", unsafe_allow_html=True)
            
            r1_col1, r1_col2 = st.columns(2)
            with r1_col1:
                skill_score = int(analysis['sub_scores']['skill_match'])
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">🎯 Skill Match</div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <div class="kpi-val">{skill_score}%</div>
                        <div style="font-size: 11px; color: #64748b; font-weight: 600;">40% Weight</div>
                    </div>
                    <div class="progress-bg" style="height: 5px; margin-top: 10px;">
                        <div class="progress-fill fill-primary" style="width: {skill_score}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with r1_col2:
                kw_score = int(analysis['sub_scores']['keyword_match'])
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">🔑 Keyword Match</div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <div class="kpi-val">{kw_score}%</div>
                        <div style="font-size: 11px; color: #64748b; font-weight: 600;">30% Weight</div>
                    </div>
                    <div class="progress-bg" style="height: 5px; margin-top: 10px;">
                        <div class="progress-fill fill-primary" style="width: {kw_score}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            r2_col1, r2_col2 = st.columns(2)
            with r2_col1:
                exp_score = int(analysis['sub_scores']['experience_match'])
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">⏳ Experience Match</div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <div class="kpi-val">{exp_score}%</div>
                        <div style="font-size: 11px; color: #64748b; font-weight: 600;">20% Weight</div>
                    </div>
                    <div class="progress-bg" style="height: 5px; margin-top: 10px;">
                        <div class="progress-fill fill-primary" style="width: {exp_score}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with r2_col2:
                edu_score = int(analysis['sub_scores']['education_match'])
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">🎓 Education Match</div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <div class="kpi-val">{edu_score}%</div>
                        <div style="font-size: 11px; color: #64748b; font-weight: 600;">10% Weight</div>
                    </div>
                    <div class="progress-bg" style="height: 5px; margin-top: 10px;">
                        <div class="progress-fill fill-primary" style="width: {edu_score}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

        # Semantic Similarity Banner
        sem_score = analysis.get("semantic_score", 0.0)
        sem_color = "#34d399" if sem_score >= 80 else "#fbbf24" if sem_score >= 60 else "#f43f5e"
        st.markdown(f"""
        <div class="custom-card" style="border-left: 5px solid {sem_color} !important; padding: 16px 20px !important; margin-top: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 24px;">🧠</span>
                <div>
                    <div style="font-size: 11px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Semantic Context Alignment</div>
                    <div style="font-size: 18px; font-weight: 700; color: #ffffff;">Concept Overlap Score: <span style="color: {sem_color};">{sem_score}%</span></div>
                </div>
            </div>
            <div style="font-size: 13px; color: #94a3b8; max-width: 500px; line-height: 1.5; text-align: left;">
                Calculated via <b>sentence-transformers (all-MiniLM-L6-v2)</b>. Represents the dense conceptual and contextual alignment between your entire profile and the job description, mapping meanings rather than just text keywords.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Summary / Candidate Info Block
        c_left, c_right = st.columns([2, 3], gap="medium")
        with c_left:
            if resume_details:
                st.markdown(f"""
                <div class="custom-card accent-card-indigo" style="height: 100%;">
                    <div class="custom-title">👤 Candidate Profile Details</div>
                    <table class="profile-table">
                        <tr><td class="label">Name</td><td><b>{resume_details.get('candidate_name') or 'N/A'}</b></td></tr>
                        <tr><td class="label">Email</td><td>{resume_details.get('email') or 'N/A'}</td></tr>
                        <tr><td class="label">Phone</td><td>{resume_details.get('phone') or 'N/A'}</td></tr>
                        <tr><td class="label">Parsed Skills</td><td><span class="pill-badge badge-skill" style="margin:0;">{analysis.get('candidate_metrics', {}).get('parsed_skills_count', 0)} Skills</span></td></tr>
                        <tr><td class="label">Experience</td><td><b>{analysis.get('candidate_metrics', {}).get('parsed_experience_years', 0)} Years</b></td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
        with c_right:
            st.markdown(f"""
            <div class="custom-card accent-card-emerald" style="height: 100%;">
                <div class="custom-title">📝 Executive Summary & Insights</div>
                <p style="line-height: 1.7; color: #cbd5e1; font-size: 14px; margin-bottom: 12px;">
                    Your profile matches <b>{ats_val}%</b> of the qualifications for the <b>{analysis['job_title']}</b> position. 
                    The resume demonstrates strong alignment in <b>{analysis['sub_scores']['experience_match']}% experience matching</b> with {analysis.get('candidate_metrics', {}).get('parsed_experience_years', 0)} years of parsed experience versus {analysis.get('job_metrics', {}).get('required_experience_years', 0)} years required.
                </p>
                <p style="line-height: 1.7; color: #cbd5e1; font-size: 14px; margin-bottom: 15px;">
                    We identified a gap of <b>{len(analysis['missing_skills'])} critical skills</b> (such as <i>{', '.join(analysis['missing_skills'][:3]) if analysis['missing_skills'] else 'None'}</i>) that are highly valued in the job description. Bridging these key technical gaps will dramatically boost your match score and initial screening viability.
                </p>
                <div style="color: #38bdf8; font-size: 13.5px; font-weight: 600; display: flex; align-items: center; gap: 6px;">
                    <span>💡</span> <span>Recommended Next Step: Review your detailed Skill Gap Analysis to create a targeted learning roadmap.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 2: SKILL GAP ANALYSIS
    # ---------------------------------------------------------
    with tab_gap:
        st.markdown("<div style='margin-bottom: 20px;'><h3>⚖️ Technical Skill Gap Analysis</h3>"
                    "<p style='color: #94a3b8; font-size: 14px;'>Compare the technical stack detected in your resume against the requirements of the job description.</p></div>", unsafe_allow_html=True)
        
        # Skill comparison cards
        col_present, col_missing = st.columns(2, gap="medium")
        
        with col_present:
            st.markdown("<div class='custom-card accent-card-emerald' style='min-height: 230px; height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<div class='custom-title'><span>✅</span> Matching Skills (Present)</div>", unsafe_allow_html=True)
            present_skills = analysis.get("skills_present", [])
            if present_skills:
                st.markdown("".join([f'<span class="pill-badge badge-present">{s}</span>' for s in present_skills]), unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size: 14px; color: #94a3b8; font-style: italic;'>No matching skills detected between your resume and the job description.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_missing:
            st.markdown("<div class='custom-card accent-card-rose' style='min-height: 230px; height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<div class='custom-title'><span>❌</span> Missing Core Skills</div>", unsafe_allow_html=True)
            missing_skills = analysis.get("missing_skills", [])
            if missing_skills:
                st.markdown("".join([f'<span class="pill-badge badge-missing">{s}</span>' for s in missing_skills]), unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size: 14px; color: #34d399; font-weight: 500;'>🎉 Perfect Match! You possess all the skills required in the job description.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Category-wise Skill Progress Bars (WOW Feature - Replaces Empty Charts)
        st.markdown("<div style='margin-top: 30px; margin-bottom: 20px;'><h4>📊 Technical Domain Alignment</h4>"
                    "<p style='color: #94a3b8; font-size: 14px;'>A granular breakdown of your compatibility across core technical areas.</p></div>", unsafe_allow_html=True)
        
        # Categorize skills dynamically
        languages_list = ["python", "sql", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "html", "css", "c", "bash", "shell"]
        frameworks_list = ["react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring boot", "spring", "express", "next.js", "bootstrap", "tailwind"]
        db_cloud_list = ["postgresql", "mysql", "mongodb", "redis", "aws", "azure", "gcp", "docker", "kubernetes", "snowflake", "sqlite", "oracle", "git", "ci/cd", "terraform"]
        
        categories_data = {
            "Programming Languages": {
                "icon": "💻",
                "present": [s for s in present_skills if s.lower() in languages_list],
                "missing": [s for s in missing_skills if s.lower() in languages_list]
            },
            "Frameworks & Core Libraries": {
                "icon": "⚙️",
                "present": [s for s in present_skills if s.lower() in frameworks_list],
                "missing": [s for s in missing_skills if s.lower() in frameworks_list]
            },
            "Databases, Cloud & DevOps": {
                "icon": "☁️",
                "present": [s for s in present_skills if s.lower() in db_cloud_list],
                "missing": [s for s in missing_skills if s.lower() in db_cloud_list]
            },
            "Other Domain Skills": {
                "icon": "🧩",
                "present": [s for s in present_skills if s.lower() not in languages_list + frameworks_list + db_cloud_list],
                "missing": [s for s in missing_skills if s.lower() not in languages_list + frameworks_list + db_cloud_list]
            }
        }
        
        active_categories = {k: v for k, v in categories_data.items() if (len(v["present"]) + len(v["missing"])) > 0}
        
        if active_categories:
            cat_cols = st.columns(len(active_categories))
            for idx, (cat_name, cat_val) in enumerate(active_categories.items()):
                with cat_cols[idx]:
                    total_cat = len(cat_val["present"]) + len(cat_val["missing"])
                    matched_cat = len(cat_val["present"])
                    ratio = int((matched_cat / total_cat) * 100) if total_cat > 0 else 0
                    
                    progress_color_class = "fill-success" if ratio >= 75 else "fill-warning" if ratio >= 50 else "fill-danger"
                    
                    st.markdown(f"""
                    <div class="custom-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <div style="font-size: 20px; margin-bottom: 8px;">{cat_val["icon"]}</div>
                            <div style="font-size: 14px; font-weight: 700; color: #ffffff; line-height: 1.3; margin-bottom: 4px;">{cat_name}</div>
                            <div style="font-size: 12px; color: #94a3b8; margin-bottom: 12px;">{matched_cat} of {total_cat} skills matched ({ratio}%)</div>
                        </div>
                        <div>
                            <div class="progress-bg" style="height: 6px; margin-bottom: 15px;">
                                <div class="progress-fill {progress_color_class}" style="width: {ratio}%;"></div>
                            </div>
                            <div style="max-height: 100px; overflow-y: auto; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px; font-size: 11px;">
                    """, unsafe_allow_html=True)
                    
                    skills_list_html = ""
                    for s in cat_val["present"]:
                        skills_list_html += f'<span style="color: #34d399; margin-right: 8px; font-weight: 500;">✓ {s}</span> '
                    for s in cat_val["missing"]:
                        skills_list_html += f'<span style="color: #f87171; margin-right: 8px; font-weight: 500;">✗ {s}</span> '
                    st.markdown(skills_list_html + "</div></div></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="custom-card" style="text-align: center; padding: 30px;">
                <span style="font-size: 36px; display: block; margin-bottom: 15px;">🔍</span>
                <h5 style="color: #ffffff; margin-bottom: 10px;">No Technical Skills Extracted</h5>
                <p style="font-size: 14px; color: #94a3b8; max-width: 500px; margin: 0 auto;">
                    We couldn't extract any explicit technical skills from the job description. Please ensure the job description contains core technical requirements or keywords.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Recommended Learning Path
        st.markdown("<div style='margin-top: 30px; margin-bottom: 20px;'><h4>🗺️ AI-Recommended Skill Bridging Roadmap</h4>"
                    "<p style='color: #94a3b8; font-size: 14px;'>A curated educational learning path designed to close your identified skill gaps.</p></div>", unsafe_allow_html=True)
        
        learning_data = analysis.get("learning_path", {})
        if learning_data and any(learning_data.get(k) for k in ["beginner", "intermediate", "advanced"]):
            tab_beg, tab_int, tab_adv = st.tabs(["🌱 Beginner Phase", "🚀 Intermediate Phase", "🏆 Advanced Phase"])
            
            with tab_beg:
                beg_list = learning_data.get("beginner", [])
                if beg_list:
                    for idx, item in enumerate(beg_list):
                        st.markdown(f"""
                        <div class="custom-card accent-card-blue" style="margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="font-weight: 700; font-size: 15px; color: #38bdf8; font-family: 'Outfit', sans-serif;">Topic {idx+1}: {item.get('topic')}</div>
                                <span class="pill-badge badge-skill" style="margin: 0; font-size: 10px; padding: 3px 8px;">Beginner</span>
                            </div>
                            <p style="margin: 8px 0 12px 0; font-size: 13.5px; color: #cbd5e1; line-height: 1.5;">{item.get('description')}</p>
                            <div style="font-size: 12px; color: #94a3b8; background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.04);">
                                <b>📚 Recommended Resource:</b> {item.get('resource_suggestion')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #94a3b8; font-style: italic; padding: 10px 0;'>No beginner-level topics required for your learning roadmap.</div>", unsafe_allow_html=True)
                    
            with tab_int:
                int_list = learning_data.get("intermediate", [])
                if int_list:
                    for idx, item in enumerate(int_list):
                        st.markdown(f"""
                        <div class="custom-card accent-card-indigo" style="margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="font-weight: 700; font-size: 15px; color: #818cf8; font-family: 'Outfit', sans-serif;">Topic {idx+1}: {item.get('topic')}</div>
                                <span class="pill-badge badge-keyword" style="margin: 0; font-size: 10px; padding: 3px 8px;">Intermediate</span>
                            </div>
                            <p style="margin: 8px 0 12px 0; font-size: 13.5px; color: #cbd5e1; line-height: 1.5;">{item.get('description')}</p>
                            <div style="font-size: 12px; color: #94a3b8; background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.04);">
                                <b>🛠️ Recommended Hands-on Project:</b> {item.get('resource_suggestion')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #94a3b8; font-style: italic; padding: 10px 0;'>No intermediate-level topics required.</div>", unsafe_allow_html=True)
                    
            with tab_adv:
                adv_list = learning_data.get("advanced", [])
                if adv_list:
                    for idx, item in enumerate(adv_list):
                        st.markdown(f"""
                        <div class="custom-card accent-card-rose" style="margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="font-weight: 700; font-size: 15px; color: #fb7185; font-family: 'Outfit', sans-serif;">Topic {idx+1}: {item.get('topic')}</div>
                                <span class="pill-badge badge-missing" style="margin: 0; font-size: 10px; padding: 3px 8px;">Advanced</span>
                            </div>
                            <p style="margin: 8px 0 12px 0; font-size: 13.5px; color: #cbd5e1; line-height: 1.5;">{item.get('description')}</p>
                            <div style="font-size: 12px; color: #94a3b8; background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.04);">
                                <b>🚀 Advanced Architecture Challenge:</b> {item.get('resource_suggestion')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #94a3b8; font-style: italic; padding: 10px 0;'>No advanced-level topics required.</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="custom-card accent-card-emerald" style="padding: 20px; display: flex; align-items: center; gap: 15px;">
                <span style="font-size: 24px;">🏆</span>
                <div>
                    <div style="font-weight: 700; color: #ffffff;">Roadmap Complete!</div>
                    <div style="font-size: 13.5px; color: #94a3b8;">No technical skill gaps were detected. You have a highly compatible technical skill set for this position.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 3: INTERVIEW PREPARATION
    # ---------------------------------------------------------
    with tab_coach:
        st.markdown("<div style='margin-bottom: 20px;'><h3>🗣️ AI Interview Coach & Simulator</h3>"
                    "<p style='color: #94a3b8; font-size: 14px;'>Prepare with custom-generated interview questions and practice your answers in our real-time feedback sandbox.</p></div>", unsafe_allow_html=True)
        
        questions_data = analysis.get("interview_questions", {})
        tech_qs = questions_data.get("technical", [])
        beh_qs = questions_data.get("behavioral", [])
        
        col_qs_list, col_sandbox = st.columns([1, 1], gap="large")
        
        with col_qs_list:
            st.markdown("<div class='custom-card accent-card-blue' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<div class='custom-title'><span>🎯</span> Practice Q&A Catalog</div>", unsafe_allow_html=True)
            
            st.markdown("<p style='font-size: 13px; color: #94a3b8; margin-bottom: 15px;'>Tailored technical and behavioral questions based on your profile. Filter by difficulty to test your readiness.</p>", unsafe_allow_html=True)
            
            diff_filter = st.radio("Difficulty Filter:", ["All", "Easy", "Medium", "Hard"], horizontal=True)
            st.write("")
            
            # Technical Questions
            st.markdown("<h5 style='border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 5px; margin-bottom: 12px; color: #38bdf8 !important;'>💻 Technical Questions</h5>", unsafe_allow_html=True)
            visible_tech = 0
            for idx, q in enumerate(tech_qs):
                if diff_filter != "All" and q.get("difficulty") != diff_filter:
                    continue
                visible_tech += 1
                diff_val = q.get("difficulty", "Medium")
                diff_color = "#34d399" if diff_val == "Easy" else "#fbbf24" if diff_val == "Medium" else "#fb7185"
                diff_bg = "rgba(16, 185, 129, 0.1)" if diff_val == "Easy" else "rgba(251, 191, 36, 0.1)" if diff_val == "Medium" else "rgba(244, 63, 94, 0.1)"
                diff_border = "rgba(16, 185, 129, 0.2)" if diff_val == "Easy" else "rgba(251, 191, 36, 0.2)" if diff_val == "Medium" else "rgba(244, 63, 94, 0.2)"
                
                st.markdown(f"""
                <div style="background: rgba(30,41,59,0.25); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 12px; margin-top: 10px; margin-bottom: 4px;">
                    <div class="question-card-header">
                        <span class="pill-badge" style="background: {diff_bg}; color: {diff_color}; border: 1px solid {diff_border}; margin: 0; font-size: 10px; padding: 2px 8px;">{diff_val}</span>
                        <span class="pill-badge badge-skill" style="margin: 0; font-size: 10px; padding: 2px 8px;">{q.get('skill')}</span>
                    </div>
                    <div class="question-text" style="margin-top: 8px;">{q.get('question')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("💡 View Ideal Answer Guidance", expanded=False):
                    st.markdown(f"""
                    <p style='font-size: 13.5px; line-height: 1.6; color: #cbd5e1; margin: 0;'>
                        <b>Ideal Answer:</b><br>{q.get('ideal_answer')}
                    </p>
                    """, unsafe_allow_html=True)
                    
            if visible_tech == 0:
                st.markdown("<p style='font-size: 13px; color: #94a3b8; font-style: italic; padding: 5px 0;'>No technical questions match this difficulty filter.</p>", unsafe_allow_html=True)
                
            st.write("")
            
            # Behavioral Questions
            st.markdown("<h5 style='border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 5px; margin-bottom: 12px; color: #818cf8 !important;'>👥 Behavioral Questions</h5>", unsafe_allow_html=True)
            visible_beh = 0
            for idx, q in enumerate(beh_qs):
                if diff_filter != "All" and q.get("difficulty") != diff_filter:
                    continue
                visible_beh += 1
                diff_val = q.get("difficulty", "Medium")
                diff_color = "#34d399" if diff_val == "Easy" else "#fbbf24" if diff_val == "Medium" else "#fb7185"
                diff_bg = "rgba(16, 185, 129, 0.1)" if diff_val == "Easy" else "rgba(251, 191, 36, 0.1)" if diff_val == "Medium" else "rgba(244, 63, 94, 0.1)"
                diff_border = "rgba(16, 185, 129, 0.2)" if diff_val == "Easy" else "rgba(251, 191, 36, 0.2)" if diff_val == "Medium" else "rgba(244, 63, 94, 0.2)"
                
                st.markdown(f"""
                <div style="background: rgba(30,41,59,0.25); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 12px; margin-top: 10px; margin-bottom: 4px;">
                    <div class="question-card-header">
                        <span class="pill-badge" style="background: {diff_bg}; color: {diff_color}; border: 1px solid {diff_border}; margin: 0; font-size: 10px; padding: 2px 8px;">{diff_val}</span>
                        <span class="pill-badge badge-keyword" style="margin: 0; font-size: 10px; padding: 2px 8px;">{q.get('scenario')}</span>
                    </div>
                    <div class="question-text" style="margin-top: 8px;">{q.get('question')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("💡 View Ideal Answer (STAR Method)", expanded=False):
                    st.markdown(f"""
                    <p style='font-size: 13.5px; line-height: 1.6; color: #cbd5e1; margin: 0;'>
                        <b>Ideal Answer (STAR Framework):</b><br>{q.get('ideal_answer')}
                    </p>
                    """, unsafe_allow_html=True)
                    
            if visible_beh == 0:
                st.markdown("<p style='font-size: 13px; color: #94a3b8; font-style: italic; padding: 5px 0;'>No behavioral questions match this difficulty filter.</p>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
                
        with col_sandbox:
            st.markdown("<div class='custom-card accent-card-indigo' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<div class='custom-title'><span>💬</span> AI Practice Sandbox</div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 13px; color: #94a3b8; margin-bottom: 15px;'>Select an interview question, type your response, and click evaluate to receive constructive AI coaching and a performance score.</p>", unsafe_allow_html=True)
            
            all_questions = []
            for q in tech_qs:
                all_questions.append(f"Technical: {q.get('question')}")
            for q in beh_qs:
                all_questions.append(f"Behavioral: {q.get('question')}")
                
            if all_questions:
                selected_q = st.selectbox("Select question to practice:", all_questions, key="sandbox_q_select")
                user_answer = st.text_area("Your Response:", height=180, placeholder="Draft your response. Remember to use the STAR method for behavioral questions and quantify your achievements (e.g. 'optimized endpoints, reducing latency by 40%')...", key="sandbox_answer_area")
                
                if st.button("Evaluate My Answer", use_container_width=True, key="sandbox_evaluate_btn"):
                    if not user_answer.strip():
                        st.error("Please enter your answer first.")
                    else:
                        with st.spinner("Analyzing your response..."):
                            try:
                                if DIRECT_SERVICES_AVAILABLE:
                                    gem = GeminiService()
                                    feedback_html = gem.evaluate_answer(selected_q, user_answer)
                                    st.markdown(feedback_html, unsafe_allow_html=True)
                                else:
                                    # Fallback if direct services are not available but genai is
                                    if diag["gemini_ok"]:
                                        import google.generativeai as genai
                                        model = genai.GenerativeModel("gemini-1.5-flash")
                                        eval_prompt = f"""
                                        You are an expert technical hiring coach. Evaluate the candidate's response to the following interview question.
                                        Provide constructive, actionable feedback and score their response out of 100.
                                        
                                        Question:
                                        {selected_q}
                                        
                                        Candidate's Response:
                                        {user_answer}
                                        
                                        Format your response beautifully using HTML. Make it feel premium, using clean headings, lists, and a bold score.
                                        Do not use standard markdown formatting. Render as pure HTML that can be placed inside an st.markdown container.
                                        Wrap the response in a container with class 'custom-card accent-card-emerald'.
                                        Within the card, include:
                                        1. Estimated Score (formatted as a prominent stat callout e.g. <div style="font-size: 24px; font-weight: 700; color: #34d399; margin-bottom: 15px;">Estimated Score: XX/100</div>)
                                        2. Key Strengths (as a bulleted list)
                                        3. Key Weaknesses / Gaps (as a bulleted list)
                                        4. Actionable Suggestions for Improvement (as a bulleted list, giving concrete phrasing suggestions)
                                        
                                        Keep the tone professional, encouraging, and highly specific.
                                        """
                                        response = model.generate_content(eval_prompt)
                                        st.markdown(response.text, unsafe_allow_html=True)
                                    else:
                                        # Use the high-quality mock evaluation from the service logic as local fallback
                                        mock_feedback = f"""
                                        <div class="custom-card accent-card-emerald" style="margin-top: 20px;">
                                            <div class="custom-title">📝 AI Coaching Feedback (Demo Mode)</div>
                                            <div style="font-size: 24px; font-weight: 700; color: #34d399; margin-bottom: 15px;">Estimated Score: 78/100</div>
                                            <p style="color: #cbd5e1; font-size: 13.5px; margin-bottom: 10px;"><b>Question Practiced:</b> {selected_q}</p>
                                            <p style="color: #cbd5e1; font-size: 13.5px; margin-bottom: 15px;"><i>Note: You are running in Demo Mode. To get personalized AI feedback, configure a valid GEMINI_API_KEY.</i></p>
                                            <div style="margin-top: 15px;">
                                                <h4 style="color: #ffffff; font-size: 14px; margin-bottom: 8px;">💪 Key Strengths</h4>
                                                <ul style="color: #cbd5e1; font-size: 13px; padding-left: 20px; margin-bottom: 15px; line-height: 1.6;">
                                                    <li><b>Structure:</b> You structured your answer well, outlining the core technical details clearly.</li>
                                                    <li><b>Keywords:</b> You mentioned critical keywords matching the target stack.</li>
                                                </ul>
                                                <h4 style="color: #ffffff; font-size: 14px; margin-bottom: 8px;">⚠️ Key Gaps</h4>
                                                <ul style="color: #cbd5e1; font-size: 13px; padding-left: 20px; margin-bottom: 15px; line-height: 1.6;">
                                                    <li><b>Lack of Quantification:</b> You did not quantify the results or impact of your work.</li>
                                                    <li><b>STAR Method:</b> The action and result phases of your answer could be more distinct.</li>
                                                </ul>
                                                <h4 style="color: #ffffff; font-size: 14px; margin-bottom: 8px;">🚀 Suggestions for Improvement</h4>
                                                <ul style="color: #cbd5e1; font-size: 13px; padding-left: 20px; line-height: 1.6;">
                                                    <li>Instead of saying <i>'helped speed up database queries'</i>, try saying: <b>'optimized SQL indexes and refactored N+1 queries, reducing endpoint latency by 35%'</b>.</li>
                                                    <li>Make sure to explicitly state the business outcome or metric that resulted from your actions.</li>
                                                </ul>
                                            </div>
                                        </div>
                                        """
                                        st.markdown(mock_feedback, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Failed to analyze response: {e}")
            else:
                st.info("Run an analysis first to populate practice questions.")
                
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 4: IMPROVEMENT SUGGESTIONS
    # ---------------------------------------------------------
    with tab_suggest:
        st.markdown("<div style='margin-bottom: 20px;'><h3>💡 Resume Optimization & Suggestions</h3>"
                    "<p style='color: #94a3b8; font-size: 14px;'>Targeted fixes, keyword enhancements, and rewritten bullet points to maximize your resume compatibility.</p></div>", unsafe_allow_html=True)
        
        suggestions_data = analysis.get("suggestions", {})
        
        col_charts, col_text = st.columns([2, 3], gap="large")
        
        with col_charts:
            st.markdown("<div class='custom-card accent-card-blue'>", unsafe_allow_html=True)
            st.markdown("<div class='custom-title'>📊 ATS Dimension Analysis</div>", unsafe_allow_html=True)
            
            categories = ['Skill Match', 'Keyword Match', 'Experience Match', 'Education Match']
            scores = [
                analysis['sub_scores']['skill_match'],
                analysis['sub_scores']['keyword_match'],
                analysis['sub_scores']['experience_match'],
                analysis['sub_scores']['education_match']
            ]
            
            categories_radar = [*categories, categories[0]]
            scores_radar = [*scores, scores[0]]
            
            fig_radar = go.Figure(
                data=[
                    go.Scatterpolar(r=scores_radar, theta=categories_radar, fill='toself', name='Sub Scores', line_color='#6366f1', fillcolor='rgba(99,102,241,0.25)')
                ],
                layout=go.Layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.08)', tickfont=dict(color='#94a3b8')),
                        angularaxis=dict(gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.08)', tickfont=dict(color='#ffffff')),
                        bgcolor='rgba(0, 0, 0, 0)'
                    ),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff', family='Inter'),
                    height=280,
                    margin=dict(l=40, r=40, t=10, b=10)
                )
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='custom-card accent-card-indigo'>", unsafe_allow_html=True)
            st.markdown("<div class='custom-title'>⚙️ Technical Category Breakdown</div>", unsafe_allow_html=True)
            
            all_skills = (analysis.get("skills_present", []) + analysis.get("missing_skills", []))
            
            languages = ["python", "sql", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "html", "css"]
            frameworks = ["react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring boot"]
            db_cloud = ["postgresql", "mysql", "mongodb", "redis", "aws", "azure", "gcp", "docker", "kubernetes", "snowflake", "sqlite"]
            
            lang_count = sum(1 for s in all_skills if s.lower() in languages)
            frame_count = sum(1 for s in all_skills if s.lower() in frameworks)
            db_count = sum(1 for s in all_skills if s.lower() in db_cloud)
            other_count = max(0, len(all_skills) - (lang_count + frame_count + db_count))
            
            pie_labels = ['Languages', 'Frameworks/APIs', 'Cloud/DevOps', 'Other Tech']
            pie_values = [lang_count, frame_count, db_count, other_count]
            
            if sum(pie_values) > 0:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=pie_labels, 
                    values=pie_values,
                    hole=.4,
                    marker_colors=['#38bdf8', '#818cf8', '#10b981', '#64748b'],
                    textfont=dict(color='#ffffff')
                )])
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8', family='Inter'),
                    height=240,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.markdown("<p style='font-size: 13.5px; color: #94a3b8; font-style: italic; text-align: center;'>No technical skills to display category breakdown.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
                
        with col_text:
            st.markdown("<div class='custom-card accent-card-blue' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<div class='custom-title'>💡 Actionable Resume Optimizations</div>", unsafe_allow_html=True)
            
            # 1. Weak Sections
            st.markdown("<h5 style='margin-bottom: 12px; color: #fb7185 !important; display: flex; align-items: center; gap: 6px;'>⚠️ Weak Sections Identified</h5>", unsafe_allow_html=True)
            weak_sections = suggestions_data.get("weak_sections", [])
            if weak_sections:
                for ws in weak_sections:
                    st.markdown(f"""
                    <div style="background-color: rgba(244, 63, 94, 0.04); border-left: 4px solid #f43f5e; padding: 14px; border-radius: 8px; margin-bottom: 12px; border-top: 1px solid rgba(255,255,255,0.02); border-right: 1px solid rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.02);">
                        <div style="font-weight: 700; color: #ffffff; font-size: 14px; font-family: 'Outfit', sans-serif;">Section: {ws.get('section')}</div>
                        <div style="font-size: 13px; margin: 6px 0; color: #cbd5e1;"><b>Issue:</b> {ws.get('issue')}</div>
                        <div style="font-size: 13px; color: #34d399; font-weight: 600;"><b>Fix:</b> {ws.get('improvement')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size: 13px; color: #34d399; font-weight: 500; margin-bottom: 15px;'>🎉 No weak sections identified! Your resume structure is highly effective.</p>", unsafe_allow_html=True)
                
            st.write("")
            
            # 2. Action Verbs
            st.markdown("<h5 style='margin-bottom: 12px; color: #38bdf8 !important; display: flex; align-items: center; gap: 6px;'>⚡ Stronger Action Verbs</h5>", unsafe_allow_html=True)
            action_verbs = suggestions_data.get("action_verbs", [])
            if action_verbs:
                for av in action_verbs:
                    st.markdown(f"""
                    <div style="background-color: rgba(56, 189, 248, 0.04); border-left: 4px solid #38bdf8; padding: 12px; border-radius: 8px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; gap: 15px; border-top: 1px solid rgba(255,255,255,0.02); border-right: 1px solid rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.02);">
                        <div style="flex-shrink: 0;">
                            <span style="text-decoration: line-through; color: #64748b; font-size: 12px; font-weight: 500; margin-right: 8px;">{av.get('original')}</span>
                            <span style="font-weight: 700; color: #38bdf8; font-size: 14px;">➡️ {av.get('suggested')}</span>
                        </div>
                        <div style="font-size: 12.5px; color: #94a3b8; text-align: right; line-height: 1.4;">
                            {av.get('context')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size: 13px; color: #94a3b8; font-style: italic; margin-bottom: 15px;'>No action verb changes recommended.</p>", unsafe_allow_html=True)
                
            st.write("")
                
            # 3. Bullet Point Enhancements (X-Y-Z formula)
            st.markdown("<h5 style='margin-bottom: 12px; color: #818cf8 !important; display: flex; align-items: center; gap: 6px;'>✍️ Bullet Point Re-writes (Google X-Y-Z Formula)</h5>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 12px; color: #94a3b8; margin-bottom: 12px;'>The X-Y-Z Formula: <b>Accomplished [X] as measured by [Y], by doing [Z]</b>. This frames your achievements with high impact.</p>", unsafe_allow_html=True)
            
            bullet_points = suggestions_data.get("bullet_points", [])
            if bullet_points:
                for bp in bullet_points:
                    st.markdown(f"""
                    <div class="rewrite-card">
                        <div class="original-box">Original: "{bp.get('original')}"</div>
                        <div class="improved-box">Optimized: "{bp.get('improved')}"</div>
                        <div class="rationale-box"><b>Why it works:</b> {bp.get('rationale')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size: 13px; color: #94a3b8; font-style: italic; margin-bottom: 15px;'>No bullet point rewrites recommended.</p>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

        # PDF/Markdown Report Export Button
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
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
            label="📥 Download Polished Resume Analysis Report (Markdown)",
            data=report_md,
            file_name=f"resume_coaching_report_{analysis['job_title'].replace(' ', '_').lower()}.md",
            mime="text/markdown",
            use_container_width=True,
            key="export_report_btn"
        )
