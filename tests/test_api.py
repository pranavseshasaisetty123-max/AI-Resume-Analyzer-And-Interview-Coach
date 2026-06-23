import os
import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup database override for tests
from src.database.session import Base, get_db
from src.api.main import app
from src.models.models import Resume, Skill, Analysis
from src.services.parser_service import ParserService
from src.services.ats_service import ATSService
from src.services.semantic_service import SemanticService
from src.services.gemini_service import GeminiService

# Temporary test database
from sqlalchemy.pool import StaticPool
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override get_db dependency
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def setup_db():
    """Initializes and drops the schema for each test to keep tests isolated."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client():
    return TestClient(app)

def test_read_root(client):
    """Sanity check: endpoint root."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_upload_resume_api(client):
    """Tests the POST /upload-resume endpoint using a mocked parser service."""
    mock_parsed_data = {
        "candidate_name": "Test Candidate",
        "email": "test@candidate.com",
        "phone": "555-0199",
        "skills": ["python", "fastapi"],
        "education": [{"degree": "B.S. Computer Science", "institution": "State University", "year": "2022"}],
        "experience": [{"job_title": "Intern", "company": "Software Corp", "years": "6 months", "description": "Wrote code"}],
        "raw_text": "Resume content"
    }

    # Mock the parser service inside the router
    with pytest.MonkeyPatch.context() as mp:
        # Patch ParserService.parse_resume
        mp.setattr(ParserService, "parse_resume", lambda self, path: mock_parsed_data)
        # Patch file saving to avoid writing to disk
        mp.setattr("shutil.copyfileobj", lambda f_src, f_dst: None)
        
        # Create a dummy PDF payload
        dummy_file = ("resume.pdf", b"%PDF-1.4...", "application/pdf")
        response = client.post(
            "/api/upload-resume",
            files={"file": dummy_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["candidate_name"] == "Test Candidate"
        assert data["email"] == "test@candidate.com"
        assert "python" in data["skills"]
        assert "fastapi" in data["skills"]
        assert data["resume_id"] is not None

def test_analyze_api(client):
    """Tests the POST /analyze endpoint with mocked ATS, Semantic, and Gemini services."""
    # 1. Create a dummy Resume in our temporary DB
    db = TestingSessionLocal()
    db_resume = Resume(
        file_name="resume.pdf",
        candidate_name="John Doe",
        email="john@doe.com",
        phone="555-5555",
        education=json.dumps([{"degree": "B.S.", "institution": "College", "year": "2020"}]),
        experience=json.dumps([{"job_title": "Developer", "company": "Co", "years": "2 years", "description": "SQL"}]),
        raw_text="John Doe. Developer. SQL."
    )
    # Add skills
    skill1 = Skill(skill_name="python")
    skill2 = Skill(skill_name="sql")
    db.add_all([skill1, skill2])
    db.commit()
    db_resume.skills.extend([skill1, skill2])
    db.add(db_resume)
    db.commit()
    db.refresh(db_resume)
    resume_id = db_resume.resume_id
    db.close()

    # 2. Setup mock return values for services
    mock_ats_result = {
        "ats_score": 85.0,
        "sub_scores": {
            "skill_match": 80.0,
            "keyword_match": 85.0,
            "experience_match": 90.0,
            "education_match": 100.0
        },
        "skills_present": ["python", "sql"],
        "missing_skills": ["docker"]
    }
    
    mock_suggestions = {
        "weak_sections": [],
        "action_verbs": [],
        "bullet_points": [],
        "missing_keywords": []
    }
    
    mock_learning_path = {
        "beginner": [],
        "intermediate": [],
        "advanced": []
    }
    
    mock_interview_qs = {
        "technical": [],
        "behavioral": []
    }

    # 3. Patch the services and call /analyze
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ATSService, "calculate_ats_score", lambda self, data, jd: mock_ats_result)
        mp.setattr(SemanticService, "calculate_similarity", lambda self, text, jd: 0.92) # 92%
        mp.setattr(GeminiService, "generate_suggestions", lambda self, text, jd: mock_suggestions)
        mp.setattr(GeminiService, "generate_learning_path", lambda self, missing: mock_learning_path)
        mp.setattr(GeminiService, "generate_interview_questions", lambda self, skills: mock_interview_qs)
        
        payload = {
            "resume_id": resume_id,
            "job_title": "Backend Engineer",
            "job_description": "Requires Python, SQL, Docker."
        }
        
        response = client.post("/api/analyze", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["ats_score"] == 85.0
        assert data["semantic_score"] == 92.0 # 0.92 * 100
        assert data["resume_id"] == resume_id
        assert data["job_title"] == "Backend Engineer"
        assert "python" in data["skills_present"]
        assert "docker" in data["missing_skills"]
        assert data["analysis_id"] is not None

def test_history_and_get_analysis_api(client):
    """Verifies history endpoint returns items and a specific analysis can be fetched."""
    db = TestingSessionLocal()
    # Add a resume and an analysis
    r = Resume(file_name="res.pdf", candidate_name="Jane")
    db.add(r)
    db.commit()
    db.refresh(r)
    
    a = Analysis(
        resume_id=r.resume_id,
        job_title="Frontend Engineer",
        job_description="React",
        ats_score=90.0,
        skill_match_score=90.0,
        keyword_match_score=90.0,
        experience_match_score=90.0,
        education_match_score=90.0,
        semantic_score=90.0
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    analysis_id = a.analysis_id
    db.close()
    
    # Test GET /history
    response_history = client.get("/api/history")
    assert response_history.status_code == 200
    history_data = response_history.json()
    assert len(history_data["resumes"]) == 1
    assert len(history_data["analyses"]) == 1
    assert history_data["resumes"][0]["candidate_name"] == "Jane"
    assert history_data["analyses"][0]["job_title"] == "Frontend Engineer"
    
    # Test GET /analysis/{id}
    response_get = client.get(f"/api/analysis/{analysis_id}")
    assert response_get.status_code == 200
    get_data = response_get.json()
    assert get_data["job_title"] == "Frontend Engineer"
    assert get_data["ats_score"] == 90.0

def test_analysis_with_missing_metrics(client):
    """Regression test: Verifies that GET /analysis/{id} handles missing/None candidate_metrics gracefully."""
    db = TestingSessionLocal()
    # Create a resume and an analysis with no metrics columns (simulating legacy data)
    r = Resume(file_name="legacy.pdf", candidate_name="Legacy Candidate")
    db.add(r)
    db.commit()
    db.refresh(r)
    
    a = Analysis(
        resume_id=r.resume_id,
        job_title="Legacy Engineer",
        job_description="Legacy description",
        ats_score=75.0,
        skill_match_score=70.0,
        keyword_match_score=70.0,
        experience_match_score=80.0,
        education_match_score=80.0,
        semantic_score=75.0,
        candidate_metrics=None,  # Explicitly None
        job_metrics=None         # Explicitly None
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    analysis_id = a.analysis_id
    db.close()
    
    response = client.get(f"/api/analysis/{analysis_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_title"] == "Legacy Engineer"
    assert data["candidate_metrics"] == {}  # Handled gracefully, returns empty dict
    assert data["job_metrics"] == {}       # Handled gracefully, returns empty dict

