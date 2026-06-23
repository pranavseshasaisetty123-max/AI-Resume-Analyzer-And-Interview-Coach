import pytest
from unittest.mock import patch, MagicMock
from src.services.ats_service import ATSService

@pytest.fixture
def ats_service():
    with patch('src.services.ats_service.settings') as mock_settings:
        mock_settings.GEMINI_API_KEY = ""
        service = ATSService()
        yield service

def test_calculate_skill_score(ats_service):
    resume_skills = ["python", "sql", "docker"]
    jd_skills = ["python", "sql", "fastapi", "kubernetes"]
    
    score, present, missing = ats_service._calculate_skill_score(resume_skills, jd_skills)
    
    # 2 matching skills (python, sql) out of 4 required = 50%
    assert score == 50.0
    assert set(present) == {"python", "sql"}
    assert set(missing) == {"fastapi", "kubernetes"}

def test_calculate_keyword_score(ats_service):
    resume_text = "Experienced software engineer working with python, fastapi and docker."
    jd_text = "Looking for a python developer who knows fastapi and docker."
    
    score, found = ats_service._calculate_keyword_score(resume_text, jd_text)
    
    assert score > 0.0
    assert "python" in found
    assert "fastapi" in found

def test_calculate_experience_score(ats_service):
    # Case 1: Candidate has more experience than required
    score1 = ats_service._calculate_experience_score(candidate_years=5.0, required_years=3)
    assert score1 == 100.0
    
    # Case 2: Candidate has less experience
    score2 = ats_service._calculate_experience_score(candidate_years=1.5, required_years=3)
    assert score2 == 50.0

def test_calculate_education_score(ats_service):
    # Case 1: Matching level
    score1 = ats_service._calculate_education_score(candidate_level="bachelor", required_level="bachelor")
    assert score1 == 100.0
    
    # Case 2: Higher level
    score2 = ats_service._calculate_education_score(candidate_level="master", required_level="bachelor")
    assert score2 == 100.0
    
    # Case 3: Lower level
    score3 = ats_service._calculate_education_score(candidate_level="high_school", required_level="bachelor")
    # Bachelor is rank 3, High School is rank 1. Difference is 2. 100 - (2 * 15) = 70.
    assert score3 == 70.0

def test_overall_ats_calculation(ats_service):
    resume_data = {
        "skills": ["python", "sql"],
        "raw_text": "Python SQL Developer. 3 years experience. Bachelor of Science.",
        "experience": [{"years": "3 years"}],
        "education": [{"degree": "Bachelor of Science"}]
    }
    
    jd = "Required: Python, SQL, Docker. Experience: 3+ years. Education: Bachelor's."
    
    # Mock JD extraction to make test deterministic
    with patch.object(ats_service, '_extract_jd_requirements', return_value={
        "skills": ["python", "sql", "docker"],
        "required_experience_years": 3,
        "required_education_level": "bachelor"
    }):
        result = ats_service.calculate_ats_score(resume_data, jd)
        
        # Verify ATS structure
        assert "ats_score" in result
        assert "sub_scores" in result
        assert result["sub_scores"]["skill_match"] == pytest.approx(66.6, 0.1) # 2 out of 3
        assert result["sub_scores"]["experience_match"] == 100.0 # 3 out of 3
        assert result["sub_scores"]["education_match"] == 100.0 # Bachelor matches Bachelor
        
        # Skill Match weight is 40%, Keyword is 30%, Exp is 20%, Edu is 10%
        # Let's ensure overall score is correctly computed
        sub = result["sub_scores"]
        expected_score = (
            (sub["skill_match"] * 0.4) + 
            (sub["keyword_match"] * 0.3) + 
            (sub["experience_match"] * 0.2) + 
            (sub["education_match"] * 0.1)
        )
        assert result["ats_score"] == round(expected_score, 1)
