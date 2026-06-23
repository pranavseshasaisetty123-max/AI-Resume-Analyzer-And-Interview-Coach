from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- Resume parsing schemas ---

class EducationSchema(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None

class ExperienceSchema(BaseModel):
    job_title: Optional[str] = None
    company: Optional[str] = None
    years: Optional[str] = None
    description: Optional[str] = None

class ResumeResponse(BaseModel):
    resume_id: int
    file_name: str
    candidate_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = []
    education: List[EducationSchema] = []
    experience: List[ExperienceSchema] = []
    upload_date: datetime

    class Config:
        from_attributes = True

# --- Analysis schemas ---

class AnalysisRequest(BaseModel):
    resume_id: int
    job_title: str
    job_description: str

class SubScoresSchema(BaseModel):
    skill_match: float
    keyword_match: float
    experience_match: float
    education_match: float

class AnalysisResponse(BaseModel):
    analysis_id: int
    resume_id: int
    job_title: str
    job_description: str
    ats_score: float
    sub_scores: SubScoresSchema
    semantic_score: float
    skills_present: List[str] = []
    missing_skills: List[str] = []
    learning_path: Dict[str, Any] = {}
    suggestions: Dict[str, Any] = {}
    interview_questions: Dict[str, Any] = {}
    candidate_metrics: Dict[str, Any] = {}
    job_metrics: Dict[str, Any] = {}
    created_at: datetime

    class Config:
        from_attributes = True

# --- Question request schema ---

class QuestionRequest(BaseModel):
    skills: List[str]

class QuestionResponse(BaseModel):
    technical: List[Dict[str, Any]] = []
    behavioral: List[Dict[str, Any]] = []

# --- History schemas ---

class ResumeSummary(BaseModel):
    resume_id: int
    file_name: str
    candidate_name: Optional[str] = None
    upload_date: datetime

class AnalysisSummary(BaseModel):
    analysis_id: int
    resume_id: int
    job_title: str
    ats_score: float
    semantic_score: float
    created_at: datetime

class HistoryResponse(BaseModel):
    resumes: List[ResumeSummary]
    analyses: List[AnalysisSummary]
