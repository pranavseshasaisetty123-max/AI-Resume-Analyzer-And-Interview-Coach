import os
import json
import shutil
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from src.database.session import get_db
from src.models.models import Resume, Skill, Analysis
from src.api import schemas
from src.services.parser_service import ParserService
from src.services.ats_service import ATSService
from src.services.semantic_service import SemanticService
from src.services.gemini_service import GeminiService
from src.utils.config import settings, logger

router = APIRouter()

# Instantiate Services
parser_service = ParserService()
ats_service = ATSService()
semantic_service = SemanticService()
gemini_service = GeminiService()

@router.post("/upload-resume", response_model=schemas.ResumeResponse)
def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Uploads a PDF resume, extracts and parses its text using PyMuPDF and Gemini,
    stores it in the database with its skills, and returns the structured profile.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    # Save the file to data/uploads
    filename = f"{int(datetime.utcnow().timestamp())}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded resume to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    # Parse resume
    try:
        parsed_data = parser_service.parse_resume(file_path)
    except Exception as e:
        logger.error(f"Resume parsing error: {e}")
        # Clean up file on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to parse resume: {str(e)}")

    # Create Resume DB model
    db_resume = Resume(
        file_name=file.filename,
        candidate_name=parsed_data.get("candidate_name"),
        email=parsed_data.get("email"),
        phone=parsed_data.get("phone"),
        education=json.dumps(parsed_data.get("education")),
        experience=json.dumps(parsed_data.get("experience")),
        raw_text=parsed_data.get("raw_text")
    )
    
    # Process Skills (many-to-many relationship)
    parsed_skills = parsed_data.get("skills", [])
    for skill_name in parsed_skills:
        skill_name_clean = skill_name.strip().lower()
        if not skill_name_clean:
            continue
        # Find or create skill
        db_skill = db.query(Skill).filter(Skill.skill_name == skill_name_clean).first()
        if not db_skill:
            db_skill = Skill(skill_name=skill_name_clean)
            db.add(db_skill)
            db.commit()
            db.refresh(db_skill)
        db_resume.skills.append(db_skill)
        
    db.add(db_resume)
    db.commit()
    db.refresh(db_resume)
    
    logger.info(f"Successfully saved Resume ID {db_resume.resume_id} in database.")
    
    # Map to response schema
    return schemas.ResumeResponse(
        resume_id=db_resume.resume_id,
        file_name=db_resume.file_name,
        candidate_name=db_resume.candidate_name,
        email=db_resume.email,
        phone=db_resume.phone,
        skills=[s.skill_name for s in db_resume.skills],
        education=parsed_data.get("education", []),
        experience=parsed_data.get("experience", []),
        upload_date=db_resume.upload_date
    )

@router.post("/analyze", response_model=schemas.AnalysisResponse)
def analyze_resume(request: schemas.AnalysisRequest, db: Session = Depends(get_db)):
    """
    Performs complete ATS and semantic analysis of a stored resume against a job description.
    Generates learning paths, suggestions, and interview questions. Stores and returns results.
    """
    # 1. Retrieve the resume from DB
    db_resume = db.query(Resume).filter(Resume.resume_id == request.resume_id).first()
    if not db_resume:
        raise HTTPException(status_code=404, detail="Resume not found.")
        
    # Reconstruct resume data for ATS service
    resume_data = {
        "skills": [s.skill_name for s in db_resume.skills],
        "raw_text": db_resume.raw_text,
        "education": json.loads(db_resume.education) if db_resume.education else [],
        "experience": json.loads(db_resume.experience) if db_resume.experience else []
    }
    
    try:
        # 2. Run ATS Calculation (Skill, Keyword, Experience, Education Match)
        ats_result = ats_service.calculate_ats_score(resume_data, request.job_description)
        
        # 3. Run Semantic Similarity (Sentence Transformers)
        semantic_score = semantic_service.calculate_similarity(db_resume.raw_text, request.job_description)
        
        # Convert semantic similarity to percentage (0.0 - 1.0 -> 0 - 100)
        semantic_percentage = round(semantic_score * 100.0, 1)
        
        # 4. Generate Suggestions (Gemini)
        suggestions = gemini_service.generate_suggestions(db_resume.raw_text, request.job_description)
        
        # 5. Generate Learning Path for Missing Skills (Gemini)
        learning_path = gemini_service.generate_learning_path(ats_result["missing_skills"])
        
        # 6. Generate Interview Questions (Gemini)
        interview_questions = gemini_service.generate_interview_questions(resume_data["skills"])
        
    except Exception as e:
        logger.error(f"Error executing resume analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis execution failed: {str(e)}")

    # Create Analysis DB model
    db_analysis = Analysis(
        resume_id=db_resume.resume_id,
        job_title=request.job_title,
        job_description=request.job_description,
        ats_score=ats_result["ats_score"],
        skill_match_score=ats_result["sub_scores"]["skill_match"],
        keyword_match_score=ats_result["sub_scores"]["keyword_match"],
        experience_match_score=ats_result["sub_scores"]["experience_match"],
        education_match_score=ats_result["sub_scores"]["education_match"],
        semantic_score=semantic_percentage,
        skills_present=json.dumps(ats_result["skills_present"]),
        missing_skills=json.dumps(ats_result["missing_skills"]),
        learning_path=json.dumps(learning_path),
        suggestions=json.dumps(suggestions),
        interview_questions=json.dumps(interview_questions),
        candidate_metrics=json.dumps(ats_result.get("candidate_metrics", {})),
        job_metrics=json.dumps(ats_result.get("job_metrics", {}))
    )
    
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    
    logger.info(f"Successfully saved Analysis ID {db_analysis.analysis_id} in database.")
    
    # Return formatted response
    return schemas.AnalysisResponse(
        analysis_id=db_analysis.analysis_id,
        resume_id=db_analysis.resume_id,
        job_title=db_analysis.job_title,
        job_description=db_analysis.job_description,
        ats_score=db_analysis.ats_score,
        sub_scores=schemas.SubScoresSchema(
            skill_match=db_analysis.skill_match_score,
            keyword_match=db_analysis.keyword_match_score,
            experience_match=db_analysis.experience_match_score,
            education_match=db_analysis.education_match_score
        ),
        semantic_score=db_analysis.semantic_score,
        skills_present=ats_result["skills_present"],
        missing_skills=ats_result["missing_skills"],
        learning_path=learning_path,
        suggestions=suggestions,
        interview_questions=interview_questions,
        candidate_metrics=ats_result.get("candidate_metrics", {}),
        job_metrics=ats_result.get("job_metrics", {}),
        created_at=db_analysis.created_at
    )

@router.post("/generate-interview-questions", response_model=schemas.QuestionResponse)
def generate_custom_interview_questions(request: schemas.QuestionRequest):
    """
    Directly generates customized technical and behavioral interview questions based on provided skills.
    """
    try:
        questions = gemini_service.generate_interview_questions(request.skills)
        return schemas.QuestionResponse(
            technical=questions.get("technical", []),
            behavioral=questions.get("behavioral", [])
        )
    except Exception as e:
        logger.error(f"Custom question generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {str(e)}")

@router.get("/analysis/{id}", response_model=schemas.AnalysisResponse)
def get_analysis(id: int, db: Session = Depends(get_db)):
    """
    Retrieves a completed analysis record by its ID.
    """
    db_analysis = db.query(Analysis).filter(Analysis.analysis_id == id).first()
    if not db_analysis:
        raise HTTPException(status_code=404, detail="Analysis record not found.")
        
    # Unpack JSON strings
    skills_present = json.loads(db_analysis.skills_present) if db_analysis.skills_present else []
    missing_skills = json.loads(db_analysis.missing_skills) if db_analysis.missing_skills else []
    learning_path = json.loads(db_analysis.learning_path) if db_analysis.learning_path else {}
    suggestions = json.loads(db_analysis.suggestions) if db_analysis.suggestions else {}
    interview_questions = json.loads(db_analysis.interview_questions) if db_analysis.interview_questions else {}
    candidate_metrics = json.loads(db_analysis.candidate_metrics) if db_analysis.candidate_metrics else {}
    job_metrics = json.loads(db_analysis.job_metrics) if db_analysis.job_metrics else {}
    
    return schemas.AnalysisResponse(
        analysis_id=db_analysis.analysis_id,
        resume_id=db_analysis.resume_id,
        job_title=db_analysis.job_title,
        job_description=db_analysis.job_description,
        ats_score=db_analysis.ats_score,
        sub_scores=schemas.SubScoresSchema(
            skill_match=db_analysis.skill_match_score,
            keyword_match=db_analysis.keyword_match_score,
            experience_match=db_analysis.experience_match_score,
            education_match=db_analysis.education_match_score
        ),
        semantic_score=db_analysis.semantic_score,
        skills_present=skills_present,
        missing_skills=missing_skills,
        learning_path=learning_path,
        suggestions=suggestions,
        interview_questions=interview_questions,
        candidate_metrics=candidate_metrics,
        job_metrics=job_metrics,
        created_at=db_analysis.created_at
    )

@router.get("/history", response_model=schemas.HistoryResponse)
def get_history(db: Session = Depends(get_db)):
    """
    Retrieves histories of all uploaded resumes and completed analyses, sorted by most recent.
    """
    resumes = db.query(Resume).order_by(Resume.upload_date.desc()).all()
    analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).all()
    
    resume_summaries = [
        schemas.ResumeSummary(
            resume_id=r.resume_id,
            file_name=r.file_name,
            candidate_name=r.candidate_name,
            upload_date=r.upload_date
        ) for r in resumes
    ]
    
    analysis_summaries = [
        schemas.AnalysisSummary(
            analysis_id=a.analysis_id,
            resume_id=a.resume_id,
            job_title=a.job_title,
            ats_score=a.ats_score,
            semantic_score=a.semantic_score,
            created_at=a.created_at
        ) for a in analyses
    ]
    
    return schemas.HistoryResponse(
        resumes=resume_summaries,
        analyses=analysis_summaries
    )
