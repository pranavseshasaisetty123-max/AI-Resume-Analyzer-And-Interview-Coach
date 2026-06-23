from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, ForeignKey, Table
from sqlalchemy.orm import relationship
from src.database.session import Base

# Many-to-many association table for Resume and Skill
resume_skills = Table(
    'resume_skills',
    Base.metadata,
    Column('resume_id', Integer, ForeignKey('resumes.resume_id', ondelete='CASCADE'), primary_key=True),
    Column('skill_id', Integer, ForeignKey('skills.skill_id', ondelete='CASCADE'), primary_key=True)
)

class Resume(Base):
    __tablename__ = "resumes"
    
    resume_id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String(255), nullable=False)
    candidate_name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    education = Column(Text, nullable=True)  # JSON-serialized text
    experience = Column(Text, nullable=True) # JSON-serialized text
    raw_text = Column(Text, nullable=True)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    skills = relationship("Skill", secondary=resume_skills, back_populates="resumes")
    analyses = relationship("Analysis", back_populates="resume", cascade="all, delete-orphan")

class Skill(Base):
    __tablename__ = "skills"
    
    skill_id = Column(Integer, primary_key=True, autoincrement=True)
    skill_name = Column(String(100), unique=True, index=True, nullable=False)
    
    # Relationships
    resumes = relationship("Resume", secondary=resume_skills, back_populates="skills")

class Analysis(Base):
    __tablename__ = "analyses"
    
    analysis_id = Column(Integer, primary_key=True, autoincrement=True)
    resume_id = Column(Integer, ForeignKey('resumes.resume_id', ondelete='CASCADE'), nullable=False)
    job_title = Column(String(255), nullable=False)
    job_description = Column(Text, nullable=False)
    
    # Scores
    ats_score = Column(Float, nullable=False)
    skill_match_score = Column(Float, nullable=False)
    keyword_match_score = Column(Float, nullable=False)
    experience_match_score = Column(Float, nullable=False)
    education_match_score = Column(Float, nullable=False)
    semantic_score = Column(Float, nullable=False)
    
    # Detailed Analysis Data (JSON-serialized strings)
    skills_present = Column(Text, nullable=True)
    missing_skills = Column(Text, nullable=True)
    learning_path = Column(Text, nullable=True)
    suggestions = Column(Text, nullable=True)
    interview_questions = Column(Text, nullable=True)
    candidate_metrics = Column(Text, nullable=True)
    job_metrics = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    resume = relationship("Resume", back_populates="analyses")
