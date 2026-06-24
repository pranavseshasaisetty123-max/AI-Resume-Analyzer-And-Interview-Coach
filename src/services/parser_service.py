import os
import re
import json
import fitz  # PyMuPDF
import google.generativeai as genai
from typing import Dict, Any, List
from src.utils.config import settings, logger

# Common technical skills for local fallback parser
COMMON_SKILLS_DB = [
    "python", "sql", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask",
    "fastapi", "spring boot", "postgresql", "mysql", "mongodb", "redis", "cassandra",
    "sqlite", "docker", "kubernetes", "aws", "azure", "gcp", "git", "github", "ci/cd",
    "jenkins", "terraform", "ansible", "linux", "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "keras", "spark", "hadoop", "airflow", "snowflake",
    "tableau", "power bi", "jira", "confluence", "graphql", "rest api", "grpc",
    "microservices", "agile", "scrum"
]

class ParserService:
    def __init__(self):
        key = settings.GEMINI_API_KEY
        self.api_enabled = bool(key and not key.startswith("your_") and len(key) > 8)
        if self.api_enabled:
            logger.info("ParserService initialized and ready using globally configured API key.")
        else:
            logger.warning("GEMINI_API_KEY not found. Resume parser will use local rule-based heuristics.")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts all text from a PDF file using PyMuPDF."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
            logger.info(f"Successfully extracted {len(text)} characters from {pdf_path}")
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")
        
        return text

    def parse_resume(self, pdf_path: str) -> Dict[str, Any]:
        """Extracts text from PDF and parses it into structured sections."""
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        if not raw_text.strip():
            logger.warning(f"Extracted text from {pdf_path} is empty.")
            return self._get_empty_parsed_structure()
            
        if self.api_enabled:
            try:
                return self._parse_with_gemini(raw_text)
            except Exception as e:
                logger.error(f"Gemini resume parsing failed: {e}. Falling back to local heuristics.")
                return self._parse_with_heuristics(raw_text)
        else:
            return self._parse_with_heuristics(raw_text)

    def _parse_with_gemini(self, text: str) -> Dict[str, Any]:
        """Uses Gemini API to parse resume text into a structured JSON format."""
        # We use gemini-1.5-flash for fast, structured parsing
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        You are an expert ATS (Applicant Tracking System) parser. Analyze the following resume raw text and extract the candidate's structured information.
        
        Your output MUST be a valid JSON object matching this schema:
        {{
            "candidate_name": "Candidate's full name (or null if not found)",
            "email": "Candidate's email address (or null if not found)",
            "phone": "Candidate's phone number (or null if not found)",
            "skills": ["List of technical and soft skills identified in the resume (lowercase, clean, e.g. 'python', 'sql')"],
            "education": [
                {{
                    "degree": "Degree name (e.g., Bachelor of Science in Computer Science)",
                    "institution": "University/School name",
                    "year": "Graduation year or date range (e.g., 2024 or 2020-2024)"
                }}
            ],
            "experience": [
                {{
                    "job_title": "Job title/Role (e.g., Software Engineer)",
                    "company": "Company name",
                    "years": "Employment duration (e.g., 2022 - Present or 2 years)",
                    "description": "Short summary or bullet points of responsibilities and achievements"
                }}
            ]
        }}

        Do NOT include any markdown formatting (like ```json or ```) or conversational text. Output ONLY the JSON string.
        
        Resume text:
        ---
        {text}
        ---
        """
        
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        response_text = response.text.strip()
        parsed_data = json.loads(response_text)
        
        # Add raw text to the return dictionary for downstream services (like semantic similarity)
        parsed_data["raw_text"] = text
        logger.info(f"Successfully parsed resume with Gemini. Candidate: {parsed_data.get('candidate_name')}")
        return parsed_data

    def _parse_with_heuristics(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based parser using regex and keywords."""
        logger.info("Running local heuristic parser.")
        
        # 1. Email extraction
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        email = emails[0] if emails else None
        
        # 2. Phone extraction
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone_matches = re.finditer(phone_pattern, text)
        phone = None
        for match in phone_matches:
            val = match.group().strip()
            # Basic validation: must contain at least 7 digits
            if sum(c.isdigit() for c in val) >= 7:
                phone = val
                break

        # 3. Name extraction (first non-empty line that doesn't look like contact info)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        candidate_name = None
        for line in lines[:5]:
            if "@" not in line and not any(char.isdigit() for char in line) and len(line.split()) >= 2:
                candidate_name = line
                break
        if not candidate_name and lines:
            candidate_name = lines[0]

        # 4. Skill extraction (keyword matching against a predefined set)
        skills_found = []
        text_lower = text.lower()
        for skill in COMMON_SKILLS_DB:
            # Word boundary matching for skills to avoid false matches (e.g. 'c' in 'cat')
            # For skills with special characters, handle carefully
            escaped_skill = re.escape(skill)
            if re.search(r'\b' + escaped_skill + r'\b', text_lower):
                skills_found.append(skill)
            elif skill in ["c++", "c#", "node.js", "ci/cd", "rest api"]:
                if skill in text_lower:
                    skills_found.append(skill)
        
        # 5. Education & Experience splitting (Heuristic based on sections)
        education_list = []
        experience_list = []
        
        # Very simple section-based extraction
        edu_keywords = ["education", "academic", "university", "college", "degree"]
        exp_keywords = ["experience", "work", "employment", "history", "professional experience"]
        
        current_section = None
        edu_text_blocks = []
        exp_text_blocks = []
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in exp_keywords) and len(line) < 30:
                current_section = "experience"
                continue
            elif any(keyword in line_lower for keyword in edu_keywords) and len(line) < 30:
                current_section = "education"
                continue
            elif len(line) < 30 and ("skills" in line_lower or "projects" in line_lower or "summary" in line_lower):
                current_section = "other"
                continue
                
            if current_section == "education":
                edu_text_blocks.append(line)
            elif current_section == "experience":
                exp_text_blocks.append(line)

        # Build basic education objects
        if edu_text_blocks:
            education_list.append({
                "degree": "Degree / Coursework",
                "institution": edu_text_blocks[0] if len(edu_text_blocks) > 0 else "University",
                "year": edu_text_blocks[1] if len(edu_text_blocks) > 1 else "N/A"
            })
        else:
            education_list.append({
                "degree": "Not specified",
                "institution": "Not specified",
                "year": "N/A"
            })

        # Build basic experience objects
        if exp_text_blocks:
            experience_list.append({
                "job_title": "Professional Experience",
                "company": "Company",
                "years": "N/A",
                "description": "\n".join(exp_text_blocks[:5])
            })
        else:
            experience_list.append({
                "job_title": "Not specified",
                "company": "Not specified",
                "years": "N/A",
                "description": "Not specified"
            })

        return {
            "candidate_name": candidate_name,
            "email": email,
            "phone": phone,
            "skills": skills_found,
            "education": education_list,
            "experience": experience_list,
            "raw_text": text
        }

    def _get_empty_parsed_structure(self) -> Dict[str, Any]:
        return {
            "candidate_name": None,
            "email": None,
            "phone": None,
            "skills": [],
            "education": [],
            "experience": [],
            "raw_text": ""
        }
