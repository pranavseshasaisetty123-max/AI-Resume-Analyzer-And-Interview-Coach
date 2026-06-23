import re
import google.generativeai as genai
from typing import Dict, Any, List, Tuple
from src.utils.config import settings, logger
from src.services.parser_service import COMMON_SKILLS_DB

class ATSService:
    def __init__(self):
        self.api_enabled = bool(settings.GEMINI_API_KEY)
        if self.api_enabled:
            genai.configure(api_key=settings.GEMINI_API_KEY)

    def calculate_ats_score(self, resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """
        Calculates the complete ATS compatibility score and sub-scores.
        Weights: Skill Match (40%), Keyword Match (30%), Experience Match (20%), Education Match (10%)
        """
        resume_skills = [s.lower() for s in resume_data.get("skills", [])]
        resume_text = resume_data.get("raw_text", "")
        
        # 1. Extract Requirements from Job Description (Skills, Experience, Education)
        jd_requirements = self._extract_jd_requirements(job_description)
        jd_skills = [s.lower() for s in jd_requirements.get("skills", [])]
        required_experience = jd_requirements.get("required_experience_years", 0)
        required_education = jd_requirements.get("required_education_level", "bachelor")

        # 2. Calculate Skill Match Score (40%)
        skill_score, skills_present, missing_skills = self._calculate_skill_score(resume_skills, jd_skills)
        
        # 3. Calculate Keyword Match Score (30%)
        keyword_score, keywords_found = self._calculate_keyword_score(resume_text, job_description)
        
        # 4. Calculate Experience Match Score (20%)
        candidate_experience = self._parse_candidate_experience_years(resume_data)
        exp_score = self._calculate_experience_score(candidate_experience, required_experience)
        
        # 5. Calculate Education Match Score (10%)
        candidate_education_level = self._parse_candidate_education_level(resume_data)
        edu_score = self._calculate_education_score(candidate_education_level, required_education)
        
        # 6. Compute Weighted Overall ATS Score
        overall_ats_score = (
            (skill_score * 0.40) +
            (keyword_score * 0.30) +
            (exp_score * 0.20) +
            (edu_score * 0.10)
        )
        
        overall_ats_score = round(overall_ats_score, 1)
        
        logger.info(
            f"ATS Calculation Complete. Overall: {overall_ats_score}/100. "
            f"Skill: {skill_score}%, Keyword: {keyword_score}%, Exp: {exp_score}%, Edu: {edu_score}%"
        )
        
        return {
            "ats_score": overall_ats_score,
            "sub_scores": {
                "skill_match": round(skill_score, 1),
                "keyword_match": round(keyword_score, 1),
                "experience_match": round(exp_score, 1),
                "education_match": round(edu_score, 1)
            },
            "skills_present": skills_present,
            "missing_skills": missing_skills,
            "candidate_metrics": {
                "parsed_skills_count": len(resume_skills),
                "parsed_experience_years": candidate_experience,
                "parsed_education_level": candidate_education_level
            },
            "job_metrics": {
                "required_experience_years": required_experience,
                "required_education_level": required_education,
                "required_skills_count": len(jd_skills)
            }
        }

    def _extract_jd_requirements(self, job_description: str) -> Dict[str, Any]:
        """Extracts skills, years of experience, and degree level required by the Job Description."""
        if self.api_enabled:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"""
                Analyze the following job description and extract the candidate requirements.
                
                Return ONLY a JSON object matching this schema:
                {{
                    "skills": ["List of technical or professional skills required (lowercase, clean, e.g., 'python')"],
                    "required_experience_years": 3,  // Integer representing the minimum years of experience required (default to 0 if not specified)
                    "required_education_level": "bachelor" // Must be one of: "none", "high_school", "associate", "bachelor", "master", "phd"
                }}

                Do NOT include markdown formatting. Output ONLY the JSON string.
                
                Job Description:
                ---
                {job_description}
                ---
                """
                response = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                import json
                return json.loads(response.text.strip())
            except Exception as e:
                logger.error(f"Gemini JD requirements extraction failed: {e}. Using local regex extraction.")
                
        # Fallback local parser for JD
        return self._extract_jd_requirements_local(job_description)

    def _extract_jd_requirements_local(self, jd: str) -> Dict[str, Any]:
        jd_lower = jd.lower()
        
        # Extract skills via keyword search
        skills = []
        for skill in COMMON_SKILLS_DB:
            escaped_skill = re.escape(skill)
            if re.search(r'\b' + escaped_skill + r'\b', jd_lower):
                skills.append(skill)
            elif skill in ["c++", "c#", "node.js", "ci/cd"]:
                if skill in jd_lower:
                    skills.append(skill)
                    
        # Extract experience years (e.g. "3+ years", "5 years", "experience of 2 years")
        exp_match = re.search(r'(\d+)\s*(?:\+|to\s*\d+)?\s*years?', jd_lower)
        required_experience = int(exp_match.group(1)) if exp_match else 0
        
        # Extract education level
        required_education = "bachelor"  # Default
        if "phd" in jd_lower or "ph.d" in jd_lower or "doctorate" in jd_lower:
            required_education = "phd"
        elif "master" in jd_lower or "m.s" in jd_lower or "mtech" in jd_lower:
            required_education = "master"
        elif "bachelor" in jd_lower or "b.s" in jd_lower or "btech" in jd_lower or "degree" in jd_lower:
            required_education = "bachelor"
        elif "associate" in jd_lower:
            required_education = "associate"
            
        return {
            "skills": skills,
            "required_experience_years": required_experience,
            "required_education_level": required_education
        }

    def _calculate_skill_score(self, resume_skills: List[str], jd_skills: List[str]) -> Tuple[float, List[str], List[str]]:
        """Compares resume skills with JD skills. Returns (score, present_skills, missing_skills)."""
        if not jd_skills:
            # If the JD has no detected skills, give a default high score
            return 100.0, resume_skills, []
            
        resume_skills_set = set(resume_skills)
        jd_skills_set = set(jd_skills)
        
        present_skills = list(jd_skills_set.intersection(resume_skills_set))
        missing_skills = list(jd_skills_set.difference(resume_skills_set))
        
        score = (len(present_skills) / len(jd_skills_set)) * 100.0
        return score, present_skills, missing_skills

    def _calculate_keyword_score(self, resume_text: str, job_description: str) -> Tuple[float, List[str]]:
        """Counts presence of important keywords from job description in the resume."""
        if not resume_text or not job_description:
            return 0.0, []
            
        # Tokenize JD to find important nouns/keywords (excluding stopwords)
        stopwords = {
            "and", "the", "for", "with", "from", "that", "this", "our", "your", "will", "have",
            "required", "preferred", "duties", "responsibilities", "requirements", "experience",
            "skills", "knowledge", "ability", "candidate", "role", "position", "team", "work"
        }
        
        # Simple extraction of capitalized words and words matching tech regex
        words_jd = re.findall(r'\b[a-zA-Z]{3,15}\b', job_description)
        words_jd_filtered = [w.lower() for w in words_jd if w.lower() not in stopwords]
        
        # Get unique words and count frequency in JD
        from collections import Counter
        word_counts = Counter(words_jd_filtered)
        
        # Get the top 15 most frequent words as key keywords
        top_keywords = [item[0] for item in word_counts.most_common(15)]
        
        if not top_keywords:
            return 100.0, []
            
        resume_text_lower = resume_text.lower()
        found_keywords = [kw for kw in top_keywords if kw in resume_text_lower]
        
        score = (len(found_keywords) / len(top_keywords)) * 100.0
        return score, found_keywords

    def _parse_candidate_experience_years(self, resume_data: Dict[str, Any]) -> float:
        """Parses and calculates the total years of experience from experience bullet points."""
        exp_list = resume_data.get("experience", [])
        if not exp_list:
            return 0.0
            
        total_years = 0.0
        for exp in exp_list:
            years_str = str(exp.get("years", "")).lower()
            # Look for patterns like "2 years", "1.5 years", "2020 - 2022", "2019 to present"
            # Pattern 1: Simple digit + years
            digit_match = re.search(r'(\d+(?:\.\d+)?)\s*years?', years_str)
            if digit_match:
                total_years += float(digit_match.group(1))
                continue
                
            # Pattern 2: Date range (e.g. 2020 - 2022)
            year_range = re.findall(r'\b(20\d{2}|19\d{2})\b', years_str)
            if len(year_range) == 2:
                y1, y2 = int(year_range[0]), int(year_range[1])
                total_years += abs(y2 - y1)
            elif len(year_range) == 1 and "present" in years_str:
                from datetime import datetime
                y1 = int(year_range[0])
                y2 = datetime.now().year
                total_years += max(0, y2 - y1)
                
        # If no years parsed, check descriptions for numbers or default to a modest heuristic
        if total_years == 0.0:
            # Let's count how many experience blocks exist. Each block represents approx 1.5 years.
            total_years = len(exp_list) * 1.5
            
        return round(total_years, 1)

    def _parse_candidate_education_level(self, resume_data: Dict[str, Any]) -> str:
        """Determines the candidate's highest level of education from the resume data."""
        edu_list = resume_data.get("education", [])
        if not edu_list:
            return "none"
            
        edu_str = " ".join([str(edu.get("degree", "")) for edu in edu_list]).lower()
        edu_str += " " + " ".join([str(edu.get("institution", "")) for edu in edu_list]).lower()
        
        if "phd" in edu_str or "ph.d" in edu_str or "doctor" in edu_str:
            return "phd"
        elif "master" in edu_str or "m.s" in edu_str or "mtech" in edu_str or "mba" in edu_str:
            return "master"
        elif "bachelor" in edu_str or "b.s" in edu_str or "btech" in edu_str or "undergraduate" in edu_str:
            return "bachelor"
        elif "associate" in edu_str or "diploma" in edu_str:
            return "associate"
        elif "school" in edu_str:
            return "high_school"
            
        return "bachelor"  # Default assumption for a tech candidate with parsed edu

    def _calculate_experience_score(self, candidate_years: float, required_years: int) -> float:
        """Calculates experience match score."""
        if required_years == 0:
            return 100.0
            
        if candidate_years >= required_years:
            return 100.0
            
        # Partial credit
        return (candidate_years / required_years) * 100.0

    def _calculate_education_score(self, candidate_level: str, required_level: str) -> float:
        """Calculates education match score based on mapped ranks."""
        levels = {
            "none": 0,
            "high_school": 1,
            "associate": 2,
            "bachelor": 3,
            "master": 4,
            "phd": 5
        }
        
        cand_rank = levels.get(candidate_level, 3)
        req_rank = levels.get(required_level, 3)
        
        if cand_rank >= req_rank:
            return 100.0
            
        # If candidate has lower education, deduct 15 points per level difference
        diff = req_rank - cand_rank
        score = 100.0 - (diff * 15.0)
        return max(0.0, score)
