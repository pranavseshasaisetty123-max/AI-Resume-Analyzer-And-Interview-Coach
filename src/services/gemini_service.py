import json
import google.generativeai as genai
from typing import Dict, Any, List
from src.utils.config import settings, logger

class GeminiService:
    def __init__(self):
        # Resolve api_enabled by checking if the key is valid (not mock, and set)
        key = settings.GEMINI_API_KEY
        self.api_enabled = bool(key and not key.startswith("your_") and len(key) > 8)
        if self.api_enabled:
            logger.info("Gemini Service initialized and ready using globally configured API key.")
        else:
            logger.warning("GEMINI_API_KEY not configured or in Demo Mode. GeminiService will use local template-based fallbacks.")

    def generate_suggestions(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Generates structured suggestions to improve the resume against a job description."""
        if self.api_enabled:
            try:
                return self._generate_suggestions_with_gemini(resume_text, job_description)
            except Exception as e:
                logger.error(f"Failed to generate suggestions with Gemini: {e}. Using mock suggestions.")
                return self._get_mock_suggestions()
        else:
            return self._get_mock_suggestions()

    def generate_interview_questions(self, skills: List[str]) -> Dict[str, Any]:
        """Generates technical and behavioral interview questions categorized by difficulty."""
        if self.api_enabled:
            try:
                return self._generate_questions_with_gemini(skills)
            except Exception as e:
                logger.error(f"Failed to generate questions with Gemini: {e}. Using mock questions.")
                return self._get_mock_questions(skills)
        else:
            return self._get_mock_questions(skills)

    def generate_learning_path(self, missing_skills: List[str]) -> Dict[str, Any]:
        """Generates a learning path (Beginner, Intermediate, Advanced) for missing skills."""
        if self.api_enabled:
            try:
                return self._generate_learning_path_with_gemini(missing_skills)
            except Exception as e:
                logger.error(f"Failed to generate learning path with Gemini: {e}. Using mock learning path.")
                return self._get_mock_learning_path(missing_skills)
        else:
            return self._get_mock_learning_path(missing_skills)

    # --- Gemini Implementations ---

    def _generate_suggestions_with_gemini(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        You are an elite career coach and resume reviewer. Compare the resume with the job description and generate specific, highly actionable improvement suggestions.
        
        Return ONLY a JSON object matching this schema:
        {{
            "weak_sections": [
                {{
                    "section": "Section Name (e.g., Projects, Experience, Summary)",
                    "issue": "Specific weakness identified in that section",
                    "improvement": "Actionable way to fix it"
                }}
            ],
            "action_verbs": [
                {{
                    "original": "Weak/Passive verb used (or phrase like 'responsible for')",
                    "suggested": "Strong, impact-driven action verb replacement",
                    "context": "Brief context on where or how to use it"
                }}
            ],
            "bullet_points": [
                {{
                    "original": "An example of a weak/average bullet point from the resume",
                    "improved": "An optimized version of that bullet point following the X-Y-Z formula (Accomplished [X] as measured by [Y], by doing [Z])",
                    "rationale": "Why this change makes the bullet point stronger"
                }}
            ],
            "missing_keywords": [
                {{
                    "keyword": "Important keyword from job description",
                    "category": "Category (e.g., Technology, Methodology, Soft Skill)",
                    "importance": "High/Medium"
                }}
            ]
        }}

        Do NOT include markdown formatting. Output ONLY the JSON string.

        Job Description:
        ---
        {job_description}
        ---

        Resume:
        ---
        {resume_text}
        ---
        """
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text.strip())

    def _generate_questions_with_gemini(self, skills: List[str]) -> Dict[str, Any]:
        model = genai.GenerativeModel("gemini-1.5-flash")
        skills_str = ", ".join(skills) if skills else "General Software Engineering"
        
        prompt = f"""
        You are a hiring manager interviewing a candidate for a technical role. Based on the candidate's skills, generate highly relevant interview questions.
        Generate both technical questions (focused on the specific skills: {skills_str}) and behavioral questions (scenarios like leadership, challenge, conflict, etc.).
        
        For each category, generate exactly:
        - 2 Easy questions
        - 2 Medium questions
        - 2 Hard questions
        
        Return ONLY a JSON object matching this schema:
        {{
            "technical": [
                {{
                    "skill": "Specific skill this question targets",
                    "question": "The interview question",
                    "difficulty": "Easy, Medium, or Hard",
                    "ideal_answer": "A concise, high-impact sample response that would impress an interviewer"
                }}
            ],
            "behavioral": [
                {{
                    "scenario": "The type of scenario (e.g., Conflict Resolution, Problem Solving, Leadership)",
                    "question": "The interview question",
                    "difficulty": "Easy, Medium, or Hard",
                    "ideal_answer": "A structured response following the STAR method (Situation, Task, Action, Result)"
                }}
            ]
        }}

        Do NOT include markdown formatting. Output ONLY the JSON string.
        """
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text.strip())

    def _generate_learning_path_with_gemini(self, missing_skills: List[str]) -> Dict[str, Any]:
        model = genai.GenerativeModel("gemini-1.5-flash")
        missing_skills_str = ", ".join(missing_skills) if missing_skills else "General technical skills"
        
        prompt = f"""
        You are a technical educator. Design a structured learning path for a candidate to acquire the following missing skills: {missing_skills_str}.
        Organize the learning path into three sequential phases: Beginner, Intermediate, and Advanced.
        
        Return ONLY a JSON object matching this schema:
        {{
            "beginner": [
                {{
                    "topic": "Fundamental topic to learn",
                    "description": "What this topic covers and why it is important",
                    "resource_suggestion": "Recommended resource (e.g., specific course names, documentation, or tutorial types)"
                }}
            ],
            "intermediate": [
                {{
                    "topic": "Intermediate topic/tooling",
                    "description": "How to apply this skill in projects or workflows",
                    "resource_suggestion": "Recommended intermediate hands-on project or guide"
                }}
            ],
            "advanced": [
                {{
                    "topic": "Advanced concepts (e.g., scaling, system design, optimization)",
                    "description": "Enterprise-grade practices or advanced architecture patterns",
                    "resource_suggestion": "Recommended advanced resource or system design challenge"
                }}
            ]
        }}

        Do NOT include markdown formatting. Output ONLY the JSON string.
        """
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text.strip())

    # --- Local Mock Fallbacks ---

    def _get_mock_suggestions(self) -> Dict[str, Any]:
        return {
            "weak_sections": [
                {
                    "section": "Professional Summary",
                    "issue": "The summary is overly generic and lacks quantifiable achievements.",
                    "improvement": "Rewrite the summary to highlight specific technical skills, years of experience, and a key metric demonstrating impact (e.g., 'Reduced API latency by 30%')."
                },
                {
                    "section": "Projects",
                    "issue": "Project descriptions focus heavily on tools used rather than personal contributions and business results.",
                    "improvement": "Use the X-Y-Z formula to rewrite bullet points, explaining what you built, how you measured success, and the specific technology stack used."
                }
            ],
            "action_verbs": [
                {
                    "original": "Responsible for developing",
                    "suggested": "Engineered",
                    "context": "Use when describing complex software systems or services you built from scratch."
                },
                {
                    "original": "Worked on a team to build",
                    "suggested": "Collaborated",
                    "context": "Show active participation and teamwork rather than passive involvement."
                },
                {
                    "original": "Helped with performance",
                    "suggested": "Optimized",
                    "context": "Emphasize improvement and efficiency gains in latency, memory, or database queries."
                }
            ],
            "bullet_points": [
                {
                    "original": "Built backend services in Python for the main company application.",
                    "improved": "Designed and deployed 5+ scalable microservices using FastAPI and PostgreSQL, reducing response latency by 40% and supporting 10k+ daily active users.",
                    "rationale": "Quantifies the scale, names specific technologies, and explicitly measures the performance enhancement."
                },
                {
                    "original": "Responsible for writing SQL queries and managing database tables.",
                    "improved": "Optimized legacy PostgreSQL queries and indexes, improving read performance by 25% and reducing database CPU utilization under peak load.",
                    "rationale": "Turns a passive responsibility statement into an active, result-oriented accomplishment."
                }
            ],
            "missing_keywords": [
                {
                    "keyword": "Docker",
                    "category": "Technology",
                    "importance": "High"
                },
                {
                    "keyword": "CI/CD",
                    "category": "Methodology",
                    "importance": "Medium"
                },
                {
                    "keyword": "System Design",
                    "category": "Methodology",
                    "importance": "High"
                }
            ]
        }

    def _get_mock_questions(self, skills: List[str]) -> Dict[str, Any]:
        tech_qs = []
        
        # Determine active skills to customize questions
        active_skills = [s.lower() for s in skills] if skills else ["general"]
        
        # Python questions
        if any(s in active_skills for s in ["python", "general"]):
            tech_qs.extend([
                {
                    "skill": "Python",
                    "question": "What is the difference between a list and a tuple in Python, and when would you use each?",
                    "difficulty": "Easy",
                    "ideal_answer": "Lists are mutable, meaning their elements can be modified, added, or removed. Tuples are immutable, meaning their state cannot be changed once created. Use lists for dynamic collections of homogeneous items, and tuples for fixed-size collections of heterogeneous elements, or as keys in dictionaries due to their immutability (hashability)."
                },
                {
                    "skill": "Python",
                    "question": "Explain the Global Interpreter Lock (GIL) in Python. How does it affect multi-threaded programs, and how do you bypass it?",
                    "difficulty": "Medium",
                    "ideal_answer": "The GIL is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecodes at once. This makes multi-threaded Python programs CPU-bound on a single core. To bypass the GIL for CPU-bound tasks, you can use the multiprocessing module (which spawns separate processes), run code in C extensions, or use alternative implementations like Jython or PyPy. For I/O-bound tasks, multi-threading or asyncio is still highly effective."
                }
            ])
            
        # SQL questions
        if any(s in active_skills for s in ["sql", "postgresql", "mysql", "general"]):
            tech_qs.extend([
                {
                    "skill": "SQL",
                    "question": "What are the differences between INNER JOIN, LEFT JOIN, and RIGHT JOIN?",
                    "difficulty": "Easy",
                    "ideal_answer": "INNER JOIN returns only the rows that have matching values in both tables. LEFT JOIN returns all rows from the left table, and the matched rows from the right table (filling with NULLs if there is no match). RIGHT JOIN is the inverse, returning all rows from the right table and matching rows from the left."
                },
                {
                    "skill": "SQL",
                    "question": "Explain database indexing. How does it speed up queries, and what are the trade-offs of having too many indexes?",
                    "difficulty": "Medium",
                    "ideal_answer": "Indexes are data structures (typically B-Trees) that allow the database engine to find rows quickly without performing a full table scan. The main trade-off is that indexes occupy extra disk space and slow down write operations (INSERT, UPDATE, DELETE) because the index must be updated every time data is modified."
                }
            ])
            
        # Cloud/Docker/DevOps questions
        if any(s in active_skills for s in ["docker", "kubernetes", "aws", "gcp", "azure", "devops", "general"]):
            tech_qs.extend([
                {
                    "skill": "Docker",
                    "question": "What is the difference between a Docker image and a Docker container?",
                    "difficulty": "Easy",
                    "ideal_answer": "A Docker image is a read-only template that contains the application code, libraries, and dependencies. A Docker container is a runnable instance of an image, which runs in isolation and has its own writeable layer."
                },
                {
                    "skill": "Docker",
                    "question": "How do you minimize the size of a Docker image for production deployments?",
                    "difficulty": "Hard",
                    "ideal_answer": "To minimize image size, you can: 1) Use multi-stage builds to keep build-time dependencies out of the final image. 2) Start from a minimal base image, like Alpine Linux or distroless. 3) Minimize the number of layers by combining RUN commands. 4) Use .dockerignore to avoid copying unnecessary files (like node_modules or local cache)."
                }
            ])

        # Fill remaining slots if needed
        while len(tech_qs) < 6:
            tech_qs.append({
                "skill": "System Design",
                "question": "How would you design a rate limiter for a public API?",
                "difficulty": "Hard",
                "ideal_answer": "A rate limiter can be implemented using algorithms like Token Bucket or Leaky Bucket. You can store rate limit counters in Redis (using key-value pairs with TTL and atomic operations like INCR) for high-speed, distributed access. The rate limiter can sit at the API Gateway level to reject requests (HTTP 429 Too Many Requests) before hitting downstream microservices."
            })
            
        # Behavioral questions
        behavioral_qs = [
            {
                "scenario": "Tell Me About Yourself",
                "question": "Tell me about yourself and your background.",
                "difficulty": "Easy",
                "ideal_answer": "Use the Present-Past-Future formula. Present: 'I am currently a software developer focusing on building backend services...' Past: 'Before this, I studied computer science and completed a project where I...' Future: 'I am excited about this role because I want to leverage my skills in FastAPI and NLP to...' Keep it under 2 minutes and focus on relevant professional highlights."
            },
            {
                "scenario": "Problem Solving",
                "question": "Describe a challenging technical problem you faced and how you resolved it.",
                "difficulty": "Medium",
                "ideal_answer": "STAR Method. Situation: 'In my last project, we noticed API latency spiked to 3 seconds during peak hours.' Task: 'I was tasked with identifying and fixing the bottleneck.' Action: 'I used APM tools to profile queries, found a missing database index on the orders table, and refactored a nested N+1 query loop into a single optimized join.' Result: 'This reduced response latency by 80% to 300ms, and CPU utilization dropped by 30%.'"
            },
            {
                "scenario": "Conflict Resolution",
                "question": "Tell me about a time you had a disagreement with a team member. How did you handle it?",
                "difficulty": "Hard",
                "ideal_answer": "STAR Method. Focus on collaboration, active listening, and objective decision-making. Situation: 'During a project, a teammate wanted to use MongoDB while I advocated for PostgreSQL.' Task: 'We needed to agree on the database architecture to meet our deadline.' Action: 'I scheduled a brief meeting. I listened to their points about MongoDB's flexible schema. Then, I presented our relational requirements, constraints, and how PostgreSQL's ACID compliance was critical for our transactional data. We decided to prototype both in a 1-day spike.' Result: 'The spike proved PostgreSQL was the optimal choice. We built it successfully, and my teammate appreciated the objective, collaborative approach to resolving the dispute.'"
            }
        ]
        
        # Ensure we return a structured output matching the format
        return {
            "technical": tech_qs[:6],
            "behavioral": behavioral_qs
        }

    def _get_mock_learning_path(self, missing_skills: List[str]) -> Dict[str, Any]:
        skills = missing_skills if missing_skills else ["Docker", "Airflow", "System Design"]
        
        beginner_topics = []
        intermediate_topics = []
        advanced_topics = []
        
        for skill in skills:
            beginner_topics.append({
                "topic": f"Foundations of {skill}",
                "description": f"Learn the fundamental concepts, terminology, and core syntax of {skill}.",
                "resource_suggestion": f"Official documentation getting started guide or introductory interactive course on YouTube/Udemy."
            })
            intermediate_topics.append({
                "topic": f"Hands-on {skill} Integration",
                "description": f"Build a practical mini-project integrating {skill} with a backend web server or pipeline.",
                "resource_suggestion": f"Build a personal GitHub project using {skill} to showcase in your portfolio."
            })
            advanced_topics.append({
                "topic": f"Advanced {skill} Architecture & Scaling",
                "description": f"Understand production-grade deployment, security best practices, and performance tuning for {skill}.",
                "resource_suggestion": f"Read advanced engineering blogs (e.g., Netflix, Uber tech blogs) explaining how they scale {skill} in production."
            })

        return {
            "beginner": beginner_topics,
            "intermediate": intermediate_topics,
            "advanced": advanced_topics
        }

    def evaluate_answer(self, question: str, answer: str) -> str:
        """
        Evaluates a candidate's answer to an interview question using Gemini.
        Returns formatted HTML with score, strengths, weaknesses, and suggestions.
        """
        if self.api_enabled:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                eval_prompt = f"""
                You are an expert technical hiring coach. Evaluate the candidate's response to the following interview question.
                Provide constructive, actionable feedback and score their response out of 100.
                
                Question:
                {question}
                
                Candidate's Response:
                {answer}
                
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
                return response.text
            except Exception as e:
                logger.error(f"Failed to evaluate answer with Gemini: {e}. Falling back to simulated feedback.")
                return self._get_mock_evaluation(question, answer)
        else:
            return self._get_mock_evaluation(question, answer)

    def _get_mock_evaluation(self, question: str, answer: str) -> str:
        """Generates high-quality mock/simulated coaching feedback in HTML format for Demo Mode."""
        return f"""
        <div class="custom-card accent-card-emerald" style="margin-top: 20px;">
            <div class="custom-title">📝 AI Coaching Feedback (Demo Mode)</div>
            <div style="font-size: 24px; font-weight: 700; color: #34d399; margin-bottom: 15px;">Estimated Score: 78/100</div>
            <p style="color: #cbd5e1; font-size: 13.5px; margin-bottom: 10px;"><b>Question Practiced:</b> {question}</p>
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
