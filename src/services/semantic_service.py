import math
from collections import Counter
import google.generativeai as genai
from src.utils.config import settings, logger

class SemanticService:
    def __init__(self):
        self.model = None
        self.is_transformer_loaded = False
        
        # Try to pre-load sentence-transformers
        self._load_transformer_model_lazy()

    def _load_transformer_model_lazy(self):
        """Attempts to load sentence-transformers. Does not raise exceptions on failure."""
        if self.is_transformer_loaded:
            return True
            
        try:
            logger.info("Attempting to load sentence-transformers and all-MiniLM-L6-v2...")
            from sentence_transformers import SentenceTransformer
            # This will download the model if not present, which takes 80-100MB
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.is_transformer_loaded = True
            logger.info("Sentence-transformers model 'all-MiniLM-L6-v2' loaded successfully.")
            return True
        except Exception as e:
            logger.warning(
                f"Could not load sentence-transformers: {e}. "
                "Will use Gemini or TF-IDF fallback for semantic matching."
            )
            self.model = None
            self.is_transformer_loaded = False
            return False

    def calculate_similarity(self, resume_text: str, job_description: str) -> float:
        """
        Calculates the semantic similarity score (0.0 to 1.0) between the resume and job description.
        Tries Sentence-Transformers first, then Gemini API, and finally a pure-Python TF-IDF fallback.
        """
        if not resume_text.strip() or not job_description.strip():
            return 0.0

        # Try sentence-transformers
        if self._load_transformer_model_lazy() and self.model is not None:
            try:
                import numpy as np
                # Generate embeddings
                embeddings = self.model.encode([resume_text, job_description])
                # Compute Cosine Similarity
                emb1, emb2 = embeddings[0], embeddings[1]
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                similarity = float(dot_product / (norm1 * norm2))
                # Normalize between 0 and 1
                similarity = max(0.0, min(1.0, similarity))
                logger.info(f"Semantic match calculated using Sentence-Transformers: {similarity:.4f}")
                return similarity
            except Exception as e:
                logger.error(f"Error calculating similarity with sentence-transformers: {e}")

        # Fallback 1: Gemini API
        if settings.GEMINI_API_KEY:
            try:
                logger.info("Using Gemini API for semantic similarity fallback...")
                similarity = self._calculate_similarity_with_gemini(resume_text, job_description)
                logger.info(f"Semantic match calculated using Gemini: {similarity:.4f}")
                return similarity
            except Exception as e:
                logger.error(f"Error calculating similarity with Gemini: {e}")

        # Fallback 2: Pure-Python TF-IDF Cosine Similarity
        logger.info("Using pure-Python TF-IDF Cosine Similarity fallback...")
        similarity = self._calculate_tfidf_similarity(resume_text, job_description)
        logger.info(f"Semantic match calculated using TF-IDF: {similarity:.4f}")
        return similarity

    def _calculate_similarity_with_gemini(self, resume_text: str, job_description: str) -> float:
        """Asks Gemini to evaluate semantic similarity between resume and job description."""
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        prompt = f"""
        Analyze the overall semantic similarity between the candidate's resume and the job description.
        Evaluate how well the candidate's background, projects, skills, and experience align with the role.
        
        Return ONLY a single JSON object with a "score" key (floating point number between 0.0 and 1.0, where 1.0 is a perfect match).
        Do not output any markdown or other text.
        
        Format:
        {{"score": 0.85}}
        
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
        
        try:
            import json
            data = json.loads(response.text.strip())
            score = float(data.get("score", 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error(f"Failed to parse Gemini similarity response: {e}")
            raise

    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Computes a basic TF-IDF based cosine similarity between two texts without external dependencies."""
        def tokenize(text):
            # Lowercase, remove non-alphanumeric, split by whitespace
            words = re.findall(r'\b\w+\b', text.lower())
            return words

        import re
        words1 = tokenize(text1)
        words2 = tokenize(text2)
        
        if not words1 or not words2:
            return 0.0
            
        # Count term frequencies
        tf1 = Counter(words1)
        tf2 = Counter(words2)
        
        # Unique vocabulary
        vocab = set(tf1.keys()).union(set(tf2.keys()))
        
        # Since we only have 2 documents, Document Frequency (DF) is either 1 or 2
        # If a word is in both, DF = 2. If in only one, DF = 1.
        # IDF = log(Total Docs / DF)
        # To avoid IDF = 0 for words in both, we can use smoothed IDF: log(1 + Total Docs / DF)
        idf = {}
        for word in vocab:
            df = 0
            if word in tf1: df += 1
            if word in tf2: df += 1
            idf[word] = math.log(1.0 + 2.0 / df)
            
        # Calculate TF-IDF vectors
        vec1 = {word: tf1[word] * idf[word] for word in tf1}
        vec2 = {word: tf2[word] * idf[word] for word in tf2}
        
        # Compute dot product
        dot_product = sum(vec1[word] * vec2.get(word, 0.0) for word in vec1)
        
        # Compute magnitudes
        magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        cosine_sim = dot_product / (magnitude1 * magnitude2)
        # Normalize/scale to make it comparable to sentence embeddings
        # Word-overlap cosine similarity is typically lower, so we apply a scaling factor/exponent
        # to align it visually with typical neural similarity (which has high baseline similarity)
        scaled_sim = math.pow(cosine_sim, 0.5) if cosine_sim > 0 else 0.0
        return float(scaled_sim)
