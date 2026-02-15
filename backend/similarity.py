"""
Similarity Engine Module
========================
Generates embeddings and calculates similarity between resumes and job descriptions.
Uses Sentence Transformers for semantic understanding and cosine similarity for scoring.

Author: AI Resume Matcher Team
Version: 1.0.0
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Data class to hold similarity calculation results."""
    match_score: float  # 0-100
    semantic_similarity: float  # Raw cosine similarity
    skill_match_score: float  # Based on skills
    combined_score: float  # Weighted combination
    explanation: str
    detailed_breakdown: Dict


class SimilarityEngine:
    """
    Calculates similarity between resumes and job descriptions using:
    1. Semantic embeddings (Sentence Transformers)
    2. Skill-based matching
    3. Combined scoring with explainability
    
    Attributes:
        model: SentenceTransformer model for embeddings
        weights: Weights for different scoring components
    """
    
    # Default weights for scoring components
    DEFAULT_WEIGHTS = {
        'semantic': 0.4,
        'skills': 0.5,
        'experience': 0.1
    }
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the SimilarityEngine.
        
        Args:
            model_name: Name of the SentenceTransformer model
            weights: Custom weights for scoring components
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.model_name = model_name
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Load the model
        self._load_model()
        
        self.logger.info(f"SimilarityEngine initialized with model: {model_name}")
    
    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Model loaded successfully")
        except ImportError:
            self.logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text.
        
        Args:
            text: Input text
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        if not text or not isinstance(text, str):
            self.logger.warning("Empty or invalid text provided for embedding")
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Generate embedding
        embedding = self.model.encode(cleaned_text, convert_to_numpy=True)
        
        return embedding
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts
            
        Returns:
            numpy.ndarray: Matrix of embeddings
        """
        cleaned_texts = [self._preprocess_text(t) for t in texts if t]
        
        if not cleaned_texts:
            return np.array([])
        
        embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True)
        return embeddings
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding generation.
        
        Args:
            text: Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?@\-()/+]', ' ', text)
        
        # Normalize whitespace
        text = text.strip()
        
        return text
    
    def calculate_similarity(
        self, 
        resume_text: str, 
        jd_text: str,
        resume_skills: List[str] = None,
        jd_skills: List[str] = None,
        matched_skills: List[str] = None,
        missing_skills: List[str] = None
    ) -> SimilarityResult:
        """
        Calculate comprehensive similarity between resume and job description.
        
        Args:
            resume_text: Full resume text
            jd_text: Full job description text
            resume_skills: Extracted skills from resume
            jd_skills: Extracted skills from job description
            matched_skills: Skills that match between both
            missing_skills: Skills in JD but not in resume
            
        Returns:
            SimilarityResult with scores and explanation
        """
        self.logger.info("Calculating similarity scores")
        
        # 1. Semantic similarity using embeddings
        semantic_score = self._calculate_semantic_similarity(resume_text, jd_text)
        
        # 2. Skill-based similarity
        skill_score = self._calculate_skill_similarity(
            resume_skills or [], 
            jd_skills or [],
            matched_skills or [],
            missing_skills or []
        )
        
        # 3. Experience similarity (if experience info available)
        exp_score = self._calculate_experience_similarity(resume_text, jd_text)
        
        # Combine scores using weights
        combined_score = (
            self.weights['semantic'] * semantic_score +
            self.weights['skills'] * skill_score +
            self.weights['experience'] * exp_score
        )
        
        # Convert to percentage (0-100)
        match_percentage = combined_score * 100
        
        # Generate explanation
        explanation = self._generate_explanation(
            match_percentage,
            semantic_score,
            skill_score,
            matched_skills or [],
            missing_skills or []
        )
        
        # Create detailed breakdown
        breakdown = {
            'semantic_similarity': {
                'score': semantic_score,
                'weight': self.weights['semantic'],
                'weighted_score': semantic_score * self.weights['semantic']
            },
            'skill_similarity': {
                'score': skill_score,
                'weight': self.weights['skills'],
                'weighted_score': skill_score * self.weights['skills'],
                'total_jd_skills': len(jd_skills) if jd_skills else 0,
                'matched_skills_count': len(matched_skills) if matched_skills else 0,
                'missing_skills_count': len(missing_skills) if missing_skills else 0
            },
            'experience_similarity': {
                'score': exp_score,
                'weight': self.weights['experience'],
                'weighted_score': exp_score * self.weights['experience']
            }
        }
        
        result = SimilarityResult(
            match_score=round(match_percentage, 2),
            semantic_similarity=round(semantic_score * 100, 2),
            skill_match_score=round(skill_score * 100, 2),
            combined_score=round(combined_score * 100, 2),
            explanation=explanation,
            detailed_breakdown=breakdown
        )
        
        self.logger.info(f"Match score calculated: {result.match_score}%")
        
        return result
    
    def _calculate_semantic_similarity(
        self, 
        text1: str, 
        text2: str
    ) -> float:
        """
        Calculate semantic similarity using cosine similarity of embeddings.
        
        Args:
            text1: First text (resume)
            text2: Second text (job description)
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Generate embeddings
        embedding1 = self.generate_embedding(text1).reshape(1, -1)
        embedding2 = self.generate_embedding(text2).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        # Normalize to 0-1 range (cosine similarity is -1 to 1)
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def _calculate_skill_similarity(
        self,
        resume_skills: List[str],
        jd_skills: List[str],
        matched_skills: List[str],
        missing_skills: List[str]
    ) -> float:
        """
        Calculate skill-based similarity score.
        
        Args:
            resume_skills: All skills from resume
            jd_skills: All skills from job description
            matched_skills: Skills that match
            missing_skills: Skills missing from resume
            
        Returns:
            float: Skill match score (0-1)
        """
        if not jd_skills:
            return 0.5  # Neutral if no skills in JD
        
        # Base score on percentage of matched skills
        match_ratio = len(matched_skills) / len(jd_skills)
        
        # Bonus for having more skills than required
        extra_skills = len(resume_skills) - len(matched_skills)
        extra_bonus = min(extra_skills * 0.02, 0.1)  # Cap at 0.1
        
        # Penalty for missing critical skills
        missing_penalty = len(missing_skills) * 0.05
        
        score = match_ratio + extra_bonus - missing_penalty
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _calculate_experience_similarity(
        self, 
        resume_text: str, 
        jd_text: str
    ) -> float:
        """
        Calculate experience level similarity.
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
            
        Returns:
            float: Experience match score (0-1)
        """
        # Extract years of experience from JD
        jd_years = self._extract_years_experience(jd_text)
        
        # Extract years of experience from resume
        resume_years = self._extract_years_experience(resume_text)
        
        if jd_years is None:
            return 0.5  # Neutral if no experience requirement
        
        if resume_years is None:
            return 0.3  # Low if can't determine experience
        
        # Calculate match
        if resume_years >= jd_years:
            return 1.0
        elif resume_years >= jd_years * 0.7:
            return 0.8
        elif resume_years >= jd_years * 0.5:
            return 0.6
        else:
            return 0.4
    
    def _extract_years_experience(self, text: str) -> Optional[int]:
        """
        Extract years of experience from text.
        
        Args:
            text: Input text
            
        Returns:
            int or None: Years of experience
        """
        # Common patterns for experience
        patterns = [
            r'(\d+)\+?\s*years?\s*(of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(of\s*)?work',
            r'experience\s*:?\s*(\d+)\+?\s*years?',
            r'minimum\s*(of\s*)?(\d+)\+?\s*years?',
            r'at\s*least\s*(\d+)\+?\s*years?'
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Extract number from match
                match = matches[0]
                if isinstance(match, tuple):
                    # Find the numeric group
                    for m in match:
                        if m.isdigit():
                            return int(m)
                else:
                    return int(match)
        
        return None
    
    def _generate_explanation(
        self,
        match_score: float,
        semantic_score: float,
        skill_score: float,
        matched_skills: List[str],
        missing_skills: List[str]
    ) -> str:
        """
        Generate human-readable explanation of the match.
        
        Args:
            match_score: Overall match percentage
            semantic_score: Semantic similarity score
            skill_score: Skill match score
            matched_skills: List of matched skills
            missing_skills: List of missing skills
            
        Returns:
            str: Human-readable explanation
        """
        # Determine match level
        if match_score >= 80:
            match_level = "excellent"
            match_desc = "strong alignment"
        elif match_score >= 60:
            match_level = "good"
            match_desc = "solid match"
        elif match_score >= 40:
            match_level = "moderate"
            match_desc = "partial alignment"
        else:
            match_level = "low"
            match_desc = "limited alignment"
        
        # Build explanation parts
        parts = []
        
        # Overall assessment
        parts.append(
            f"Candidate shows a {match_level} match ({match_score:.0f}%) with the job requirements. "
            f"There is {match_desc} between the resume and job description."
        )
        
        # Semantic similarity explanation
        if semantic_score >= 0.7:
            parts.append("The overall content and context show strong semantic similarity.")
        elif semantic_score >= 0.5:
            parts.append("The content shows moderate semantic alignment.")
        else:
            parts.append("The content alignment could be improved.")
        
        # Skills explanation
        if matched_skills:
            top_skills = matched_skills[:5]
            parts.append(
                f"Key strengths include: {', '.join(top_skills)}."
            )
        
        # Missing skills explanation
        if missing_skills:
            critical_missing = missing_skills[:5]
            parts.append(
                f"Areas for development: {', '.join(critical_missing)}."
            )
        
        # Recommendations
        if match_score >= 80:
            parts.append("This candidate is highly recommended for the position.")
        elif match_score >= 60:
            parts.append("This candidate is worth considering for an interview.")
        elif match_score >= 40:
            parts.append("Consider this candidate if other options are limited.")
        else:
            parts.append("This candidate may not be the best fit for this role.")
        
        return " ".join(parts)
    
    def rank_resumes(
        self, 
        resumes: List[Dict], 
        jd_text: str,
        jd_skills: List[str] = None
    ) -> List[Dict]:
        """
        Rank multiple resumes against a job description.
        
        Args:
            resumes: List of dicts with 'text', 'filename', and 'skills'
            jd_text: Job description text
            jd_skills: Skills from job description
            
        Returns:
            List of resumes with match scores, sorted by score
        """
        self.logger.info(f"Ranking {len(resumes)} resumes")
        
        ranked_resumes = []
        
        for resume in resumes:
            result = self.calculate_similarity(
                resume_text=resume['text'],
                jd_text=jd_text,
                resume_skills=resume.get('skills', []),
                jd_skills=jd_skills or []
            )
            
            ranked_resumes.append({
                'filename': resume.get('filename', 'Unknown'),
                'text': resume['text'],
                'skills': resume.get('skills', []),
                'match_score': result.match_score,
                'semantic_similarity': result.semantic_similarity,
                'skill_match_score': result.skill_match_score,
                'explanation': result.explanation,
                'detailed_breakdown': result.detailed_breakdown
            })
        
        # Sort by match score (descending)
        ranked_resumes.sort(key=lambda x: x['match_score'], reverse=True)
        
        self.logger.info("Ranking complete")
        
        return ranked_resumes


# Convenience functions
def calculate_match_score(
    resume_text: str, 
    jd_text: str,
    resume_skills: List[str] = None,
    jd_skills: List[str] = None
) -> SimilarityResult:
    """
    Quick function to calculate match score.
    
    Args:
        resume_text: Resume text
        jd_text: Job description text
        resume_skills: Skills from resume
        jd_skills: Skills from job description
        
    Returns:
        SimilarityResult with scores
    """
    engine = SimilarityEngine()
    return engine.calculate_similarity(
        resume_text, jd_text, resume_skills, jd_skills
    )


def rank_multiple_resumes(
    resumes: List[str], 
    jd_text: str
) -> List[Tuple[str, float]]:
    """
    Quick function to rank multiple resume texts.
    
    Args:
        resumes: List of resume texts
        jd_text: Job description text
        
    Returns:
        List of (resume_text, score) tuples sorted by score
    """
    engine = SimilarityEngine()
    resume_dicts = [{'text': r, 'filename': f'Resume_{i+1}'} 
                    for i, r in enumerate(resumes)]
    
    ranked = engine.rank_resumes(resume_dicts, jd_text)
    
    return [(r['filename'], r['match_score']) for r in ranked]


if __name__ == "__main__":
    # Test the similarity engine
    sample_resume = """
    Senior Software Engineer with 7 years of experience in Python, Django, and Machine Learning.
    Expert in TensorFlow, PyTorch, and AWS. Strong background in NLP and computer vision.
    Led a team of 5 developers and delivered 10+ successful projects.
    """
    
    sample_jd = """
    We are looking for a Senior Python Developer with 5+ years of experience.
    Required skills: Python, Django, Machine Learning, TensorFlow, AWS.
    Nice to have: Docker, Kubernetes, NLP experience.
    """
    
    engine = SimilarityEngine()
    result = engine.calculate_similarity(
        resume_text=sample_resume,
        jd_text=sample_jd,
        resume_skills=['python', 'django', 'machine learning', 'tensorflow', 'aws', 'pytorch', 'nlp'],
        jd_skills=['python', 'django', 'machine learning', 'tensorflow', 'aws', 'docker', 'kubernetes', 'nlp'],
        matched_skills=['python', 'django', 'machine learning', 'tensorflow', 'aws', 'nlp'],
        missing_skills=['docker', 'kubernetes']
    )
    
    print(f"\n{'='*60}")
    print("Similarity Calculation Results")
    print(f"{'='*60}")
    print(f"Match Score: {result.match_score}%")
    print(f"Semantic Similarity: {result.semantic_similarity}%")
    print(f"Skill Match Score: {result.skill_match_score}%")
    print(f"\nExplanation:\n{result.explanation}")
    print(f"\nDetailed Breakdown:")
    for key, value in result.detailed_breakdown.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
