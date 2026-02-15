"""
Skill Extractor Module
======================
Extracts technical and soft skills from text using SpaCy NER and 
predefined skill dictionaries. Supports both resume and job description parsing.

Author: AI Resume Matcher Team
Version: 1.0.0
"""

import re
import logging
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("SpaCy not available. Using fallback skill extraction.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SkillExtractionResult:
    """Data class to hold skill extraction results."""
    technical_skills: List[str]
    soft_skills: List[str]
    all_skills: List[str]
    skill_categories: Dict[str, List[str]]
    confidence_scores: Dict[str, float]


class SkillExtractor:
    """
    Extracts skills from text using multiple methods:
    1. Predefined skill dictionary matching
    2. SpaCy NER (if available)
    3. Pattern-based extraction
    
    Attributes:
        nlp: SpaCy language model
        technical_skills: Set of technical skills to match
        soft_skills: Set of soft skills to match
    """
    
    # Comprehensive Technical Skills Database
    TECHNICAL_SKILLS = {
        # Programming Languages
        'python', 'java', 'javascript', 'js', 'typescript', 'ts', 'c++', 'c', 'c#', 
        'csharp', 'go', 'golang', 'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala',
        'r', 'matlab', 'perl', 'shell', 'bash', 'powershell', 'sql', 'plsql', 'tsql',
        'html', 'css', 'sass', 'less', 'xml', 'json', 'yaml',
        
        # Web Development
        'react', 'reactjs', 'angular', 'vue', 'vuejs', 'svelte', 'nextjs', 'nuxtjs',
        'django', 'flask', 'fastapi', 'spring', 'spring boot', 'express', 'nodejs',
        'laravel', 'rails', 'aspnet', 'wordpress', 'drupal',
        
        # Databases
        'mysql', 'postgresql', 'postgres', 'mongodb', 'sqlite', 'oracle', 'sql server',
        'redis', 'elasticsearch', 'cassandra', 'dynamodb', 'firebase', 'couchdb',
        
        # Cloud & DevOps
        'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'heroku',
        'docker', 'kubernetes', 'k8s', 'jenkins', 'gitlab ci', 'github actions',
        'terraform', 'ansible', 'puppet', 'chef', 'vagrant', 'nginx', 'apache',
        
        # Data Science & ML
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 
        'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'jupyter', 'opencv',
        'nltk', 'spacy', 'huggingface', 'transformers', 'xgboost', 'lightgbm',
        'machine learning', 'deep learning', 'nlp', 'computer vision', 'ai',
        'data science', 'data analysis', 'statistics', 'big data', 'hadoop', 'spark',
        
        # Mobile Development
        'android', 'ios', 'react native', 'flutter', 'xamarin', 'cordova', 'ionic',
        
        # Tools & Platforms
        'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'trello',
        'slack', 'teams', 'zoom', 'postman', 'insomnia', 'figma', 'sketch',
        'photoshop', 'illustrator', 'xd', 'vscode', 'intellij', 'pycharm',
        
        # Methodologies
        'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd', 'oop',
        'rest api', 'graphql', 'soap', 'microservices', 'serverless',
        
        # Operating Systems
        'linux', 'ubuntu', 'centos', 'debian', 'windows', 'macos', 'unix',
        
        # Security
        'cybersecurity', 'penetration testing', 'owasp', 'ssl', 'oauth', 'jwt',
        
        # Other Technologies
        'blockchain', 'ethereum', 'solidity', 'web3', 'iot', 'arduino', 'raspberry pi'
    }
    
    # Soft Skills Database
    SOFT_SKILLS = {
        'communication', 'leadership', 'teamwork', 'collaboration', 'problem solving',
        'critical thinking', 'creativity', 'innovation', 'adaptability', 'flexibility',
        'time management', 'organization', 'planning', 'project management',
        'decision making', 'analytical skills', 'attention to detail', 'multitasking',
        'interpersonal skills', 'customer service', 'presentation skills',
        'negotiation', 'conflict resolution', 'mentoring', 'coaching',
        'self-motivated', 'proactive', 'reliable', 'punctual', 'dedicated',
        'english', 'spanish', 'french', 'german', 'chinese', 'japanese'
    }
    
    # Skill categories for better organization
    SKILL_CATEGORIES = {
        'Programming': ['python', 'java', 'javascript', 'js', 'typescript', 'ts', 'c++', 'c', 'c#', 
                       'csharp', 'go', 'golang', 'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala',
                       'r', 'matlab', 'perl', 'shell', 'bash', 'powershell', 'sql', 'plsql', 'tsql',
                       'html', 'css', 'sass', 'less', 'xml', 'json', 'yaml'],
        
        'Web Frameworks': ['react', 'reactjs', 'angular', 'vue', 'vuejs', 'svelte', 'nextjs', 
                          'nuxtjs', 'django', 'flask', 'fastapi', 'spring', 'spring boot', 
                          'express', 'nodejs', 'laravel', 'rails', 'aspnet'],
        
        'Databases': ['mysql', 'postgresql', 'postgres', 'mongodb', 'sqlite', 'oracle', 
                     'sql server', 'redis', 'elasticsearch', 'cassandra', 'dynamodb', 
                     'firebase', 'couchdb'],
        
        'Cloud & DevOps': ['aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 
                          'heroku', 'docker', 'kubernetes', 'k8s', 'jenkins', 'gitlab ci', 
                          'github actions', 'terraform', 'ansible', 'puppet', 'chef', 
                          'vagrant', 'nginx', 'apache'],
        
        'Data Science & ML': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 
                             'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 
                             'jupyter', 'opencv', 'nltk', 'spacy', 'huggingface', 'transformers',
                             'xgboost', 'lightgbm', 'machine learning', 'deep learning', 
                             'nlp', 'computer vision', 'ai', 'data science', 'data analysis',
                             'statistics', 'big data', 'hadoop', 'spark'],
        
        'Mobile': ['android', 'ios', 'react native', 'flutter', 'xamarin', 'cordova', 'ionic'],
        
        'Tools': ['git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'trello',
                 'slack', 'teams', 'zoom', 'postman', 'insomnia', 'figma', 'sketch',
                 'photoshop', 'illustrator', 'xd', 'vscode', 'intellij', 'pycharm'],
        
        'Methodologies': ['agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd', 
                         'oop', 'rest api', 'graphql', 'soap', 'microservices', 'serverless'],
        
        'Soft Skills': ['communication', 'leadership', 'teamwork', 'collaboration', 
                       'problem solving', 'critical thinking', 'creativity', 'innovation',
                       'adaptability', 'flexibility', 'time management', 'organization',
                       'planning', 'project management', 'decision making', 'analytical skills',
                       'attention to detail', 'multitasking', 'interpersonal skills']
    }
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the SkillExtractor.
        
        Args:
            use_spacy: Whether to use SpaCy NER (if available)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.nlp = None
        
        # Try to load SpaCy model
        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                self.logger.info("SpaCy model loaded successfully")
            except OSError:
                self.logger.warning(
                    "SpaCy model 'en_core_web_sm' not found. "
                    "Install with: python -m spacy download en_core_web_sm"
                )
                self.logger.info("Falling back to dictionary-based extraction")
        
        # Compile regex patterns for skill matching
        self._compile_patterns()
        self.logger.info("SkillExtractor initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient skill matching."""
        # Create word boundary patterns for each skill
        self.tech_patterns = {
            skill: re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
            for skill in self.TECHNICAL_SKILLS
        }
        
        self.soft_patterns = {
            skill: re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
            for skill in self.SOFT_SKILLS
        }
    
    def extract_skills(self, text: str, extract_categories: bool = True) -> SkillExtractionResult:
        """
        Extract skills from text using multiple methods.
        
        Args:
            text: Input text (resume or job description)
            extract_categories: Whether to categorize skills
            
        Returns:
            SkillExtractionResult: Extracted skills with metadata
        """
        if not text or not isinstance(text, str):
            self.logger.warning("Empty or invalid text provided")
            return SkillExtractionResult([], [], [], {}, {})
        
        self.logger.info(f"Extracting skills from text ({len(text)} characters)")
        
        # Normalize text
        normalized_text = text.lower()
        
        # Extract using dictionary matching
        tech_skills = self._extract_technical_skills(normalized_text)
        soft_skills = self._extract_soft_skills(normalized_text)
        
        # Extract using SpaCy if available
        if self.nlp:
            spacy_skills = self._extract_with_spacy(text)
            tech_skills.update(spacy_skills)
        
        # Combine all skills
        all_skills = list(tech_skills.union(soft_skills))
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence(
            text, tech_skills, soft_skills
        )
        
        # Categorize skills if requested
        skill_categories = {}
        if extract_categories:
            skill_categories = self._categorize_skills(all_skills)
        
        result = SkillExtractionResult(
            technical_skills=sorted(list(tech_skills)),
            soft_skills=sorted(list(soft_skills)),
            all_skills=sorted(all_skills),
            skill_categories=skill_categories,
            confidence_scores=confidence_scores
        )
        
        self.logger.info(
            f"Extracted {len(tech_skills)} technical skills, "
            f"{len(soft_skills)} soft skills"
        )
        
        return result
    
    def _extract_technical_skills(self, text: str) -> Set[str]:
        """
        Extract technical skills using dictionary matching.
        
        Args:
            text: Normalized input text
            
        Returns:
            Set of matched technical skills
        """
        matched_skills = set()
        
        for skill, pattern in self.tech_patterns.items():
            if pattern.search(text):
                matched_skills.add(skill.lower())
        
        return matched_skills
    
    def _extract_soft_skills(self, text: str) -> Set[str]:
        """
        Extract soft skills using dictionary matching.
        
        Args:
            text: Normalized input text
            
        Returns:
            Set of matched soft skills
        """
        matched_skills = set()
        
        for skill, pattern in self.soft_patterns.items():
            if pattern.search(text):
                matched_skills.add(skill.lower())
        
        return matched_skills
    
    def _extract_with_spacy(self, text: str) -> Set[str]:
        """
        Extract potential skills using SpaCy NER.
        
        Args:
            text: Input text
            
        Returns:
            Set of potential skills from NER
        """
        matched_skills = set()
        
        if not self.nlp:
            return matched_skills
        
        doc = self.nlp(text)
        
        # Extract organizations and products that might be technologies
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART']:
                # Check if it's in our skill database
                ent_text = ent.text.lower()
                if ent_text in self.TECHNICAL_SKILLS:
                    matched_skills.add(ent_text)
        
        return matched_skills
    
    def _calculate_confidence(
        self, text: str, tech_skills: Set[str], soft_skills: Set[str]
    ) -> Dict[str, float]:
        """
        Calculate confidence scores for extracted skills.
        
        Args:
            text: Original text
            tech_skills: Extracted technical skills
            soft_skills: Extracted soft skills
            
        Returns:
            Dictionary of skill -> confidence score
        """
        confidence_scores = {}
        text_lower = text.lower()
        
        all_skills = tech_skills.union(soft_skills)
        
        for skill in all_skills:
            score = 0.5  # Base confidence
            
            # Higher confidence if skill appears multiple times
            count = text_lower.count(skill.lower())
            if count > 1:
                score += min(0.2 * count, 0.3)  # Cap at 0.3 bonus
            
            # Higher confidence for technical skills (more specific)
            if skill in tech_skills:
                score += 0.1
            
            # Cap at 1.0
            confidence_scores[skill] = min(score, 1.0)
        
        return confidence_scores
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Categorize skills into predefined categories.
        
        Args:
            skills: List of extracted skills
            
        Returns:
            Dictionary of category -> skills
        """
        categorized = defaultdict(list)
        
        for skill in skills:
            skill_lower = skill.lower()
            found_category = False
            
            for category, category_skills in self.SKILL_CATEGORIES.items():
                if skill_lower in [s.lower() for s in category_skills]:
                    categorized[category].append(skill)
                    found_category = True
                    break
            
            if not found_category:
                categorized['Other'].append(skill)
        
        return dict(categorized)
    
    def compare_skills(
        self, 
        resume_skills: SkillExtractionResult, 
        jd_skills: SkillExtractionResult
    ) -> Dict:
        """
        Compare skills between resume and job description.
        
        Args:
            resume_skills: Skills extracted from resume
            jd_skills: Skills extracted from job description
            
        Returns:
            Dictionary with matched, missing, and extra skills
        """
        resume_set = set(resume_skills.all_skills)
        jd_set = set(jd_skills.all_skills)
        
        # Calculate intersections and differences
        matched = resume_set.intersection(jd_set)
        missing = jd_set - resume_set  # In JD but not in resume
        extra = resume_set - jd_set    # In resume but not in JD
        
        # Calculate match percentages
        tech_matched = set(resume_skills.technical_skills) & set(jd_skills.technical_skills)
        tech_missing = set(jd_skills.technical_skills) - set(resume_skills.technical_skills)
        
        soft_matched = set(resume_skills.soft_skills) & set(jd_skills.soft_skills)
        soft_missing = set(jd_skills.soft_skills) - set(resume_skills.soft_skills)
        
        return {
            'matched_skills': sorted(list(matched)),
            'missing_skills': sorted(list(missing)),
            'extra_skills': sorted(list(extra)),
            'technical': {
                'matched': sorted(list(tech_matched)),
                'missing': sorted(list(tech_missing)),
                'match_percentage': len(tech_matched) / len(jd_skills.technical_skills) * 100 
                                   if jd_skills.technical_skills else 0
            },
            'soft': {
                'matched': sorted(list(soft_matched)),
                'missing': sorted(list(soft_missing)),
                'match_percentage': len(soft_matched) / len(jd_skills.soft_skills) * 100 
                                   if jd_skills.soft_skills else 0
            },
            'overall_match_percentage': len(matched) / len(jd_set) * 100 if jd_set else 0
        }
    
    def get_skill_importance(
        self, 
        skill: str, 
        jd_text: str
    ) -> str:
        """
        Determine the importance of a skill in the job description.
        
        Args:
            skill: Skill to check
            jd_text: Job description text
            
        Returns:
            Importance level: 'high', 'medium', or 'low'
        """
        jd_lower = jd_text.lower()
        skill_lower = skill.lower()
        
        # Count occurrences
        count = jd_lower.count(skill_lower)
        
        # Check for importance indicators
        importance_patterns = [
            rf'required\s*:?\s*[^.]*{re.escape(skill_lower)}',
            rf'must have\s*:?\s*[^.]*{re.escape(skill_lower)}',
            rf'essential\s*:?\s*[^.]*{re.escape(skill_lower)}',
            rf'critical\s*:?\s*[^.]*{re.escape(skill_lower)}',
        ]
        
        for pattern in importance_patterns:
            if re.search(pattern, jd_lower):
                return 'high'
        
        # Check for preferred/nice to have
        preferred_patterns = [
            rf'preferred\s*:?\s*[^.]*{re.escape(skill_lower)}',
            rf'nice to have\s*:?\s*[^.]*{re.escape(skill_lower)}',
            rf'bonus\s*:?\s*[^.]*{re.escape(skill_lower)}',
        ]
        
        for pattern in preferred_patterns:
            if re.search(pattern, jd_lower):
                return 'low'
        
        # Based on frequency
        if count >= 3:
            return 'high'
        elif count >= 2:
            return 'medium'
        
        return 'low'


# Convenience functions
def extract_skills_from_text(text: str) -> SkillExtractionResult:
    """
    Quick function to extract skills from text.
    
    Args:
        text: Input text
        
    Returns:
        SkillExtractionResult with extracted skills
    """
    extractor = SkillExtractor()
    return extractor.extract_skills(text)


def compare_resume_with_jd(resume_text: str, jd_text: str) -> Dict:
    """
    Quick function to compare resume skills with job description.
    
    Args:
        resume_text: Resume text
        jd_text: Job description text
        
    Returns:
        Dictionary with skill comparison results
    """
    extractor = SkillExtractor()
    
    resume_skills = extractor.extract_skills(resume_text)
    jd_skills = extractor.extract_skills(jd_text)
    
    return extractor.compare_skills(resume_skills, jd_skills)


if __name__ == "__main__":
    # Test the skill extractor
    sample_text = """
    Senior Python Developer with 5+ years of experience in Django, Flask, and FastAPI.
    Strong expertise in Machine Learning, TensorFlow, and PyTorch.
    Proficient with AWS, Docker, Kubernetes, and CI/CD pipelines.
    Excellent communication skills and team leadership experience.
    """
    
    extractor = SkillExtractor()
    result = extractor.extract_skills(sample_text)
    
    print(f"\n{'='*60}")
    print("Skill Extraction Results")
    print(f"{'='*60}")
    print(f"\nTechnical Skills ({len(result.technical_skills)}):")
    print(", ".join(result.technical_skills))
    print(f"\nSoft Skills ({len(result.soft_skills)}):")
    print(", ".join(result.soft_skills))
    print(f"\nSkill Categories:")
    for category, skills in result.skill_categories.items():
        if skills:
            print(f"  {category}: {', '.join(skills)}")
    print(f"{'='*60}\n")
