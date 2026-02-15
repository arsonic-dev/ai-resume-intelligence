"""
AI Resume-Job Matching System - FastAPI Backend
===============================================
Main FastAPI application providing RESTful API endpoints for:
- Resume parsing (PDF/DOCX)
- Skill extraction
- Similarity calculation
- Batch resume ranking
- Report generation

Author: AI Resume Matcher Team
Version: 1.0.0
"""

import os
import io
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import our modules
from parser import DocumentParser, parse_resume_bytes
from skill_extractor import SkillExtractor, SkillExtractionResult
from similarity import SimilarityEngine, SimilarityResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Resume-Job Matching System",
    description="Intelligent resume matching with explainable AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
parser = DocumentParser()
skill_extractor = SkillExtractor()
similarity_engine = SimilarityEngine()

# Ensure reports directory exists
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ============== Pydantic Models ==============

class JobDescriptionRequest(BaseModel):
    """Model for job description input."""
    text: str = Field(..., description="Job description text", min_length=10)
    title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")


class SkillExtractionResponse(BaseModel):
    """Model for skill extraction response."""
    technical_skills: List[str]
    soft_skills: List[str]
    all_skills: List[str]
    skill_categories: Dict[str, List[str]]
    confidence_scores: Dict[str, float]


class MatchRequest(BaseModel):
    """Model for match calculation request."""
    resume_text: str = Field(..., description="Resume text content")
    jd_text: str = Field(..., description="Job description text")


class MatchResponse(BaseModel):
    """Model for match calculation response."""
    match_score: float
    semantic_similarity: float
    skill_match_score: float
    combined_score: float
    explanation: str
    detailed_breakdown: Dict[str, Any]
    matched_skills: List[str]
    missing_skills: List[str]
    extra_skills: List[str]


class BatchMatchResponse(BaseModel):
    """Model for batch resume ranking response."""
    rankings: List[Dict[str, Any]]
    total_resumes: int
    jd_title: Optional[str]


class ReportRequest(BaseModel):
    """Model for report generation request."""
    resume_text: str
    jd_text: str
    candidate_name: Optional[str] = "Candidate"
    job_title: Optional[str] = "Position"


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]


# ============== API Endpoints ==============

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Resume-Job Matching System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify all components are working."""
    components = {
        "parser": "operational",
        "skill_extractor": "operational",
        "similarity_engine": "operational"
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        components=components
    )


@app.post("/parse-resume", response_model=Dict[str, Any])
async def parse_resume_endpoint(file: UploadFile = File(...)):
    """
    Parse a resume file (PDF or DOCX) and extract text content.
    
    Args:
        file: Uploaded resume file (PDF or DOCX)
        
    Returns:
        Dict with extracted text and metadata
    """
    logger.info(f"Received file: {file.filename}")
    
    # Validate file type
    allowed_extensions = ['.pdf', '.docx', '.doc']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Parse the document
        result = parse_resume_bytes(content, file.filename)
        
        if not result['success']:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse document: {result.get('error', 'Unknown error')}"
            )
        
        logger.info(f"Successfully parsed {file.filename}")
        
        return {
            "success": True,
            "filename": result['filename'],
            "format": result['format'],
            "page_count": result['page_count'],
            "text_length": len(result['text']),
            "text": result['text']
        }
        
    except Exception as e:
        logger.error(f"Error parsing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-skills", response_model=SkillExtractionResponse)
async def extract_skills_endpoint(text: str = Form(...)):
    """
    Extract skills from text (resume or job description).
    
    Args:
        text: Text content to analyze
        
    Returns:
        SkillExtractionResponse with extracted skills
    """
    logger.info(f"Extracting skills from text ({len(text)} characters)")
    
    try:
        result = skill_extractor.extract_skills(text)
        
        return SkillExtractionResponse(
            technical_skills=result.technical_skills,
            soft_skills=result.soft_skills,
            all_skills=result.all_skills,
            skill_categories=result.skill_categories,
            confidence_scores=result.confidence_scores
        )
        
    except Exception as e:
        logger.error(f"Error extracting skills: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate-match", response_model=MatchResponse)
async def calculate_match_endpoint(request: MatchRequest):
    """
    Calculate match score between resume and job description.
    
    Args:
        request: MatchRequest with resume_text and jd_text
        
    Returns:
        MatchResponse with scores and explanation
    """
    logger.info("Calculating match score")
    
    try:
        # Extract skills from both texts
        resume_skills_result = skill_extractor.extract_skills(request.resume_text)
        jd_skills_result = skill_extractor.extract_skills(request.jd_text)
        
        # Compare skills
        skill_comparison = skill_extractor.compare_skills(
            resume_skills_result, 
            jd_skills_result
        )
        
        # Calculate similarity
        similarity_result = similarity_engine.calculate_similarity(
            resume_text=request.resume_text,
            jd_text=request.jd_text,
            resume_skills=resume_skills_result.all_skills,
            jd_skills=jd_skills_result.all_skills,
            matched_skills=skill_comparison['matched_skills'],
            missing_skills=skill_comparison['missing_skills']
        )
        
        return MatchResponse(
            match_score=similarity_result.match_score,
            semantic_similarity=similarity_result.semantic_similarity,
            skill_match_score=similarity_result.skill_match_score,
            combined_score=similarity_result.combined_score,
            explanation=similarity_result.explanation,
            detailed_breakdown=similarity_result.detailed_breakdown,
            matched_skills=skill_comparison['matched_skills'],
            missing_skills=skill_comparison['missing_skills'],
            extra_skills=skill_comparison['extra_skills']
        )
        
    except Exception as e:
        logger.error(f"Error calculating match: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-rank-resumes", response_model=BatchMatchResponse)
async def batch_rank_resumes_endpoint(
    files: List[UploadFile] = File(...),
    jd_text: str = Form(...),
    jd_title: Optional[str] = Form(None)
):
    """
    Upload multiple resumes and rank them against a job description.
    
    Args:
        files: List of resume files (PDF or DOCX)
        jd_text: Job description text
        jd_title: Optional job title
        
    Returns:
        BatchMatchResponse with ranked resumes
    """
    logger.info(f"Received {len(files)} resumes for ranking")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    try:
        # Extract skills from JD
        jd_skills_result = skill_extractor.extract_skills(jd_text)
        
        # Parse all resumes
        resumes = []
        for file in files:
            content = await file.read()
            parsed = parse_resume_bytes(content, file.filename)
            
            if parsed['success']:
                resume_skills = skill_extractor.extract_skills(parsed['text'])
                resumes.append({
                    'filename': parsed['filename'],
                    'text': parsed['text'],
                    'skills': resume_skills.all_skills
                })
        
        # Rank resumes
        ranked = similarity_engine.rank_resumes(
            resumes=resumes,
            jd_text=jd_text,
            jd_skills=jd_skills_result.all_skills
        )
        
        return BatchMatchResponse(
            rankings=ranked,
            total_resumes=len(ranked),
            jd_title=jd_title
        )
        
    except Exception as e:
        logger.error(f"Error ranking resumes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-report")
async def generate_report_endpoint(
    background_tasks: BackgroundTasks,
    resume_file: UploadFile = File(...),
    jd_text: str = Form(...),
    candidate_name: Optional[str] = Form("Candidate"),
    job_title: Optional[str] = Form("Position")
):
    """
    Generate a PDF report for a resume-job description match.
    
    Args:
        resume_file: Resume file (PDF or DOCX)
        jd_text: Job description text
        candidate_name: Candidate name for the report
        job_title: Job title for the report
        
    Returns:
        PDF file download
    """
    logger.info(f"Generating report for {candidate_name}")
    
    try:
        # Parse resume
        content = await resume_file.read()
        parsed = parse_resume_bytes(content, resume_file.filename)
        
        if not parsed['success']:
            raise HTTPException(
                status_code=422,
                detail="Failed to parse resume"
            )
        
        # Calculate match
        resume_skills = skill_extractor.extract_skills(parsed['text'])
        jd_skills = skill_extractor.extract_skills(jd_text)
        skill_comparison = skill_extractor.compare_skills(resume_skills, jd_skills)
        
        similarity_result = similarity_engine.calculate_similarity(
            resume_text=parsed['text'],
            jd_text=jd_text,
            resume_skills=resume_skills.all_skills,
            jd_skills=jd_skills.all_skills,
            matched_skills=skill_comparison['matched_skills'],
            missing_skills=skill_comparison['missing_skills']
        )
        
        # Generate PDF report
        report_path = generate_pdf_report(
            candidate_name=candidate_name,
            job_title=job_title,
            similarity_result=similarity_result,
            skill_comparison=skill_comparison,
            resume_skills=resume_skills,
            jd_skills=jd_skills
        )
        
        # Schedule cleanup of old reports
        background_tasks.add_task(cleanup_old_reports)
        
        return FileResponse(
            path=report_path,
            filename=f"match_report_{candidate_name.replace(' ', '_')}.pdf",
            media_type="application/pdf"
        )
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/full-analysis")
async def full_analysis_endpoint(
    resume_file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    """
    Complete analysis endpoint - parse resume, extract skills, and calculate match.
    
    Args:
        resume_file: Resume file (PDF or DOCX)
        jd_text: Job description text
        
    Returns:
        Complete analysis with all components
    """
    logger.info("Running full analysis")
    
    try:
        # Parse resume
        content = await resume_file.read()
        parsed = parse_resume_bytes(content, resume_file.filename)
        
        if not parsed['success']:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse resume: {parsed.get('error', 'Unknown error')}"
            )
        
        # Extract skills
        resume_skills = skill_extractor.extract_skills(parsed['text'])
        jd_skills = skill_extractor.extract_skills(jd_text)
        
        # Compare skills
        skill_comparison = skill_extractor.compare_skills(resume_skills, jd_skills)
        
        # Calculate similarity
        similarity_result = similarity_engine.calculate_similarity(
            resume_text=parsed['text'],
            jd_text=jd_text,
            resume_skills=resume_skills.all_skills,
            jd_skills=jd_skills.all_skills,
            matched_skills=skill_comparison['matched_skills'],
            missing_skills=skill_comparison['missing_skills']
        )
        
        return {
            "resume_info": {
                "filename": parsed['filename'],
                "format": parsed['format'],
                "page_count": parsed['page_count'],
                "text_length": len(parsed['text'])
            },
            "skills": {
                "resume": {
                    "technical": resume_skills.technical_skills,
                    "soft": resume_skills.soft_skills,
                    "all": resume_skills.all_skills,
                    "categories": resume_skills.skill_categories
                },
                "job_description": {
                    "technical": jd_skills.technical_skills,
                    "soft": jd_skills.soft_skills,
                    "all": jd_skills.all_skills,
                    "categories": jd_skills.skill_categories
                }
            },
            "skill_comparison": skill_comparison,
            "match_analysis": {
                "match_score": similarity_result.match_score,
                "semantic_similarity": similarity_result.semantic_similarity,
                "skill_match_score": similarity_result.skill_match_score,
                "explanation": similarity_result.explanation,
                "detailed_breakdown": similarity_result.detailed_breakdown
            }
        }
        
    except Exception as e:
        logger.error(f"Error in full analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Helper Functions ==============

def generate_pdf_report(
    candidate_name: str,
    job_title: str,
    similarity_result: SimilarityResult,
    skill_comparison: Dict,
    resume_skills: SkillExtractionResult,
    jd_skills: SkillExtractionResult
) -> str:
    """
    Generate a PDF report for the match analysis.
    
    Args:
        candidate_name: Name of the candidate
        job_title: Job title
        similarity_result: Similarity calculation result
        skill_comparison: Skill comparison results
        resume_skills: Skills from resume
        jd_skills: Skills from job description
        
    Returns:
        str: Path to generated PDF file
    """
    try:
        from fpdf import FPDF
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, 'AI Resume Match Report', 0, 1, 'C')
                self.ln(5)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title section
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f'Candidate: {candidate_name}', 0, 1)
        pdf.cell(0, 10, f'Position: {job_title}', 0, 1)
        pdf.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d")}', 0, 1)
        pdf.ln(10)
        
        # Match Score
        pdf.set_font('Arial', 'B', 16)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 15, f'Match Score: {similarity_result.match_score}%', 0, 1, 'C', True)
        pdf.ln(10)
        
        # Score Breakdown
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Score Breakdown:', 0, 1)
        pdf.set_font('Arial', '', 11)
        
        breakdown = similarity_result.detailed_breakdown
        pdf.cell(0, 8, f"Semantic Similarity: {similarity_result.semantic_similarity}%", 0, 1)
        pdf.cell(0, 8, f"Skill Match: {similarity_result.skill_match_score}%", 0, 1)
        pdf.ln(5)
        
        # Explanation
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Analysis:', 0, 1)
        pdf.set_font('Arial', '', 11)
        
        # Wrap explanation text
        explanation = similarity_result.explanation
        pdf.multi_cell(0, 8, explanation)
        pdf.ln(10)
        
        # Skills Section
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Skills Analysis:', 0, 1)
        
        # Matched Skills
        pdf.set_font('Arial', 'B', 11)
        pdf.set_text_color(0, 128, 0)
        matched_count = len(skill_comparison['matched_skills'])
        pdf.cell(0, 8, f"Matched Skills ({matched_count}):", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(0, 0, 0)
        
        matched_text = ', '.join(skill_comparison['matched_skills'][:15])
        pdf.multi_cell(0, 6, matched_text if matched_text else 'None')
        pdf.ln(5)
        
        # Missing Skills
        pdf.set_font('Arial', 'B', 11)
        pdf.set_text_color(255, 0, 0)
        matched_count = len(skill_comparison['matched_skills'])
        pdf.cell(0, 8, f"Matched Skills ({matched_count}):", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(0, 0, 0)
        
        missing_text = ', '.join(skill_comparison['missing_skills'][:15])
        pdf.multi_cell(0, 6, missing_text if missing_text else 'None')
        pdf.ln(5)
        
        # Save PDF
        report_filename = f"match_report_{candidate_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = REPORTS_DIR / report_filename
        pdf.output(str(report_path))
        
        logger.info(f"Report generated: {report_path}")
        return str(report_path)
        
    except ImportError:
        logger.error("fpdf not installed. Cannot generate PDF report.")
        raise HTTPException(status_code=500, detail="PDF generation not available")


def cleanup_old_reports():
    """Clean up old report files (older than 1 hour)."""
    try:
        import time
        current_time = time.time()
        
        for report_file in REPORTS_DIR.glob("*.pdf"):
            file_age = current_time - report_file.stat().st_mtime
            if file_age > 3600:  # 1 hour
                report_file.unlink()
                logger.info(f"Cleaned up old report: {report_file}")
    except Exception as e:
        logger.warning(f"Error cleaning up reports: {e}")


# ============== Main Entry Point ==============

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
