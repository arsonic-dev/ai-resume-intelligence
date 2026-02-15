#!/usr/bin/env python
"""
Quick Start Script for AI Resume-Job Matching System
====================================================
Provides easy commands to run different parts of the system.

Usage:
    python run.py backend     # Start FastAPI backend
    python run.py frontend    # Start Streamlit frontend
    python run.py test        # Run tests
    python run.py demo        # Run demo analysis
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting FastAPI Backend...")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔴 Press Ctrl+C to stop\n")
    
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])


def run_frontend():
    """Start the Streamlit frontend."""
    print("🎨 Starting Streamlit Frontend...")
    print("🌐 Open: http://localhost:8501")
    print("🔴 Press Ctrl+C to stop\n")
    
    subprocess.run([
        sys.executable, "-m", "streamlit",
        "run", "frontend/app.py"
    ])


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest not installed. Install with: pip install pytest")
        return
    
    subprocess.run([sys.executable, "-m", "pytest", "-v"])


def run_demo():
    """Run a demo analysis."""
    print("🎯 Running Demo Analysis...\n")
    
    # Import backend modules
    sys.path.insert(0, str(Path(__file__).parent / "backend"))
    
    from parser import DocumentParser
    from skill_extractor import SkillExtractor
    from similarity import SimilarityEngine
    
    # Sample data
    sample_resume = """
    Senior Python Developer with 7 years of experience in Django, FastAPI, and AWS.
    Expert in Docker, Kubernetes, and microservices architecture.
    Strong background in Machine Learning with TensorFlow and PyTorch.
    Excellent communication and leadership skills.
    """
    
    sample_jd = """
    We are looking for a Senior Python Developer with 5+ years of experience.
    Required: Python, Django, AWS, Docker, Kubernetes.
    Nice to have: Machine Learning, TensorFlow.
    Must have strong communication skills.
    """
    
    print("=" * 60)
    print("DEMO: Resume-Job Matching Analysis")
    print("=" * 60)
    
    # Initialize components
    print("\n📦 Initializing components...")
    skill_extractor = SkillExtractor()
    similarity_engine = SimilarityEngine()
    
    # Extract skills
    print("🔧 Extracting skills from resume...")
    resume_skills = skill_extractor.extract_skills(sample_resume)
    
    print("🔧 Extracting skills from job description...")
    jd_skills = skill_extractor.extract_skills(sample_jd)
    
    # Compare skills
    print("⚖️  Comparing skills...")
    skill_comparison = skill_extractor.compare_skills(resume_skills, jd_skills)
    
    # Calculate similarity
    print("🧮 Calculating similarity...")
    similarity_result = similarity_engine.calculate_similarity(
        resume_text=sample_resume,
        jd_text=sample_jd,
        resume_skills=resume_skills.all_skills,
        jd_skills=jd_skills.all_skills,
        matched_skills=skill_comparison['matched_skills'],
        missing_skills=skill_comparison['missing_skills']
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Match Score: {similarity_result.match_score}%")
    print(f"🧠 Semantic Similarity: {similarity_result.semantic_similarity}%")
    print(f"🔧 Skill Match Score: {similarity_result.skill_match_score}%")
    
    print(f"\n✅ Matched Skills ({len(skill_comparison['matched_skills'])}):")
    print(", ".join(skill_comparison['matched_skills']))
    
    print(f"\n❌ Missing Skills ({len(skill_comparison['missing_skills'])}):")
    print(", ".join(skill_comparison['missing_skills']) or "None")
    
    print(f"\n📝 Explanation:")
    print(similarity_result.explanation)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Resume-Job Matching System - Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py backend     # Start the API server
  python run.py frontend    # Start the web UI
  python run.py test        # Run tests
  python run.py demo        # Run demo analysis
        """
    )
    
    parser.add_argument(
        "command",
        choices=["backend", "frontend", "test", "demo"],
        help="Command to run"
    )
    
    args = parser.parse_args()
    
    commands = {
        "backend": run_backend,
        "frontend": run_frontend,
        "test": run_tests,
        "demo": run_demo,
    }
    
    commands[args.command]()


if __name__ == "__main__":
    main()
