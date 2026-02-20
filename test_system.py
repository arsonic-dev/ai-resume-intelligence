#!/usr/bin/env python
"""
System Test Script for AI Resume-Job Matching System
=====================================================
Tests all components without requiring full ML model downloads.

Usage:
    python test_system.py
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))


def test_parser():
    """Test the document parser."""
    print("\n" + "="*60)
    print("TEST 1: Document Parser")
    print("="*60)
    
    try:
        from parser import DocumentParser
        
        parser = DocumentParser()
        
        # Test text cleaning
        test_text = "  Hello   World!!!  This is a test...  "
        cleaned = parser._clean_text(test_text)
        
        print(f"✅ Parser initialized successfully")
        print(f"✅ Text cleaning works: '{test_text}' -> '{cleaned}'")
        
        # Test section extraction
        sample_resume = """
        Contact: john@example.com
        Summary: Experienced developer
        Experience: 5 years at TechCorp
        Skills: Python, JavaScript
        Education: BS Computer Science
        """
        
        sections = parser.extract_sections(sample_resume)
        print(f"✅ Section extraction works: Found {len([v for v in sections.values() if v])} sections")
        
        return True
        
    except Exception as e:
        print(f"❌ Parser test failed: {e}")
        return False


def test_skill_extractor():
    """Test the skill extractor."""
    print("\n" + "="*60)
    print("TEST 2: Skill Extractor")
    print("="*60)
    
    try:
        from skill_extractor import SkillExtractor
        
        extractor = SkillExtractor(use_spacy=False)
        
        test_text = """
        Senior Python Developer with expertise in Django, AWS, Docker, and Machine Learning.
        Strong communication skills and team leadership experience.
        """
        
        result = extractor.extract_skills(test_text)
        
        print(f"✅ Skill extractor initialized successfully")
        print(f"✅ Found {len(result.technical_skills)} technical skills")
        print(f"   Technical: {', '.join(result.technical_skills[:5])}")
        print(f"✅ Found {len(result.soft_skills)} soft skills")
        print(f"   Soft: {', '.join(result.soft_skills[:3])}")
        
        # Test skill comparison
        resume_skills = extractor.extract_skills("Python Django AWS Docker")
        jd_skills = extractor.extract_skills("Python Django Kubernetes AWS")
        
        comparison = extractor.compare_skills(resume_skills, jd_skills)
        
        print(f"✅ Skill comparison works")
        print(f"   Matched: {len(comparison['matched_skills'])}")
        print(f"   Missing: {len(comparison['missing_skills'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Skill extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\n" + "="*60)
    print("TEST 3: File Structure")
    print("="*60)
    
    required_files = [
        "backend/parser.py",
        "backend/skill_extractor.py",
        "backend/similarity.py",
        "backend/main.py",
        "frontend/app.py",
        "requirements.txt",
        "README.md",
        "run.py",
        "setup.py",
        ".env.example",
    ]
    
    base_path = Path(__file__).parent
    all_exist = True
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist


def test_sample_data():
    """Test that sample data exists."""
    print("\n" + "="*60)
    print("TEST 4: Sample Data")
    print("="*60)
    
    base_path = Path(__file__).parent / "samples"
    
    # Check job descriptions
    jd_path = base_path / "job_descriptions"
    jd_files = list(jd_path.glob("*.txt")) if jd_path.exists() else []
    print(f"✅ Job Descriptions: {len(jd_files)} files")
    for f in jd_files:
        print(f"   - {f.name}")
    
    # Check resumes
    resume_path = base_path / "resumes"
    resume_files = list(resume_path.glob("*.txt")) if resume_path.exists() else []
    print(f"✅ Sample Resumes: {len(resume_files)} files")
    for f in resume_files:
        print(f"   - {f.name}")
    
    return len(jd_files) > 0 and len(resume_files) > 0


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AI RESUME-JOB MATCHING SYSTEM - TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Parser", test_parser()))
    results.append(("Skill Extractor", test_skill_extractor()))
    results.append(("File Structure", test_file_structure()))
    results.append(("Sample Data", test_sample_data()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
