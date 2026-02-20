# 🚀 Quick Start Guide

Get the AI Resume-Job Matching System running in minutes!

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB free disk space (for ML models)

---

## ⚡ Installation (5 minutes)

### Step 1: Navigate to Project Directory

```bash
cd ai-resume-matcher
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Language Model

```bash
python -m spacy download en_core_web_sm
```

---

## 🎮 Running the System

### Option 1: Streamlit Web Interface (Recommended)

```bash
python run.py frontend
```

Then open: **http://localhost:8501**

**Features:**
- 📄 Upload and analyze single resumes
- 📊 Batch rank multiple candidates
- 📈 Interactive visualizations
- 📑 Download PDF reports

### Option 2: FastAPI Backend

```bash
python run.py backend
```

API available at: **http://localhost:8000**

**Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Option 3: Quick Demo

```bash
python run.py demo
```

Runs a sample analysis without uploading files.

---

## 📊 Using the Web Interface

### Single Resume Analysis

1. Go to **"Single Match"** tab
2. Upload a resume (PDF or DOCX)
3. Paste a job description
4. Click **"Analyze Match"**
5. View results with:
   - Match score gauge
   - Skill comparison
   - AI explanation

### Batch Ranking

1. Go to **"Batch Ranking"** tab
2. Paste job description
3. Upload multiple resumes
4. Click **"Rank All Resumes"**
5. View ranked list with top candidates

---

## 🔌 API Usage Examples

### Parse Resume

```bash
curl -X POST "http://localhost:8000/parse-resume" \
  -F "file=@resume.pdf"
```

### Calculate Match

```bash
curl -X POST "http://localhost:8000/calculate-match" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Python developer with 5 years...",
    "jd_text": "Looking for Python expert..."
  }'
```

### Full Analysis

```bash
curl -X POST "http://localhost:8000/full-analysis" \
  -F "resume_file=@resume.pdf" \
  -F "jd_text=Looking for Python developer..."
```

---

## 🐍 Python API Usage

```python
from backend.parser import parse_resume
from backend.skill_extractor import SkillExtractor
from backend.similarity import SimilarityEngine

# Parse resume
parsed = parse_resume("resume.pdf")

# Extract skills
extractor = SkillExtractor()
resume_skills = extractor.extract_skills(parsed['text'])

# Calculate match
engine = SimilarityEngine()
result = engine.calculate_similarity(
    resume_text=parsed['text'],
    jd_text=job_description
)

print(f"Match Score: {result.match_score}%")
print(f"Explanation: {result.explanation}")
```

---

## 🧪 Testing

Run the test suite:

```bash
python test_system.py
```

---

## 📁 Project Structure

```
ai-resume-matcher/
├── backend/           # FastAPI + Core Logic
│   ├── main.py       # API endpoints
│   ├── parser.py     # PDF/DOCX parsing
│   ├── skill_extractor.py
│   └── similarity.py
├── frontend/          # Streamlit UI
│   └── app.py
├── samples/           # Test data
│   ├── resumes/
│   └── job_descriptions/
├── requirements.txt   # Dependencies
└── README.md         # Full documentation
```

---

## 🆘 Troubleshooting

### Issue: "Module not found"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "SpaCy model not found"

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Issue: "Port already in use"

**Solution:**
```bash
# Find and kill process using port 8000
# Or use different port
uvicorn backend.main:app --port 8001
```

### Issue: "Out of memory"

**Solution:**
The ML model requires ~500MB RAM. Close other applications or use a machine with more RAM.

---

## 📞 Support

- 📖 Full documentation: [README.md](README.md)
- 🐛 Report issues: GitHub Issues
- 💬 Questions: GitHub Discussions

---

**Ready to go!** 🚀 Start with `python run.py demo` to see it in action.
