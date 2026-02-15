# AI Resume Intelligence Platform

### Explainable NLP-Based Candidate--Job Matching System

An end-to-end AI-powered hiring intelligence platform that combines
transformer-based semantic similarity with structured skill analysis to
evaluate resume--job alignment.

This project bridges modern NLP research with production-oriented ML
system design, demonstrating how embedding models can be deployed within
an interpretable, modular architecture suitable for real-world
applications.

------------------------------------------------------------------------

## 🚀 Overview

Traditional applicant tracking systems rely heavily on keyword matching,
often missing deeper semantic alignment between resumes and job
descriptions.

This platform implements:

-   Transformer-based sentence embeddings (MiniLM)
-   Structured skill extraction and categorization
-   Hybrid weighted similarity scoring
-   Explainable AI breakdown
-   Batch candidate ranking
-   Interactive SaaS-style visualization dashboard

The system is modular, scalable, and designed for extensibility.

------------------------------------------------------------------------

## 🧠 System Architecture

    Resume (PDF/DOCX)
            ↓
    Document Parser
            ↓
    Skill Extraction Engine
            ↓
    Sentence Embedding Model (MiniLM)
            ↓
    Cosine Similarity
            ↓
    Hybrid Scoring Engine
            ↓
    Explainability Layer
            ↓
    Interactive Dashboard (Streamlit)

------------------------------------------------------------------------

## 🔬 Modeling Approach

### 1️⃣ Semantic Similarity

-   Model: `all-MiniLM-L6-v2` (SentenceTransformers)
-   Embedding dimension: 384
-   Similarity metric: Cosine similarity

Why MiniLM?

-   Balanced accuracy and inference speed
-   Lightweight for CPU deployment
-   Strong benchmark performance

------------------------------------------------------------------------

### 2️⃣ Skill Extraction

-   Technical and soft skill categorization
-   Category-based grouping
-   Matched / Missing / Extra comparison

Structured signals improve interpretability and robustness.

------------------------------------------------------------------------

### 3️⃣ Hybrid Scoring Engine

Final match score combines:

-   Semantic similarity
-   Skill overlap score
-   Experience alignment heuristic

------------------------------------------------------------------------

## 📊 Experimental Evaluation

Small-scale manual evaluation:

  Resume Type      Job Description   Expected   Predicted
  ---------------- ----------------- ---------- -----------
  Python Dev       Python Backend    High       87%
  Java Dev         Python Backend    Low        34%
  Data Scientist   ML Role           High       82%

Observations:

-   Embeddings capture contextual alignment effectively.
-   Skill overlap enhances explainability.
-   Hybrid scoring reduces false positives.

------------------------------------------------------------------------

## 🧩 Core Features

-   Explainable match scoring
-   Skill gap analysis
-   Radar & gauge visualization
-   Batch ranking
-   Dark SaaS-style UI
-   CPU-friendly inference

------------------------------------------------------------------------

## 🛠 Tech Stack

  Layer           Technology
  --------------- -------------------------------
  NLP Model       SentenceTransformers (MiniLM)
  Backend         Python, FastAPI
  Frontend        Streamlit
  Visualization   Plotly
  Parsing         PyPDF2, python-docx
  ML Utilities    scikit-learn

------------------------------------------------------------------------

## ⚙ Installation

``` bash
git clone https://github.com/yourusername/ai-resume-intelligence.git
cd ai-resume-intelligence

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Run backend:

``` bash
cd backend
uvicorn main:app --reload
```

Run frontend:

``` bash
cd frontend
streamlit run app.py
```

------------------------------------------------------------------------

## 🔌 API Overview

-   POST /analyze → Analyze single resume
-   POST /batch → Rank multiple resumes
-   GET /health → Health check

------------------------------------------------------------------------

## 📁 Project Structure

    AI-Resume-Intelligence/
    │
    ├── backend/
    │   ├── main.py
    │   ├── similarity.py
    │   ├── skill_extractor.py
    │   └── parser.py
    │
    ├── frontend/
    │   └── app.py
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## ⚠️ Limitations

-   No domain-specific fine-tuning
-   Partially rule-based skill extraction
-   Small evaluation dataset
-   No fairness audit
-   Simplified experience heuristic

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Fine-tuned embeddings
-   Cross-encoder re-ranking
-   Fairness-aware scoring
-   Feedback-driven retraining
-   Dockerized deployment

------------------------------------------------------------------------

## 💼 Industry Relevance

Demonstrates:

-   End-to-end ML system design
-   Transformer integration
-   Hybrid feature engineering
-   Explainable AI
-   Modular architecture
-   Production-oriented thinking

------------------------------------------------------------------------

## 👤 Author

Ankit Kumar\
AI / ML Engineer\
LinkedIn: https://linkedin.com/in/ankit-kumar-btech-cse/   
GitHub: https://github.com/arsonic-dev
