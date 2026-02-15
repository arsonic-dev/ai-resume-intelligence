"""
AI Resume-Job Matching System - Streamlit Frontend
==================================================
Interactive web interface for the AI Resume Matcher.
Provides an intuitive UI for uploading resumes, entering job descriptions,
and viewing explainable match results with visualizations.

Author: AI Resume Matcher Team
Version: 1.0.0
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import backend modules directly for standalone mode
from parser import DocumentParser, parse_resume_bytes
from skill_extractor import SkillExtractor
from similarity import SimilarityEngine

# Page configuration
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    .score-value {
        font-size: 4rem;
        font-weight: bold;
    }
    .score-label {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .skill-matched {
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .skill-missing {
        background-color: #f44336;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .skill-extra {
        background-color: #2196F3;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .explanation-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def get_parser():
    return DocumentParser()

@st.cache_resource
def get_skill_extractor():
    return SkillExtractor()

@st.cache_resource
def get_similarity_engine():
    return SimilarityEngine()

parser = get_parser()
skill_extractor = get_skill_extractor()
similarity_engine = get_similarity_engine()


def create_gauge_chart(score: float, title: str = "Match Score") -> go.Figure:
    """Create a gauge chart for match score visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1E88E5"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [0, 40], 'color': '#ffebee'},
                {'range': [40, 60], 'color': '#fff3e0'},
                {'range': [60, 80], 'color': '#e8f5e9'},
                {'range': [80, 100], 'color': '#e3f2fd'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig


def create_radar_chart(
    resume_categories: Dict[str, List[str]], 
    jd_categories: Dict[str, List[str]]
) -> go.Figure:
    """Create a radar chart comparing skill categories."""
    categories = list(set(list(resume_categories.keys()) + list(jd_categories.keys())))
    categories = [c for c in categories if c != 'Other']
    
    resume_counts = [len(resume_categories.get(cat, [])) for cat in categories]
    jd_counts = [len(jd_categories.get(cat, [])) for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=resume_counts + [resume_counts[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Resume',
        line_color='#2196F3',
        fillcolor='rgba(33, 150, 243, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=jd_counts + [jd_counts[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Job Description',
        line_color='#4CAF50',
        fillcolor='rgba(76, 175, 80, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(resume_counts, default=0), max(jd_counts, default=0)) + 1]
            )
        ),
        showlegend=True,
        title="Skill Categories Comparison",
        height=450
    )
    
    return fig


def create_skills_bar_chart(
    matched: List[str],
    missing: List[str],
    extra: List[str]
) -> go.Figure:
    """Create a bar chart showing skill comparison."""
    categories = ['Matched', 'Missing', 'Extra']
    counts = [len(matched), len(missing), len(extra)]
    colors = ['#4CAF50', '#f44336', '#2196F3']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Skills Overview",
        xaxis_title="Skill Type",
        yaxis_title="Count",
        height=350,
        showlegend=False
    )
    
    return fig


def display_skill_tags(skills: List[str], tag_type: str = "matched"):
    """Display skills as colored tags."""
    css_class = f"skill-{tag_type}"
    tags_html = " ".join([f'<span class="{css_class}">{skill}</span>' for skill in skills])
    st.markdown(tags_html, unsafe_allow_html=True)


def analyze_single_resume(resume_bytes: bytes, filename: str, jd_text: str) -> Dict:
    """Perform full analysis on a single resume."""
    # Parse resume
    parsed = parse_resume_bytes(resume_bytes, filename)
    
    if not parsed['success']:
        st.error(f"Failed to parse resume: {parsed.get('error', 'Unknown error')}")
        return None
    
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
        'resume_info': parsed,
        'resume_skills': resume_skills,
        'jd_skills': jd_skills,
        'skill_comparison': skill_comparison,
        'similarity': similarity_result
    }


def render_home_page():
    """Render the home page with overview."""
    st.markdown('<h1 class="main-header">📄 AI Resume Matcher</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Resume-Job Matching with Explainable AI</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI-Powered</h3>
            <p>Uses state-of-the-art NLP models (Sentence Transformers) for semantic understanding</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Explainable</h3>
            <p>Get clear explanations of why a candidate matches or doesn't match</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Fast & Accurate</h3>
            <p>Process multiple resumes in seconds with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("How It Works")
    
    steps = [
        ("1️⃣", "Upload Resume", "Upload a PDF or DOCX resume file"),
        ("2️⃣", "Enter Job Description", "Paste the job description text"),
        ("3️⃣", "Get Match Score", "Receive a detailed match analysis with explanations"),
        ("4️⃣", "Download Report", "Generate and download a PDF report")
    ]
    
    for emoji, title, desc in steps:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"<h2 style='text-align: center;'>{emoji}</h2>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{title}** {desc}")

    
    st.markdown("---")
    
    st.info("👈 Use the sidebar to navigate to different features: Single Match, Batch Ranking, or Skills Analysis")


def render_single_match_page():
    """Render the single resume matching page."""
    st.markdown('<h1 class="main-header">Single Resume Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a resume and compare with a job description</p>', unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Upload Resume")
        resume_file = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=['pdf', 'docx', 'doc'],
            help="Upload the candidate's resume"
        )
        
        if resume_file:
            st.success(f"✅ Uploaded: {resume_file.name}")
    
    with col2:
        st.subheader("💼 Job Description")
        jd_text = st.text_area(
            "Paste the job description here",
            height=200,
            help="Enter the full job description text"
        )
        
        # Sample job descriptions
        with st.expander("📋 Use Sample Job Description"):
            sample_jds = {
                "Python Developer": """We are looking for a Senior Python Developer with 5+ years of experience.
Required skills: Python, Django, FastAPI, SQL, Docker, AWS.
Experience with microservices and REST APIs is essential.
Knowledge of Machine Learning is a plus.""",
                
                "Data Scientist": """Seeking a Data Scientist with expertise in Python, Machine Learning, and Deep Learning.
Required: TensorFlow, PyTorch, Pandas, NumPy, SQL.
Experience with NLP and Computer Vision preferred.
Strong statistical background required.""",
                
                "DevOps Engineer": """Hiring a DevOps Engineer with strong experience in AWS, Docker, Kubernetes.
Required: CI/CD, Jenkins, Terraform, Linux, Python.
Experience with monitoring tools and cloud infrastructure.
Knowledge of security best practices."""
            }
            
            selected_sample = st.selectbox(
                "Select a sample job description",
                ["None"] + list(sample_jds.keys())
            )
            
            if selected_sample != "None":
                jd_text = sample_jds[selected_sample]
                st.info(f"Loaded sample: {selected_sample}")
    
    # Analysis button
    if resume_file and jd_text:
        if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
            with st.spinner("Analyzing resume... This may take a moment"):
                # Read resume bytes
                resume_bytes = resume_file.getvalue()
                
                # Perform analysis
                result = analyze_single_resume(resume_bytes, resume_file.name, jd_text)
                
                if result:
                    display_analysis_results(result)
    elif resume_file or jd_text:
        st.warning("⚠️ Please provide both resume and job description")


def display_analysis_results(result: Dict):
    """Display the analysis results."""
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = result['similarity'].match_score
        color = "#4CAF50" if score >= 80 else "#FFC107" if score >= 60 else "#f44336"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #666;">Match Score</h4>
            <h1 style="color: {color}; font-size: 3rem; margin: 0;">{score:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #666;">Semantic Similarity</h4>
            <h1 style="color: #2196F3; font-size: 2.5rem; margin: 0;">{result['similarity'].semantic_similarity:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        matched_count = len(result['skill_comparison']['matched_skills'])
        total_jd_skills = len(result['jd_skills'].all_skills)
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #666;">Skills Matched</h4>
            <h1 style="color: #4CAF50; font-size: 2.5rem; margin: 0;">{matched_count}/{total_jd_skills}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        missing_count = len(result['skill_comparison']['missing_skills'])
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #666;">Missing Skills</h4>
            <h1 style="color: #f44336; font-size: 2.5rem; margin: 0;">{missing_count}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for detailed analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔧 Skills Analysis", "📝 Explanation", "📋 Raw Data"])
    
    with tab1:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Gauge chart
            fig = create_gauge_chart(result['similarity'].match_score)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Radar chart
            fig = create_radar_chart(
                result['resume_skills'].skill_categories,
                result['jd_skills'].skill_categories
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Matched Skills")
            if result['skill_comparison']['matched_skills']:
                display_skill_tags(result['skill_comparison']['matched_skills'], "matched")
            else:
                st.info("No matched skills found")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.subheader("❌ Missing Skills")
            if result['skill_comparison']['missing_skills']:
                display_skill_tags(result['skill_comparison']['missing_skills'], "missing")
            else:
                st.success("No missing skills - perfect match!")
        
        with col2:
            # Skills bar chart
            fig = create_skills_bar_chart(
                result['skill_comparison']['matched_skills'],
                result['skill_comparison']['missing_skills'],
                result['skill_comparison']['extra_skills']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Skill categories comparison
            st.subheader("📊 Skill Categories")
            comparison_data = []
            all_categories = set(
                list(result['resume_skills'].skill_categories.keys()) + 
                list(result['jd_skills'].skill_categories.keys())
            )
            
            for cat in all_categories:
                if cat != 'Other':
                    comparison_data.append({
                        'Category': cat,
                        'Resume': len(result['resume_skills'].skill_categories.get(cat, [])),
                        'Job Description': len(result['jd_skills'].skill_categories.get(cat, []))
                    })
            
            if comparison_data:
                import pandas as pd
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("🤖 AI Explanation")
        st.markdown(f"""
        <div class="explanation-box">
            {result['similarity'].explanation}
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📊 Score Breakdown")
        breakdown = result['similarity'].detailed_breakdown
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Semantic Similarity",
                f"{result['similarity'].semantic_similarity:.1f}%",
                f"Weight: {breakdown['semantic_similarity']['weight']:.0%}"
            )
        
        with col2:
            st.metric(
                "Skill Match",
                f"{result['similarity'].skill_match_score:.1f}%",
                f"Weight: {breakdown['skill_similarity']['weight']:.0%}"
            )
        
        with col3:
            st.metric(
                "Experience Match",
                f"{breakdown['experience_similarity']['score']*100:.1f}%",
                f"Weight: {breakdown['experience_similarity']['weight']:.0%}"
            )
    
    with tab4:
        st.json({
            "resume_info": {
                "filename": result['resume_info']['filename'],
                "format": result['resume_info']['format'],
                "page_count": result['resume_info']['page_count'],
                "text_length": len(result['resume_info']['text'])
            },
            "skills": {
                "resume_technical": result['resume_skills'].technical_skills,
                "resume_soft": result['resume_skills'].soft_skills,
                "jd_technical": result['jd_skills'].technical_skills,
                "jd_soft": result['jd_skills'].soft_skills
            },
            "match_scores": {
                "overall": result['similarity'].match_score,
                "semantic": result['similarity'].semantic_similarity,
                "skill": result['similarity'].skill_match_score
            }
        })


def render_batch_ranking_page():
    """Render the batch resume ranking page."""
    st.markdown('<h1 class="main-header">Batch Resume Ranking</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload multiple resumes and rank them against a job description</p>', unsafe_allow_html=True)
    
    # Job description input
    st.subheader("💼 Job Description")
    jd_text = st.text_area(
        "Paste the job description here",
        height=150,
        key="batch_jd"
    )
    
    # Multiple file upload
    st.subheader("📄 Upload Resumes")
    resume_files = st.file_uploader(
        "Choose multiple PDF or DOCX files",
        type=['pdf', 'docx', 'doc'],
        accept_multiple_files=True,
        help="Upload multiple resumes to compare"
    )
    
    if resume_files:
        st.success(f"✅ Uploaded {len(resume_files)} resume(s)")
        
        # List uploaded files
        with st.expander("📋 Uploaded Files"):
            for i, file in enumerate(resume_files, 1):
                st.write(f"{i}. {file.name}")
    
    # Analyze button
    if resume_files and jd_text:
        if st.button("🔍 Rank All Resumes", type="primary", use_container_width=True):
            with st.spinner("Analyzing all resumes... This may take a moment"):
                # Extract JD skills once
                jd_skills = skill_extractor.extract_skills(jd_text)
                
                # Process each resume
                results = []
                progress_bar = st.progress(0)
                
                for i, resume_file in enumerate(resume_files):
                    resume_bytes = resume_file.getvalue()
                    parsed = parse_resume_bytes(resume_bytes, resume_file.filename)
                    
                    if parsed['success']:
                        resume_skills = skill_extractor.extract_skills(parsed['text'])
                        skill_comparison = skill_extractor.compare_skills(resume_skills, jd_skills)
                        
                        similarity_result = similarity_engine.calculate_similarity(
                            resume_text=parsed['text'],
                            jd_text=jd_text,
                            resume_skills=resume_skills.all_skills,
                            jd_skills=jd_skills.all_skills,
                            matched_skills=skill_comparison['matched_skills'],
                            missing_skills=skill_comparison['missing_skills']
                        )
                        
                        results.append({
                            'filename': resume_file.name,
                            'match_score': similarity_result.match_score,
                            'semantic_similarity': similarity_result.semantic_similarity,
                            'skill_match_score': similarity_result.skill_match_score,
                            'matched_skills': skill_comparison['matched_skills'],
                            'missing_skills': skill_comparison['missing_skills'],
                            'explanation': similarity_result.explanation
                        })
                    
                    progress_bar.progress((i + 1) / len(resume_files))
                
                # Sort by match score
                results.sort(key=lambda x: x['match_score'], reverse=True)
                
                # Display results
                display_batch_results(results)


def display_batch_results(results: List[Dict]):
    """Display batch ranking results."""
    st.markdown("---")
    st.subheader("📊 Ranking Results")
    
    # Top 3 podium
    if len(results) >= 3:
        st.markdown("### 🏆 Top Candidates")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(results) > 1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #C0C0C0 0%, #E8E8E8 100%); 
                            padding: 1.5rem; border-radius: 15px; text-align: center;">
                    <h2>🥈 2nd Place</h2>
                    <h4>{results[1]['filename']}</h4>
                    <h1 style="color: #1E88E5;">{results[1]['match_score']:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center;
                        transform: scale(1.1);">
                <h2>🥇 1st Place</h2>
                <h4>{results[0]['filename']}</h4>
                <h1 style="color: #1E88E5; font-size: 3rem;">{results[0]['match_score']:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if len(results) > 2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #CD7F32 0%, #D2691E 100%); 
                            padding: 1.5rem; border-radius: 15px; text-align: center;">
                    <h2>🥉 3rd Place</h2>
                    <h4>{results[2]['filename']}</h4>
                    <h1 style="color: #1E88E5;">{results[2]['match_score']:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Full ranking table
    st.markdown("### 📋 Complete Rankings")
    
    import pandas as pd
    
    df_data = []
    for i, r in enumerate(results, 1):
        df_data.append({
            'Rank': i,
            'Resume': r['filename'],
            'Match Score': f"{r['match_score']:.1f}%",
            'Semantic': f"{r['semantic_similarity']:.1f}%",
            'Skills': f"{r['skill_match_score']:.1f}%",
            'Matched': len(r['matched_skills']),
            'Missing': len(r['missing_skills'])
        })
    
    df = pd.DataFrame(df_data)
    
    # Color code the match score column
    def color_score(val):
        score = float(val.replace('%', ''))
        if score >= 80:
            return 'background-color: #c8e6c9'
        elif score >= 60:
            return 'background-color: #fff9c4'
        else:
            return 'background-color: #ffcdd2'
    
    styled_df = df.style.applymap(color_score, subset=['Match Score'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Detailed view for each candidate
    st.markdown("### 🔍 Detailed Analysis")
    
    for i, r in enumerate(results, 1):
        with st.expander(f"#{i} - {r['filename']} ({r['match_score']:.1f}% match)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ Matched Skills:**")
                if r['matched_skills']:
                    display_skill_tags(r['matched_skills'][:10], "matched")
                else:
                    st.info("No matched skills")
            
            with col2:
                st.markdown("**❌ Missing Skills:**")
                if r['missing_skills']:
                    display_skill_tags(r['missing_skills'][:10], "missing")
                else:
                    st.success("No missing skills")
            
            st.markdown("**📝 Explanation:**")
            st.write(r['explanation'])


def render_skills_analysis_page():
    """Render the skills analysis page."""
    st.markdown('<h1 class="main-header">Skills Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extract and analyze skills from any text</p>', unsafe_allow_html=True)
    
    text_input = st.text_area(
        "Paste text to analyze (resume or job description)",
        height=300,
        help="Enter any text to extract skills from it"
    )
    
    if text_input:
        if st.button("🔍 Extract Skills", type="primary"):
            with st.spinner("Analyzing text..."):
                skills_result = skill_extractor.extract_skills(text_input)
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔧 Technical Skills")
                    st.markdown(f"**Count: {len(skills_result.technical_skills)}**")
                    if skills_result.technical_skills:
                        display_skill_tags(skills_result.technical_skills, "matched")
                    else:
                        st.info("No technical skills found")
                
                with col2:
                    st.subheader("🤝 Soft Skills")
                    st.markdown(f"**Count: {len(skills_result.soft_skills)}**")
                    if skills_result.soft_skills:
                        display_skill_tags(skills_result.soft_skills, "extra")
                    else:
                        st.info("No soft skills found")
                
                st.markdown("---")
                
                st.subheader("📊 Skill Categories")
                
                import pandas as pd
                
                category_data = []
                for category, skills in skills_result.skill_categories.items():
                    if skills:
                        category_data.append({
                            'Category': category,
                            'Count': len(skills),
                            'Skills': ', '.join(skills[:5]) + ('...' if len(skills) > 5 else '')
                        })
                
                if category_data:
                    df = pd.DataFrame(category_data)
                    df = df.sort_values('Count', ascending=False)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Bar chart
                    fig = px.bar(
                        df,
                        x='Category',
                        y='Count',
                        title='Skills by Category',
                        color='Count',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No categorized skills found")


def main():
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.title("📄 AI Resume Matcher")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔍 Single Match", "📊 Batch Ranking", "🔧 Skills Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About**
    
    This AI-powered tool uses NLP and machine learning to:
    - Parse resumes (PDF/DOCX)
    - Extract skills automatically
    - Calculate semantic similarity
    - Provide explainable match scores
    
    **Tech Stack:**
    - FastAPI + Streamlit
    - Sentence Transformers
    - SpaCy + scikit-learn
    """)
    
    # Render selected page
    if page == "🏠 Home":
        render_home_page()
    elif page == "🔍 Single Match":
        render_single_match_page()
    elif page == "📊 Batch Ranking":
        render_batch_ranking_page()
    elif page == "🔧 Skills Analysis":
        render_skills_analysis_page()


if __name__ == "__main__":
    main()
