"""
AI Resume-Job Matching System - Streamlit Frontend
==================================================
Redesigned with premium dark SaaS aesthetic inspired by EOTY Labs.
Glassmorphism, neon accents, and smooth interactions.

Author: AI Resume Matcher Team
Version: 2.0.0 - Premium UI Update
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

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Resume Matcher | Intelligent Hiring",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hidden for premium feel
)

# ==========================================
# CUSTOM CSS - DARK GLASSMORPHISM THEME
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Base Theme Override */
    .stApp {
        background: #000000;
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Animated Background Gradient Mesh */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(ellipse at 20% 80%, rgba(16, 185, 129, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(52, 211, 153, 0.1) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(6, 78, 59, 0.2) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
        animation: pulseGradient 8s ease-in-out infinite;
    }

    @keyframes pulseGradient {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
    }

    /* Floating Navigation Bar */
    .nav-container {
        position: fixed;
        top: 1rem;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 100px;
        padding: 0.75rem 1.5rem;
        display: flex;
        gap: 2rem;
        align-items: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .nav-logo {
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        letter-spacing: -0.02em;
    }

    .nav-logo span {
        color: #10B981;
    }

    .nav-links {
        display: flex;
        gap: 1.5rem;
        font-size: 0.875rem;
        color: rgba(255, 255, 255, 0.7);
    }

    .nav-links a {
        color: inherit;
        text-decoration: none;
        transition: color 0.2s;
    }

    .nav-links a:hover {
        color: white;
    }

    .nav-cta {
        background: white;
        color: black;
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.875rem;
        font-weight: 500;
        border: none;
        cursor: pointer;
        transition: all 0.2s;
    }

    .nav-cta:hover {
        background: #10B981;
        color: white;
    }

    /* Main Content Spacing */
    .main-content {
        padding-top: 6rem;
        position: relative;
        z-index: 1;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 4rem 2rem 6rem;
        max-width: 900px;
        margin: 0 auto;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.875rem;
        color: #34D399;
        margin-bottom: 2rem;
        animation: fadeInDown 0.8s ease-out;
    }

    .hero-badge::before {
        content: "◆";
        color: #10B981;
    }

    .hero-title {
        font-size: clamp(2.5rem, 6vw, 4.5rem);
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.03em;
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }

    .hero-title .gradient-text {
        background: linear-gradient(135deg, #ffffff 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 1.25rem;
        color: rgba(255, 255, 255, 0.6);
        line-height: 1.6;
        max-width: 600px;
        margin: 0 auto 2.5rem;
        animation: fadeInUp 0.8s ease-out 0.4s both;
    }

    /* Glowing CTA Button */
    /* Glowing CTA Button */
.glow-button {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: #FFFFFF !important;   /* Force white */
    text-decoration: none !important;  /* Remove link styling */
    padding: 1rem 2rem;
    border-radius: 12px;   /* More professional than 100px */
    font-weight: 600;
    font-size: 1rem;
    border: none;
    cursor: pointer;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.35);
    transition: all 0.3s ease;
    animation: fadeInUp 0.8s ease-out 0.6s both;
}

/* Force white in all link states */
.glow-button:link,
.glow-button:visited,
.glow-button:hover,
.glow-button:active {
    color: #FFFFFF !important;
    text-decoration: none !important;
}


    .glow-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.6);
    }

    .glow-button::after {
        content: "→";
        transition: transform 0.2s;
    }

    .glow-button:hover::after {
        transform: translateX(4px);
    }

    /* Feature Cards Grid */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .feature-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #10B981, transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }

    .feature-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(16, 185, 129, 0.3);
        transform: translateY(-4px);
    }

    .feature-card:hover::before {
        opacity: 1;
    }

    .feature-icon {
        width: 48px;
        height: 48px;
        background: rgba(16, 185, 129, 0.1);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .feature-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
    }

    .feature-desc {
        font-size: 0.875rem;
        color: rgba(255, 255, 255, 0.5);
        line-height: 1.5;
    }

    /* Section Headers */
    .section-header {
        text-align: center;
        padding: 4rem 2rem 2rem;
    }

    .section-label {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.5rem 1rem;
        border-radius: 100px;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
    }

    .section-title .accent {
        color: #10B981;
    }

    /* Split Screen Layouts */
    .split-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 4rem;
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
        align-items: center;
    }

    @media (max-width: 768px) {
        .split-container {
            grid-template-columns: 1fr;
            gap: 2rem;
        }
    }

    .visual-side {
        position: relative;
        height: 400px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(0, 0, 0, 0) 100%);
        border-radius: 24px;
        border: 1px solid rgba(16, 185, 129, 0.2);
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .visual-side::before {
        content: "";
        position: absolute;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.2) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .content-side h3 {
        font-size: 1.875rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1.5rem;
    }

    .content-side p {
        color: rgba(255, 255, 255, 0.6);
        line-height: 1.7;
        margin-bottom: 1.5rem;
    }

    /* Stats/Metrics Cards */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        max-width: 1000px;
        margin: 0 auto;
        padding: 2rem;
    }

    @media (max-width: 768px) {
        .metrics-row {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    .metric-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 0.875rem;
        color: rgba(255, 255, 255, 0.5);
        margin-top: 0.5rem;
    }

    /* Process Steps */
    .process-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 2rem;
    }

    .process-step {
        display: flex;
        gap: 2rem;
        margin-bottom: 3rem;
        position: relative;
    }

    .process-step::before {
        content: "";
        position: absolute;
        left: 24px;
        top: 60px;
        width: 2px;
        height: calc(100% + 1rem);
        background: linear-gradient(180deg, #10B981 0%, transparent 100%);
    }

    .process-step:last-child::before {
        display: none;
    }

    .step-number {
        width: 50px;
        height: 50px;
        background: rgba(16, 185, 129, 0.1);
        border: 2px solid #10B981;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: #10B981;
        flex-shrink: 0;
        position: relative;
        z-index: 1;
    }

    .step-content h4 {
        font-size: 1.25rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
    }

    .step-content p {
        color: rgba(255, 255, 255, 0.5);
        line-height: 1.6;
    }

    /* Dark Section (Solutions) */
    .dark-section {
        background: linear-gradient(180deg, rgba(6, 78, 59, 0.3) 0%, rgba(0, 0, 0, 0) 100%);
        margin: 4rem 0;
        padding: 4rem 2rem;
        border-top: 1px solid rgba(16, 185, 129, 0.2);
        border-bottom: 1px solid rgba(16, 185, 129, 0.2);
    }

    /* Skills Tags */
    .skill-tag {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
        transition: all 0.2s;
    }

    .skill-matched {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .skill-missing {
        background: rgba(239, 68, 68, 0.2);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .skill-extra {
        background: rgba(59, 130, 246, 0.2);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Scroll Reveal Animation Classes */
    .reveal {
        opacity: 0;
        transform: translateY(30px);
        transition: all 0.8s ease-out;
    }

    .reveal.active {
        opacity: 1;
        transform: translateY(0);
    }

    /* Custom Streamlit Overrides */
    .stButton > button {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;   /* 100px looks too pill-like */
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.25) !important;
    transition: all 0.2s ease-in-out !important;
}


    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.5) !important;
    }

    .stTextArea > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 2px dashed rgba(16, 185, 129, 0.3) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255, 255, 255, 0.03);
        padding: 0.5rem;
        border-radius: 100px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255, 255, 255, 0.6);
        border-radius: 100px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(16, 185, 129, 0.2) !important;
        color: #10B981 !important;
    }

    /* Dataframe Styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    /* Success/Info/Warning Boxes */
    .stSuccess, .stInfo, .stWarning {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    .stSuccess { border-left: 4px solid #10B981 !important; }
    .stInfo { border-left: 4px solid #3B82F6 !important; }
    .stWarning { border-left: 4px solid #F59E0B !important; }
</style>

<script>
// Scroll reveal animation
document.addEventListener('DOMContentLoaded', function() {
    const reveals = document.querySelectorAll('.reveal');

    function reveal() {
        reveals.forEach(element => {
            const windowHeight = window.innerHeight;
            const elementTop = element.getBoundingClientRect().top;
            const elementVisible = 150;

            if (elementTop < windowHeight - elementVisible) {
                element.classList.add('active');
            }
        });
    }

    window.addEventListener('scroll', reveal);
    reveal();
});
</script>
""", unsafe_allow_html=True)


# ==========================================
# INITIALIZE COMPONENTS
# ==========================================
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


# ==========================================
# NAVIGATION COMPONENT
# ==========================================
def render_navigation():
    st.markdown("""
    <div class="nav-container">
        <div class="nav-logo">RESUME<span>MATCHER</span></div>
        <div class="nav-links">
            <a href="#home">Overview</a>
            <a href="#analyze">Analyze</a>
            <a href="#batch">Batch</a>
            <a href="#skills">Skills</a>
        </div>
        <button class="nav-cta" onclick="document.getElementById('analyze').scrollIntoView({behavior: 'smooth'})">Get Started</button>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# HERO SECTION
# ==========================================
def render_hero():
    st.markdown("""
    <div class="hero-container" id="home">
        <div class="hero-badge">AI-Powered Hiring Intelligence</div>
        <h1 class="hero-title">
            Verify to Trust<br>
            <span class="gradient-text">AI Resume Matching</span>
        </h1>
        <p class="hero-subtitle">
            Introducing explainable AI for recruitment. Verify candidate-job fit 
            with semantic analysis, skill extraction, and transparent scoring.
        </p>
        <a href="#analyze" class="glow-button">Start Analysis</a>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# FEATURES GRID
# ==========================================
def render_features():
    features = [
        ("◉", "AI-Powered", "State-of-the-art NLP models for semantic understanding of resumes and job descriptions."),
        ("⌕", "Explainable", "Clear explanations of matching scores with detailed breakdowns and reasoning."),
        ("⚡", "Lightning Fast", "Process multiple resumes in seconds with high-accuracy batch ranking."),
        ("⚔", "Privacy First", "Local processing ensures your candidate data never leaves your infrastructure.")
    ]

    st.markdown('<div class="features-grid">', unsafe_allow_html=True)
    for icon, title, desc in features:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# CHART CREATION FUNCTIONS
# ==========================================
def create_gauge_chart(score: float, title: str = "Match Score") -> go.Figure:
    """Create a futuristic gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': 'white', 'family': 'Inter'}},
        number={'font': {'size': 48, 'color': '#10B981', 'family': 'Inter'}, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': '#0B0F1A'},
            'bar': {'color': "#10B981", 'thickness': 0.75},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [40, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [60, 80], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.4)'}
            ],
            'threshold': {
                'line': {'color': "#10B981", 'width': 2},
                'thickness': 0.8,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': 'Inter', 'color': 'white'}
    )

    return fig


def create_radar_chart(resume_categories: Dict[str, List[str]],
                       jd_categories: Dict[str, List[str]]) -> go.Figure:
    """Create a dark-themed radar chart."""
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
        line_color='#10B981',
        fillcolor='rgba(16, 185, 129, 0.2)',
        line={'width': 2}
    ))

    fig.add_trace(go.Scatterpolar(
        r=jd_counts + [jd_counts[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Job Description',
        line_color='#3B82F6',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line={'width': 2}
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(resume_counts, default=0), max(jd_counts, default=0)) + 1],
                tickfont={'color': 'rgba(255,255,255,0.5)'},
                gridcolor='rgba(255,255,255,0.1)'
            ),
            angularaxis=dict(
                tickfont={'color': 'white', 'size': 12},
                gridcolor='rgba(255,255,255,0.1)'
            ),
            bgcolor='rgba(255,255,255,0.03)'
        ),
        showlegend=True,
        legend=dict(
            font={'color': 'white'},
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0
        ),
        title=dict(
            text="Skill Categories",
            font={'color': 'white', 'size': 16},
            x=0.5
        ),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def create_skills_bar_chart(matched: List[str], missing: List[str], extra: List[str]) -> go.Figure:
    """Create a dark-themed bar chart."""
    categories = ['Matched', 'Missing', 'Extra']
    counts = [len(matched), len(missing), len(extra)]
    colors = ['#10B981', '#EF4444', '#3B82F6']

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='auto',
            textfont={'color': 'white', 'size': 14},
            marker=dict(
                line=dict(color='rgba(255,255,255,0.1)', width=1)
            )
        )
    ])

    fig.update_layout(
        title=dict(
            text="Skills Overview",
            font={'color': 'white', 'size': 16},
            x=0.5
        ),
        xaxis=dict(
            tickfont={'color': 'white'},
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            tickfont={'color': 'rgba(255,255,255,0.5)'},
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font={'family': 'Inter'}
    )

    return fig


# ==========================================
# ANALYSIS FUNCTIONS
# ==========================================
def analyze_single_resume(resume_bytes: bytes, filename: str, jd_text: str) -> Dict:
    """Perform full analysis on a single resume."""
    parsed = parse_resume_bytes(resume_bytes, filename)

    if not parsed['success']:
        st.error(f"Failed to parse resume: {parsed.get('error', 'Unknown error')}")
        return None

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

    return {
        'resume_info': parsed,
        'resume_skills': resume_skills,
        'jd_skills': jd_skills,
        'skill_comparison': skill_comparison,
        'similarity': similarity_result
    }


# ==========================================
# DISPLAY FUNCTIONS
# ==========================================
def display_skill_tags(skills: List[str], tag_type: str = "matched"):
    """Display skills as styled tags."""
    css_class = f"skill-tag skill-{tag_type}"
    tags_html = "".join([f'<span class="{css_class}">{skill}</span>' for skill in skills])
    st.markdown(f'<div style="line-height: 2;">{tags_html}</div>', unsafe_allow_html=True)


def display_analysis_results(result: Dict):
    """Display analysis results with premium styling."""
    score = result['similarity'].match_score

    # Metrics Row
    st.markdown("""
    <div class="section-header" style="padding-top: 2rem;">
        <div class="section-label">Analysis Results</div>
        <h2 class="section-title">Match <span class="accent">Verification</span></h2>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    metrics = [
        (f"{score:.1f}%", "Match Score", "#10B981" if score >= 60 else "#EF4444"),
        (f"{result['similarity'].semantic_similarity:.1f}%", "Semantic", "#3B82F6"),
        (f"{len(result['skill_comparison']['matched_skills'])}", "Skills Matched", "#10B981"),
        (f"{len(result['skill_comparison']['missing_skills'])}", "Gaps", "#EF4444")
    ]

    for col, (value, label, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="background: linear-gradient(135deg, {color} 0%, #ffffff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Tabs for detailed analysis
    tab1, tab2, tab3, tab4 = st.tabs(["🗒 Overview", "⚙ Skills", "ℹ Explanation", "⧉ Data"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = create_gauge_chart(result['similarity'].match_score)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with col2:
            fig = create_radar_chart(
                result['resume_skills'].skill_categories,
                result['jd_skills'].skill_categories
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✓ Matched Skills")
            if result['skill_comparison']['matched_skills']:
                display_skill_tags(result['skill_comparison']['matched_skills'], "matched")
            else:
                st.info("No matched skills found", icon="⚠")

            st.markdown("#### ↔ Skill Gaps")
            if result['skill_comparison']['missing_skills']:
                display_skill_tags(result['skill_comparison']['missing_skills'], "missing")
            else:
                st.success("No missing skills - perfect match!")

        with col2:
            fig = create_skills_bar_chart(
                result['skill_comparison']['matched_skills'],
                result['skill_comparison']['missing_skills'],
                result['skill_comparison']['extra_skills']
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with tab3:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); 
                    border-radius: 16px; padding: 2rem; margin: 1rem 0;">
            <h4 style="color: #10B981; margin-bottom: 1rem;">⦿ AI Explanation</h4>
            <p style="color: rgba(255,255,255,0.8); line-height: 1.7; font-size: 1rem;">
                {explanation}
            </p>
        </div>
        """.format(explanation=result['similarity'].explanation), unsafe_allow_html=True)

        breakdown = result['similarity'].detailed_breakdown
        cols = st.columns(3)
        with cols[0]:
            st.metric("Semantic", f"{result['similarity'].semantic_similarity:.1f}%",
                      f"Weight: {breakdown['semantic_similarity']['weight']:.0%}")
        with cols[1]:
            st.metric("Skills", f"{result['similarity'].skill_match_score:.1f}%",
                      f"Weight: {breakdown['skill_similarity']['weight']:.0%}")
        with cols[2]:
            st.metric("Experience", f"{breakdown['experience_similarity']['score'] * 100:.1f}%",
                      f"Weight: {breakdown['experience_similarity']['weight']:.0%}")

    with tab4:
        st.json({
            "resume": {
                "filename": result['resume_info']['filename'],
                "pages": result['resume_info']['page_count'],
                "text_length": len(result['resume_info']['text'])
            },
            "skills": {
                "matched": len(result['skill_comparison']['matched_skills']),
                "missing": len(result['skill_comparison']['missing_skills']),
                "extra": len(result['skill_comparison']['extra_skills'])
            },
            "scores": {
                "overall": result['similarity'].match_score,
                "semantic": result['similarity'].semantic_similarity,
                "skill": result['similarity'].skill_match_score
            }
        })


# ==========================================
# PAGE RENDERERS
# ==========================================
def render_single_match():
    """Render the single resume matching page."""
    st.markdown("""
    <div class="section-header" id="analyze">
        <div class="section-label">Single Analysis</div>
        <h2 class="section-title">Analyze <span class="accent">Resume</span></h2>
        <p style="color: rgba(255,255,255,0.5); max-width: 600px; margin: 0 auto;">
            Upload a candidate resume and compare it against a job description 
            to get detailed matching insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⇪ Resume Upload")
        resume_file = st.file_uploader(
            "Drop resume here (PDF/DOCX)",
            type=['pdf', 'docx', 'doc'],
            help="Upload candidate resume"
        )
        if resume_file:
            st.success(f"✓ {resume_file.name}")

    with col2:
        st.markdown("### ⧉ Job Description")

        # Sample selector
        samples = {
            "Python Developer": """Senior Python Developer role requiring 5+ years experience.
Skills: Python, Django, FastAPI, PostgreSQL, Docker, AWS, Microservices, REST APIs.""",
            "Data Scientist": """Data Scientist position. Required: Python, Machine Learning, 
TensorFlow, PyTorch, SQL, Statistics, NLP, Computer Vision.""",
            "DevOps Engineer": """DevOps Engineer. Must have: AWS, Docker, Kubernetes, 
CI/CD, Jenkins, Terraform, Linux, Python, Security."""
        }

        sample = st.selectbox("Load sample", ["Custom"] + list(samples.keys()))
        jd_text = samples[sample] if sample != "Custom" else ""

        jd_text = st.text_area(
            "Job description text",
            value=jd_text,
            height=200
        )

    if resume_file and jd_text:
        if st.button("⌕ Analyze Match", use_container_width=True):
            with st.spinner("Processing..."):
                result = analyze_single_resume(
                    resume_file.getvalue(),
                    resume_file.name,
                    jd_text
                )
                if result:
                    display_analysis_results(result)


def render_batch_ranking():
    """Render batch ranking with premium UI."""
    st.markdown("""
    <div class="section-header" id="batch">
        <div class="section-label">Batch Processing</div>
        <h2 class="section-title">Rank <span class="accent">Candidates</span></h2>
    </div>
    """, unsafe_allow_html=True)

    jd_text = st.text_area("Job Description", height=100, key="batch_jd")
    files = st.file_uploader(
        "Upload multiple resumes",
        type=['pdf', 'docx'],
        accept_multiple_files=True
    )

    if files and jd_text:
        st.success(f"📎 {len(files)} files ready")

        if st.button("⊞ Rank All Candidates", use_container_width=True):
            with st.spinner("Analyzing batch..."):
                jd_skills = skill_extractor.extract_skills(jd_text)
                results = []
                progress = st.progress(0)

                for i, file in enumerate(files):
                    parsed = parse_resume_bytes(file.getvalue(), file.name)
                    if parsed['success']:
                        r_skills = skill_extractor.extract_skills(parsed['text'])
                        comparison = skill_extractor.compare_skills(r_skills, jd_skills)
                        sim = similarity_engine.calculate_similarity(
                            parsed['text'], jd_text,
                            r_skills.all_skills, jd_skills.all_skills,
                            comparison['matched_skills'], comparison['missing_skills']
                        )
                        results.append({
                            'name': file.name,
                            'score': sim.match_score,
                            'matched': len(comparison['matched_skills']),
                            'explanation': sim.explanation
                        })
                    progress.progress((i + 1) / len(files))

                results.sort(key=lambda x: x['score'], reverse=True)

                # Display leaderboard
                st.markdown("### ♕ Candidate Rankings")
                for i, r in enumerate(results[:5], 1):
                    medal = "①" if i == 1 else "②" if i == 2 else "③" if i == 3 else f"#{i}"
                    color = "#10B981" if r['score'] >= 80 else "#F59E0B" if r['score'] >= 60 else "#EF4444"

                    with st.expander(f"{medal} {r['name']} — {r['score']:.1f}%"):
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; 
                                        background: rgba(255,255,255,0.05); border-radius: 12px;">
                                <div style="font-size: 2rem; font-weight: 800; color: {color};">
                                    {r['score']:.0f}%
                                </div>
                                <div style="font-size: 0.875rem; color: rgba(255,255,255,0.5);">
                                    {r['matched']} skills
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        with cols[1]:
                            st.write(r['explanation'])


def render_skills_analysis():
    """Render skills extraction page."""
    st.markdown("""
    <div class="section-header" id="skills">
        <div class="section-label">Skills Engine</div>
        <h2 class="section-title">Extract <span class="accent">Skills</span></h2>
    </div>
    """, unsafe_allow_html=True)

    text = st.text_area("Paste text to analyze", height=250)

    if text and st.button("⌬ Extract Skills", use_container_width=True):
        with st.spinner("Processing..."):
            skills = skill_extractor.extract_skills(text)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ⚙ Technical")
                if skills.technical_skills:
                    display_skill_tags(skills.technical_skills, "matched")
                else:
                    st.info("No technical skills detected")

            with col2:
                st.markdown("### ⌘ Soft Skills")
                if skills.soft_skills:
                    display_skill_tags(skills.soft_skills, "extra")
                else:
                    st.info("No soft skills detected")

            # Categories
            st.markdown("### ⟡ Categories")
            import pandas as pd
            cat_data = [{'Category': k, 'Count': len(v)}
                        for k, v in skills.skill_categories.items() if v and k != 'Other']
            if cat_data:
                df = pd.DataFrame(cat_data).sort_values('Count', ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)


# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    render_navigation()

    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # Hero Section
    render_hero()
    render_features()

    # Analysis Section
    st.markdown('<div style="margin-top: 4rem;">', unsafe_allow_html=True)
    render_single_match()
    st.markdown('</div>', unsafe_allow_html=True)

    # Batch Section
    st.markdown('<div class="dark-section">', unsafe_allow_html=True)
    render_batch_ranking()
    st.markdown('</div>', unsafe_allow_html=True)

    # Skills Section
    render_skills_analysis()

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 4rem;">
        <p style="color: rgba(255,255,255,0.4); font-size: 0.875rem;">
            AI Resume Matcher • Explainable AI for Intelligent Hiring
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()