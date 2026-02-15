import streamlit as st
import time
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Resume Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# FUTURISTIC CSS
# ---------------------------------------------------
st.markdown("""
<style>
body, .main {
    background-color: #0B0F1A;
    color: white;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A, #111827);
    border-right: 1px solid #00F5FF;
}

.neon-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #00F5FF;
    text-shadow: 0 0 10px #00F5FF, 0 0 25px #8B5CF6;
}

.subtext {
    text-align: center;
    color: #9CA3AF;
    font-size: 18px;
}

.glass-card {
    background: rgba(17, 24, 39, 0.6);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(0, 245, 255, 0.3);
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.2);
}

.stButton>button {
    background: linear-gradient(90deg, #00F5FF, #8B5CF6);
    color: white;
    border-radius: 14px;
    border: none;
    padding: 12px 28px;
    font-weight: bold;
    font-size: 16px;
    box-shadow: 0 0 15px #00F5FF;
    transition: 0.3s;
}

.stButton>button:hover {
    box-shadow: 0 0 30px #8B5CF6;
    transform: scale(1.05);
}

[data-testid="stMetricValue"] {
    color: #00FF9C;
    font-size: 28px;
}

.divider {
    border-top: 1px solid #00F5FF;
    margin: 40px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="neon-title">🤖 AI Resume Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Next-Generation Resume & Job Matching Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📄 Upload Resume")
    st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📌 Job Description")
    st.text_area("Paste Job Description Here", height=200)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

center = st.columns([3,2,3])
with center[1]:
    analyze = st.button("⚡ Analyze Match")

# ---------------------------------------------------
# RESULTS SECTION
# ---------------------------------------------------
if analyze:
    with st.spinner("🤖 AI Neural Engine Processing..."):
        time.sleep(2)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.subheader("⚡ Neural Match Confidence")

    match_score = 87

    progress_bar = st.progress(0)
    for percent in range(match_score + 1):
        time.sleep(0.01)
        progress_bar.progress(percent)

    st.markdown(f"""
    <h2 style='text-align:center; color:#00FF9C;
    text-shadow:0 0 15px #00FF9C;'>
    {match_score}%
    </h2>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.subheader("📊 AI Analysis Metrics")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Match Score", "87%")

    with r2:
        st.metric("Semantic Similarity", "91%")

    with r3:
        st.metric("Skills Matched", "12 / 15")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.subheader("📈 Skill Distribution Radar")

    categories = ['Frontend', 'Backend', 'Database', 'AI/ML', 'System Design']
    values = [90, 70, 60, 75, 65]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line=dict(color='#00F5FF')
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='#0B0F1A',
            radialaxis=dict(visible=True)
        ),
        paper_bgcolor='#0B0F1A',
        font_color='white'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.subheader("🧠 AI Optimization Suggestions")

    st.markdown("""
    <div class="glass-card">
    <ul>
    <li>🔹 Add measurable achievements</li>
    <li>🔹 Include advanced TypeScript knowledge</li>
    <li>🔹 Mention CI/CD tools</li>
    <li>🔹 Add deployment experience</li>
    <li>🔹 Highlight team collaboration impact</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
