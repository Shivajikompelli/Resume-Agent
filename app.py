"""
app.py — Resume Screening Agent — Streamlit Web Application

Run with:  streamlit run app.py
"""

import os
import json
import tempfile
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screening Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #1B2A4A 0%, #2E5FA3 100%);
    padding: 2rem; border-radius: 12px; margin-bottom: 2rem; color: white;
  }
  .metric-card {
    background: #F3F5F8; border-radius: 8px; padding: 1rem;
    border-left: 4px solid #2E5FA3; margin-bottom: 0.5rem;
  }
  .strongly-eligible { border-left-color: #1A6B3C !important; background: #D4EDDA !important; }
  .eligible          { border-left-color: #1A7B8C !important; background: #D4EEF2 !important; }
  .partially-eligible{ border-left-color: #B7680E !important; background: #FDE9C3 !important; }
  .not-eligible      { border-left-color: #C0392B !important; background: #FAD7D3 !important; }
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600;
  }
  .badge-green  { background: #D4EDDA; color: #1A6B3C; }
  .badge-red    { background: #FAD7D3; color: #C0392B; }
  .badge-amber  { background: #FDE9C3; color: #B7680E; }
  .badge-blue   { background: #D6E4F7; color: #2E5FA3; }
  .section-header {
    font-size: 1.1rem; font-weight: 700; color: #1B2A4A;
    border-bottom: 2px solid #2E5FA3; padding-bottom: 6px; margin: 1.5rem 0 0.8rem;
  }
  .skill-tag {
    display: inline-block; background: #D6E4F7; color: #2E5FA3;
    padding: 3px 10px; border-radius: 12px; font-size: 13px;
    margin: 2px; font-weight: 500;
  }
  .skill-tag.missing { background: #FAD7D3; color: #C0392B; }
  .skill-tag.matched { background: #D4EDDA; color: #1A6B3C; }
  .step-bar {
    background: #e8f0fb; border-radius: 6px; padding: 0.6rem 1rem;
    margin: 3px 0; font-size: 13px; color: #1B2A4A;
  }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1 style="margin:0;font-size:2rem;"> Resume Screening Agent</h1>
  <p style="margin:4px 0 0;opacity:0.85;">
    Powered by <strong>Groq API</strong> (Free) · Llama 3 70B · Deterministic Scoring · EEOC Compliant
  </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar — Configuration ────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get your free key at https://console.groq.com",
    )

    model = st.selectbox(
        "Groq Model",
        options=["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0,
        help="llama3-70b gives best results. Use 8b for faster response.",
    )

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        os.environ["GROQ_MODEL"]   = model
        st.success(" API key set")
    else:
        st.warning(" Enter your Groq API key to start")
        st.markdown("[Get free key →](https://console.groq.com)")

    st.divider()
    st.markdown("###  How It Works")
    steps = [
        "1️  Upload resume (PDF/DOCX/TXT)",
        "2️  Compliance gate strips PII",
        "3️  Groq LLM extracts candidate data",
        "4️  Groq LLM parses job description",
        "5️  3-layer skill matching engine",
        "6️  Deterministic scoring (no LLM)",
        "7️  Full explainability report",
    ]
    for step in steps:
        st.markdown(f'<div class="step-bar">{step}</div>', unsafe_allow_html=True)

    st.divider()
    st.caption(" Protected attributes are redacted before any AI processing")


# ── Main layout ────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("###  Resume")
    upload_tab, text_tab = st.tabs(["Upload File", "Paste Text"])

    resume_path = None
    resume_text_input = ""

    with upload_tab:
        uploaded = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
        )
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded.name).suffix,
            )
            tmp.write(uploaded.read())
            tmp.close()
            resume_path = tmp.name
            st.success(f" {uploaded.name} ({uploaded.size/1024:.1f} KB)")

    with text_tab:
        resume_text_input = st.text_area(
            "Paste resume content",
            height=280,
            placeholder="Paste the full text of the resume here...",
            label_visibility="collapsed",
        )

with col_right:
    st.markdown("###  Job Description")
    jd_text = st.text_area(
        "Paste job description",
        height=340,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
    )

# ── Sample data buttons ────────────────────────────────────────────
with st.expander(" Load sample data for demo"):
    if st.button("Load Sample Resume + JD"):
        st.session_state["sample_jd"] = """
Senior Machine Learning Engineer

We are looking for a Senior ML Engineer to join our AI Platform team.

Required Skills:
- Python (must have)
- Machine Learning / Deep Learning (must have)
- PyTorch or TensorFlow
- MLOps (Kubeflow, MLflow, or similar)
- SQL and data pipeline experience
- Docker and Kubernetes

Preferred Skills:
- LLM fine-tuning and prompt engineering
- Distributed training
- AWS SageMaker or Google Vertex AI
- Experience with transformers (HuggingFace)

Requirements:
- Minimum 5 years of software engineering experience
- Minimum 3 years of ML/AI experience
- Bachelor's degree in Computer Science, Statistics, or related field (or equivalent experience)
- AWS Certified Machine Learning – Specialty is a plus

Responsibilities:
- Design and deploy ML models at scale
- Build MLOps pipelines for training and inference
- Collaborate with data scientists and engineers
- Mentor junior team members
        """
        st.session_state["sample_resume"] = """
Alex Johnson
alex.johnson@email.com | +1-555-0123 | San Francisco, CA
LinkedIn: linkedin.com/in/alexjohnson

SUMMARY
ML Engineer with 7 years of experience building and deploying large-scale machine learning systems.

SKILLS
Python, PyTorch, TensorFlow, Scikit-learn, Pandas, NumPy, SQL, Docker, Kubernetes,
MLflow, AWS, GCP, HuggingFace, LangChain, Apache Spark, Kafka, Git, Linux

EXPERIENCE

Senior ML Engineer — TechCorp AI (2021–Present) [3 years]
- Built end-to-end MLOps platform using Kubeflow and MLflow reducing deployment time by 60%
- Fine-tuned LLMs (Llama 2, Mistral) for domain-specific NLP tasks
- Deployed real-time inference serving 10M+ predictions/day on AWS SageMaker

ML Engineer — DataDriven Inc (2019–2021) [2 years]
- Developed fraud detection models (XGBoost, neural networks) saving $2M annually
- Built distributed training pipelines with PyTorch on AWS EC2 clusters

Software Engineer — StartupCo (2017–2019) [2 years]
- Developed data pipelines in Python and Apache Spark processing 5TB/day
- Designed PostgreSQL and Redis schemas for high-throughput applications

EDUCATION
Master of Science in Computer Science — Stanford University, 2017
Bachelor of Science in Statistics — UC Berkeley, 2015

CERTIFICATIONS
AWS Certified Machine Learning – Specialty
Google Professional Machine Learning Engineer
        """
        st.rerun()

if "sample_jd" in st.session_state:
    jd_text = st.session_state["sample_jd"]
    resume_text_input = st.session_state["sample_resume"]

# ── Evaluate button ────────────────────────────────────────────────
st.divider()
btn_col, info_col = st.columns([1, 3])
with btn_col:
    evaluate_btn = st.button(
        " Evaluate Candidate",
        type="primary",
        use_container_width=True,
        disabled=not api_key,
    )
with info_col:
    if not api_key:
        st.info("Enter your Groq API key in the sidebar to enable evaluation")


# ── Run evaluation ─────────────────────────────────────────────────
if evaluate_btn:
    # Validate inputs
    has_resume = resume_path or resume_text_input.strip()
    if not has_resume:
        st.error("Please upload a resume or paste resume text.")
        st.stop()
    if not jd_text.strip():
        st.error("Please paste a job description.")
        st.stop()

    # Update env with sidebar values
    os.environ["GROQ_API_KEY"] = api_key
    os.environ["GROQ_MODEL"]   = model

    # Import here so env vars are set before module load
    import importlib, config as cfg
    cfg.GROQ_API_KEY = api_key
    cfg.GROQ_MODEL   = model

    from agent.orchestrator import EvaluationPipeline

    progress_bar = st.progress(0)
    status_text  = st.empty()

    def on_progress(msg: str, pct: int):
        progress_bar.progress(pct)
        status_text.markdown(f"⏳ **{msg}**")

    try:
        pipeline = EvaluationPipeline()

        with st.spinner("Running evaluation pipeline..."):
            if resume_path:
                report = pipeline.evaluate(resume_path, jd_text, on_progress)
            else:
                report = pipeline.evaluate_from_text(
                    resume_text_input, jd_text, on_progress
                )

        progress_bar.progress(100)
        status_text.markdown(" **Evaluation complete!**")

        # Store in session state
        st.session_state["report"] = report

    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        st.stop()


# ── Display Results ────────────────────────────────────────────────
if "report" in st.session_state:
    report = st.session_state["report"]

    if report.get("status") == "FAILED":
        st.error(f" Evaluation Error: {report.get('error', 'Unknown error')}")
        st.stop()

    scoring = report.get("scoring", {})
    expl    = report.get("explainability", {})
    skills  = report.get("skills_analysis", {})
    exp_a   = report.get("experience_analysis", {})
    edu_a   = report.get("education_analysis", {})
    profile = report.get("candidate_profile", {})
    job_req = report.get("job_requirements", {})
    comp    = report.get("compliance", {})

    cat     = scoring.get("eligibility_category", "UNKNOWN")
    score   = scoring.get("composite_score", 0)

    # ── Decision Banner ────────────────────────────────────────────
    st.divider()
    cat_css = {
        "STRONGLY_ELIGIBLE": ("strongly-eligible", "✅", "#1A6B3C"),
        "ELIGIBLE":          ("eligible",           "✔️",  "#1A7B8C"),
        "PARTIALLY_ELIGIBLE":("partially-eligible", "⚠️",  "#B7680E"),
        "NOT_ELIGIBLE":      ("not-eligible",       "❌",  "#C0392B"),
    }.get(cat, ("metric-card", "ℹ️", "#4A5568"))

    st.markdown(f"""
    <div class="metric-card {cat_css[0]}" style="padding:1.5rem;border-radius:10px;">
      <div style="font-size:2rem;font-weight:800;color:{cat_css[2]};">
        {cat_css[1]} {cat.replace("_", " ").title()}
      </div>
      <div style="font-size:1.1rem;margin-top:4px;color:#333;">
        Composite Score: <strong>{score}/100</strong> &nbsp;|&nbsp;
        Confidence: <strong>{expl.get('confidence_score', 0):.0%}</strong> &nbsp;|&nbsp;
        Evaluation ID: <code>{report.get('evaluation_id','')[:8]}...</code>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"> {expl.get('reasoning_summary', '')}")

    # ── Score breakdown ────────────────────────────────────────────
    st.markdown('<div class="section-header"> Score Breakdown</div>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    metrics = [
        (sc1, "Required Skills",  scoring.get("required_skill_score",0),  40, "blue"),
        (sc2, "Experience",       scoring.get("experience_score",0),       25, "teal"),
        (sc3, "Education",        scoring.get("education_score",0),        20, "purple"),
        (sc4, "Preferred Skills", scoring.get("preferred_skill_score",0),  10, "amber"),
        (sc5, "Certifications",   scoring.get("certification_bonus",0),     5, "green"),
    ]
    for col, label, val, mx, color in metrics:
        with col:
            st.metric(label, f"{val}/{mx}")
            st.progress(val / mx if mx else 0)

    # ── Hard disqualifier warning ──────────────────────────────────
    if scoring.get("hard_disqualifier_triggered"):
        st.error(f" Hard Disqualifier: {scoring.get('hard_disqualifier_reason')}")

    # ── Tabs for details ───────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🎯 Skills", "💼 Experience", "🎓 Education",
        "👤 Profile", "🛡️ Compliance", "📄 JSON Report"
    ])

    with t1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"####  Matched Required ({skills.get('required_match_pct',0):.0f}%)")
            matched = skills.get("required_skills_matched", [])
            if matched:
                html = " ".join(f'<span class="skill-tag matched">{s}</span>' for s in matched)
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("None matched")

            if skills.get("preferred_skills_matched"):
                st.markdown(f"####  Preferred Skills Matched")
                html = " ".join(f'<span class="skill-tag">{s}</span>'
                                for s in skills["preferred_skills_matched"])
                st.markdown(html, unsafe_allow_html=True)

        with col2:
            st.markdown(f"####  Missing Required ({len(skills.get('required_skills_missing',[]))} skills)")
            missing = skills.get("required_skills_missing", [])
            if missing:
                html = " ".join(f'<span class="skill-tag missing">{s}</span>' for s in missing)
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.success("No required skills missing!")

        if skills.get("similarity_pairs"):
            st.markdown("####  Skill Similarity Pairs")
            pairs_data = []
            for p in skills["similarity_pairs"]:
                pairs_data.append({
                    "JD Skill":        p["jd_skill"],
                    "Candidate Skill": p["candidate_skill"],
                    "Score":           f"{p['score']:.3f}",
                    "Method":          p.get("method", "—"),
                })
            st.dataframe(pairs_data, use_container_width=True)

    with t2:
        cand_yrs = exp_a.get("candidate_years", 0)
        req_yrs  = exp_a.get("required_years",  0)
        gap      = exp_a.get("experience_gap_years")
        meets    = exp_a.get("meets_requirement", False)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Candidate Experience", f"{cand_yrs:.1f} years")
        mc2.metric("Required Experience",  f"{req_yrs:.0f} years")
        mc3.metric("Gap", f"{gap:.1f} years" if gap else "None ✅")

        if not meets and gap:
            st.warning(f" {gap:.1f} years below the minimum requirement")
        elif meets:
            st.success(" Experience requirement met")

        if profile.get("experience"):
            st.markdown("#### Work History")
            for exp in profile["experience"]:
                st.markdown(
                    f"**{exp.get('title')}** @ {exp.get('company')}  "
                    f"— {exp.get('years', 0):.1f} years  "
                    f"({exp.get('start_date','?')} – {exp.get('end_date','?')})"
                )

    with t3:
        highest = edu_a.get("highest_level")
        req_edu = job_req.get("required_edu_level")
        meets_e = edu_a.get("meets_requirement", True)

        ec1, ec2 = st.columns(2)
        ec1.metric("Candidate's Highest", highest or "Not specified")
        ec2.metric("Required Level",       req_edu  or "Not specified")

        if meets_e:
            st.success(" Education requirement met")
        else:
            st.warning(f" Education below required level ({req_edu})")

        if edu_a.get("relevant_degrees"):
            st.markdown("#### Degrees")
            for deg in edu_a["relevant_degrees"]:
                st.markdown(f"• {deg}")

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Contact Info")
            contact = profile.get("contact", {})
            st.markdown(f"**Name:** {profile.get('name', '—')}")
            st.markdown(f"**Email:** {contact.get('email', '—')}")
            st.markdown(f"**Phone:** {contact.get('phone', '—')}")
            st.markdown(f"**Location:** {profile.get('location', '—')}")

        with c2:
            st.markdown("#### All Skills")
            if profile.get("skills"):
                html = " ".join(f'<span class="skill-tag">{s}</span>'
                                for s in profile["skills"])
                st.markdown(html, unsafe_allow_html=True)

        if profile.get("certifications"):
            st.markdown("#### Certifications")
            for c in profile["certifications"]:
                st.markdown(f"• {c}")

        if expl.get("strengths"):
            st.markdown("####  Strengths")
            for s in expl["strengths"]:
                st.markdown(f" {s}")

        if expl.get("gaps"):
            st.markdown("####  Gaps")
            for g in expl["gaps"]:
                st.markdown(f" {g}")

    with t5:
        st.markdown("####  Compliance Gate Results")
        redacted = comp.get("pii_attributes_redacted", [])
        if redacted:
            st.warning(f"Protected attributes detected and redacted: **{', '.join(redacted)}**")
        else:
            st.success(" No protected attributes detected in resume")

        jd_issues = report.get("compliance", {}).get("jd_issues", [])
        if jd_issues:
            st.error(f" Job description issues: {', '.join(jd_issues)}")
        else:
            st.success(" No discriminatory language detected in job description")

        st.markdown("#### Audit Record")
        st.json({
            "evaluation_id":  report.get("evaluation_id"),
            "evaluated_at":   report.get("evaluated_at"),
            "gate_passed":    comp.get("gate_passed", True),
            "redacted_attrs": redacted,
        })

    with t6:
        st.markdown("#### Full EvaluationResult (Canonical JSON)")
        json_str = json.dumps(report, indent=2, default=str)
        st.code(json_str, language="json")
        st.download_button(
            " Download JSON Report",
            data=json_str,
            file_name=f"evaluation_{report.get('evaluation_id','')[:8]}.json",
            mime="application/json",
        )
