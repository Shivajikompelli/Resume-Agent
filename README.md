# 🤖 Resume Screening & Candidate Eligibility Validation Agent

Powered by **Groq API** (Free tier) · Llama 3 70B · Deterministic Scoring · EEOC Compliant

---

## ✅ Quick Start (5 minutes)

### 1. Get your FREE Groq API Key
→ Sign up at **https://console.groq.com** (no credit card required)

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** For OCR support on scanned PDFs, also install Tesseract:
> - macOS: `brew install tesseract`
> - Ubuntu: `sudo apt install tesseract-ocr`
> - Windows: Download from https://github.com/tesseract-ocr/tesseract

### 3. Set your API key
```bash
cp .env.example .env
# Edit .env and add your key:
# GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

### 4a. Launch the Web UI
```bash
streamlit run app.py
```
Opens at: **http://localhost:8501**

### 4b. Or use the CLI
```bash
python run.py --resume path/to/resume.pdf --jd path/to/job_description.txt
```

---

## 🏗️ Architecture

```
resume_agent/
├── app.py                 ← Streamlit web UI
├── run.py                 ← Command-line interface
├── config.py              ← Skill taxonomy, scoring weights, constants
├── requirements.txt
├── .env.example
│
└── agent/
    ├── ingestion.py       ← PDF/DOCX/TXT text extraction + OCR
    ├── compliance.py      ← PII detection & redaction (runs FIRST)
    ├── extraction.py      ← Groq LLM: structured resume extraction
    ├── jd_parser.py       ← Groq LLM: job description parsing
    ├── skill_matcher.py   ← 3-layer skill matching (no GPU needed)
    ├── scoring.py         ← Deterministic scoring engine (zero LLM)
    ├── report.py          ← EvaluationResult JSON assembly
    └── orchestrator.py    ← Pipeline coordinator (9 steps)
```

---

## 🔄 9-Step Pipeline

| Step | Service | Description |
|------|---------|-------------|
| 1 | Ingestion | PDF/DOCX/TXT extraction, OCR fallback |
| 2 | **Compliance Gate** | PII redaction **before any LLM** |
| 3 | LLM Extraction | Groq Llama 3 extracts candidate data |
| 4 | JD Parser | Groq Llama 3 parses job requirements |
| 5 | Skill Matcher | 3-layer: exact → TF-IDF cosine → Jaccard |
| 6 | Scoring | Deterministic weighted scoring (zero LLM) |
| 7 | Report | Full JSON report + explainability |

---

## ⚖️ Scoring Model

| Dimension | Weight | Hard Disqualifier? |
|-----------|--------|--------------------|
| Required Skill Match | 40 pts | Yes — if <50% + must-have missing |
| Experience Validation | 25 pts | No (proportional) |
| Education Validation | 20 pts | Configurable |
| Preferred Skill Match | 10 pts | No |
| Certification Bonus | 5 pts | No |
| **Total** | **100 pts** | |

### Eligibility Bands
| Score | Category |
|-------|----------|
| 85–100 | ✅ STRONGLY_ELIGIBLE |
| 70–84 | ✔️ ELIGIBLE |
| 50–69 | ⚠️ PARTIALLY_ELIGIBLE |
| <50 | ❌ NOT_ELIGIBLE |

---

## 🛡️ Bias Mitigation

- Protected attributes (gender, age, religion, marital status, nationality, disability) are **redacted before any AI processing**
- The scoring engine receives **only**: skill lists, experience years, education level enum, JD constraints
- It has **zero access** to: names, institutions, graduation years, addresses, or any proxy attributes
- Discriminatory language detection on job descriptions

---

## 🤖 Groq Models (All Free)

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `llama3-70b-8192` | Medium | ⭐⭐⭐⭐⭐ | Best extraction quality |
| `llama3-8b-8192` | Fast | ⭐⭐⭐⭐ | Quick evaluations |
| `mixtral-8x7b-32768` | Medium | ⭐⭐⭐⭐ | Long resumes (32K context) |

---

## 📋 CLI Usage

```bash
# Basic
python run.py --resume resume.pdf --jd job.txt

# With specific model
python run.py --resume resume.pdf --jd job.txt --model mixtral-8x7b-32768

# Save report to specific path
python run.py --resume resume.pdf --jd job.txt --output reports/candidate_1.json

# Verbose mode
python run.py --resume resume.pdf --jd job.txt --verbose

# Pass API key directly
python run.py --resume resume.pdf --jd job.txt --api-key gsk_xxx
```

---

## 📄 Output JSON Schema

```json
{
  "evaluation_id": "uuid-v4",
  "candidate_profile": { "name", "skills", "experience", "education", "certifications" },
  "job_requirements":  { "required_skills", "preferred_skills", "min_experience_years", ... },
  "skills_analysis":   { "matched", "missing", "similarity_pairs", "match_pct" },
  "experience_analysis": { "candidate_years", "required_years", "gap", "meets_requirement" },
  "education_analysis":  { "highest_level", "meets_requirement" },
  "scoring": {
    "required_skill_score", "experience_score", "education_score",
    "composite_score", "eligibility_category",
    "hard_disqualifier_triggered", "hard_disqualifier_reason"
  },
  "explainability": { "confidence_score", "reasoning_summary", "strengths", "gaps" },
  "compliance":     { "pii_attributes_redacted", "gate_passed" }
}
```

---

## 🔧 Customisation

Edit `config.py` to:
- Adjust **scoring weights** per job family
- Add skills to the **taxonomy dictionary**
- Change **similarity thresholds**
- Modify **PII detection patterns**

---

## 📦 Dependencies

```
groq             ← Groq API client (free LLM inference)
streamlit        ← Web UI
pdfplumber       ← PDF text extraction
python-docx      ← DOCX extraction
scikit-learn     ← TF-IDF cosine similarity for skill matching
pydantic         ← Schema validation (anti-hallucination)
python-dotenv    ← .env file support
pytesseract      ← OCR for scanned PDFs (optional)
Pillow           ← Image processing for OCR
```
