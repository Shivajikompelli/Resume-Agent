"""
config.py — Central configuration for the Resume Screening Agent
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Groq API ────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# ─── Scoring Weights ─────────────────────────────────────────────
SCORING_WEIGHTS = {
    "required_skills": 40,   # Max 40 pts
    "experience":      25,   # Max 25 pts
    "education":       20,   # Max 20 pts
    "preferred_skills":10,   # Max 10 pts
    "certifications":   5,   # Max 5 pts (bonus)
}

# ─── Eligibility Thresholds ───────────────────────────────────────
THRESHOLDS = {
    "strongly_eligible": 85,
    "eligible":          70,
    "partially_eligible":50,
}

# ─── Hard Disqualifier Rules ──────────────────────────────────────
HARD_DISQUALIFY_REQUIRED_MATCH_BELOW = 0.50  # <50% required skills → NOT_ELIGIBLE

# ─── Cosine Similarity Threshold for Skill Matching ─────────────
SKILL_MATCH_THRESHOLD = 0.72   # TF-IDF cosine (lower than embedding threshold)
SKILL_JACCARD_THRESHOLD = 0.50  # Fallback token-overlap threshold

# ─── Education Level Scoring ──────────────────────────────────────
# Points awarded when JD requires BACHELORS
EDU_SCORE_MAP = {
    "PHD":         20,
    "MASTERS":     20,
    "BACHELORS":   20,
    "ASSOCIATE":   12,
    "HIGH_SCHOOL":  5,
    None:           0,
}

# EDU levels ordered lowest → highest
EDU_LEVELS = ["HIGH_SCHOOL", "ASSOCIATE", "BACHELORS", "MASTERS", "PHD"]

# ─── Skill Taxonomy ───────────────────────────────────────────────
# Maps common aliases → canonical skill name
SKILL_TAXONOMY = {
    # JavaScript ecosystem
    "js": "JavaScript", "javascript": "JavaScript", "es6": "JavaScript",
    "es2015": "JavaScript", "es2020": "JavaScript", "ecmascript": "JavaScript",
    "nodejs": "Node.js", "node": "Node.js", "node.js": "Node.js",
    "reactjs": "React", "react.js": "React", "react js": "React",
    "vuejs": "Vue.js", "vue.js": "Vue.js", "vue js": "Vue.js",
    "angularjs": "Angular", "angular js": "Angular", "angular 2+": "Angular",
    "nextjs": "Next.js", "next.js": "Next.js",
    "expressjs": "Express.js", "express": "Express.js",
    "typescript": "TypeScript", "ts": "TypeScript",

    # Python
    "python3": "Python", "python 3": "Python", "py": "Python",
    "django rest": "Django", "drf": "Django REST Framework",
    "flask": "Flask", "fastapi": "FastAPI",
    "pandas": "Pandas", "numpy": "NumPy", "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn", "sci-kit learn": "scikit-learn",

    # Cloud
    "aws": "Amazon Web Services", "amazon web services": "Amazon Web Services",
    "amazon aws": "Amazon Web Services",
    "gcp": "Google Cloud Platform", "google cloud": "Google Cloud Platform",
    "azure": "Microsoft Azure", "microsoft azure": "Microsoft Azure",

    # Containers & Orchestration
    "k8s": "Kubernetes", "kube": "Kubernetes",
    "docker": "Docker", "containerization": "Docker",
    "helm": "Helm",

    # Databases
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "psql": "PostgreSQL",
    "mongo": "MongoDB", "mongodb": "MongoDB",
    "mysql": "MySQL", "mssql": "SQL Server", "ms sql": "SQL Server",
    "redis": "Redis",
    "elasticsearch": "Elasticsearch", "elastic search": "Elasticsearch",

    # ML / AI
    "ml": "Machine Learning", "machine learning": "Machine Learning",
    "dl": "Deep Learning", "deep learning": "Deep Learning",
    "nlp": "Natural Language Processing", "natural language processing": "Natural Language Processing",
    "llm": "Large Language Models", "large language model": "Large Language Models",
    "tensorflow": "TensorFlow", "tf": "TensorFlow",
    "pytorch": "PyTorch", "torch": "PyTorch",
    "huggingface": "HuggingFace", "hugging face": "HuggingFace",
    "openai": "OpenAI API", "langchain": "LangChain",

    # DevOps / CI-CD
    "ci/cd": "CI/CD", "cicd": "CI/CD", "continuous integration": "CI/CD",
    "github actions": "GitHub Actions", "gha": "GitHub Actions",
    "jenkins": "Jenkins", "gitlab ci": "GitLab CI",
    "terraform": "Terraform", "iac": "Infrastructure as Code",
    "ansible": "Ansible",

    # Data
    "apache spark": "Spark", "pyspark": "Spark",
    "kafka": "Apache Kafka", "apache kafka": "Apache Kafka",
    "airflow": "Apache Airflow", "apache airflow": "Apache Airflow",
    "dbt": "dbt", "data build tool": "dbt",
    "bigquery": "BigQuery", "bq": "BigQuery",
    "snowflake": "Snowflake",

    # Other
    "git": "Git", "github": "Git/GitHub", "version control": "Git",
    "rest": "REST APIs", "rest api": "REST APIs", "restful": "REST APIs",
    "graphql": "GraphQL", "grpc": "gRPC",
    "microservices": "Microservices", "micro services": "Microservices",
    "agile": "Agile", "scrum": "Scrum", "kanban": "Kanban",
    "linux": "Linux", "unix": "Linux", "bash": "Bash/Shell",
    "shell scripting": "Bash/Shell", "shell script": "Bash/Shell",
}

# ─── PII Patterns to Redact ───────────────────────────────────────
PII_PATTERNS = {
    "gender_pronouns": [
        r"\b(he/him|she/her|they/them|he/she|mr\.|mrs\.|ms\.|miss)\b",
    ],
    "marital_status": [
        r"\b(married|single|divorced|widowed|domestic partner|unmarried)\b",
    ],
    "religion": [
        r"\b(christian|muslim|jewish|hindu|buddhist|sikh|atheist|catholic|"
        r"protestant|evangelical|orthodox|baptist|methodist)\b",
    ],
    "nationality_keywords": [
        r"\b(citizen of|national of|nationality:|citizenship:)\b",
    ],
    "photo_references": [
        r"\b(photo|picture|photograph|headshot|linkedin\.com/in/[^/\s]+/photo)\b",
    ],
    "age_dob": [
        r"\b(age:\s*\d+|d\.?o\.?b\.?|date of birth|born in|born on)\b",
        r"\b(aged?\s+\d+)\b",
    ],
    "disability": [
        r"\b(disability|disabled|handicapped|impairment)\b",
    ],
}

# ─── Domain Taxonomy ──────────────────────────────────────────────
DOMAINS = [
    "Software Engineering", "Data Science", "Machine Learning / AI",
    "DevOps / SRE", "Cloud Infrastructure", "Cybersecurity",
    "Product Management", "UI/UX Design", "Mobile Development",
    "Embedded Systems", "Blockchain", "Networking",
    "Finance / FinTech", "Healthcare / MedTech", "E-commerce / Retail",
    "Manufacturing / Industry 4.0", "Education / EdTech", "Legal / Compliance",
]
