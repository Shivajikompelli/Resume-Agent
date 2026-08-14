"""
agent/jd_parser.py — Job Description Parser Service (Groq API)

Extracts structured job requirements from raw JD text.
Identifies required/preferred/must-have skills, experience, education,
domain, seniority level, and discriminatory language.
"""

import json
import logging
import re
from typing import Optional
from groq import Groq
from pydantic import BaseModel, Field
from config import GROQ_API_KEY, GROQ_MODEL, DOMAINS

logger = logging.getLogger(__name__)


# ── Pydantic Schema ───────────────────────────────────────────────

class JobRequirements(BaseModel):
    job_title:            str            = "Unknown Role"
    required_skills:      list[str]      = Field(default_factory=list)
    preferred_skills:     list[str]      = Field(default_factory=list)
    must_have_skills:     list[str]      = Field(default_factory=list)
    min_experience_years: float          = 0.0
    required_edu_level:   Optional[str]  = None  # PHD|MASTERS|BACHELORS|ASSOCIATE|HIGH_SCHOOL
    edu_flexible:         bool           = False  # "or equivalent experience"
    required_certs:       list[str]      = Field(default_factory=list)
    domain:               str            = "Software Engineering"
    seniority:            str            = "MID"  # JUNIOR|MID|SENIOR|STAFF|PRINCIPAL
    nice_to_have:         list[str]      = Field(default_factory=list)
    responsibilities:     list[str]      = Field(default_factory=list)


# ── Parser Service ────────────────────────────────────────────────

class JDParserService:
    """
    Uses Groq LLM to extract structured requirements from job descriptions.
    Results are cached by job_title + hash for efficiency.
    """

    SYSTEM_PROMPT = """You are a precise job description analyst. Extract structured requirements from the job description.

STRICT RULES:
1. Return ONLY valid JSON. No markdown, no explanation.
2. must_have_skills = skills explicitly marked as "required", "must have", "essential", "non-negotiable"
3. required_skills = all other required technical and professional skills
4. preferred_skills = skills marked as "preferred", "nice to have", "bonus", "plus"
5. seniority must be one of: JUNIOR, MID, SENIOR, STAFF, PRINCIPAL
6. required_edu_level must be one of: PHD, MASTERS, BACHELORS, ASSOCIATE, HIGH_SCHOOL, or null
7. Extract exact skill names as written in the JD

Return this exact JSON:
{
  "job_title": string,
  "required_skills": [list of strings],
  "preferred_skills": [list of strings],
  "must_have_skills": [list of strings],
  "min_experience_years": float,
  "required_edu_level": "PHD"|"MASTERS"|"BACHELORS"|"ASSOCIATE"|"HIGH_SCHOOL"|null,
  "edu_flexible": boolean (true if "or equivalent experience" is mentioned),
  "required_certs": [list of certification strings],
  "domain": string (e.g. "Software Engineering", "Data Science", "Machine Learning / AI"),
  "seniority": "JUNIOR"|"MID"|"SENIOR"|"STAFF"|"PRINCIPAL",
  "nice_to_have": [list of strings],
  "responsibilities": [top 5 key responsibilities as strings]
}"""

    def __init__(self):
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self._cache: dict[str, JobRequirements] = {}

    def parse(self, jd_text: str, max_retries: int = 3) -> JobRequirements:
        """Parse a job description and return structured requirements."""
        cache_key = str(hash(jd_text.strip()))
        if cache_key in self._cache:
            logger.info("JD parse: cache hit")
            return self._cache[cache_key]

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                raw_json = self._call_groq(jd_text)
                data = self._parse_json(raw_json)
                # Validate edu_level
                valid_edu = {"PHD", "MASTERS", "BACHELORS", "ASSOCIATE", "HIGH_SCHOOL"}
                if data.get("required_edu_level") not in valid_edu:
                    data["required_edu_level"] = None
                # Validate seniority
                valid_sen = {"JUNIOR", "MID", "SENIOR", "STAFF", "PRINCIPAL"}
                if data.get("seniority") not in valid_sen:
                    data["seniority"] = "MID"

                result = JobRequirements(**data)
                self._cache[cache_key] = result
                logger.info(f"JD parsed on attempt {attempt}: {result.job_title}")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"JD parse attempt {attempt} failed: {e}")

        logger.error(f"JD parsing failed: {last_error}")
        return JobRequirements()

    def _call_groq(self, jd_text: str) -> str:
        max_chars = 8000
        if len(jd_text) > max_chars:
            jd_text = jd_text[:max_chars]

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": f"Parse this job description:\n\n{jd_text}"},
            ],
        )
        return response.choices[0].message.content.strip()

    def _parse_json(self, raw: str) -> dict:
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1:
            raise ValueError(f"No JSON in JD parse response: {raw[:200]}")
        return json.loads(raw[start:end])
