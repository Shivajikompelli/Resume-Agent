"""
agent/extraction.py — LLM Extraction Service (Groq API)

Extracts structured candidate data from sanitised resume text.
Uses Groq's free-tier API with llama3-70b or mixtral.
Anti-hallucination: strict JSON schema + null-for-absent rule.
"""

import json
import logging
import re
from typing import Optional
from groq import Groq
from pydantic import BaseModel, Field, validator
from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


# ── Pydantic Schema (post-LLM validation) ────────────────────────

class ExperienceEntry(BaseModel):
    title:      str
    company:    str
    years:      float = 0.0
    start_date: Optional[str] = None  # YYYY-MM or null
    end_date:   Optional[str] = None  # YYYY-MM or "Present" or null
    description:Optional[str] = None


class EducationEntry(BaseModel):
    degree:      str
    institution: str
    edu_level:   Optional[str] = None  # PHD|MASTERS|BACHELORS|ASSOCIATE|HIGH_SCHOOL
    year:        Optional[int] = None

    @validator("edu_level", pre=True, always=True)
    def validate_edu_level(cls, v):
        valid = {"PHD", "MASTERS", "BACHELORS", "ASSOCIATE", "HIGH_SCHOOL"}
        if v and str(v).upper() in valid:
            return str(v).upper()
        return None


class CandidateProfile(BaseModel):
    name:                    Optional[str]  = None
    email:                   Optional[str]  = None
    phone:                   Optional[str]  = None
    location:                Optional[str]  = None
    linkedin:                Optional[str]  = None
    summary:                 Optional[str]  = None
    skills:                  list[str]      = Field(default_factory=list)
    experience:              list[ExperienceEntry] = Field(default_factory=list)
    education:               list[EducationEntry]  = Field(default_factory=list)
    certifications:          list[str]      = Field(default_factory=list)
    languages:               list[str]      = Field(default_factory=list)
    total_experience_years:  float          = 0.0


# ── Extraction Service ────────────────────────────────────────────

class ExtractionService:
    """
    Uses Groq LLM to extract structured data from resume text.
    Temperature=0 for determinism. Pydantic validates every field.
    """

    SYSTEM_PROMPT = """You are a precise resume data extractor. Your ONLY job is to extract information that is EXPLICITLY present in the resume text.

STRICT RULES:
1. Return ONLY valid JSON. No markdown, no explanation, no code blocks.
2. If a field is NOT present in the resume → use null (never guess or infer).
3. Never fabricate skills, companies, dates, or qualifications.
4. Skills must be extracted verbatim as listed in the resume.
5. Calculate total_experience_years by summing all work experience durations.
6. edu_level must be one of: PHD, MASTERS, BACHELORS, ASSOCIATE, HIGH_SCHOOL (or null).

Return this exact JSON structure:
{
  "name": string or null,
  "email": string or null,
  "phone": string or null,
  "location": string or null,
  "linkedin": string or null,
  "summary": string or null,
  "skills": [list of skill strings],
  "experience": [
    {
      "title": string,
      "company": string,
      "years": float (duration in years),
      "start_date": "YYYY-MM" or null,
      "end_date": "YYYY-MM" or "Present" or null,
      "description": string or null
    }
  ],
  "education": [
    {
      "degree": string,
      "institution": string,
      "edu_level": "PHD"|"MASTERS"|"BACHELORS"|"ASSOCIATE"|"HIGH_SCHOOL"|null,
      "year": integer or null
    }
  ],
  "certifications": [list of certification strings],
  "languages": [list of language strings],
  "total_experience_years": float
}"""

    def __init__(self):
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY not set. Get your free key at https://console.groq.com "
                "and add it to .env file."
            )
        self.client = Groq(api_key=GROQ_API_KEY)

    def extract(self, sanitised_text: str, max_retries: int = 3) -> CandidateProfile:
        """
        Extract structured candidate profile from sanitised resume text.
        Retries up to max_retries times on validation failure.
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                raw_json = self._call_groq(sanitised_text)
                data = self._parse_json(raw_json)
                profile = CandidateProfile(**data)
                logger.info(f"Extraction succeeded on attempt {attempt}")
                return profile
            except Exception as e:
                last_error = e
                logger.warning(f"Extraction attempt {attempt} failed: {e}")

        # All retries exhausted — return minimal profile rather than crashing
        logger.error(f"Extraction failed after {max_retries} attempts: {last_error}")
        return CandidateProfile(
            summary=f"[Extraction failed: {last_error}]",
            skills=[],
        )

    def _call_groq(self, text: str) -> str:
        """Call Groq API and return raw response string."""
        # Truncate to stay within context limits
        max_chars = 12000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... truncated for context limit ...]"

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": f"Extract structured data from this resume:\n\n{text}"},
            ],
        )
        return response.choices[0].message.content.strip()

    def _parse_json(self, raw: str) -> dict:
        """Parse JSON from LLM response, handling common formatting issues."""
        # Strip markdown code blocks if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        # Find the outermost JSON object
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in response: {raw[:200]}")

        return json.loads(raw[start:end])
