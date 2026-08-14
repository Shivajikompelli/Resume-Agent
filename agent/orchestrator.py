"""
agent/orchestrator.py — Pipeline Orchestrator

Runs the full 9-step evaluation pipeline in sequence.
This is the single entry point for evaluating a resume against a JD.
"""

import uuid
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Callable, Optional

from agent.ingestion    import IngestionService
from agent.compliance   import ComplianceService
from agent.extraction   import ExtractionService
from agent.jd_parser    import JDParserService
from agent.skill_matcher import SkillMatcher
from agent.report       import ReportService

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Orchestrates the 9-step resume evaluation pipeline.

    Step 1 — File ingestion (PDF/DOCX/TXT)
    Step 2 — Compliance gate (PII redaction)
    Step 3 — LLM extraction (Groq)
    Step 4 — JD parsing (Groq)
    Step 5 — Skill normalisation + matching
    Step 6 — Deterministic scoring (no LLM)
    Step 7 — Report assembly
    """

    def __init__(self):
        self.ingestion   = IngestionService()
        self.compliance  = ComplianceService()
        self.extractor   = ExtractionService()
        self.jd_parser   = JDParserService()
        self.skill_match = SkillMatcher()
        self.reporter    = ReportService()

    def evaluate(
        self,
        resume_path: str | Path,
        jd_text:     str,
        progress_cb: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline.

        Args:
            resume_path: Path to resume file (PDF, DOCX, or TXT)
            jd_text:     Raw job description text
            progress_cb: Optional callback(message, pct) for progress updates

        Returns:
            EvaluationResult dict conforming to canonical JSON schema
        """
        evaluation_id = str(uuid.uuid4())

        def progress(msg: str, pct: int):
            logger.info(f"[{pct}%] {msg}")
            if progress_cb:
                progress_cb(msg, pct)

        try:
            # ── Step 1: Ingestion ──────────────────────────────────
            progress("Extracting text from resume...", 10)
            raw_text, ingestion_meta = self.ingestion.extract(resume_path)

            # ── Step 2: Compliance Gate (MUST run before LLM) ──────
            progress("Running compliance gate (PII redaction)...", 20)
            scrubbed_text, redacted_attrs = self.compliance.scrub(raw_text)
            jd_clean, jd_issues = self.compliance.check_jd(jd_text)
            if not jd_clean:
                logger.warning(f"JD contains potential issues: {jd_issues}")

            compliance_info = {
                "redacted_attrs": redacted_attrs,
                "gate_passed":    True,
                "jd_issues":      jd_issues,
            }

            # ── Step 3: LLM Extraction ─────────────────────────────
            progress("Extracting candidate profile with Groq LLM...", 35)
            profile = self.extractor.extract(scrubbed_text)

            # ── Step 4: JD Parsing ─────────────────────────────────
            progress("Parsing job description with Groq LLM...", 50)
            job_req = self.jd_parser.parse(jd_text)

            # ── Step 5: Skill Matching ─────────────────────────────
            progress("Matching skills (3-layer semantic matching)...", 65)
            skill_result = self.skill_match.match(
                candidate_skills = profile.skills,
                required_skills  = job_req.required_skills,
                preferred_skills = job_req.preferred_skills,
                must_have_skills = job_req.must_have_skills,
                required_certs   = job_req.required_certs,
                candidate_certs  = profile.certifications,
            )

            # ── Step 6 + 7: Scoring + Report ──────────────────────
            progress("Computing eligibility score...", 80)
            report = self.reporter.build(
                profile         = profile,
                job_req         = job_req,
                skill_result    = skill_result,
                compliance_info = compliance_info,
                ingestion_meta  = ingestion_meta,
                evaluation_id   = evaluation_id,
            )

            progress("Evaluation complete!", 100)
            return report

        except Exception as e:
            logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
            return {
                "evaluation_id":   evaluation_id,
                "error":           str(e),
                "status":          "FAILED",
                "scoring":         {"eligibility_category": "ERROR", "composite_score": 0},
                "explainability":  {"reasoning_summary": f"Evaluation failed: {e}"},
            }

    def evaluate_from_text(
        self,
        resume_text: str,
        jd_text: str,
        progress_cb: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate from raw text directly (no file upload needed).
        Useful for testing and Streamlit text input.
        """
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(resume_text)
            tmp_path = f.name
        try:
            return self.evaluate(tmp_path, jd_text, progress_cb)
        finally:
            os.unlink(tmp_path)
