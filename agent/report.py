"""
agent/report.py — Report Assembly Service

Assembles the final EvaluationResult from all pipeline outputs.
Produces the canonical JSON schema + a human-readable summary.
"""

import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from agent.extraction  import CandidateProfile
from agent.jd_parser   import JobRequirements
from agent.skill_matcher import SkillMatchResult
from agent.scoring     import ScoringInput, ScoringResult, ScoringEngine

logger = logging.getLogger(__name__)


class ReportService:
    """
    Assembles the canonical EvaluationResult JSON from all pipeline stages.
    This is the single source of truth for every hiring decision.
    """

    def __init__(self):
        self.scoring_engine = ScoringEngine()

    def build(
        self,
        profile:         CandidateProfile,
        job_req:         JobRequirements,
        skill_result:    SkillMatchResult,
        compliance_info: Dict,
        ingestion_meta:  Dict,
        evaluation_id:   str = None,
    ) -> Dict[str, Any]:
        """
        Build the complete EvaluationResult.
        Returns a dict conforming to the canonical JSON schema.
        """
        if not evaluation_id:
            evaluation_id = str(uuid.uuid4())

        # ── Build scoring input (no text fields, no demographics) ──
        must_have_all = all(
            skill in skill_result.matched_required
            for skill in job_req.must_have_skills
        )

        scoring_input = ScoringInput(
            matched_required_count=  len(skill_result.matched_required),
            total_required_count=    len(job_req.required_skills),
            matched_preferred_count= len(skill_result.matched_preferred),
            total_preferred_count=   len(job_req.preferred_skills),
            matched_cert_count=      len(skill_result.matched_certs),
            must_have_skills_all_matched= must_have_all,
            candidate_years=         profile.total_experience_years,
            required_years=          job_req.min_experience_years,
            candidate_edu_level=     self._highest_edu(profile),
            required_edu_level=      job_req.required_edu_level,
            edu_flexible=            job_req.edu_flexible,
        )

        scoring_result = self.scoring_engine.score(scoring_input)

        reasoning = self.scoring_engine.build_reasoning(
            scoring_input, scoring_result, skill_result.missing_required
        )

        strengths = self._build_strengths(scoring_input, scoring_result, skill_result)
        gaps      = self._build_gaps(scoring_input, scoring_result, skill_result)

        # ── Assemble canonical report ──────────────────────────────
        report = {
            "evaluation_id":   evaluation_id,
            "schema_version":  "1.0.0",
            "evaluated_at":    datetime.now(timezone.utc).isoformat(),

            "candidate_profile": {
                "name":    profile.name,
                "contact": {"email": profile.email, "phone": profile.phone},
                "location":profile.location,
                "skills":  profile.skills,
                "experience": [
                    {
                        "title":       e.title,
                        "company":     e.company,
                        "years":       e.years,
                        "start_date":  e.start_date,
                        "end_date":    e.end_date,
                    }
                    for e in profile.experience
                ],
                "education": [
                    {
                        "degree":      ed.degree,
                        "institution": ed.institution,
                        "edu_level":   ed.edu_level,
                        "year":        ed.year,
                    }
                    for ed in profile.education
                ],
                "certifications":         profile.certifications,
                "total_experience_years": profile.total_experience_years,
            },

            "job_requirements": {
                "job_title":            job_req.job_title,
                "required_skills":      job_req.required_skills,
                "preferred_skills":     job_req.preferred_skills,
                "must_have_skills":     job_req.must_have_skills,
                "min_experience_years": job_req.min_experience_years,
                "required_edu_level":   job_req.required_edu_level,
                "domain":               job_req.domain,
                "seniority":            job_req.seniority,
            },

            "skills_analysis": {
                "required_skills_matched":  skill_result.matched_required,
                "required_skills_missing":  skill_result.missing_required,
                "preferred_skills_matched": skill_result.matched_preferred,
                "required_match_pct":       skill_result.required_match_pct,
                "preferred_match_pct":      skill_result.preferred_match_pct,
                "similarity_pairs":         skill_result.similarity_pairs,
            },

            "experience_analysis": {
                "candidate_years":      profile.total_experience_years,
                "required_years":       job_req.min_experience_years,
                "experience_gap_years": scoring_result.experience_gap_years
                                        if scoring_result.experience_gap_years > 0 else None,
                "meets_requirement":    scoring_result.meets_experience,
            },

            "education_analysis": {
                "highest_level":     self._highest_edu(profile),
                "meets_requirement": scoring_result.meets_education,
                "relevant_degrees":  [e.degree for e in profile.education],
            },

            "scoring": {
                "required_skill_score":    scoring_result.required_skill_score,
                "experience_score":        scoring_result.experience_score,
                "education_score":         scoring_result.education_score,
                "preferred_skill_score":   scoring_result.preferred_skill_score,
                "certification_bonus":     scoring_result.certification_bonus,
                "composite_score":         scoring_result.composite_score,
                "eligibility_category":    scoring_result.eligibility_category,
                "hard_disqualifier_triggered": scoring_result.hard_disqualifier,
                "hard_disqualifier_reason":    scoring_result.hard_disqualifier_reason,
            },

            "explainability": {
                "confidence_score":  scoring_result.confidence_score,
                "reasoning_summary": reasoning,
                "missing_skills":    skill_result.missing_required,
                "strengths":         strengths,
                "gaps":              gaps,
            },

            "compliance": {
                "pii_attributes_redacted":            compliance_info.get("redacted_attrs", []),
                "protected_attrs_found_and_removed":  compliance_info.get("redacted_attrs", []),
                "gate_passed":                        compliance_info.get("gate_passed", True),
            },

            "ingestion_meta": ingestion_meta,
        }

        return report

    # ── Helpers ───────────────────────────────────────────────────

    def _highest_edu(self, profile: CandidateProfile) -> str | None:
        from config import EDU_LEVELS
        levels = [e.edu_level for e in profile.education if e.edu_level in EDU_LEVELS]
        if not levels:
            return None
        return max(levels, key=lambda l: EDU_LEVELS.index(l))

    def _build_strengths(self, inp, result, skill_result) -> list[str]:
        s = []
        if skill_result.required_match_pct >= 80:
            s.append(f"Strong required skill match ({skill_result.required_match_pct:.0f}%)")
        if result.meets_experience and inp.required_years > 0:
            s.append(f"{inp.candidate_years:.1f} years of experience meets requirement")
        if result.meets_education:
            s.append("Education meets or exceeds requirement")
        if skill_result.matched_preferred:
            s.append(f"Matches {len(skill_result.matched_preferred)} preferred skill(s)")
        if skill_result.matched_certs:
            s.append(f"Holds {len(skill_result.matched_certs)} relevant certification(s)")
        return s

    def _build_gaps(self, inp, result, skill_result) -> list[str]:
        g = []
        if skill_result.missing_required:
            g.append(f"Missing {len(skill_result.missing_required)} required skill(s): "
                     f"{', '.join(skill_result.missing_required[:4])}")
        if result.experience_gap_years > 0:
            g.append(f"Experience gap: {result.experience_gap_years:.1f} years short")
        if not result.meets_education and inp.required_edu_level:
            g.append(f"Education below required {inp.required_edu_level} level")
        return g

    def to_json(self, report: Dict, indent: int = 2) -> str:
        return json.dumps(report, indent=indent, default=str)

    def summary_text(self, report: Dict) -> str:
        """One-paragraph plain-English summary for display."""
        s = report["scoring"]
        e = report["explainability"]
        sa = report["skills_analysis"]
        cat = s["eligibility_category"].replace("_", " ").title()
        return (
            f"**{cat}** (Score: {s['composite_score']}/100)\n\n"
            f"{e['reasoning_summary']}\n\n"
            f"Required skills matched: **{sa['required_match_pct']}%**  |  "
            f"Preferred skills matched: **{sa['preferred_match_pct']}%**"
        )
