"""
agent/scoring.py — Deterministic Eligibility Scoring Engine

ZERO LLM calls in this module.
Receives only normalised, structured data — no text, no names, no demographics.
Fully unit-testable and reproducible.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from config import SCORING_WEIGHTS, THRESHOLDS, EDU_SCORE_MAP, EDU_LEVELS, \
                   HARD_DISQUALIFY_REQUIRED_MATCH_BELOW

logger = logging.getLogger(__name__)


# ── Score Input / Output Models ───────────────────────────────────

@dataclass
class ScoringInput:
    """Only structured, bias-free fields are passed to the scorer."""
    # Skill data
    matched_required_count:  int   = 0
    total_required_count:    int   = 0
    matched_preferred_count: int   = 0
    total_preferred_count:   int   = 0
    matched_cert_count:      int   = 0
    must_have_skills_all_matched: bool = True  # True if every must-have is matched

    # Experience
    candidate_years:  float = 0.0
    required_years:   float = 0.0

    # Education
    candidate_edu_level: Optional[str] = None  # PHD|MASTERS|BACHELORS|...
    required_edu_level:  Optional[str] = None
    edu_flexible:        bool = False

    # JD config overrides (per job family)
    weight_required_skills: int = field(default_factory=lambda: SCORING_WEIGHTS["required_skills"])
    weight_experience:      int = field(default_factory=lambda: SCORING_WEIGHTS["experience"])
    weight_education:       int = field(default_factory=lambda: SCORING_WEIGHTS["education"])
    weight_preferred_skills:int = field(default_factory=lambda: SCORING_WEIGHTS["preferred_skills"])
    weight_certs:           int = field(default_factory=lambda: SCORING_WEIGHTS["certifications"])


@dataclass
class ScoringResult:
    required_skill_score:    float = 0.0
    experience_score:        float = 0.0
    education_score:         float = 0.0
    preferred_skill_score:   float = 0.0
    certification_bonus:     float = 0.0
    composite_score:         float = 0.0
    eligibility_category:    str   = "NOT_ELIGIBLE"
    hard_disqualifier:       bool  = False
    hard_disqualifier_reason:Optional[str] = None
    experience_gap_years:    float = 0.0
    meets_experience:        bool  = False
    meets_education:         bool  = False
    required_match_pct:      float = 0.0
    confidence_score:        float = 0.0


# ── Scoring Engine ────────────────────────────────────────────────

class ScoringEngine:
    """
    Deterministic, rules-based eligibility scoring.
    No randomness. No LLM. Same inputs → same outputs every time.
    """

    def score(self, inp: ScoringInput) -> ScoringResult:
        result = ScoringResult()

        # ── Check hard disqualifiers FIRST ────────────────────────
        disq_reason = self._check_hard_disqualifiers(inp)
        if disq_reason:
            result.hard_disqualifier = True
            result.hard_disqualifier_reason = disq_reason
            result.eligibility_category = "NOT_ELIGIBLE"
            result.composite_score = 0.0
            result.confidence_score = 1.0
            # Still compute subscores for the report
            result.required_skill_score = self._score_required_skills(inp)
            result.experience_score     = self._score_experience(inp, result)
            result.education_score      = self._score_education(inp, result)
            logger.info(f"Hard disqualifier: {disq_reason}")
            return result

        # ── Compute dimension scores ──────────────────────────────
        result.required_skill_score  = self._score_required_skills(inp)
        result.experience_score      = self._score_experience(inp, result)
        result.education_score       = self._score_education(inp, result)
        result.preferred_skill_score = self._score_preferred_skills(inp)
        result.certification_bonus   = self._score_certs(inp)

        # ── Composite ──────────────────────────────────────────────
        result.composite_score = round(
            result.required_skill_score +
            result.experience_score +
            result.education_score +
            result.preferred_skill_score +
            result.certification_bonus,
            2
        )

        # ── Eligibility category ──────────────────────────────────
        if result.composite_score >= THRESHOLDS["strongly_eligible"]:
            result.eligibility_category = "STRONGLY_ELIGIBLE"
        elif result.composite_score >= THRESHOLDS["eligible"]:
            result.eligibility_category = "ELIGIBLE"
        elif result.composite_score >= THRESHOLDS["partially_eligible"]:
            result.eligibility_category = "PARTIALLY_ELIGIBLE"
        else:
            result.eligibility_category = "NOT_ELIGIBLE"

        # ── Confidence ─────────────────────────────────────────────
        result.confidence_score = self._compute_confidence(inp, result)

        logger.info(
            f"Score: {result.composite_score} → {result.eligibility_category}"
        )
        return result

    # ── Dimension Scorers ─────────────────────────────────────────

    def _score_required_skills(self, inp: ScoringInput) -> float:
        if inp.total_required_count == 0:
            return float(inp.weight_required_skills)  # No required skills → full credit
        ratio = inp.matched_required_count / inp.total_required_count
        score = ratio * inp.weight_required_skills
        inp_ref = inp  # keep for result
        # Store for report
        return round(score, 2)

    def _score_experience(self, inp: ScoringInput, result: ScoringResult) -> float:
        result.experience_gap_years = max(0.0, inp.required_years - inp.candidate_years)
        result.meets_experience = inp.candidate_years >= inp.required_years

        if inp.required_years == 0:
            result.meets_experience = True
            return float(inp.weight_experience)

        ratio = inp.candidate_years / inp.required_years
        # Cap at 1.5× to give seniority bonus but prevent reverse weighting
        capped_ratio = min(ratio, 1.5)
        score = capped_ratio * (inp.weight_experience / 1.5)
        return round(min(score, inp.weight_experience), 2)

    def _score_education(self, inp: ScoringInput, result: ScoringResult) -> float:
        max_pts = inp.weight_education

        if inp.required_edu_level is None or inp.edu_flexible:
            result.meets_education = True
            return float(max_pts)  # No education requirement

        cand_idx = EDU_LEVELS.index(inp.candidate_edu_level) if \
                   inp.candidate_edu_level in EDU_LEVELS else -1
        req_idx  = EDU_LEVELS.index(inp.required_edu_level)  if \
                   inp.required_edu_level  in EDU_LEVELS else -1

        result.meets_education = cand_idx >= req_idx and cand_idx >= 0

        # Get base score from map (relative to requiring BACHELORS)
        base_score = EDU_SCORE_MAP.get(inp.candidate_edu_level, 0)

        # Scale to actual weight
        scaled = base_score / 20.0 * max_pts
        return round(min(scaled, max_pts), 2)

    def _score_preferred_skills(self, inp: ScoringInput) -> float:
        if inp.total_preferred_count == 0:
            return 0.0
        ratio = inp.matched_preferred_count / inp.total_preferred_count
        return round(ratio * inp.weight_preferred_skills, 2)

    def _score_certs(self, inp: ScoringInput) -> float:
        return round(min(float(inp.matched_cert_count), float(inp.weight_certs)), 2)

    # ── Hard Disqualifier Check ───────────────────────────────────

    def _check_hard_disqualifiers(self, inp: ScoringInput) -> Optional[str]:
        """Return a reason string if candidate should be immediately disqualified."""
        if inp.total_required_count > 0:
            ratio = inp.matched_required_count / inp.total_required_count
            if ratio < HARD_DISQUALIFY_REQUIRED_MATCH_BELOW and not inp.must_have_skills_all_matched:
                return (
                    f"Matched only {ratio*100:.0f}% of required skills "
                    f"({inp.matched_required_count}/{inp.total_required_count}) "
                    f"with one or more must-have skills missing."
                )
        return None

    # ── Confidence Score ─────────────────────────────────────────

    def _compute_confidence(self, inp: ScoringInput, result: ScoringResult) -> float:
        """
        Confidence in the eligibility decision (0–1).
        Lower when: few required skills in JD, no experience years specified, etc.
        """
        factors = []
        factors.append(1.0 if inp.total_required_count >= 3 else 0.6)
        factors.append(1.0 if inp.required_years > 0 else 0.7)
        factors.append(1.0 if inp.candidate_years > 0 else 0.75)
        factors.append(1.0 if inp.required_edu_level else 0.85)
        return round(sum(factors) / len(factors), 2)

    # ── Explanation Builder ───────────────────────────────────────

    def build_reasoning(
        self,
        inp: ScoringInput,
        result: ScoringResult,
        missing_skills: list[str],
    ) -> str:
        """Generate a plain-English reasoning summary."""
        lines = []

        # Decision
        cat_map = {
            "STRONGLY_ELIGIBLE": "strongly eligible",
            "ELIGIBLE":          "eligible",
            "PARTIALLY_ELIGIBLE":"partially eligible",
            "NOT_ELIGIBLE":      "not eligible",
        }
        lines.append(
            f"Candidate is {cat_map.get(result.eligibility_category, 'not eligible')} "
            f"with a composite score of {result.composite_score}/100."
        )

        # Hard disqualifier
        if result.hard_disqualifier:
            lines.append(f"Hard disqualifier triggered: {result.hard_disqualifier_reason}")
            return " ".join(lines)

        # Required skills
        if inp.total_required_count > 0:
            pct = round(inp.matched_required_count / inp.total_required_count * 100)
            lines.append(
                f"Required skills: matched {inp.matched_required_count} of "
                f"{inp.total_required_count} ({pct}%)."
            )
            if missing_skills:
                lines.append(f"Missing: {', '.join(missing_skills[:5])}{'...' if len(missing_skills)>5 else ''}.")

        # Experience
        if inp.required_years > 0:
            if result.meets_experience:
                lines.append(
                    f"Experience: {inp.candidate_years:.1f} years meets the "
                    f"{inp.required_years:.0f}-year requirement."
                )
            else:
                lines.append(
                    f"Experience gap: {result.experience_gap_years:.1f} years short of "
                    f"the {inp.required_years:.0f}-year requirement."
                )

        # Education
        if inp.required_edu_level:
            status = "meets" if result.meets_education else "does not meet"
            lines.append(
                f"Education {status} the {inp.required_edu_level.title().replace('_', ' ')} requirement."
            )

        # Preferred skills
        if inp.total_preferred_count > 0:
            lines.append(
                f"Preferred skills: matched {inp.matched_preferred_count} of "
                f"{inp.total_preferred_count}."
            )

        return " ".join(lines)
