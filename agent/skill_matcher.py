"""
agent/skill_matcher.py — Skill Matching Engine

3-Layer matching hierarchy (no GPU required):
  Layer 1 — Exact match after taxonomy normalization
  Layer 2 — TF-IDF cosine similarity (sklearn)
  Layer 3 — Jaccard token overlap fallback
"""

import logging
from typing import Dict, List, Tuple
from config import SKILL_TAXONOMY, SKILL_MATCH_THRESHOLD, SKILL_JACCARD_THRESHOLD

logger = logging.getLogger(__name__)


class SkillMatchResult:
    def __init__(self):
        self.matched_required:  List[str] = []
        self.missing_required:  List[str] = []
        self.matched_preferred: List[str] = []
        self.matched_certs:     List[str] = []
        self.similarity_pairs:  List[Dict] = []  # [{jd_skill, candidate_skill, score, method}]
        self.required_match_pct:  float = 0.0
        self.preferred_match_pct: float = 0.0


class SkillMatcher:
    """
    Deterministic, no-LLM skill matching using a 3-layer strategy.
    Works completely offline — no model downloads required.
    """

    def __init__(self):
        self._tfidf = None  # Lazy init

    # ── Public API ────────────────────────────────────────────────

    def match(
        self,
        candidate_skills: List[str],
        required_skills:  List[str],
        preferred_skills: List[str],
        must_have_skills: List[str],
        required_certs:   List[str],
        candidate_certs:  List[str],
    ) -> SkillMatchResult:
        """
        Full skill matching run.
        Returns a SkillMatchResult with all matched/missing info.
        """
        result = SkillMatchResult()

        # Normalise all skill lists via taxonomy
        norm_candidate  = {s: self._normalise(s) for s in candidate_skills}
        norm_required   = [(s, self._normalise(s)) for s in required_skills]
        norm_preferred  = [(s, self._normalise(s)) for s in preferred_skills]

        # ── Match required skills ──────────────────────────────────
        for orig_req, norm_req in norm_required:
            matched, cand_skill, score, method = self._find_best_match(
                norm_req, norm_candidate
            )
            if matched:
                result.matched_required.append(orig_req)
                result.similarity_pairs.append({
                    "jd_skill":        orig_req,
                    "candidate_skill": cand_skill,
                    "score":           round(score, 3),
                    "method":          method,
                })
            else:
                result.missing_required.append(orig_req)

        # ── Match preferred skills ────────────────────────────────
        for orig_pref, norm_pref in norm_preferred:
            matched, cand_skill, score, method = self._find_best_match(
                norm_pref, norm_candidate
            )
            if matched:
                result.matched_preferred.append(orig_pref)

        # ── Match certifications (exact/fuzzy) ─────────────────────
        for cert in required_certs:
            norm_cert = self._normalise(cert)
            for cand_cert in candidate_certs:
                if norm_cert in self._normalise(cand_cert) or \
                   self._normalise(cand_cert) in norm_cert:
                    result.matched_certs.append(cert)
                    break

        # ── Compute percentages ───────────────────────────────────
        if required_skills:
            result.required_match_pct = round(
                len(result.matched_required) / len(required_skills) * 100, 1
            )
        if preferred_skills:
            result.preferred_match_pct = round(
                len(result.matched_preferred) / len(preferred_skills) * 100, 1
            )

        logger.info(
            f"Skill matching: {len(result.matched_required)}/{len(required_skills)} required, "
            f"{len(result.matched_preferred)}/{len(preferred_skills)} preferred"
        )
        return result

    # ── Matching Logic ────────────────────────────────────────────

    def _find_best_match(
        self,
        target_norm: str,
        candidate_norms: Dict[str, str],  # {original: normalised}
    ) -> Tuple[bool, str, float, str]:
        """
        Try all 3 layers and return (matched, candidate_original, score, method).
        """
        # Layer 1: Exact normalised match
        for orig, norm in candidate_norms.items():
            if target_norm == norm:
                return True, orig, 1.0, "exact"

        # Layer 1b: Substring match (e.g. "Python 3" in "Python")
        for orig, norm in candidate_norms.items():
            if target_norm in norm or norm in target_norm:
                return True, orig, 0.95, "substring"

        # Layer 2: TF-IDF cosine similarity
        best_score, best_cand = self._tfidf_best(target_norm, candidate_norms)
        if best_score >= SKILL_MATCH_THRESHOLD:
            return True, best_cand, best_score, "tfidf_cosine"

        # Layer 3: Jaccard token overlap fallback
        target_tokens = set(target_norm.lower().split())
        for orig, norm in candidate_norms.items():
            cand_tokens = set(norm.lower().split())
            union = target_tokens | cand_tokens
            if not union:
                continue
            jaccard = len(target_tokens & cand_tokens) / len(union)
            if jaccard >= SKILL_JACCARD_THRESHOLD:
                return True, orig, jaccard, "jaccard"

        return False, "", 0.0, "none"

    def _tfidf_best(
        self, target: str, candidates: Dict[str, str]
    ) -> Tuple[float, str]:
        """Compute TF-IDF cosine similarity between target and all candidates."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
        except ImportError:
            return 0.0, ""

        if not candidates:
            return 0.0, ""

        cand_list = list(candidates.items())   # [(original, normalised)]
        corpus = [target] + [n for _, n in cand_list]

        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            tfidf = vec.fit_transform(corpus)
            sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
            best_idx = int(np.argmax(sims))
            return float(sims[best_idx]), cand_list[best_idx][0]
        except Exception:
            return 0.0, ""

    def _normalise(self, skill: str) -> str:
        """Normalise a skill string via taxonomy lookup + lowercasing."""
        clean = skill.strip().lower()
        # Direct taxonomy lookup
        if clean in SKILL_TAXONOMY:
            return SKILL_TAXONOMY[clean].lower()
        # Remove special chars and retry
        clean_simple = clean.replace(".", "").replace("-", "").replace(" ", "")
        for key, value in SKILL_TAXONOMY.items():
            if key.replace(".", "").replace("-", "").replace(" ", "") == clean_simple:
                return value.lower()
        return clean
