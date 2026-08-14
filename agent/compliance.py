"""
agent/compliance.py — PII Detection & Protected Attribute Redaction

Runs BEFORE any LLM processing. Strips bias-inducing content.
"""

import re
import logging
from typing import Dict, List, Tuple
from config import PII_PATTERNS

logger = logging.getLogger(__name__)


class ComplianceService:
    """
    Detects and redacts protected attributes from resume text.
    Must be called before any LLM or scoring step.
    """

    def __init__(self):
        # Pre-compile all regex patterns for efficiency
        self._compiled: Dict[str, List[re.Pattern]] = {}
        for category, patterns in PII_PATTERNS.items():
            self._compiled[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def scrub(self, text: str) -> Tuple[str, List[str]]:
        """
        Redact all protected attributes from resume text.

        Returns:
            scrubbed_text: Text with PII replaced by [REDACTED]
            redacted_attrs: List of attribute categories that were found and removed
        """
        redacted_categories = []
        scrubbed = text

        for category, patterns in self._compiled.items():
            found = False
            for pattern in patterns:
                if pattern.search(scrubbed):
                    found = True
                    scrubbed = pattern.sub("[REDACTED]", scrubbed)
            if found:
                redacted_categories.append(category)

        # Remove lines that contain only [REDACTED] or whitespace
        lines = scrubbed.split("\n")
        lines = [l for l in lines if l.strip() not in ("", "[REDACTED]")]
        scrubbed = "\n".join(lines)

        if redacted_categories:
            logger.info(f"Compliance: redacted {redacted_categories}")

        return scrubbed, redacted_categories

    def check_jd(self, jd_text: str) -> Tuple[bool, List[str]]:
        """
        Check a job description for discriminatory / exclusionary language.

        Returns:
            is_clean: True if no issues found
            issues: List of problem descriptions
        """
        issues = []
        discriminatory_patterns = [
            (r"\b(only (males?|females?|men|women) (need|should) apply)\b", "Gender-exclusive language"),
            (r"\b(must be under|must be over|age (limit|requirement))\b",   "Age discrimination"),
            (r"\b(native (english|speaker))\b",                              "National origin proxy"),
            (r"\b(recent (grad|graduate)s? only)\b",                         "Age-proxy (recent grad only)"),
            (r"\b(physically (fit|able|capable))\b",                         "Disability-adjacent language"),
        ]
        for pat, label in discriminatory_patterns:
            if re.search(pat, jd_text, re.IGNORECASE):
                issues.append(label)

        return len(issues) == 0, issues

    def audit_record(self, evaluation_id: str, redacted: List[str]) -> Dict:
        """Generate an audit record for the compliance log."""
        from datetime import datetime, timezone
        return {
            "evaluation_id":   evaluation_id,
            "audited_at":      datetime.now(timezone.utc).isoformat(),
            "redacted_attrs":  redacted,
            "gate_passed":     True,
        }
