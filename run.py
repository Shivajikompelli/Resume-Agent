"""
run.py — Command-Line Interface for the Resume Screening Agent

Usage:
  python run.py --resume path/to/resume.pdf --jd path/to/jd.txt
  python run.py --resume resume.pdf --jd jd.txt --output report.json
"""

import argparse
import json
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Resume Screening & Eligibility Validation Agent (Groq-powered)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --resume john_doe.pdf --jd senior_ml_engineer.txt
  python run.py --resume resume.docx --jd jd.txt --output result.json --verbose

Get your free Groq API key at: https://console.groq.com
        """,
    )
    parser.add_argument("--resume",  required=True, help="Path to resume file (PDF/DOCX/TXT)")
    parser.add_argument("--jd",      required=True, help="Path to job description text file")
    parser.add_argument("--output",  default=None,  help="Path to save JSON report (optional)")
    parser.add_argument("--model",   default="llama3-70b-8192",
                        choices=["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
                        help="Groq model to use (default: llama3-70b-8192)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    parser.add_argument("--api-key", default=None,
                        help="Groq API key (or set GROQ_API_KEY env var / .env file)")

    args = parser.parse_args()

    # Set API key
    if args.api_key:
        os.environ["GROQ_API_KEY"] = args.api_key
    os.environ["GROQ_MODEL"] = args.model

    # Validate paths
    resume_path = Path(args.resume)
    jd_path     = Path(args.jd)

    if not resume_path.exists():
        print(f" Resume file not found: {resume_path}", file=sys.stderr)
        sys.exit(1)
    if not jd_path.exists():
        print(f" JD file not found: {jd_path}", file=sys.stderr)
        sys.exit(1)

    jd_text = jd_path.read_text(encoding="utf-8")

    # Progress callback
    def progress(msg: str, pct: int):
        if args.verbose:
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"  [{bar}] {pct:3d}%  {msg}")
        else:
            print(f"  [{pct:3d}%] {msg}")

    # Run pipeline
    print(f"\n Resume Screening Agent")
    print(f"   Resume : {resume_path}")
    print(f"   JD     : {jd_path}")
    print(f"   Model  : {args.model}")
    print()

    from agent.orchestrator import EvaluationPipeline
    pipeline = EvaluationPipeline()
    report   = pipeline.evaluate(resume_path, jd_text, progress)

    # Print summary
    scoring = report.get("scoring", {})
    expl    = report.get("explainability", {})
    skills  = report.get("skills_analysis", {})

    cat     = scoring.get("eligibility_category", "UNKNOWN")
    score   = scoring.get("composite_score", 0)

    COLORS = {
        "STRONGLY_ELIGIBLE":  "\033[92m",  # Green
        "ELIGIBLE":           "\033[96m",  # Cyan
        "PARTIALLY_ELIGIBLE": "\033[93m",  # Yellow
        "NOT_ELIGIBLE":       "\033[91m",  # Red
    }
    RESET = "\033[0m"
    color = COLORS.get(cat, "")

    print(f"\n{'='*60}")
    print(f"  {color}▶ {cat.replace('_',' ').title()} — Score: {score}/100{RESET}")
    print(f"{'='*60}")
    print(f"\n  {expl.get('reasoning_summary','')}\n")

    # Score breakdown
    s = report.get("scoring", {})
    print("  Score Breakdown:")
    print(f"    Required Skills  : {s.get('required_skill_score',0):5.1f} / 40")
    print(f"    Experience       : {s.get('experience_score',0):5.1f} / 25")
    print(f"    Education        : {s.get('education_score',0):5.1f} / 20")
    print(f"    Preferred Skills : {s.get('preferred_skill_score',0):5.1f} / 10")
    print(f"    Certifications   : {s.get('certification_bonus',0):5.1f} /  5")
    print(f"    ─────────────────────────────")
    print(f"    TOTAL            : {score:5.1f} / 100")

    if skills.get("required_skills_missing"):
        print(f"\n  Missing Skills: {', '.join(skills['required_skills_missing'])}")

    if expl.get("strengths"):
        print(f"\n  Strengths:")
        for s in expl["strengths"]:
            print(f"     {s}")

    if expl.get("gaps"):
        print(f"\n  Gaps:")
        for g in expl["gaps"]:
            print(f"    ⚠️  {g}")

    if scoring.get("hard_disqualifier_triggered"):
        print(f"\n   HARD DISQUALIFIER: {scoring.get('hard_disqualifier_reason')}")

    # Save report
    json_str = json.dumps(report, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"\n   Report saved → {args.output}")
    else:
        # Auto-save to reports/ directory
        os.makedirs("reports", exist_ok=True)
        out_path = f"reports/evaluation_{report.get('evaluation_id','')[:8]}.json"
        Path(out_path).write_text(json_str, encoding="utf-8")
        print(f"\n   Report saved → {out_path}")

    print(f"   Evaluation ID: {report.get('evaluation_id','')}\n")


if __name__ == "__main__":
    main()
