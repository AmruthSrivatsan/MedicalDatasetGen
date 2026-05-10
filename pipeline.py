#!/usr/bin/env python3
"""
Medical Training Data Pipeline  (v2 — with MedGemma judge)
===========================================================
Converts raw clinical discharge summary .txt files into HIPAA-compliant,
abbreviation-expanded, MedGemma-judged instruction-following training data.

Full pipeline per file:
  1. HIPAA de-identification      (regex, Safe Harbor)
  2. Section parsing              (regex)
  3. Abbreviation expansion       (dictionary, longest-match)
  4. Training example generation  (rule-based templates)
  5. Rule-based validation        (PHI safety, format, length)
  6. MedGemma judge               (accuracy / completeness / abbreviations)
  7. Write PASS → JSONL  |  REJECT → rejected log

Usage:
    python pipeline.py --input <dir_or_file> --output <output_dir> [options]

Options:
    --input       Path to a .txt file or directory of .txt files
    --output      Output directory (default: ./output)
    --no-judge    Skip MedGemma judging (rule-based only, faster)
    --verbose     Print per-example judge scores
"""

import argparse
import json
import logging
import sys
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

from deidentify  import deidentify, stable_case_id
from parser      import parse_note
from generator   import generate_training_examples
from validator   import validate_batch
from judge       import judge_batch, JUDGE_MODEL, OLLAMA_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# ── Ollama health check ───────────────────────────────────────────────────────

def _check_ollama() -> bool:
    """Return True if Ollama is reachable and the judge model is available."""
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            models = [m["name"] for m in body.get("models", [])]
            available = any(JUDGE_MODEL in m or m in JUDGE_MODEL for m in models)
            if not available:
                log.warning(
                    "Judge model '%s' not found in Ollama. Available: %s",
                    JUDGE_MODEL, models
                )
            return available
    except Exception as e:
        log.warning("Ollama not reachable: %s", e)
        return False


# ── Per-file processing ───────────────────────────────────────────────────────

def process_file(
    path: Path,
    use_judge: bool,
    verbose: bool,
) -> tuple[list[dict], list[dict], dict]:
    """
    Full pipeline for one .txt file.
    Returns (passed_examples, rejected_examples, stats_dict).
    """
    raw     = path.read_text(encoding="utf-8", errors="replace")
    case_id = stable_case_id(raw)

    log.info("── %s  (case %s)", path.name, case_id)

    # Step 1 — De-identify
    clean = deidentify(raw)

    # Step 2 & 3 — Parse + expand
    note = parse_note(clean, case_id)

    # Step 4 — Generate
    examples = generate_training_examples(note)
    log.info("   Generated : %d examples", len(examples))

    # Step 5 — Rule-based validation
    rule_passed, rule_rejected = validate_batch(examples)
    log.info("   Rule check: %d passed, %d rejected", len(rule_passed), len(rule_rejected))

    # Step 6 — MedGemma judge
    if use_judge and rule_passed:
        log.info("   MedGemma judging %d examples ...", len(rule_passed))
        judge_passed, judge_rejected = judge_batch(
            rule_passed, clean, show_progress=verbose
        )
        log.info(
            "   Judge     : %d passed, %d rejected",
            len(judge_passed), len(judge_rejected)
        )
        all_passed   = judge_passed
        all_rejected = rule_rejected + judge_rejected
    else:
        all_passed   = rule_passed
        all_rejected = rule_rejected

    total = len(all_passed) + len(all_rejected)
    stats = {
        "file":            path.name,
        "case_id":         case_id,
        "generated":       len(examples),
        "rule_passed":     len(rule_passed),
        "rule_rejected":   len(rule_rejected),
        "judge_passed":    len(all_passed)                          if use_judge else None,
        "judge_rejected":  len(all_rejected) - len(rule_rejected)  if use_judge else None,
        "final_passed":    len(all_passed),
        "final_rejected":  len(all_rejected),
        "pass_rate_pct":   round(len(all_passed) / max(1, total) * 100, 1),
        "judge_used":      use_judge,
    }

    # Per-example score summary in verbose mode
    if verbose and use_judge:
        for ex in all_passed:
            j = ex.get("_judge", {})
            log.info(
                "     PASS  task=%-35s score=%s/9 (acc=%s cmp=%s abv=%s)",
                ex.get("task"), j.get("total_score", "?"),
                j.get("score_accuracy", "?"),
                j.get("score_completeness", "?"),
                j.get("score_abbrev", "?"),
            )
        for ex in all_rejected:
            j = ex.get("_judge", {})
            if j:
                log.info(
                    "     FAIL  task=%-35s score=%s/9 | acc: %s | cmp: %s | abv: %s",
                    ex.get("task"), j.get("total_score", 0),
                    j.get("reason_accuracy", ""),
                    j.get("reason_completeness", ""),
                    j.get("reason_abbrev", ""),
                )

    return all_passed, all_rejected, stats


# ── Main run ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    if input_path.is_dir():
        files = sorted(input_path.glob("*.txt"))
    elif input_path.suffix == ".txt":
        files = [input_path]
    else:
        log.error("--input must be a .txt file or directory of .txt files")
        sys.exit(1)

    if not files:
        log.warning("No .txt files found in %s", input_path)
        sys.exit(0)

    # Judge availability check
    use_judge = not args.no_judge
    if use_judge:
        log.info("Checking Ollama + %s ...", JUDGE_MODEL)
        if not _check_ollama():
            log.warning(
                "MedGemma judge unavailable — falling back to rule-based only.\n"
                "  Start Ollama : ollama serve\n"
                "  Pull model   : ollama pull %s", JUDGE_MODEL
            )
            use_judge = False
        else:
            log.info("MedGemma judge ready")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file   = output_dir / f"training_data_{timestamp}.jsonl"
    rej_file   = output_dir / f"rejected_{timestamp}.jsonl"
    stats_file = output_dir / f"pipeline_report_{timestamp}.json"

    all_passed:   list[dict] = []
    all_rejected: list[dict] = []
    per_file_stats: list[dict] = []

    for f in files:
        try:
            passed, rejected, stats = process_file(f, use_judge, args.verbose)
            all_passed.extend(passed)
            all_rejected.extend(rejected)
            per_file_stats.append(stats)
        except Exception as exc:
            log.error("Failed to process %s: %s", f.name, exc, exc_info=True)
            per_file_stats.append({"file": f.name, "error": str(exc)})

    # ── Write outputs ─────────────────────────────────────────────────────────

    # Strip internal metadata from final training file
    clean_passed = [
        {k: v for k, v in ex.items() if not k.startswith("_")}
        for ex in all_passed
    ]

    with out_file.open("w", encoding="utf-8") as fh:
        for ex in clean_passed:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info("Wrote %d training examples → %s", len(clean_passed), out_file)

    # Rejected log keeps full judge metadata for human review
    with rej_file.open("w", encoding="utf-8") as fh:
        for ex in all_rejected:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    if all_rejected:
        log.info("Wrote %d rejected examples → %s", len(all_rejected), rej_file)

    # Score distribution (for judged runs)
    scores = [
        ex.get("_judge", {}).get("total_score")
        for ex in all_passed
        if "_judge" in ex
    ]
    score_dist = {}
    for s in scores:
        if s is not None:
            score_dist[str(s)] = score_dist.get(str(s), 0) + 1

    total = len(all_passed) + len(all_rejected)
    report = {
        "run_timestamp":            timestamp,
        "judge_model":              JUDGE_MODEL if use_judge else None,
        "judge_enabled":            use_judge,
        "input":                    str(input_path),
        "files_processed":          len(files),
        "total_generated":          sum(s.get("generated", 0) for s in per_file_stats),
        "total_passed":             len(all_passed),
        "total_rejected":           len(all_rejected),
        "overall_pass_rate_pct":    round(len(all_passed) / max(1, total) * 100, 1),
        "judge_score_distribution": score_dist,
        "per_file":                 per_file_stats,
    }
    stats_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Pipeline report → %s", stats_file)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Files processed   : {len(files)}")
    print(f"  Examples generated: {report['total_generated']}")
    print(f"  Passed            : {len(all_passed)}")
    print(f"  Rejected          : {len(all_rejected)}")
    print(f"  Pass rate         : {report['overall_pass_rate_pct']}%")
    print(f"  Judge             : {'MedGemma (' + JUDGE_MODEL + ')' if use_judge else 'rule-based only'}")
    if score_dist:
        dist_str = "  ".join(
            f"{k}/9 x{v}" for k, v in sorted(score_dist.items(), reverse=True)
        )
        print(f"  Score distribution: {dist_str}")
    print(f"  Training JSONL    : {out_file}")
    print(f"  Rejected log      : {rej_file}")
    print(f"  Report            : {stats_file}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Medical discharge summary → HIPAA-compliant + MedGemma-judged training data"
    )
    parser.add_argument("--input",    required=True, help="Input .txt file or directory")
    parser.add_argument("--output",   default="output", help="Output directory")
    parser.add_argument("--no-judge", action="store_true", help="Skip MedGemma judging")
    parser.add_argument("--verbose",  action="store_true", help="Show per-example judge scores")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
