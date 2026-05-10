"""
MedGemma Judge Module
=====================
Uses medgemma1.5:4b via Ollama to evaluate each training example on:
  1. Clinical accuracy  — does the assistant answer match the source note?
  2. Completeness       — are key clinical facts missing?
  3. Abbreviation cleanliness — any unexpanded medical abbreviations remaining?

Returns a structured verdict: PASS or REJECT with per-dimension scores and reasons.
PHI never reaches the model — only de-identified text is judged.
"""

from __future__ import annotations
import json
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
import urllib.request
import urllib.error

log = logging.getLogger(__name__)

OLLAMA_URL   = "http://localhost:11434/api/generate"
JUDGE_MODEL  = "medgemma1.5:4b"
TIMEOUT_SECS = 120   # generous for a 4b model on CPU
MAX_RETRIES  = 2
MAX_SOURCE_CHARS = 7000


# ── Verdict dataclass ─────────────────────────────────────────────────────────

@dataclass
class JudgeVerdict:
    passed: bool
    score_accuracy:     int = 0   # 0-3
    score_completeness: int = 0   # 0-3
    score_abbrev:       int = 0   # 0-3
    total_score:        int = 0   # 0-9
    reason_accuracy:     str = ""
    reason_completeness: str = ""
    reason_abbrev:       str = ""
    raw_response:        str = ""
    error:               Optional[str] = None


# ── Prompt builder ────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """You are a clinical NLP quality auditor evaluating de-identified medical training examples.
You must be strict and precise. Your job is to catch errors, not to be lenient.
Respond ONLY with valid JSON — no markdown, no prose outside the JSON object."""

_JUDGE_PROMPT_TEMPLATE = """You are evaluating a training example derived from a de-identified clinical discharge summary.

SOURCE NOTE (de-identified):
\"\"\"
{source}
\"\"\"

TRAINING EXAMPLE:
Task: {task}
User question: {user}
Assistant answer: {assistant}

Evaluate the assistant answer on EXACTLY these three dimensions. Score each 0-3:

ACCURACY (0-3): Does every clinical fact in the answer correctly reflect the source note?
  3 = fully accurate, no errors
  2 = mostly accurate, minor imprecision
  1 = contains an inaccuracy that could mislead a clinician
  0 = factually wrong or contradicts the source

COMPLETENESS (0-3): Does the answer include all key clinical information relevant to the question?
  3 = complete, nothing important missing
  2 = mostly complete, minor omission
  1 = missing a clinically significant fact
  0 = critically incomplete

ABBREVIATIONS (0-3): Are all medical abbreviations properly expanded in the assistant answer?
  3 = no unexpanded abbreviations
  2 = 1-2 minor abbreviations remain (non-critical)
  1 = several abbreviations unexpanded
  0 = pervasive unexpanded abbreviations

Respond with ONLY this JSON object:
{{
  "score_accuracy": <0-3>,
  "reason_accuracy": "<one sentence>",
  "score_completeness": <0-3>,
  "reason_completeness": "<one sentence>",
  "score_abbrev": <0-3>,
  "reason_abbrev": "<one sentence>"
}}"""


# ── Ollama call ───────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, system: str) -> str:
    """Send a request to the local Ollama API and return the response text."""
    payload = json.dumps({
        "model":  JUDGE_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,   # deterministic judging
            "num_predict": 300,
            "num_ctx": 8192,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECS) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "").strip()


def _parse_verdict(raw: str) -> dict:
    """Extract JSON object from model response, tolerating minor formatting issues."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Find first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw[:200]}")
    return json.loads(match.group())


def _window_from(source_note: str, labels: tuple[str, ...]) -> str:
    """Return a task-relevant source window, falling back to a head/tail excerpt."""
    lower = source_note.lower()
    positions = [lower.find(label.lower()) for label in labels]
    positions = [p for p in positions if p >= 0]
    if positions:
        start = max(0, min(positions) - 500)
        return source_note[start:start + MAX_SOURCE_CHARS]

    if len(source_note) <= MAX_SOURCE_CHARS:
        return source_note
    half = MAX_SOURCE_CHARS // 2
    return (
        source_note[:half]
        + "\n\n[... middle of source note omitted for judge context ...]\n\n"
        + source_note[-half:]
    )


def _source_excerpt_for_task(source_note: str, task: str) -> str:
    if task in {"medication_reconciliation"}:
        return _window_from(source_note, ("Medications on Admission", "Discharge Medications"))
    if task in {"discharge_plan", "qa_discharge_diagnosis"}:
        return _window_from(source_note, ("Discharge Condition", "Discharge Instructions", "Discharge Diagnosis"))
    if task in {"lab_interpretation", "qa_imaging"}:
        return _window_from(source_note, ("Pertinent Results", "ADMISSION LABS", "IMAGING"))
    if task in {"problem_list", "drug_interaction_rationale"}:
        return _window_from(source_note, ("Brief Hospital Course", "Hospital Course"))
    if task in {"admission_summary", "qa_chief_complaint", "qa_pmh", "allergy_profile"}:
        return source_note[:MAX_SOURCE_CHARS]
    return _window_from(source_note, ())


# ── Public interface ──────────────────────────────────────────────────────────

PASS_THRESHOLD = 6   # out of 9; any dimension scoring 0 auto-rejects

def judge_example(example: dict, source_note: str) -> JudgeVerdict:
    """
    Judge a single training example against its source note.
    Returns a JudgeVerdict. Never raises — errors are captured in verdict.error.
    """
    messages  = example.get("messages", [])
    user_msg  = next((m["content"] for m in messages if m["role"] == "user"), "")
    asst_msg  = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    task      = example.get("task", "unknown")

    source_excerpt = _source_excerpt_for_task(source_note, task)

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        source=source_excerpt,
        task=task,
        user=user_msg[:400],
        assistant=asst_msg[:800],
    )

    raw = ""
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            raw = _call_ollama(prompt, _JUDGE_SYSTEM)
            parsed = _parse_verdict(raw)
            break
        except urllib.error.URLError as e:
            last_error = f"Ollama connection error: {e}"
            log.warning("Judge attempt %d failed: %s", attempt + 1, last_error)
            time.sleep(2)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"JSON parse error: {e} | raw: {raw[:200]}"
            log.warning("Judge attempt %d parse failed: %s", attempt + 1, last_error)
            time.sleep(1)
        except Exception as e:
            last_error = f"Unexpected error: {e}"
            log.warning("Judge attempt %d unexpected error: %s", attempt + 1, last_error)
            time.sleep(1)
    else:
        # All retries exhausted
        return JudgeVerdict(
            passed=False,
            error=last_error,
            raw_response=raw,
        )

    s_acc  = int(parsed.get("score_accuracy",     0))
    s_comp = int(parsed.get("score_completeness", 0))
    s_abbr = int(parsed.get("score_abbrev",       0))
    total  = s_acc + s_comp + s_abbr

    # Auto-reject if any dimension is 0 (critical failure)
    any_zero = (s_acc == 0 or s_comp == 0 or s_abbr == 0)
    passed   = (total >= PASS_THRESHOLD) and not any_zero

    return JudgeVerdict(
        passed=passed,
        score_accuracy=s_acc,
        score_completeness=s_comp,
        score_abbrev=s_abbr,
        total_score=total,
        reason_accuracy=parsed.get("reason_accuracy", ""),
        reason_completeness=parsed.get("reason_completeness", ""),
        reason_abbrev=parsed.get("reason_abbrev", ""),
        raw_response=raw,
    )


def judge_batch(
    examples: list[dict],
    source_note: str,
    show_progress: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Judge a list of examples against a source note.
    Returns (passed_examples, rejected_examples).
    Rejected examples get a '_judge' key with the verdict details.
    """
    passed, rejected = [], []

    for i, ex in enumerate(examples):
        if show_progress:
            log.info("  Judging [%d/%d] task=%s ...", i + 1, len(examples), ex.get("task"))

        verdict = judge_example(ex, source_note)

        verdict_dict = {
            "passed":              verdict.passed,
            "total_score":         verdict.total_score,
            "score_accuracy":      verdict.score_accuracy,
            "score_completeness":  verdict.score_completeness,
            "score_abbrev":        verdict.score_abbrev,
            "reason_accuracy":     verdict.reason_accuracy,
            "reason_completeness": verdict.reason_completeness,
            "reason_abbrev":       verdict.reason_abbrev,
            "error":               verdict.error,
        }

        if verdict.error:
            log.warning("    Judge error for task=%s: %s", ex.get("task"), verdict.error)
            # On judge failure, mark as rejected for human review
            ex["_judge"] = verdict_dict
            rejected.append(ex)
        elif verdict.passed:
            ex["_judge"] = verdict_dict
            passed.append(ex)
        else:
            log.info(
                "    REJECTED task=%s score=%d/9 (acc=%d cmp=%d abv=%d)",
                ex.get("task"), verdict.total_score,
                verdict.score_accuracy, verdict.score_completeness, verdict.score_abbrev
            )
            ex["_judge"] = verdict_dict
            rejected.append(ex)

    return passed, rejected
