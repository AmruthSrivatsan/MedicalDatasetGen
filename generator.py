"""
Training data generator.
Converts a parsed ClinicalNote into a rich set of instruction-following
training examples covering different clinical reasoning tasks.
"""

from __future__ import annotations
import re
from typing import Optional
from parser import ClinicalNote
from expander import expand_text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(s: Optional[str]) -> str:
    if not s:
        return ""
    return expand_text(s.strip())


def _list_to_prose(items: list[str], conjunction: str = "and") -> str:
    items = [expand_text(i) for i in items if i.strip()]
    if not items:
        return "none documented"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", {conjunction} " + items[-1]


def _record(system: str, user: str, assistant: str, case_id: str, task: str) -> dict:
    return {
        "case_id": case_id,
        "task": task,
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
            {"role": "assistant","content": assistant},
        ]
    }


_MED_KEY_STOP = re.compile(
    r'\b(?:\d+(?:\.\d+)?|mg|mcg|g|mL|units?|UNIT|PO|oral|IH|INH|NEB|'
    r'TAB|CAP|DROP|SPRY|NASAL|Q\d+H|BID|TID|QID|QHS|QAM|QPM|DAILY|'
    r'PRN|ASDIR)\b',
    re.I,
)
_FORMULATION_TOKENS = {
    "er", "sr", "xr", "dr", "extended", "release", "delayed", "sustained"
}


def _med_key(med: str) -> str:
    """Create a conservative medication identity key for reconciliation."""
    text = re.sub(r'\[[^\]]+\]', ' ', med)
    text = re.sub(r'\([^)]*\)', ' ', text)
    match = _MED_KEY_STOP.search(text)
    name = text[:match.start()] if match else text
    name = re.sub(r'[^A-Za-z0-9]+', ' ', name).lower().strip()
    tokens = [t for t in name.split() if t not in _FORMULATION_TOKENS]
    if not tokens:
        tokens = re.sub(r'[^A-Za-z0-9]+', ' ', med).lower().split()[:2]
    return " ".join(tokens[:4])


def _med_map(meds: list[str]) -> dict[str, list[str]]:
    mapped: dict[str, list[str]] = {}
    for med in meds:
        key = _med_key(med)
        if key:
            mapped.setdefault(key, []).append(med)
    return mapped


def _med_signature(meds: list[str]) -> set[str]:
    return {re.sub(r'\s+', ' ', m.lower()).strip() for m in meds}


_SYSTEM_CLINICAL = (
    "You are a clinical decision-support assistant. "
    "You provide accurate, thorough answers based solely on the provided "
    "de-identified medical record. Do not speculate beyond the available information."
)

_SYSTEM_SUMMARY = (
    "You are a medical documentation assistant. "
    "Produce clear, structured summaries from de-identified clinical notes "
    "for use in care coordination and handoffs."
)

_SYSTEM_EDUCATOR = (
    "You are a medical education assistant helping clinicians understand "
    "treatment rationale and drug interactions documented in clinical cases."
)


# ── Individual example generators ────────────────────────────────────────────

def gen_admission_summary(note: ClinicalNote) -> Optional[dict]:
    hpi = _clean(note.history_of_present_illness)
    cc  = _clean(note.chief_complaint)
    pmh = _list_to_prose(note.past_medical_history)
    if not hpi:
        return None
    sex = note.sex or "the patient"
    answer = (
        f"Chief complaint: {cc}\n\n"
        f"History of present illness: {hpi}\n\n"
        f"Relevant past medical history: {pmh}"
    )
    return _record(
        _SYSTEM_SUMMARY,
        "Summarise the admission history and chief complaint for this de-identified patient.",
        answer,
        note.case_id, "admission_summary"
    )


def gen_problem_list(note: ClinicalNote) -> Optional[dict]:
    dx = _clean(note.discharge_diagnosis)
    problems = note.acute_problems
    if not dx and not problems:
        return None
    lines = [f"Discharge diagnosis:\n{dx}\n"] if dx else []
    for title, body in problems.items():
        lines.append(f"Problem – {expand_text(title)}:\n{expand_text(body)}")
    return _record(
        _SYSTEM_CLINICAL,
        "List the clinical problems addressed during this hospitalisation and briefly describe how each was managed.",
        "\n\n".join(lines),
        note.case_id, "problem_list"
    )


def gen_medication_changes(note: ClinicalNote) -> Optional[dict]:
    adm  = note.admission_medications
    disc = note.discharge_medications
    if not adm or not disc:
        return None

    adm_map = _med_map(adm)
    disc_map = _med_map(disc)
    adm_set = set(adm_map)
    disc_set = set(disc_map)

    new_meds = [
        expand_text(m)
        for key in sorted(disc_set - adm_set)
        for m in disc_map[key]
    ]
    stop_meds = [
        expand_text(m)
        for key in sorted(adm_set - disc_set)
        for m in adm_map[key]
    ]
    changed_meds = []
    unchanged_meds = []
    for key in sorted(adm_set & disc_set):
        if _med_signature(adm_map[key]) != _med_signature(disc_map[key]):
            before = "; ".join(expand_text(m) for m in adm_map[key])
            after = "; ".join(expand_text(m) for m in disc_map[key])
            changed_meds.append(f"{key}: admission regimen was {before}; discharge regimen was {after}")
        else:
            unchanged_meds.extend(expand_text(m) for m in disc_map[key])

    answer = ""
    if new_meds:
        answer += "Newly started medications:\n" + "\n".join(f"  • {m}" for m in new_meds) + "\n\n"
    if stop_meds:
        answer += "Discontinued medications:\n" + "\n".join(f"  • {m}" for m in stop_meds) + "\n\n"
    if changed_meds:
        answer += "Dose or regimen changes:\n" + "\n".join(f"  • {m}" for m in changed_meds) + "\n\n"
    if unchanged_meds:
        answer += "Continued medications (selected):\n" + "\n".join(f"  • {m}" for m in unchanged_meds[:10])

    return _record(
        _SYSTEM_CLINICAL,
        "Compare the admission and discharge medication lists. Identify any new medications started, any medications discontinued, and any dose changes.",
        answer.strip(),
        note.case_id, "medication_reconciliation"
    )


def gen_discharge_plan(note: ClinicalNote) -> Optional[dict]:
    ti   = _clean(note.transitional_issues)
    inst = _clean(note.discharge_instructions)
    fup  = _clean(note.followup_instructions)
    cond = _clean(note.discharge_condition)
    if not ti and not inst and not fup:
        return None
    answer = ""
    if cond:
        answer += f"Discharge condition: {cond}\n\n"
    if ti:
        answer += f"Transitional issues and follow-up plan:\n{ti}\n\n"
    if inst:
        answer += f"Patient instructions summary:\n{inst}\n\n"
    if fup:
        answer += f"Follow-up instructions:\n{fup}"
    return _record(
        _SYSTEM_SUMMARY,
        "Describe the discharge plan and follow-up requirements for this patient.",
        answer.strip(),
        note.case_id, "discharge_plan"
    )


def gen_lab_interpretation(note: ClinicalNote) -> Optional[dict]:
    adm_labs  = _clean(note.admission_labs)
    pert_labs = _clean(note.pertinent_labs)
    disc_labs = _clean(note.discharge_labs)
    if not adm_labs:
        return None
    answer = ""
    if adm_labs:
        answer += f"Admission laboratory results:\n{adm_labs}\n\n"
    if pert_labs:
        answer += f"Pertinent inpatient laboratory results:\n{pert_labs}\n\n"
    if disc_labs:
        answer += f"Discharge laboratory results:\n{disc_labs}"
    return _record(
        _SYSTEM_CLINICAL,
        "Summarise the key laboratory findings from admission through discharge and note any clinically significant trends.",
        answer.strip(),
        note.case_id, "lab_interpretation"
    )


def gen_drug_interaction_rationale(note: ClinicalNote) -> Optional[dict]:
    """Generate a reasoning example around drug interactions documented in the note."""
    hc = _clean(note.hospital_course)
    if not hc or 'theophylline' not in hc.lower():
        return None
    q = (
        "The hospital course mentions adjusting theophylline dosing. "
        "Based on the documented clinical reasoning, explain why theophylline "
        "was adjusted and what interactions or risks were considered."
    )
    paragraphs = [
        re.sub(r'\s+', ' ', p).strip()
        for p in re.split(r'\n\s*\n+', hc)
        if p.strip()
    ]
    keywords = ("theophylline", "amiodarone", "QT interval", "corrected QT interval")
    relevant = [p for p in paragraphs if any(k.lower() in p.lower() for k in keywords)]
    rationale = "\n\n".join(relevant[:3]) if relevant else hc
    return _record(
        _SYSTEM_EDUCATOR,
        q,
        f"Based on the clinical note:\n\n{rationale}",
        note.case_id, "drug_interaction_rationale"
    )


def gen_allergy_profile(note: ClinicalNote) -> Optional[dict]:
    if not note.allergies:
        return None
    expanded = [expand_text(a) for a in note.allergies]
    answer = "Documented allergies:\n" + "\n".join(f"  • {a}" for a in expanded)
    return _record(
        _SYSTEM_CLINICAL,
        "List all documented allergies for this patient.",
        answer,
        note.case_id, "allergy_profile"
    )


def gen_clinical_qa(note: ClinicalNote) -> list[dict]:
    """Generate targeted clinical QA pairs from structured fields."""
    examples = []
    hpi = _clean(note.history_of_present_illness)
    hc  = _clean(note.hospital_course)

    # Chief complaint
    if note.chief_complaint:
        examples.append(_record(
            _SYSTEM_CLINICAL,
            "What was the patient's chief complaint on admission?",
            _clean(note.chief_complaint),
            note.case_id, "qa_chief_complaint"
        ))

    # PMH
    if note.past_medical_history:
        examples.append(_record(
            _SYSTEM_CLINICAL,
            "What is the patient's relevant past medical history?",
            _list_to_prose(note.past_medical_history),
            note.case_id, "qa_pmh"
        ))

    # Imaging
    if note.imaging:
        examples.append(_record(
            _SYSTEM_CLINICAL,
            "What did the imaging studies show during this hospitalisation?",
            _clean(note.imaging),
            note.case_id, "qa_imaging"
        ))

    # Discharge diagnosis
    if note.discharge_diagnosis:
        examples.append(_record(
            _SYSTEM_CLINICAL,
            "What were the primary and secondary diagnoses at discharge?",
            _clean(note.discharge_diagnosis),
            note.case_id, "qa_discharge_diagnosis"
        ))

    return examples


# ── Main entry ────────────────────────────────────────────────────────────────

def generate_training_examples(note: ClinicalNote) -> list[dict]:
    """Return all training examples for a single ClinicalNote."""
    generators = [
        gen_admission_summary(note),
        gen_problem_list(note),
        gen_medication_changes(note),
        gen_discharge_plan(note),
        gen_lab_interpretation(note),
        gen_drug_interaction_rationale(note),
        gen_allergy_profile(note),
    ]
    examples = [ex for ex in generators if ex is not None]
    examples.extend(gen_clinical_qa(note))
    return examples
