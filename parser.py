"""
Clinical note parser.
Extracts labelled sections from de-identified discharge summaries.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClinicalNote:
    case_id: str = ""
    sex: Optional[str] = None
    service: Optional[str] = None
    chief_complaint: Optional[str] = None
    allergies: list[str] = field(default_factory=list)
    history_of_present_illness: Optional[str] = None
    past_medical_history: list[str] = field(default_factory=list)
    admission_physical_exam: Optional[str] = None
    discharge_physical_exam: Optional[str] = None
    admission_labs: Optional[str] = None
    pertinent_labs: Optional[str] = None
    discharge_labs: Optional[str] = None
    imaging: Optional[str] = None
    hospital_course: Optional[str] = None
    acute_problems: dict[str, str] = field(default_factory=dict)
    transitional_issues: Optional[str] = None
    discharge_medications: list[str] = field(default_factory=list)
    admission_medications: list[str] = field(default_factory=list)
    discharge_diagnosis: Optional[str] = None
    discharge_condition: Optional[str] = None
    discharge_instructions: Optional[str] = None
    followup_instructions: Optional[str] = None


# Section header patterns → canonical key
_SECTION_MAP = [
    (re.compile(r'Chief Complaint\s*:', re.I),            'chief_complaint'),
    (re.compile(r'Major Surgical or Invasive Procedure\s*:', re.I), '_skip'),
    (re.compile(r'History of Present Illness\s*:', re.I), 'history_of_present_illness'),
    (re.compile(r'Review of Systems\s*:', re.I),          '_skip'),
    (re.compile(r'Past Medical History\s*:', re.I),       'past_medical_history'),
    (re.compile(r'Social History\s*:', re.I),             'social_history'),
    (re.compile(r'Family History\s*:', re.I),             'family_history'),
    (re.compile(r'^\s*PHYSICAL EXAMINATION ON ADMISSION\s*:?\s*$', re.I), 'admission_physical_exam'),
    (re.compile(r'^\s*PHYSICAL EXAMINATION ON DISCHARGE\s*:?\s*$', re.I), 'discharge_physical_exam'),
    (re.compile(r'^\s*ADMISSION EXAM\s*:?\s*$', re.I),    'admission_physical_exam'),
    (re.compile(r'^\s*DISCHARGE EXAM\s*:?\s*$', re.I),    'discharge_physical_exam'),
    (re.compile(r'^\s*ADMISSION PHYSICAL(?: EXAM)?\s*:?\s*$', re.I), 'admission_physical_exam'),
    (re.compile(r'^\s*DISCHARGE PHYSICAL(?: EXAM)?\s*:?\s*$', re.I), 'discharge_physical_exam'),
    (re.compile(r'Physical\s+(?:Exam|\[REDACTED\]|[_\w]+)\s*:', re.I), 'admission_physical_exam'),
    (re.compile(r'Pertinent Results\s*:', re.I),          'pertinent_results'),
    (re.compile(r'^\s*LABS ON ADMISSION\s*:?\s*$', re.I), 'admission_labs'),
    (re.compile(r'^\s*LABS ON DISCHARGE\s*:?\s*$', re.I), 'discharge_labs'),
    (re.compile(r'^\s*LABS ON DISHCHARGE\s*:?\s*$', re.I), 'discharge_labs'),
    (re.compile(r'^\s*ADMISSION LABS\s*:?\s*$', re.I),   'admission_labs'),
    (re.compile(r'^\s*PERTINENT LABS\s*:?\s*$', re.I),   'pertinent_labs'),
    (re.compile(r'^\s*DISCHARGE LABS\s*:?\s*$', re.I),   'discharge_labs'),
    (re.compile(r'^\s*IMAGING STUDIES\s*:?\s*$', re.I),  'imaging'),
    (re.compile(r'^\s*IMAGING\s*:?\s*$', re.I),          'imaging'),
    (re.compile(r'(?:Brief )?Hospital Course\s*:', re.I), 'hospital_course'),
    (re.compile(r'Transitional [Ii]ssues\s*:', re.I),    'transitional_issues'),
    (re.compile(r'(?:___|\[REDACTED\])\s+on Admission\s*:|Medications on Admission\s*:', re.I), 'admission_medications_raw'),
    (re.compile(r'Discharge Medications\s*:', re.I),      'discharge_medications_raw'),
    (re.compile(r'Discharge Disposition\s*:', re.I),      'discharge_disposition'),
    (re.compile(r'Discharge Diagnosis\s*:', re.I),        'discharge_diagnosis'),
    (re.compile(r'Discharge Condition\s*:', re.I),        'discharge_condition'),
    (re.compile(r'Discharge Instructions\s*:', re.I),     'discharge_instructions'),
    (re.compile(r'Allergies\s*:', re.I),                  'allergies_raw'),
    (re.compile(r'Followup Instructions\s*:', re.I),      'followup_instructions'),
    (re.compile(r'Attending\s*:', re.I),                  '_skip'),
    (re.compile(r'Service\s*:', re.I),                    'service'),
    (re.compile(r'(?:Sex|Gender)\s*:', re.I),             'sex_raw'),
]


def _split_sections(text: str) -> dict[str, str]:
    """Split raw note text into labelled sections."""
    lines = text.splitlines()
    sections: dict[str, str] = {}
    current_key = '__header__'
    buffer: list[str] = []

    for line in lines:
        matched = False
        for pattern, key in _SECTION_MAP:
            if pattern.search(line):
                # save previous buffer
                sections[current_key] = '\n'.join(buffer).strip()
                buffer = []
                current_key = key
                # strip the header from the line itself
                remainder = pattern.sub('', line).strip()
                if remainder:
                    buffer.append(remainder)
                matched = True
                break
        if not matched:
            buffer.append(line)

    sections[current_key] = '\n'.join(buffer).strip()
    return sections


def _parse_medication_list(raw: str) -> list[str]:
    """Return clean list of medication strings from numbered list text."""
    meds: list[str] = []
    current: list[str] = []
    skipping_rx_detail = False
    skipping_non_med_detail = False
    skip_prefixes = (
        'The Preadmission', 'Tapered dose',
        'Disp #', '#*', 'Refills:', 'Duration:',
    )
    non_med_prefixes = (
        'Outpatient Physical Therapy', 'Physical Therapy',
        'Occupational Therapy', 'Speech Therapy',
    )

    def flush_current() -> None:
        if current:
            med = ' '.join(current)
            med = re.sub(r'\s+', ' ', med).strip()
            if len(med) > 5:
                meds.append(med)
            current.clear()

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        numbered = re.match(r'^(\d+)\.\s*(.*)$', line)

        if skipping_non_med_detail:
            if numbered:
                skipping_non_med_detail = False
            else:
                continue

        if re.match(r'^RX\s*\*', line, re.I):
            skipping_rx_detail = True
            continue
        if skipping_rx_detail:
            if numbered:
                skipping_rx_detail = False
            else:
                continue

        if any(line.startswith(p) for p in skip_prefixes):
            continue

        if numbered:
            flush_current()
            line = numbered.group(2).strip()

        if any(line.startswith(p) for p in non_med_prefixes):
            skipping_non_med_detail = True
            continue

        if line:
            current.append(line)

    flush_current()
    return meds


def _parse_pmh(raw: str) -> list[str]:
    items = []
    for line in raw.splitlines():
        line = line.strip().lstrip('-').strip()
        if line:
            items.append(line)
    return items


def _extract_acute_problems(hospital_course: str) -> dict[str, str]:
    """Extract named problem sections from hospital course narrative."""
    problems = {}
    # Match headers like "#COPD exacerbation:" or "# Atrial fibrillation:"
    pattern = re.compile(r'#\s*([^\n:]+?)\s*(?::|$)', re.MULTILINE)
    matches = list(pattern.finditer(hospital_course))
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(hospital_course)
        body = hospital_course[start:end].strip()
        problems[title] = body
    return problems


def parse_note(text: str, case_id: str) -> ClinicalNote:
    note = ClinicalNote(case_id=case_id)
    sections = _split_sections(text)

    note.chief_complaint = sections.get('chief_complaint')
    note.history_of_present_illness = sections.get('history_of_present_illness')
    note.admission_physical_exam = sections.get('admission_physical_exam')
    note.discharge_physical_exam = sections.get('discharge_physical_exam')
    note.admission_labs = sections.get('admission_labs')
    note.pertinent_labs = sections.get('pertinent_labs')
    note.discharge_labs = sections.get('discharge_labs')
    note.imaging = sections.get('imaging')
    note.transitional_issues = sections.get('transitional_issues')
    note.discharge_diagnosis = sections.get('discharge_diagnosis')
    note.discharge_condition = sections.get('discharge_condition')
    note.discharge_instructions = sections.get('discharge_instructions')
    note.followup_instructions = sections.get('followup_instructions')
    note.service = sections.get('service', '').strip()

    sex_raw = sections.get('sex_raw', '')
    if 'F' in sex_raw:
        note.sex = 'female'
    elif 'M' in sex_raw:
        note.sex = 'male'

    allergy_raw = sections.get('allergies_raw', '')
    note.allergies = [a.strip() for a in re.split(r'[/,]', allergy_raw) if a.strip()]

    pmh_raw = sections.get('past_medical_history', '')
    note.past_medical_history = _parse_pmh(pmh_raw)

    hc = sections.get('hospital_course', '')
    note.hospital_course = hc
    note.acute_problems = _extract_acute_problems(hc)

    note.admission_medications = _parse_medication_list(
        sections.get('admission_medications_raw', ''))
    note.discharge_medications = _parse_medication_list(
        sections.get('discharge_medications_raw', ''))

    return note
