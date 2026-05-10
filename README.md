# Medical Training Data Pipeline

This repository converts MIMIC-style clinical discharge summary `.txt` files into
de-identified JSONL instruction-following training examples.

The pipeline:

1. De-identifies obvious HIPAA Safe Harbor identifiers with regex rules.
2. Parses common discharge-summary sections.
3. Expands common clinical abbreviations.
4. Generates training examples for summaries, QA, medications, labs, discharge plans, and problem lists.
5. Runs rule-based validation for PHI-like patterns, formatting, short outputs, and residual abbreviations.
6. Optionally judges examples with `medgemma1.5:4b` through local Ollama.

## Requirements

- Python 3.10 or newer.
- No required Python packages.
- Optional for stronger quality checks: Ollama with `medgemma1.5:4b`.

Optional judge setup:

```bash
ollama serve
ollama pull medgemma1.5:4b
```

## Quick Start

Run the portable rule-based pipeline on the included synthetic example:

```bash
python3 pipeline.py --input examples --output output --no-judge
```

Run with the MedGemma judge if Ollama is available:

```bash
python3 pipeline.py --input examples --output output
```

Use a directory of your own similar `.txt` files:

```bash
python3 pipeline.py --input /path/to/txt_notes --output output --no-judge
```

Use a single file:

```bash
python3 pipeline.py --input /path/to/note.txt --output output --no-judge
```

## Outputs

Each run writes timestamped files to the output directory:

- `training_data_YYYYMMDD_HHMMSS.jsonl`: accepted training examples.
- `rejected_YYYYMMDD_HHMMSS.jsonl`: rejected examples with validation or judge metadata.
- `pipeline_report_YYYYMMDD_HHMMSS.json`: run summary and per-file pass rates.

Each JSONL row has this shape:

```json
{
  "case_id": "CASE-...",
  "task": "admission_summary",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Verification Results

Verified locally on five de-identified MIMIC-style `.txt` examples. Those local
clinical-note files are intentionally not committed to this public repository.
The repository includes `examples/synthetic_discharge_note.txt` for a safe
smoke test.

Rule-based run:

```bash
python3 -m py_compile *.py
python3 pipeline.py --input /path/to/local_test_notes --output output_verify --no-judge
```

Result:

- Files processed: 5
- Examples generated: 49
- Rule-based passed: 49
- Rejected: 0
- Pass rate: 100.0%
- Additional scan found 0 tracked PHI-pattern hits and 0 tracked high-risk abbreviation hits.

Judged run with local `medgemma1.5:4b`:

```bash
python3 pipeline.py --input /path/to/local_test_notes --output output_judged_verify
```

Result:

- Files processed: 5
- Examples generated: 49
- Judge passed: 49
- Judge rejected: 0
- Pass rate: 100.0%
- Score distribution: `9/9 x43`, `8/9 x3`, `7/9 x3`

Each judged run writes its report to `pipeline_report_YYYYMMDD_HHMMSS.json` in
the chosen output directory.

## Input Format Expectations

The parser is tuned for discharge summaries with headers similar to:

- `Chief Complaint:`
- `History of Present Illness:`
- `Past Medical History:`
- `Pertinent Results:`
- `ADMISSION LABS:`
- `DISCHARGE LABS:`
- `IMAGING:` or `IMAGING STUDIES:`
- `Brief Hospital Course:`
- `Medications on Admission:`
- `Discharge Medications:`
- `Discharge Diagnosis:`
- `Discharge Instructions:`
- `Followup Instructions:`

It also handles some common variants such as underlined all-caps headers,
`LABS ON DISHCHARGE:`, and de-identified `[REDACTED] on Admission:`.

## Quality Notes

The included verification shows the code works well on the bundled sample
format and should work on similar discharge-summary `.txt` files. For a new
hospital format, first run a small batch and inspect:

- `pipeline_report_*.json` for low pass-rate files.
- `rejected_*.jsonl` for validation or judge failures.
- A few accepted rows in `training_data_*.jsonl` for section bleed or awkward abbreviation expansion.

The de-identification and judge are quality controls, not formal clinical or
HIPAA certification. Manually review outputs before using them in a production
training set.

The `.gitignore` excludes root-level `.txt` files and generated outputs by
default to avoid accidentally publishing local clinical notes or derived
datasets.

## Troubleshooting

If the judge is unavailable, the pipeline logs a warning and falls back to
rule-based validation. Use `--no-judge` for fast, portable runs.

If similar files produce poor output, update:

- `parser.py` for new section header variants.
- `abbreviations.py` for remaining abbreviations.
- `validator.py` for project-specific rejection rules.
