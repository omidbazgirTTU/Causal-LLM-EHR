# Causal-LLM-EHR

Research workspace for studying how LLM-induced measurement error affects downstream causal estimates in EHR-based analyses.

## Current Scope

- cohort and study design documentation
- Phase 1 ICU cohort builder backed by DuckDB
- OCI-backed LLM smoke-test scripts
- minimal local OCI client helpers for future development

## Why DuckDB

DuckDB is the default engine for the Phase 1 cohort build because the local
MIMIC-IV snapshot is already large enough to justify an analytical SQL engine:
the current `hosp` and `icu` exports are about 5.9 GB and 4.1 GB locally.
DuckDB can query the compressed CSV files directly, express the cohort logic
cleanly in SQL, and materialize reproducible local `.duckdb` and Parquet
artifacts without requiring a separate database service.

The longer rationale is documented in `.agents/IMPLEMENTATION_DECISIONS.md`.

## Repository Safety

- raw EHR data is intentionally excluded from git
- local `.env` files and credential material are ignored
- generated caches and local analysis artifacts are ignored

## Local Setup

1. Create a local `.env` from `.env.example`.
2. Activate `cllm-env`.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run `python llm-script.py`.
5. Run `python oci_model_panel_smoke_test.py`.
6. Run `python build_initial_cohort.py`.
7. Run `python audit_note_sources.py`.
8. Run `python prepare_gold_annotations.py --notes-parquet <linked_notes.parquet>` once note data is available.

## Key Files

- `llm-script.py`
- `build_initial_cohort.py`
- `audit_note_sources.py`
- `prepare_gold_annotations.py`
- `oci_model_panel_smoke_test.py`
- `requirements.txt`
- `.agents/`
- `.agents/MODEL_SELECTION.md`
- `EVALUATION_CHECKLIST.md`
