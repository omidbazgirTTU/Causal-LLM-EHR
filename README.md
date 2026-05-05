# Causal-LLM-EHR

Research workspace for studying how LLM-induced measurement error affects downstream causal estimates in EHR-based analyses.

## Current Scope

- cohort and study design documentation
- Phase 1 ICU cohort builder backed by DuckDB
- PhysioNet MIMIC-IV-Note download helper with restricted-access checks
- Phase 2 annotation task preparation, reviewer assignment, and gold-label finalization
- deterministic rule-based baseline (`X_rule`) from structured MIMIC-IV tables
- Phase 3 OCI-backed three-model extraction runner with fixed output artifacts
- Phase 4 extraction-vs-gold evaluation with `X_rule` baseline comparison
- Phase 4b synthetic time-shift perturbation generator for robustness checks
- Phase 5 matched analysis-dataset builder for gold, `X_rule`, and all models
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
7. Run `python build_rule_based_baseline.py`.
8. Run `python download_mimic_iv_note.py`.
9. Run `python audit_note_sources.py`.
10. If the note audit approves a note source for gold labels, run `python prepare_gold_annotations.py --notes-parquet <linked_notes.parquet>`.
11. Run `python assign_gold_annotations.py --reviewers reviewer_1 reviewer_2 reviewer_3`.
12. After reviewers finish, run `python finalize_gold_annotations.py --annotation-csv <reviewer_1.csv> --annotation-csv <reviewer_2.csv>`.
13. Run `python run_llm_extraction.py` after gold-label finalization.
14. Run `python evaluate_llm_extraction.py` to compare all three model outputs and `X_rule` against the same gold set.
15. Run `python build_analysis_datasets.py` to derive matched treatment/outcome datasets for Gold, `X_rule`, and all three model outputs.
16. Run `python generate_time_perturbations.py` to materialize synthetic shifted-time datasets for robustness analyses.

## Phase 3 And 4

- `python run_llm_extraction.py` reads finalized `gold_label_tasks.jsonl`, loads the
  selected model panel from `.agents/MODEL_SELECTION.md`, and writes
  `extracted_model_1.json`, `extracted_model_2.json`, `extracted_model_3.json`,
  per-model CSVs, and `extraction_run_manifest.json`.
- use `--model-panel dev` only for OCI plumbing checks and smoke tests
- keep `--model-panel paper` for the matched-scale paper panel after the imported
  OCI endpoints are provisioned
- `python evaluate_llm_extraction.py` compares every extraction output against
  finalized `gold_labels.csv` and evaluates `derived_data/rule_based/rule_labels.csv`
  as the deterministic `X_rule` baseline
- evaluation fails if the extraction outputs or `X_rule` rows do not line up with
  the gold-label task set

## Phase 5

- `python build_analysis_datasets.py` joins Gold, `X_rule`, and all extracted
  model outputs to the same sampled cohort and derives downstream treatment and
  outcome fields
- the builder emits one CSV per predictor plus a summary CSV, manifest JSON, and
  phase report under `derived_data/analysis` and `reports/analysis`
- treatment timing is derived relative to predictor-specific `suspicion_time`
  using a configurable early-treatment window
- the builder fails on alignment problems, invalid predictor rows, and
  `suspicion_time` values that fall outside the audited study window

## Phase 4b

- `python generate_time_perturbations.py` reads the analysis-dataset manifest and
  generates synthetic datasets with signed hour shifts applied to `suspicion_time`
- the default perturbation set is `-3`, `-2`, `-1`, `+1`, `+2`, and `+3` hours
  on the Gold dataset
- each shifted dataset recomputes treatment timing, 28-day mortality follow-up,
  and analysis eligibility without changing the underlying cohort rows

## Key Files

- `llm-script.py`
- `build_initial_cohort.py`
- `download_mimic_iv_note.py`
- `audit_note_sources.py`
- `build_rule_based_baseline.py`
- `prepare_gold_annotations.py`
- `assign_gold_annotations.py`
- `finalize_gold_annotations.py`
- `run_llm_extraction.py`
- `evaluate_llm_extraction.py`
- `build_analysis_datasets.py`
- `generate_time_perturbations.py`
- `oci_model_panel_smoke_test.py`
- `requirements.txt`
- `.agents/`
- `.agents/MODEL_SELECTION.md`
- `.agents/ANNOTATION_WORKFLOW.md`
- `.agents/rule_based_annotator/AGENTS.md`
- `EVALUATION_CHECKLIST.md`
