# Annotation Workflow

This file defines the Phase 2 gold-label workflow for the repository.

## Goal

- create a reviewer-auditable gold dataset before any three-model extraction run
- keep annotation and adjudication reproducible enough for a paper-grade methods section

## Command Order

1. Run `python download_mimic_iv_note.py`
2. Run `python audit_note_sources.py`
3. If the note-domain manifest approves a gold-label source, run
   `python prepare_gold_annotations.py --notes-parquet <linked_notes.parquet>`
4. Run `python assign_gold_annotations.py --reviewers reviewer_1 reviewer_2 reviewer_3`
5. Have reviewers fill their reviewer-specific CSV files from
   `derived_data/annotations/assignments/`
6. Run `python finalize_gold_annotations.py --annotation-csv <reviewer_1.csv> --annotation-csv <reviewer_2.csv> ...`
7. If `needs_adjudication.csv` is non-empty, have an adjudicator complete it and rerun with
   `--adjudication-csv <needs_adjudication_completed.csv>`

## Note-Domain Gate

- do not silently substitute radiology or discharge notes for the intended
  physician / nursing note design
- if `derived_data/notes/note_domain_manifest.json` marks a domain as not
  approved for gold labels, stop and revise the design explicitly before using
  it
- `--allow-nonprimary-notes` is reserved for an intentional, documented design
  revision

## Reviewer Rules

- annotate from the provided note bundle only
- do not use post-treatment information outside the task packet
- `suspected_sepsis` must be yes or no
- if `suspected_sepsis=yes`, provide:
  - `suspicion_time`
  - `infection_source`
  - `evidence_span`
- if `suspected_sepsis=no`, leave `suspicion_time` and `infection_source` blank
- `confidence` must stay in `[0, 1]`

## Agreement Rules

- core agreement is based on:
  - `suspected_sepsis`
  - `suspicion_time`
  - `infection_source`
- if core labels disagree, the task must be adjudicated
- if only `evidence_span` differs while the core label agrees, the task can still finalize
  and the final record will preserve the distinct evidence spans

## Outputs

- reviewer packets: `derived_data/annotations/assignments/*`
- final gold labels: `derived_data/annotations/finalized/gold_labels.csv`
- model-ready gold tasks:
  `derived_data/annotations/finalized/gold_label_tasks.jsonl`
- adjudication template:
  `derived_data/annotations/finalized/needs_adjudication.csv`
