# Implementation Decisions

This file records implementation choices that are important enough to justify
once and reuse later.

## Why DuckDB For Phase 1 Cohort Building

DuckDB was chosen for the initial cohort builder for pragmatic reasons tied to
the local dataset and the current workflow.

- the local MIMIC-IV snapshot already includes large `hosp` and `icu` exports
  of about 5.9 GB and 4.1 GB respectively, so Phase 1 needs an analytical
  engine rather than ad hoc row-by-row Python processing
- DuckDB can query compressed CSV inputs directly, which lets the cohort
  builder work from the local PhysioNet files without first loading everything
  into pandas or standing up a separate database server
- the cohort logic is easier to express and audit as SQL because the pipeline
  is mostly joins, time-window filters, and cohort-level aggregations
- DuckDB produces a single local database file plus Parquet outputs, which fits
  the repo's reproducibility goal and works well with git-ignored derived
  artifacts
- repeated local reruns are cheaper because the logic stays in one query engine
  instead of mixing shell scripts, pandas transforms, and temporary CSV files

## Why Not Pandas As The Primary Engine

- the Phase 1 inputs are large enough that eager in-memory dataframe workflows
  are a worse default for repeated local full-table scans
- SQL is a clearer representation for the ICU cohort definition, the
  pre/post-window logic, and the loose sepsis signal joins

## Why Not SQLite As The Primary Engine

- SQLite is a strong transactional store, but it is less convenient than
  DuckDB for analytical scans over compressed CSV files and columnar Parquet
  exports
- this phase is analytical cohort construction, not OLTP-style application
  storage

## Revisit Condition

Revisit this choice only if a later phase requires a distributed execution
engine or if the local workflow becomes bottlenecked in a way DuckDB cannot
handle cleanly.

## MIMIC-IV-Note Scope And Note-Source Audit

The open MIMIC-IV-Note module is not assumed to satisfy the intended
physician-plus-nursing note design by default.

- note-source availability must be audited before any extraction run
- if the available open note source does not match the intended note domain, do
  not silently substitute it into the main workflow
- discharge summaries can be linked for auditing, but they are not valid
  pre-decision inputs for the primary extraction pipeline
- any note-domain substitution requires an explicit design decision and should
  be documented before model runs begin
