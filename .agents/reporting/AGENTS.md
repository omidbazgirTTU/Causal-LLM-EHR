# Reporting Agent

## Purpose
Generate structured summary reports, machine-readable progress logs, and
publication-grade figure bundles after each phase when applicable

## Output Path
reports/<phase>/<timestamp>_<phase>_report.md

## Additional Output Paths
- artifacts/logs/<phase>/<timestamp>_<phase>_log.json
- artifacts/figures/<phase>/<figure_id>/

## Format

# <TITLE>
Date: <timestamp>

## Inputs
- dataset
- size

## Key Metrics
- phase-specific metrics

## Sanity Checks
- dataset size
- missing values

## Rules
- no fabrication
- no interpretation
- only computed values
- follow `../PROGRESS_LOGGING.md` for required run-log fields
- follow `../PUBLICATION_FIGURES.md` for figure styling and bundle contents
- when a phase yields comparison-ready results, save the figure manifest and
  source-data path alongside the report
