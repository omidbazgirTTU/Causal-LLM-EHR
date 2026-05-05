# Progress Logging Skill

This file defines the default logging rule for research progress, reproducible
runs, and publication support artifacts.

## Goal

- preserve a compact audit trail for each substantive phase run
- make it easy to reconstruct methods, outputs, and validation state later
- support paper writing without relying on memory or terminal scrollback

## Default Rule

- after each substantive run, write a machine-readable progress log even if the
  run fails
- pair the machine-readable log with the human-readable markdown report for the
  same phase

## Default Outputs

- markdown summary: `reports/<phase>/<timestamp>_<phase>_report.md`
- machine-readable run log: `artifacts/logs/<phase>/<timestamp>_<phase>_log.json`
- when figures are produced, include the associated figure manifest paths in the
  run log

## Required Log Fields

- `timestamp_utc`
- `phase`
- `status`
- `purpose`
- `entrypoint`
- `command_args`
- `conda_env` or runtime environment
- `git_commit`
- `git_dirty`
- `input_artifacts`
- `output_artifacts`
- `row_counts` or cohort counts when applicable
- `key_metrics`
- `validation_checks`
- `warnings`
- `failure_reason` when status is not `ok`

## Study-Specific Fields

- selected model panel and concrete model ids when model runs are involved
- prompt or schema version when extraction outputs are involved
- cohort version or manifest path when downstream datasets are involved
- causal contract version, truncation rule, and overlap threshold for causal
  runs
- figure ids or manifest paths when publication figures are generated

## Privacy And Safety

- never log raw note text, patient identifiers, secrets, OCI credentials, or
  `.env` contents
- log aggregate counts and artifact paths, not protected content

## Maintenance Rules

- append new timestamped logs instead of overwriting prior runs
- log failed runs explicitly; failed work is part of the audit trail
- keep field names stable so later scripts can summarize logs across phases

## Do Not

- do not leave a successful phase with only terminal output and no saved log
- do not fabricate metrics or backfill unknown values
- do not mix multiple unrelated runs into one log file
