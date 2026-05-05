# Agents Overview

`.agents/` is the source of truth for the study design, execution rules, and
phase-specific instructions in this repository.

## Goal

Quantify how LLM-induced measurement error affects causal inference in
healthcare data.

## Core Idea

- `X*` = gold / true variable
- `X_rule` = deterministic rule-based baseline measurement
- `X̂_m` = extracted variable from model `m`
- `X̃_m` = reasoning-based variable from model `m`
- `Bias(HR_m) = HR(X̂_m) - HR(X*)`

LLMs are treated as noisy measurement functions.

## Model Policy

See `MODEL_SELECTION.md`.

- final paper runs use the matched-scale paper panel
- OCI smoke tests and API debugging use the development panel only

## Document Map

- `AGENTS.md`: global execution rules and hard constraints
- `ANNOTATION_WORKFLOW.md`: reviewer assignment, validation, and adjudication workflow
- `CODE_COMMENTING.md`: default docstring and commenting rules for touched code
- `GIT_WORKFLOW.md`: default commit/push policy after completed tasks
- `IMPLEMENTATION_DECISIONS.md`: rationale for major implementation choices
- `MODEL_SELECTION.md`: paper panel and development panel definitions
- `PUBLICATION_FIGURES.md`: figure-quality rules for paper-ready plots and figure bundles
- `PROGRESS_LOGGING.md`: machine-readable run-log policy for progress and publication support
- `SEMANTIC_LLM_POLICY.md`: semantic-first modeling rule for implementation
- `ANNOTATION_WORKFLOW.md`: includes the note-download and note-domain gating rules
- `routing.md`: workflow order and validation gates
- `PROBLEM.md`: formal causal question
- `COHORT_AND_SCHEMA.md`: cohort definition and observation window
- `NOTE_EXTRACTION_PLAN.md`: extraction schema and model constraints
- `CAUSAL_PLAN.md`: datasets, estimands, and robustness outputs
- `TASKS.md`: phase breakdown
- `note_extraction/AGENTS.md`: extraction schema, output names, and prompt rules
- `evaluation/AGENTS.md`: extraction-vs-gold metrics and baseline comparison rules
- `rule_based_annotator/AGENTS.md`: deterministic baseline measurement rules
- `*/AGENTS.md`: phase-specific agent rules

## Agent Roles

- `note_extraction` -> measurement
- `reasoning` -> controlled interpretation
- `evaluation` -> error quantification
- `causal_analysis` -> robustness and HR estimation
- `reporting` -> audit, progress logs, and publication figures
