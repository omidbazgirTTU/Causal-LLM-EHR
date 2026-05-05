# Rule-Based Annotator Agent

## Purpose
Create a deterministic reference measurement (X_rule) for the clinical variable of interest.

## Important Definition
X_rule is a reproducible baseline measurement, not clinical ground truth.

## Primary Use Case
Estimate suspicion_time and related baseline variables using transparent rules on structured data.

## Inputs
- ICU cohort
- admissions
- icustays
- prescriptions
- microbiology events
- diagnoses
- labs
- note metadata (optional, only if explicitly permitted by the rule)

## Output Schema
{
  "patient_id": "...",
  "hadm_id": "...",
  "icustay_id": "...",
  "suspicion_time_rule": "...",
  "infection_source_rule": "...",
  "rule_confidence": "high | medium | low",
  "rule_trigger": ["..."],
  "evidence_source": ["table_name:field_name", "..."]
}

## Rules
- Use only explicit, deterministic rules
- Never infer missing values
- Never use future outcomes
- Never use LLM outputs
- Never fabricate timestamps
- If no rule fires, return null values and log why

## Rule v2: Clinically Informed Baseline Proxy

### Goal
Estimate suspicion_time_rule as the earliest observable sign that clinicians were acting or documenting as if infection/sepsis was being considered, not the exact moment of internal suspicion.

### Priority order
Use the earliest qualifying timestamp from the following signals:

#### Tier 1: Direct note evidence
- First physician or nursing note containing explicit suspicion language such as:
  - "sepsis"
  - "suspected infection"
  - "possible pneumonia"
  - "rule out infection"
  - "empiric antibiotics"
  - "concern for source"
- Only count if the phrase appears in a note timestamped within the early window around the ICU stay.

#### Tier 2: Early clinical action
- First antibiotic administration
- First blood culture order
- First infection-directed imaging or order if it is clearly part of an infection workup

#### Tier 3: Supporting physiologic context
- These support confidence, but do not define suspicion time on their own:
  - lactate measurement
  - fever / hypotension / tachycardia
  - abnormal WBC

### Exclude as primary anchors
- Do not use diagnosis codes alone as suspicion_time_rule.
- Diagnosis codes may support context but are often delayed documentation artifacts.

### Proposed Definition
suspicion_time_rule = earliest timestamp among qualifying Tier 1 note evidence,
or, if no Tier 1 evidence exists, earliest Tier 2 clinical action,
with Tier 3 signals used only to support confidence.

## Confidence Levels
- high: Tier 1 note evidence and Tier 2 action agree within a short window
- medium: only Tier 1 or only Tier 2 exists
- low: only weak supporting signals exist, or the rule is triggered by a broad proxy

## Recommended Rule Behavior
- If multiple Tier 1 signals exist, use the earliest qualifying note timestamp.
- If no Tier 1 signal exists, use the earliest Tier 2 action timestamp.
- If only Tier 3 signals exist, return null and low confidence unless an explicit rule is provided to combine them with another tier.

## Recommended Rules for Sepsis Baseline
- suspicion_time_rule = earliest of:
  - first explicit note evidence of suspected infection/sepsis
  - else first antibiotic administration time
  - else first positive culture order time
  - else null
- infection_source_rule = mapped from explicit note text, culture context, or infection-related diagnosis only when unambiguous
- rule_confidence = high only when multiple signals agree

## Outputs
- rule_labels.json
- rule_labels.csv
- rule_execution_log.json

## Validation
- all timestamps must come from source tables
- required fields must exist
- output must be schema-valid
- rule_trigger must state which tier fired

## Forbidden
- no probabilistic inference
- no reasoning about what "probably happened"
- no natural-language summary in the data file

## Goal
Provide a deterministic baseline for comparing LLM extraction and reasoning outputs.
