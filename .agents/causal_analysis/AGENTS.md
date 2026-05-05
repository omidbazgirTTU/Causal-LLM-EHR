# Causal Analysis Agent (Pre-Specified Analysis Contract)

## Purpose
Estimate treatment effects (hazard ratio) and quantify robustness to measurement error.

---

## Inputs

- dataset_gold (X*)
- dataset_model_A (X̂_A)
- dataset_model_B (X̂_B)
- dataset_model_Qwen (X̂_Qwen)
- dataset_synthetic (X̃)
- dataset_reasoning (X̃_m, optional)

All datasets must have identical schema except for measurement variable.

---

## Study Setup

- Treatment: early vs delayed antibiotics
- Outcome: 28-day mortality
- Time zero: suspicion_time (dataset-specific)

---

## 🔒 Analysis Contract (MUST NOT CHANGE)

### Covariate Policy

Include ONLY variables measured strictly before time zero (t0).

#### Include:
- demographics:
  - age
  - sex
- admission context:
  - admission type
  - ICU unit
  - calendar time
- comorbidity:
  - Charlson or equivalent
- prior utilization:
  - prior admissions (if available)
- baseline severity:
  - vitals (HR, BP, temp)
  - labs (WBC, lactate, creatinine)
  - organ support (ventilation, vasopressors BEFORE t0)
- note-derived baseline proxies (if available and pre-t0)

#### Exclude:
- any post-treatment variables
- any outcome-related variables
- any mediator variables
- anything measured after antibiotics start

---

### Propensity Model

Primary model:
- Logistic regression with L2 regularization

Specification:
- same covariates across ALL datasets
- no dataset-specific tuning
- continuous variables:
  - optionally include splines (pre-specified)
- interactions:
  - only pre-specified (do NOT add dynamically)

---

### Weighting Strategy

Primary:
- Stabilized Inverse Probability of Treatment Weighting (IPTW)

Formula:
- standard stabilized weights using marginal treatment probability

---

### Weight Truncation

Primary rule:
- truncate weights at 1st and 99th percentile

Required outputs:
- % of weights truncated
- weight distribution summary

---

### Positivity Check

Before model fitting:

- plot / compute propensity score overlap
- compute:
  - min/max PS
  - overlap region

If severe lack of overlap:
- DO NOT change primary analysis
- flag in report
- run sensitivity (see below)

---

### Outcome Model

- Cox proportional hazards model
- same specification across all datasets
- no dataset-specific adjustments

---

## Outputs (PER DATASET)

- hazard ratio (HR)
- confidence interval (CI)
- survival curve
- number of patients
- number of events

---

## Robustness Metrics

Compute:

### Bias
Bias_m = HR(X̂_m) − HR(X*)

### Stability
Stability = max(HR_m) − min(HR_m)

### Variance
Variance across models:
- Var(HR_A, HR_B, HR_Qwen)

---

## Sensitivity Analyses

### 1. Weight Sensitivity
- alternative truncation:
  - 5th / 95th percentile
- compare HR changes

---

### 2. Overlap Weighting (secondary)
- run overlap weighting as sensitivity
- DO NOT replace primary IPTW result

---

### 3. Synthetic Perturbation
- run model on X̃ datasets
- compute HR vs perturbation magnitude

---

### 4. Disagreement Subset
- restrict to patients where models disagree
- compute HR_disagreement
- compare with full cohort

---

## Reporting Requirements

Each run must output:

- dataset name
- N patients
- N events
- HR + CI
- weight summary
- % truncated
- convergence status

---

## Guardrails

- DO NOT modify datasets
- DO NOT drop patients silently
- DO NOT impute missing covariates unless pre-specified
- DO NOT tune model per dataset
- DO NOT change covariate set after seeing results

---

## Failure Conditions (STOP)

- missing covariates
- model non-convergence
- extreme weights (e.g., > threshold)
- dataset mismatch across runs

---

## Goal

Quantify how different measurement strategies (X*, X̂, X̃)
affect causal effect estimates under a fixed, pre-specified analysis pipeline.