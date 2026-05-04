# Causal Analysis Agent

## Purpose
Estimate HR and robustness

## Inputs
- Gold
- Model 1 (`MODEL_SELECTION.md`)
- Model 2 (`MODEL_SELECTION.md`)
- Model 3 (`MODEL_SELECTION.md`)
- Synthetic
- Reasoning datasets (optional `X̃`)

## Study Setup
- Treatment: early vs delayed antibiotics
- Outcome: 28-day mortality
- Time zero: suspicion_time

## Outputs
- HR
- CI
- Survival curves
- Bias + variance summaries

## Rules
- identical pipeline
- no tuning per dataset
- compare `HR(X*)`, `HR(X̂)`, and when available `HR(X̃)`
