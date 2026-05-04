# Causal Analysis Agent

## Purpose
Estimate HR and robustness

## Inputs
- Gold
- Model A
- Model B
- Qwen
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
