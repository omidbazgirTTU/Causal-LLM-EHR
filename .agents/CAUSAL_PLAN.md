# Causal Plan

## Datasets

- Gold
- Model A
- Model B
- Qwen
- Synthetic
- reasoning datasets (`X̃`) when available

## Study Setup

- treatment: early vs delayed antibiotics
- outcome: 28-day mortality
- time zero: `suspicion_time`

## Outputs

- `HR(X*)`
- `HR(X̂)`
- `HR(X̃)`
- confidence intervals
- survival curves

## Measure

- bias
- variance
- stability of direction
- overlap of confidence intervals

## Rules

- identical pipeline across datasets
- no tuning per dataset
