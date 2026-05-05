# Causal Plan

## Datasets

- Gold
- Rule-based baseline (`X_rule`)
- Model 1 (`MODEL_SELECTION.md`)
- Model 2 (`MODEL_SELECTION.md`)
- Model 3 (`MODEL_SELECTION.md`)
- Synthetic
- reasoning datasets (`X̃`) when available

## Study Setup

- treatment: early vs delayed antibiotics
- outcome: 28-day mortality
- time zero: `suspicion_time`

## Outputs

- `HR(X*)`
- `HR(X_rule)`
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
