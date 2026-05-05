# Evaluation Agent

## Purpose
Compare X̂ and X* and quantify error

## Metrics
- precision / recall / F1
- mean / median time error
- % within 1 hour
- % within 3 hours
- infection_source agreement
- model disagreement tables

## Rules
- no modification of inputs
- fail on mismatch
- compare all models against the same gold set
- evaluate `X_rule` as a deterministic baseline against the same gold set
