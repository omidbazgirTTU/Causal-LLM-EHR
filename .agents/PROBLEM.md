# Formal Problem

- `X*` = true / gold variable
- `X̂` = LLM-extracted variable
- `X̃` = reasoning-based variable

## Study Question

How much do downstream causal conclusions change when `X*` is replaced by noisy
LLM-derived measurements?

## Primary Quantity

`Bias(HR_m) = HR(X̂_m) - HR(X*)`

## Success Criterion

Answer whether causal conclusions remain stable across the selected three-model
panel in `MODEL_SELECTION.md`.
