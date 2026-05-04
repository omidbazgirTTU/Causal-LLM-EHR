# Agent Routing Guide (Research-Grade)

## Workflow Order

1. cohort filtering
2. gold dataset
3. note extraction
4. reasoning
5. evaluation
6. perturbation
7. analysis dataset construction
8. causal analysis
9. bootstrap + disagreement analysis
10. reporting after each phase and final comparison

## Models
- Model A
- Model B
- Qwen

## Rules
- NEVER skip filtering
- NEVER run on full dataset
- NEVER include post-treatment notes
- NEVER change prompts across models
- NEVER tune after causal results
- ALWAYS generate report after each phase
- STOP on any validation failure
- ALWAYS compare models on the same notes
- ALWAYS use the same downstream causal pipeline across datasets
