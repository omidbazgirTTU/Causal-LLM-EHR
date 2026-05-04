# Global Agent Rules

1. Build cohort.
2. Build gold dataset.
3. Run the selected three-model panel from `MODEL_SELECTION.md` on the same
   notes.
4. Evaluate extraction against gold labels.
5. Run perturbations and robustness checks.
6. Build analysis datasets.
7. Run causal models.
8. Compare hazard ratios and robustness across datasets.
9. Generate a report after each phase.

## Do Not

- run on full data
- skip filtering
- skip the gold dataset
- include post-treatment notes
- change prompts across models
- tune extractors after seeing causal results

## Always

- stop on any validation failure
- use identical prompt structure across models
- use the same notes and time window for all models
- use the same downstream causal pipeline across Gold, all three selected model
  datasets, Synthetic, and reasoning datasets
- compare `HR(X*)`, `HR(X̂)`, and when available `HR(X̃)`
