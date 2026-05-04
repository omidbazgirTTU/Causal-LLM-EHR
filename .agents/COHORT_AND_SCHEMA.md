# Cohort And Schema

## Base Cohort

- ICU patients
- age >= 18
- first ICU stay per patient

## Time Zero And Window

- `t0 = ICU admission`
- retain data in `[t0 - 24h, t0 + 48h]`
- use only pre-decision data for extraction and reasoning

## Loose Candidate Sepsis Filter

Keep patients with any of:

- antibiotic exposure
- lactate measurement
- blood culture

## Notes

- notes must be within the study window
- physician and nursing notes are the intended note sources

## Initial Scale

- build an initial filtered dataset of roughly 500 to 1000 patients before any
  large LLM run
