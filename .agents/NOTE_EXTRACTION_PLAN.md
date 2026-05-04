# Note Extraction Plan

LLM output is treated as noisy measurement.

## Output Schema

```json
{
  "suspected_sepsis": true,
  "suspicion_time": "YYYY-MM-DD HH:MM:SS",
  "infection_source": "string",
  "evidence_span": "string",
  "confidence": 0.0
}
```

## Models

Use the selected panel defined in `MODEL_SELECTION.md`.

## Rules

- identical prompt across models
- JSON only
- no fabrication
- evidence must match input text
- run all models on the same notes

## Outputs

- `extracted_model_1.json`
- `extracted_model_2.json`
- `extracted_model_3.json`
