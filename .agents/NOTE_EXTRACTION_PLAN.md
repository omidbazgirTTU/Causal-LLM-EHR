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

- Model A
- Model B
- Qwen

## Rules

- identical prompt across models
- JSON only
- no fabrication
- evidence must match input text
- run all models on the same notes

## Outputs

- `extracted_model_A.json`
- `extracted_model_B.json`
- `extracted_qwen.json`
