# Note Extraction Agent

## Purpose
Extract structured variables (X̂)

## Models
- Model A
- Model B
- Qwen

## Schema
- suspected_sepsis
- suspicion_time
- infection_source
- evidence_span
- confidence

## Rules
- identical prompt
- JSON only
- no fabrication
- evidence must match input text
- same notes and time window for every model

## Outputs
- extracted_model_A.json
- extracted_model_B.json
- extracted_qwen.json
