# Note Extraction Agent

## Purpose
Extract structured variables (X̂)

## Models

Use the selected panel defined in `../MODEL_SELECTION.md`.

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
- extracted_model_1.json
- extracted_model_2.json
- extracted_model_3.json
