# Model Selection

This file is the source of truth for model IDs used in this repository.

## Paper Panel

Do not start final paper runs until all three models below are provisioned and
reachable from OCI.

- `paper_model_1`: `meta.llama-3.3-70b-instruct`
- `paper_model_2`: `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF`
- `paper_model_3`: `Qwen/Qwen2.5-72B-Instruct`

## Why This Panel

- all three are instruction-tuned general LLMs in the same ~70B-class range
- the design keeps model scale tightly matched, which is the cleanest
  reviewer-facing defense for causal robustness comparisons
- the same notes, prompts, schema, and downstream causal pipeline must be used
  for all three models

## OCI Constraint

- `paper_model_1` is OCI-managed and callable directly in this compartment
- `paper_model_2` and `paper_model_3` require OCI imported-model and endpoint
  provisioning before final experiments can begin
- until those imported models are provisioned, final paper experiments are
  blocked

## Development Smoke-Test Panel

Use this panel only for OCI authentication checks, API plumbing, and pipeline
debugging.

- `dev_model_1`: `openai.gpt-oss-20b`
- `dev_model_2`: `meta.llama-4-scout-17b-16e-instruct`
- `dev_model_3`: `meta.llama-4-maverick-17b-128e-instruct-fp8`

## Run Policy

- every extraction and evaluation run must record the exact model IDs used
- do not mix dev-panel results into paper tables
- if the paper panel changes, update this file and the methods section before
  running experiments
