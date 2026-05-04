# Causal-LLM-EHR

Research workspace for studying how LLM-induced measurement error affects downstream causal estimates in EHR-based analyses.

## Current Scope

- cohort and study design documentation
- OCI-backed LLM smoke-test script
- minimal local OCI client helpers for future development

## Repository Safety

- raw EHR data is intentionally excluded from git
- local `.env` files and credential material are ignored
- generated caches and local analysis artifacts are ignored

## Local Setup

1. Create a local `.env` from `.env.example`.
2. Activate `cllm-env`.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run `python llm-script.py`.

## Key Files

- `llm-script.py`
- `requirements.txt`
- `.agents/`
- `EVALUATION_CHECKLIST.md`
