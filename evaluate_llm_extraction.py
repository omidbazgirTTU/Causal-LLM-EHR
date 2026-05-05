#!/usr/bin/env python3
"""Repo-root entrypoint for extraction-vs-gold evaluation."""

from causal_llm.evaluation.extraction_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
