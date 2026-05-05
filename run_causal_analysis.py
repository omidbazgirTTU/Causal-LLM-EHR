#!/usr/bin/env python3
"""Repo-root entrypoint for the pre-specified causal analysis pipeline."""

from causal_llm.causal_analysis.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
