#!/usr/bin/env python3
"""Repo-root entrypoint for deterministic baseline generation."""

from causal_llm.rule_based.baseline_annotator import main


if __name__ == "__main__":
    raise SystemExit(main())
