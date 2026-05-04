#!/usr/bin/env python3
"""Repo-root entrypoint for reviewer assignment generation."""

from causal_llm.annotation.reviewer_assignment import main


if __name__ == "__main__":
    raise SystemExit(main())
