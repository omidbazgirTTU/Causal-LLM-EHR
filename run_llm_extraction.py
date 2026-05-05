#!/usr/bin/env python3
"""Repo-root entrypoint for OCI-backed three-model extraction runs."""

from causal_llm.extraction.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
