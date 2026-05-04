#!/usr/bin/env python3
"""Repo-root entrypoint for grouped gold-label task preparation."""

from causal_llm.annotation.gold_task_prep import main


if __name__ == "__main__":
    raise SystemExit(main())
