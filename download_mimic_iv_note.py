#!/usr/bin/env python3
"""Repo-root entrypoint for PhysioNet MIMIC-IV-Note downloads."""

from causal_llm.notes.physionet_note_download import main


if __name__ == "__main__":
    raise SystemExit(main())
