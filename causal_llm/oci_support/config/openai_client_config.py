"""Load repo-local OCI client settings for local scripts and smoke tests."""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_config() -> dict[str, str | None]:
    """Load OCI client configuration from the repo-local `.env` file."""
    load_dotenv(Path.cwd() / ".env")

    return {
        "profile": os.getenv("PROFILE"),
        "region": os.getenv("REGION"),
        "compartment_id": os.getenv("COMPARTMENT_ID"),
        "oci_model_name": os.getenv("OCI_MODEL_NAME", "openai.gpt-5.4"),
        "oci_stage": os.getenv("OCI_STAGE", "ppe"),
    }
