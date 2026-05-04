from openai import AsyncOpenAI

from causal_llm.oci_support.config.openai_client_config import load_config
from causal_llm.oci_support.utils.oci_openai import AsyncOciOpenAI, OciOpenAI


def _load_required_oci_settings() -> tuple[str, str, str, str]:
    config = load_config()
    profile = config.get("profile")
    region = config.get("region")
    compartment_id = config.get("compartment_id")
    stage = config.get("oci_stage") or "ppe"

    if not profile:
        raise ValueError("PROFILE is not set in the environment or .env file.")
    if not region:
        raise ValueError("REGION is not set in the environment or .env file.")
    if not compartment_id:
        raise ValueError("COMPARTMENT_ID is not set in the environment or .env file.")

    return profile, region, compartment_id, stage


def get_oci_openai_client() -> OciOpenAI:
    """Build a sync OpenAI-compatible client backed by OCI GenAI."""
    profile, region, compartment_id, stage = _load_required_oci_settings()
    return OciOpenAI(
        profile=profile,
        region=region,
        compartment_id=compartment_id,
        stage=stage,
    )


def get_oci_async_openai_client(local: bool = False) -> AsyncOciOpenAI | AsyncOpenAI:
    """Build an async OpenAI-compatible client backed by OCI GenAI."""
    if local:
        return AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="<NOTUSED>")

    profile, region, compartment_id, stage = _load_required_oci_settings()
    return AsyncOciOpenAI(
        profile=profile,
        region=region,
        compartment_id=compartment_id,
        stage=stage,
    )
