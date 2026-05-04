"""Simple OCI smoke test and agent wiring example for local development."""

from pprint import pprint

from agents import Agent, set_default_openai_api, set_default_openai_client, set_tracing_disabled

from causal_llm.oci_support.config.openai_client_config import load_config
from causal_llm.oci_support.runtime.api_client import get_oci_async_openai_client, get_oci_openai_client


def smoke_test_chat() -> None:
    """Send a basic chat request through the OCI-backed OpenAI-compatible client."""
    config = load_config()
    model_name = config.get("oci_model_name") or "openai.gpt-5.4"
    client = get_oci_openai_client()
    messages = [
        {
            "role": "system",
            "content": "You are a research agent. Answer the questions asked by the user in a concise manner.",
        },
        {"role": "user", "content": "What is quantum entanglement?"},
    ]
    response = client.chat.completions.create(model=model_name, messages=messages)
    pprint(response.choices[0].message.content)


def build_guardrail_agent() -> Agent:
    """Construct a minimal local `agents` SDK object backed by OCI chat completions."""
    set_default_openai_client(get_oci_async_openai_client())
    set_tracing_disabled(True)  # Avoid the Tracing client error 401 in local runs.
    set_default_openai_api("chat_completions")

    input_guardrail_prompt = (
        "You are a topic classifier for a healthcare application. "
        "Your task is to decide whether a user's message is requesting patient health data."
    )
    return Agent(name="prompt_creator", instructions=input_guardrail_prompt, model="openai.gpt-5.4")


def main() -> None:
    """Run the local smoke test and verify agent construction does not fail."""
    smoke_test_chat()
    _ = build_guardrail_agent()


if __name__ == "__main__":
    main()
