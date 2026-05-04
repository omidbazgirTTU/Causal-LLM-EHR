#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from openai import BadRequestError

from agent_apps.agents_gym.config.openai_client_config import load_config
from agent_apps.agents_gym.utils.oci_openai import OciOpenAI

DEFAULT_MODELS = [
    "openai.gpt-oss-20b",
    "meta.llama-4-scout-17b-16e-instruct",
    "meta.llama-4-maverick-17b-128e-instruct-fp8",
]
DEFAULT_PROMPT = "Reply with exactly OK."
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_COMPLETION_TOKENS = 128


@dataclass
class SmokeTestResult:
    model: str
    ok: bool
    latency_seconds: float
    preview: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test OCI-hosted LLM APIs with the current local OCI credentials."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Explicit model IDs to test. Defaults to OCI_SMOKE_MODELS or the repo development trio.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User prompt sent to each model.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help="Token budget used for the smoke test response.",
    )
    return parser.parse_args()


def resolve_models(cli_models: list[str] | None) -> list[str]:
    if cli_models:
        return cli_models

    env_value = os.getenv("OCI_SMOKE_MODELS", "")
    if env_value.strip():
        return [item.strip() for item in env_value.split(",") if item.strip()]

    return DEFAULT_MODELS.copy()


def build_client(timeout_seconds: float) -> tuple[OciOpenAI, dict[str, str | None]]:
    config = load_config()
    client = OciOpenAI(
        profile=config["profile"],
        region=config["region"],
        compartment_id=config["compartment_id"],
        stage=config.get("oci_stage") or "ppe",
        timeout=timeout_seconds,
        max_retries=0,
    )
    return client, config


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def build_request_variants(
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }

    max_completion_variant = dict(base)
    max_completion_variant["max_completion_tokens"] = max_completion_tokens
    if model == "openai.gpt-oss-20b":
        max_completion_variant["reasoning_effort"] = "low"

    max_tokens_variant = dict(base)
    max_tokens_variant["max_tokens"] = max_completion_tokens

    if model.startswith("openai."):
        return [max_completion_variant, max_tokens_variant]
    return [max_tokens_variant, max_completion_variant]


def extract_text(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", None)

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        if text_parts:
            return " ".join(text_parts)

    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return f"[reasoning_only] {reasoning_content.strip()}"

    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        return f"[refusal] {refusal.strip()}"

    return "[empty_response]"


def short_preview(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def smoke_test_model(
    client: OciOpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
) -> SmokeTestResult:
    last_error: Exception | None = None
    last_latency_seconds = 0.0

    for request_kwargs in build_request_variants(model, messages, max_completion_tokens):
        start = perf_counter()
        try:
            response = client.chat.completions.create(**request_kwargs)
            latency_seconds = perf_counter() - start
            preview = extract_text(response)
            usage = getattr(response, "usage", None)
            total_tokens = getattr(usage, "total_tokens", None)
            detail = f"tokens={total_tokens}" if total_tokens is not None else "tokens=unknown"
            return SmokeTestResult(
                model=model,
                ok=True,
                latency_seconds=latency_seconds,
                preview=short_preview(preview),
                detail=detail,
            )
        except BadRequestError as exc:
            last_error = exc
            last_latency_seconds = perf_counter() - start
            message = str(exc)
            if "Unsupported parameter" in message or "not supported with this model" in message:
                continue
            break
        except Exception as exc:  # pragma: no cover - integration path
            last_error = exc
            last_latency_seconds = perf_counter() - start
            break

    assert last_error is not None
    return SmokeTestResult(
        model=model,
        ok=False,
        latency_seconds=last_latency_seconds,
        preview="-",
        detail=f"{type(last_error).__name__}: {last_error}",
    )


def print_results(results: list[SmokeTestResult], config: dict[str, str | None]) -> None:
    region = config.get("region") or "<missing>"
    stage = config.get("oci_stage") or "ppe"
    print(f"OCI smoke test against region={region} stage={stage}")
    print()

    for result in results:
        status = "OK" if result.ok else "FAIL"
        latency = f"{result.latency_seconds:.2f}s" if result.ok else "-"
        print(f"[{status}] {result.model}")
        print(f"  latency: {latency}")
        print(f"  preview: {result.preview}")
        print(f"  detail: {result.detail}")
        print()


def main() -> int:
    args = parse_args()
    models = resolve_models(args.models)
    client, config = build_client(timeout_seconds=args.timeout)
    messages = build_messages(args.prompt)

    try:
        results = [
            smoke_test_model(
                client=client,
                model=model,
                messages=messages,
                max_completion_tokens=args.max_completion_tokens,
            )
            for model in models
        ]
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    print_results(results, config)

    failures = [result for result in results if not result.ok]
    if failures:
        print(f"{len(failures)} model call(s) failed.", file=sys.stderr)
        return 1

    print(f"All {len(results)} model call(s) succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
