"""OpenAI-compatible OCI clients plus request signing helpers for local use."""

import base64
import hashlib
from email.utils import formatdate
from pathlib import Path
from typing import Generator, Mapping

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes
from openai import (
    DEFAULT_MAX_RETRIES,
    NOT_GIVEN,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
    DefaultHttpxClient,
    NotGiven,
    OpenAI,
    Timeout,
)


class OciOpenAI(OpenAI):
    """Sync OpenAI client configured to talk to OCI Generative AI inference."""

    def __init__(
        self,
        *,
        profile: str,
        region: str,
        compartment_id: str,
        stage: str = "ppe",
        service_endpoint: str | None = None,
        timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize a sync OCI-backed OpenAI-compatible client."""
        if service_endpoint is None:
            service_endpoint = resolve_default_service_endpoint(region, stage)
        super().__init__(
            api_key="<NOTUSED>",
            base_url=f"{service_endpoint}/20231130/actions/v1",
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=DefaultHttpxClient(
                auth=OCISessionAuth(profile),
                headers={"CompartmentId": compartment_id},
            ),
        )


class AsyncOciOpenAI(AsyncOpenAI):
    """Async OpenAI client configured to talk to OCI Generative AI inference."""

    def __init__(
        self,
        *,
        profile: str,
        region: str,
        compartment_id: str,
        stage: str = "ppe",
        service_endpoint: str | None = None,
        timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize an async OCI-backed OpenAI-compatible client."""
        if service_endpoint is None:
            service_endpoint = resolve_default_service_endpoint(region, stage)
        super().__init__(
            api_key="<NOTUSED>",
            base_url=f"{service_endpoint}/20231130/actions/v1",
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=DefaultAsyncHttpxClient(
                auth=OCISessionAuth(profile),
                headers={"CompartmentId": compartment_id},
            ),
        )


class OCISessionAuth(httpx.Auth):
    """Sign OCI inference requests with the local security-token session files."""

    def __init__(self, profile: str) -> None:
        """Load the token and private key for a named local OCI session profile."""
        self.profile = profile
        self.token_path = Path.home() / ".oci" / "sessions" / profile / "token"
        self.key_path = Path.home() / ".oci" / "sessions" / profile / "oci_api_key.pem"
        self.token = self._load_token()
        self.private_key = self._load_private_key()

    def _load_token(self) -> str:
        """Read the short-lived OCI security token for the active session."""
        with open(self.token_path, "r", encoding="utf-8") as file:
            return file.read().strip()

    def _load_private_key(self) -> PrivateKeyTypes:
        """Read the private key paired with the local OCI security token."""
        with open(self.key_path, "rb") as file:
            return serialization.load_pem_private_key(file.read(), password=None)

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        """Attach OCI Signature Version 1 headers to each outgoing request."""
        method = request.method.lower()
        path = request.url.raw_path.decode("utf-8")
        host = request.url.host or ""

        headers = {"host": host}
        headers_to_sign = ["(request-target)", "host"]
        auth_data = f"(request-target): {method} {path}\nhost: {host}"

        if request.content:
            # OCI requires body metadata headers to be included in both the
            # request and the signed payload when a JSON body is present.
            content_length = str(len(request.content))
            content_type = "application/json"
            content_sha256 = base64.b64encode(hashlib.sha256(request.content).digest()).decode("utf-8")

            headers["content-length"] = content_length
            headers["content-type"] = content_type
            headers["x-content-sha256"] = content_sha256
            headers_to_sign.extend(["content-length", "content-type", "x-content-sha256"])
            auth_data += f"\ncontent-length: {content_length}"
            auth_data += f"\ncontent-type: {content_type}"
            auth_data += f"\nx-content-sha256: {content_sha256}"

        x_date = formatdate(timeval=None, localtime=False, usegmt=True)
        headers["x-date"] = x_date
        headers_to_sign.append("x-date")
        auth_data += f"\nx-date: {x_date}"

        signature = self.private_key.sign(auth_data.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())
        b64_signature = base64.b64encode(signature).decode("utf-8")
        authorization = (
            'Signature version="1",algorithm="rsa-sha256",'
            f'headers="{" ".join(headers_to_sign)}",'
            f'keyId="ST${self.token}",signature="{b64_signature}"'
        )

        for key, value in headers.items():
            request.headers[key] = value
        request.headers["Authorization"] = authorization
        yield request


def resolve_service_endpoint(region: str, stage: str) -> str:
    """Map a deployment stage name to its OCI inference base URL."""
    if stage == "prod":
        return f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    if stage == "dev":
        return f"https://dev.inference.generativeai.{region}.oci.oraclecloud.com"
    if stage == "ppe":
        return f"https://ppe.inference.generativeai.{region}.oci.oraclecloud.com"
    raise ValueError(f"Invalid stage: {stage}")


def resolve_default_service_endpoint(region: str, stage: str = "ppe") -> str:
    """Resolve the default OCI inference endpoint for a region and stage."""
    return resolve_service_endpoint(region, stage)
