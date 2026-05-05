"""Download restricted MIMIC-IV-Note files with a login-backed PhysioNet session."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_FILES_ROOT = Path("physionet.org/files")
DEFAULT_OUTPUT_DIR = Path("derived_data/notes")
DEFAULT_REPORT_DIR = Path("reports/notes")
DEFAULT_DATASET_SLUG = "mimic-iv-note"
DEFAULT_DATASET_VERSION = "2.2"
DEFAULT_NOTE_SUBDIRECTORY = "note"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_NOTE_FILES = (
    "discharge.csv.gz",
    "discharge_detail.csv.gz",
    "radiology.csv.gz",
    "radiology_detail.csv.gz",
)

LOGIN_URL = "https://physionet.org/login/"
CONTENT_URL_TEMPLATE = "https://physionet.org/content/{dataset_slug}/{dataset_version}/"
FILE_URL_TEMPLATE = (
    "https://physionet.org/files/{dataset_slug}/{dataset_version}/"
    "{note_subdirectory}/{filename}"
)
SIGN_DUA_URL_TEMPLATE = "https://physionet.org/sign-dua/{dataset_slug}/{dataset_version}/"
USER_AGENT = "Causal-LLM-EHR/1.0"

CSRF_REGEX = re.compile(r'name="csrfmiddlewaretoken" value="([^"]+)"')
TAG_REGEX = re.compile(r"<[^>]+>")
LIST_ITEM_REGEX = re.compile(r"<li>(.*?)</li>", re.DOTALL)
ACCESS_ALERT_REGEX = re.compile(
    r'<div class="alert alert-danger[^"]*"[^>]*>(.*?)</div>',
    re.DOTALL,
)
RESTRICTED_ACCESS_MARKER = "This is a restricted-access resource."
INVALID_LOGIN_MARKER = "please enter a correct username and password"


@dataclass(frozen=True)
class PhysioNetNoteDownloadConfig:
    """Configuration for the restricted MIMIC-IV-Note download workflow."""

    files_root: Path
    output_dir: Path
    report_dir: Path
    dataset_slug: str
    dataset_version: str
    timeout_seconds: float
    force: bool
    files: tuple[str, ...]


@dataclass(frozen=True)
class DownloadedNoteFile:
    """Result for one requested note-domain file."""

    filename: str
    destination_path: str
    status: str
    bytes_downloaded: int
    detail: str


@dataclass(frozen=True)
class PhysioNetNoteDownloadSummary:
    """Summary artifact for a PhysioNet note-download attempt."""

    generated_at_utc: str
    dataset_slug: str
    dataset_version: str
    files_root: str
    dataset_root: str
    output_dir: str
    report_path: str
    summary_json_path: str
    content_page_url: str
    sign_dua_url: str
    requested_files: list[str]
    login_ok: bool
    access_granted: bool
    all_requested_files_ready: bool
    access_gate_reason: str | None
    downloaded_file_count: int
    skipped_existing_file_count: int
    file_results: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the PhysioNet note downloader."""

    parser = argparse.ArgumentParser(
        description="Download restricted MIMIC-IV-Note CSV files using PhysioNet credentials from .env."
    )
    parser.add_argument(
        "--files-root",
        type=Path,
        default=DEFAULT_FILES_ROOT,
        help="Local PhysioNet files root. Downloads land under <files-root>/<dataset>/<version>/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for download-attempt summaries.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for markdown download reports.",
    )
    parser.add_argument(
        "--dataset-version",
        default=DEFAULT_DATASET_VERSION,
        help="PhysioNet MIMIC-IV-Note version to fetch.",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Specific file to download. Repeat to limit the download set.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when the local destination already exists.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PhysioNetNoteDownloadConfig:
    """Resolve CLI arguments into a normalized note-download config."""

    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")

    requested_files = tuple(args.files) if args.files else DEFAULT_NOTE_FILES
    if not requested_files:
        raise ValueError("At least one note file must be requested.")

    return PhysioNetNoteDownloadConfig(
        files_root=args.files_root.resolve(),
        output_dir=args.output_dir.resolve(),
        report_dir=args.report_dir.resolve(),
        dataset_slug=DEFAULT_DATASET_SLUG,
        dataset_version=args.dataset_version.strip(),
        timeout_seconds=args.timeout_seconds,
        force=args.force,
        files=requested_files,
    )


def load_physionet_credentials() -> tuple[str, str]:
    """Load PhysioNet credentials from the local `.env` file."""

    load_dotenv(Path.cwd() / ".env")
    username = os.getenv("PHYSIONET_USERNAME", "").strip()
    password = os.getenv("PHYSIONET_PASSWORD", "").strip()

    if not username or not password:
        raise ValueError(
            "Missing PHYSIONET_USERNAME or PHYSIONET_PASSWORD in .env. "
            "Add the credentials before attempting note downloads."
        )

    return username, password


def build_opener() -> tuple[urllib.request.OpenerDirector, CookieJar]:
    """Create a cookie-backed opener for the PhysioNet login session."""

    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    opener.addheaders = [("User-Agent", USER_AGENT)]
    return opener, cookie_jar


def fetch_text_response(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
) -> str:
    """Fetch one HTML/text response through the opener."""

    request = urllib.request.Request(
        url=url,
        data=data,
        headers=headers or {},
        method="POST" if data is not None else "GET",
    )
    with opener.open(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", errors="replace")


def strip_html(value: str) -> str:
    """Collapse HTML into readable plain text for logs and reports."""

    return " ".join(html.unescape(TAG_REGEX.sub(" ", value)).split())


def extract_csrf_token(login_page_html: str) -> str:
    """Extract the CSRF token from the PhysioNet login form."""

    match = CSRF_REGEX.search(login_page_html)
    if not match:
        raise RuntimeError("Could not find a CSRF token on the PhysioNet login page.")
    return match.group(1)


def has_cookie(cookie_jar: CookieJar, name: str) -> bool:
    """Return whether the cookie jar currently holds a named cookie."""

    return any(cookie.name == name for cookie in cookie_jar)


def login_to_physionet(
    opener: urllib.request.OpenerDirector,
    cookie_jar: CookieJar,
    *,
    username: str,
    password: str,
    timeout_seconds: float,
) -> None:
    """Establish an authenticated PhysioNet session using username/password."""

    login_page_html = fetch_text_response(
        opener,
        LOGIN_URL,
        timeout_seconds=timeout_seconds,
    )
    csrf_token = extract_csrf_token(login_page_html)
    payload = urllib.parse.urlencode(
        {
            "csrfmiddlewaretoken": csrf_token,
            "username": username,
            "password": password,
            "next": "",
        }
    ).encode("utf-8")
    post_login_html = fetch_text_response(
        opener,
        LOGIN_URL,
        timeout_seconds=timeout_seconds,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": LOGIN_URL,
        },
        data=payload,
    )

    lowered_html = post_login_html.lower()
    if INVALID_LOGIN_MARKER in lowered_html or "<title>login</title>" in lowered_html:
        raise PermissionError("PhysioNet login failed. Check the username/password pair in .env.")
    if not has_cookie(cookie_jar, "sessionid"):
        raise PermissionError("PhysioNet login did not create a session cookie.")


def extract_access_requirements(content_page_html: str) -> list[str]:
    """Extract the current restricted-access gating bullets from the dataset page."""

    for alert_fragment in ACCESS_ALERT_REGEX.findall(content_page_html):
        alert_text = strip_html(alert_fragment)
        if RESTRICTED_ACCESS_MARKER.lower() not in alert_text.lower():
            continue
        requirements = [
            strip_html(item)
            for item in LIST_ITEM_REGEX.findall(alert_fragment)
            if strip_html(item)
        ]
        return requirements
    return []


def download_one_file(
    opener: urllib.request.OpenerDirector,
    *,
    url: str,
    destination_path: Path,
    timeout_seconds: float,
    force: bool,
) -> DownloadedNoteFile:
    """Download one restricted file to a local temp path and atomically move it into place."""

    if destination_path.exists() and not force:
        return DownloadedNoteFile(
            filename=destination_path.name,
            destination_path=str(destination_path),
            status="skipped_existing",
            bytes_downloaded=destination_path.stat().st_size,
            detail="Existing file preserved.",
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_suffix(destination_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    try:
        request = urllib.request.Request(url=url, headers={"Accept": "*/*"})
        total_bytes = 0
        with opener.open(request, timeout=timeout_seconds) as response:
            with open(temp_path, "wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    file.write(chunk)

        temp_path.replace(destination_path)
        return DownloadedNoteFile(
            filename=destination_path.name,
            destination_path=str(destination_path),
            status="downloaded",
            bytes_downloaded=total_bytes,
            detail="Download completed.",
        )
    except urllib.error.HTTPError as exc:
        if temp_path.exists():
            temp_path.unlink()
        detail = f"HTTP {exc.code}"
        response_body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403:
            blocked_requirements = extract_access_requirements(response_body)
            if blocked_requirements:
                detail = "; ".join(blocked_requirements)
            else:
                detail = "PhysioNet denied access to the requested file."
            status = "access_denied"
        else:
            status = "http_error"
        return DownloadedNoteFile(
            filename=destination_path.name,
            destination_path=str(destination_path),
            status=status,
            bytes_downloaded=0,
            detail=detail,
        )
    except Exception as exc:  # pragma: no cover - integration path
        if temp_path.exists():
            temp_path.unlink()
        return DownloadedNoteFile(
            filename=destination_path.name,
            destination_path=str(destination_path),
            status="error",
            bytes_downloaded=0,
            detail=f"{type(exc).__name__}: {exc}",
        )


def write_outputs(
    summary: PhysioNetNoteDownloadSummary,
    report_path: Path,
    summary_json_path: Path,
) -> None:
    """Persist the JSON and markdown artifacts for one note-download attempt."""

    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)

    report_lines = [
        "# PhysioNet Note Download Report",
        f"Date: {summary.generated_at_utc}",
        "",
        "## Dataset",
        f"- dataset_slug: `{summary.dataset_slug}`",
        f"- dataset_version: `{summary.dataset_version}`",
        f"- dataset_root: `{summary.dataset_root}`",
        "",
        "## Session",
        f"- login_ok: {summary.login_ok}",
        f"- access_granted: {summary.access_granted}",
        f"- all_requested_files_ready: {summary.all_requested_files_ready}",
        f"- access_gate_reason: {summary.access_gate_reason or 'none'}",
        "",
        "## Requested Files",
        f"- requested_files: {', '.join(summary.requested_files)}",
        f"- downloaded_file_count: {summary.downloaded_file_count}",
        f"- skipped_existing_file_count: {summary.skipped_existing_file_count}",
        "",
        "## Links",
        f"- content_page_url: `{summary.content_page_url}`",
        f"- sign_dua_url: `{summary.sign_dua_url}`",
        "",
        "## Per-File Results",
    ]
    for file_result in summary.file_results:
        report_lines.append(
            "- "
            + ", ".join(
                [
                    f"filename={file_result['filename']}",
                    f"status={file_result['status']}",
                    f"bytes_downloaded={file_result['bytes_downloaded']}",
                    f"detail={file_result['detail']}",
                ]
            )
        )

    report_lines.extend(
        [
            "",
            "## Outputs",
            f"- `{summary.summary_json_path}`",
            f"- `{summary.report_path}`",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines) + "\n")


def run_note_download(config: PhysioNetNoteDownloadConfig) -> PhysioNetNoteDownloadSummary:
    """Execute the PhysioNet login, access check, and note-file download flow."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc)
    report_path = config.report_dir / (
        f"{generated_at.strftime('%Y%m%dT%H%M%SZ')}_mimic_iv_note_download_report.md"
    )
    summary_json_path = config.output_dir / "mimic_iv_note_download_summary.json"
    # PhysioNet exposes the MIMIC-IV-Note tables under a nested `note/`
    # directory for v2.2, so the local mirror follows that layout.
    dataset_root = (
        config.files_root
        / config.dataset_slug
        / config.dataset_version
        / DEFAULT_NOTE_SUBDIRECTORY
    )
    content_page_url = CONTENT_URL_TEMPLATE.format(
        dataset_slug=config.dataset_slug,
        dataset_version=config.dataset_version,
    )
    sign_dua_url = SIGN_DUA_URL_TEMPLATE.format(
        dataset_slug=config.dataset_slug,
        dataset_version=config.dataset_version,
    )

    login_ok = False
    access_granted = False
    access_gate_reason: str | None = None
    file_results: list[DownloadedNoteFile] = []

    try:
        username, password = load_physionet_credentials()
        opener, cookie_jar = build_opener()
        login_to_physionet(
            opener,
            cookie_jar,
            username=username,
            password=password,
            timeout_seconds=config.timeout_seconds,
        )
        login_ok = True

        content_page_html = fetch_text_response(
            opener,
            content_page_url,
            timeout_seconds=config.timeout_seconds,
        )
        access_requirements = extract_access_requirements(content_page_html)
        if access_requirements:
            access_gate_reason = "; ".join(access_requirements)
        else:
            access_granted = True
            for filename in config.files:
                file_url = FILE_URL_TEMPLATE.format(
                    dataset_slug=config.dataset_slug,
                    dataset_version=config.dataset_version,
                    note_subdirectory=DEFAULT_NOTE_SUBDIRECTORY,
                    filename=filename,
                )
                file_results.append(
                    download_one_file(
                        opener,
                        url=file_url,
                        destination_path=dataset_root / filename,
                        timeout_seconds=config.timeout_seconds,
                        force=config.force,
                    )
                )

            if any(result.status not in {"downloaded", "skipped_existing"} for result in file_results):
                access_granted = False
                first_failure = next(
                    (result.detail for result in file_results if result.status not in {"downloaded", "skipped_existing"}),
                    None,
                )
                access_gate_reason = first_failure or "One or more requested note files could not be downloaded."
    except Exception as exc:
        access_gate_reason = str(exc)

    if not file_results:
        file_results = [
            DownloadedNoteFile(
                filename=filename,
                destination_path=str(dataset_root / filename),
                status="not_downloaded",
                bytes_downloaded=0,
                detail=access_gate_reason or "Download was not attempted.",
            )
            for filename in config.files
        ]

    downloaded_file_count = sum(result.status == "downloaded" for result in file_results)
    skipped_existing_file_count = sum(result.status == "skipped_existing" for result in file_results)
    all_requested_files_ready = all(
        result.status in {"downloaded", "skipped_existing"} for result in file_results
    )
    summary = PhysioNetNoteDownloadSummary(
        generated_at_utc=generated_at.isoformat(),
        dataset_slug=config.dataset_slug,
        dataset_version=config.dataset_version,
        files_root=str(config.files_root),
        dataset_root=str(dataset_root),
        output_dir=str(config.output_dir),
        report_path=str(report_path),
        summary_json_path=str(summary_json_path),
        content_page_url=content_page_url,
        sign_dua_url=sign_dua_url,
        requested_files=list(config.files),
        login_ok=login_ok,
        access_granted=access_granted,
        all_requested_files_ready=all_requested_files_ready,
        access_gate_reason=access_gate_reason,
        downloaded_file_count=downloaded_file_count,
        skipped_existing_file_count=skipped_existing_file_count,
        file_results=[asdict(result) for result in file_results],
    )
    write_outputs(summary, report_path, summary_json_path)
    return summary


def main() -> int:
    """CLI entrypoint for the restricted MIMIC-IV-Note downloader."""

    config = build_config(parse_args())
    summary = run_note_download(config)

    print("PhysioNet note download completed.")
    print(f"Login OK: {summary.login_ok}")
    print(f"Access granted: {summary.access_granted}")
    print(f"All requested files ready: {summary.all_requested_files_ready}")
    print(f"Access gate reason: {summary.access_gate_reason or 'none'}")
    print(f"Dataset root: {summary.dataset_root}")
    print(f"Summary JSON: {summary.summary_json_path}")
    print(f"Report: {summary.report_path}")

    return 0 if summary.all_requested_files_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
