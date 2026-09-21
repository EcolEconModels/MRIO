#!/usr/bin/env python3
"""Install selected public Box EXIOBASE beta ZIPs into the canonical source cache.

The ordinary MRIO source path remains DOI-first. This utility is an explicit opt-in
for an upstream beta that has no DOI yet. It resolves exact file IDs from the public
Box shared-folder page, downloads only requested ``IOT_YYYY_pxp.zip`` archives,
validates each archive, writes exact SHA-256/source provenance, and atomically puts
those bytes at the canonical yearly source-ZIP path.

No Box account or API token is used. If Box changes its public-page HTML contract,
the utility fails before replacing any canonical source ZIP.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecolecon_mrio.data import (  # noqa: E402
    DEFAULT_EXIOBASE3_DOI,
    get_mrio_data_root,
    resolve_exiobase_cache_paths,
    write_exiobase_local_source_manifest,
)
from ecolecon_mrio.download_utils import (  # noqa: E402
    DownloadSpec,
    download_file_resumable,
    zip_contains_member,
)

DEFAULT_BOX_FOLDER = "https://xio.app.box.com/s/xtx6f6z34gsgk3o4d0vh00r3x0yu23ti"
DEFAULT_RELEASE_LABEL = "3.11.2-beta"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36"


def parse_years(spec: str) -> tuple[int, ...]:
    """Parse ``2012:2022`` or comma-separated years into a unique ordered tuple."""

    raw = str(spec).strip()
    if not raw:
        raise ValueError("--years must not be empty")
    years: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start_raw, end_raw = token.split(":", 1)
            start, end = int(start_raw), int(end_raw)
            if end < start:
                raise ValueError(f"invalid descending year span: {token}")
            years.extend(range(start, end + 1))
        else:
            years.append(int(token))
    out = tuple(dict.fromkeys(years))
    if not out:
        raise ValueError("--years selected no years")
    return out


def _balanced_json_object(text: str, start: int) -> str:
    """Return one JSON object beginning at ``start`` with quote-aware brace matching."""

    if start >= len(text) or text[start] != "{":
        raise ValueError("balanced JSON scan must start at an opening brace")
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise ValueError("unterminated Box.postStreamData JSON object")


def parse_box_post_stream_data(page_text: str) -> dict[str, Any]:
    """Parse Box's public shared-page bootstrap object without executing JavaScript."""

    text = html.unescape(str(page_text))
    marker = "Box.postStreamData"
    marker_pos = text.find(marker)
    if marker_pos < 0:
        raise ValueError("public Box page does not contain Box.postStreamData")
    equals_pos = text.find("=", marker_pos + len(marker))
    if equals_pos < 0:
        raise ValueError("Box.postStreamData assignment is malformed")
    start = text.find("{", equals_pos + 1)
    if start < 0:
        raise ValueError("Box.postStreamData has no JSON object")
    raw = _balanced_json_object(text, start)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Box.postStreamData must decode to a JSON object")
    return payload


def _walk_json(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def box_file_index(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return exact filename -> Box numeric file ID from public-page bootstrap data."""

    out: dict[str, str] = {}
    for item in _walk_json(payload):
        if str(item.get("type", "")).strip().lower() != "file":
            continue
        name = str(item.get("name", "")).strip()
        raw_id = item.get("id")
        if raw_id is None:
            raw_id = item.get("typedID", item.get("typedId"))
        token = str(raw_id or "").strip()
        if token.startswith("f_"):
            token = token[2:]
        if not name or not token.isdigit():
            continue
        previous = out.get(name)
        if previous is not None and previous != token:
            raise ValueError(f"public Box page contains duplicate filename with different IDs: {name}")
        out[name] = token
    return out


def _shared_name(url: str) -> str:
    parsed = urlparse(str(url))
    match = re.search(r"/s/([A-Za-z0-9_-]+)", parsed.path)
    if match is None:
        raise ValueError(f"Box URL does not contain a /s/<shared-name> token: {url}")
    return match.group(1)


def box_direct_download_url(folder_url: str, file_id: str) -> str:
    parsed = urlparse(str(folder_url))
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"invalid Box shared-folder URL: {folder_url}")
    shared = _shared_name(folder_url)
    return (
        f"{parsed.scheme}://{parsed.netloc}/index.php?rm=box_download_shared_file"
        f"&shared_name={shared}&file_id=f_{str(file_id)}"
    )


def box_folder_page_url(folder_url: str, page_number: int) -> str:
    """Return the public shared-folder URL for one validated one-based page."""

    page = int(page_number)
    if page < 1:
        raise ValueError(f"Box page number must be positive, got {page_number!r}")
    parsed = urlsplit(str(folder_url))
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"invalid Box shared-folder URL: {folder_url}")
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["page"] = str(page)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


def _shared_folder_page(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return Box's declared shared-folder pagination object or fail closed."""

    value = payload.get("/app-api/enduserapp/shared-folder")
    if not isinstance(value, Mapping):
        raise RuntimeError("public Box bootstrap has no shared-folder pagination object")
    return value


def _positive_page_field(page: Mapping[str, Any], field: str) -> int:
    """Validate one positive integer field from Box's public folder bootstrap."""

    raw = page.get(field)
    if isinstance(raw, bool):
        raise RuntimeError(f"public Box folder field {field} is not a positive integer: {raw!r}")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"public Box folder field {field} is not a positive integer: {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"public Box folder field {field} is not a positive integer: {raw!r}")
    return value


def fetch_public_box_index(folder_url: str, *, timeout_seconds: float = 60.0) -> tuple[dict[str, str], dict[str, str]]:
    """Fetch every declared open Box folder page and return exact file IDs/cookies.

    Box lists a public shared folder in finite HTML-bootstrap pages.  The first page
    is not evidence that absent requested years do not exist, so each declared page
    is fetched and checked before filename-to-ID resolution.  No file bytes are
    requested here.
    """

    response = requests.get(
        str(folder_url),
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        timeout=float(timeout_seconds),
    )
    response.raise_for_status()
    first_payload = parse_box_post_stream_data(response.text)
    first_page = _shared_folder_page(first_payload)
    page_count = _positive_page_field(first_page, "pageCount")
    if _positive_page_field(first_page, "pageNumber") != 1:
        raise RuntimeError("public Box initial folder response did not identify itself as page 1")
    files = box_file_index(first_payload)
    if not files:
        raise RuntimeError("public Box page exposed no file metadata in Box.postStreamData")
    cookies = response.cookies.get_dict()
    for page_number in range(2, page_count + 1):
        page_response = requests.get(
            box_folder_page_url(folder_url, page_number),
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            cookies=cookies,
            timeout=float(timeout_seconds),
        )
        page_response.raise_for_status()
        page_payload = parse_box_post_stream_data(page_response.text)
        page = _shared_folder_page(page_payload)
        if _positive_page_field(page, "pageCount") != page_count:
            raise RuntimeError("public Box folder page count changed during metadata enumeration")
        if _positive_page_field(page, "pageNumber") != page_number:
            raise RuntimeError(
                f"public Box requested page {page_number} but bootstrap reported "
                f"page {_positive_page_field(page, 'pageNumber')}"
            )
        for name, file_id in box_file_index(page_payload).items():
            prior = files.get(name)
            if prior is not None and prior != file_id:
                raise RuntimeError(f"public Box pages disagree on file ID for {name}: {prior} vs {file_id}")
            files[name] = file_id
        cookies.update(page_response.cookies.get_dict())
    return files, cookies


def install(args: argparse.Namespace) -> None:
    years = parse_years(args.years)
    if str(args.system).strip().lower() != "pxp":
        raise ValueError("this beta installer is intentionally limited to the requested pxp source family")
    data_root = get_mrio_data_root(args.data_root)
    folder_url = str(args.box_folder_url).strip()
    file_index, cookies = fetch_public_box_index(folder_url, timeout_seconds=args.timeout_seconds)

    wanted = {year: f"IOT_{year}_pxp.zip" for year in years}
    missing = [name for name in wanted.values() if name not in file_index]
    if missing:
        sample = ", ".join(sorted(file_index)[:20])
        raise RuntimeError(
            "requested EXIOBASE beta files are absent from the Box page bootstrap data: "
            f"{missing}. Discovered sample: {sample}. Box may have changed pagination/page HTML; "
            "refusing to guess file IDs."
        )

    print(
        f"EXIOBASE beta install plan: release={args.release_label}; years={years}; "
        f"data_root={data_root}; source={folder_url}",
        flush=True,
    )
    if args.dry_run:
        for year, name in wanted.items():
            print(f"DRY RUN {year}: {name} file_id={file_index[name]}", flush=True)
        return

    for year, name in wanted.items():
        paths = resolve_exiobase_cache_paths(year, system="pxp", data_root=data_root)
        data_dir = Path(paths["data_dir"])
        canonical = Path(paths["zip"])
        data_dir.mkdir(parents=True, exist_ok=True)
        file_id = file_index[name]
        direct_url = box_direct_download_url(folder_url, file_id)
        staging = data_dir / f".{name}.{re.sub(r'[^A-Za-z0-9_.-]+', '_', str(args.release_label))}.download"

        print(f"[{year}] public Box file_id={file_id}; downloading {name}", flush=True)
        result = download_file_resumable(
            DownloadSpec(
                url=direct_url,
                dest_path=staging,
                label=f"EXIOBASE {args.release_label} {year} pxp",
                headers={"User-Agent": USER_AGENT, "Referer": folder_url},
                cookies=cookies,
                integrity_check=lambda p: zip_contains_member(p, "file_parameters.json"),
            ),
            overwrite_existing=False,
            resume=True,
            progress=True,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
        )
        if result.status == "failed" or not staging.is_file():
            raise RuntimeError(f"beta download failed for {year}: {result.reason}")

        # Write the manifest first. If the process dies before os.replace(), the
        # loader sees a hash mismatch and fails closed instead of mislabelling old
        # DOI bytes as the declared beta. The following replace is atomic per year.
        write_exiobase_local_source_manifest(
            data_dir=data_dir,
            zip_path=staging,
            year=year,
            system="pxp",
            release_label=str(args.release_label),
            source_url=folder_url,
            source_item_id=file_id,
            locator_doi=str(args.locator_doi),
        )
        os.replace(staging, canonical)

        # Rewrite against the canonical filename because source_zip is part of the
        # manifest identity, then validate the canonical bytes once more.
        manifest = write_exiobase_local_source_manifest(
            data_dir=data_dir,
            zip_path=canonical,
            year=year,
            system="pxp",
            release_label=str(args.release_label),
            source_url=folder_url,
            source_item_id=file_id,
            locator_doi=str(args.locator_doi),
        )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        print(
            f"[{year}] installed {args.release_label}: {canonical}; "
            f"sha256={payload['source_zip_sha256']}",
            flush=True,
        )

    print("EXIOBASE public-Box beta install complete", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install selected EXIOBASE 3.11.2-beta pxp source ZIPs from the public XIO Box "
            "folder with exact local-byte provenance; ordinary DOI loading remains the default "
            "for years without a local override manifest."
        )
    )
    parser.add_argument("--years", required=True, help="Year list/span, e.g. 2012:2022 or 2022,2023,2024")
    parser.add_argument("--system", default="pxp", choices=("pxp",))
    parser.add_argument("--box-folder-url", default=DEFAULT_BOX_FOLDER)
    parser.add_argument("--release-label", default=DEFAULT_RELEASE_LABEL)
    parser.add_argument("--locator-doi", default=DEFAULT_EXIOBASE3_DOI)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Resolve file IDs but do not download or modify source caches")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    install(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
