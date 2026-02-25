from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping
import zipfile

import requests
from pymrio.tools.ioutil import filename_from_url

ProgressCallback = Callable[["DownloadProgressEvent"], None]
IntegrityCheck = Callable[[Path], bool]


@dataclass
class DownloadProgressEvent:
    label: str
    phase: str
    bytes_downloaded: int
    total_bytes: int | None
    percent: float | None
    rate_bps: float | None
    eta_seconds: float | None


@dataclass
class DownloadSpec:
    url: str
    dest_path: Path
    label: str | None = None
    headers: Mapping[str, str] | None = None
    cookies: Mapping[str, str] | None = None
    chunk_size: int = 1024 * 1024
    integrity_check: IntegrityCheck | None = None


@dataclass
class DownloadResult:
    status: str
    bytes_written: int
    total_bytes: int | None
    elapsed_seconds: float
    resumed: bool
    range_supported: bool | None
    path: str
    reason: str


def _infer_total_bytes(resp: requests.Response, *, existing_size: int, resumed: bool) -> int | None:
    if resumed:
        content_range = resp.headers.get("Content-Range")
        if content_range and "/" in content_range:
            try:
                return int(content_range.rsplit("/", 1)[1])
            except Exception:
                pass
    content_len = resp.headers.get("Content-Length")
    if content_len is None:
        return None
    try:
        clen = int(content_len)
    except Exception:
        return None
    return existing_size + clen if resumed and resp.status_code == 206 else clen


def _default_progress_logger() -> ProgressCallback:
    last_emit = {"t": 0.0, "bucket": -1}

    def _cb(ev: DownloadProgressEvent) -> None:
        now = time.time()
        if ev.phase in {"starting", "verifying", "done"}:
            msg = f"[{ev.label}] {ev.phase}"
            if ev.total_bytes:
                msg += f" {ev.bytes_downloaded}/{ev.total_bytes}"
            print(msg, flush=True)
            return
        # emit each 5% bucket or every 5s when total unknown
        if ev.percent is not None:
            bucket = int(ev.percent // 5)
            if bucket == last_emit["bucket"] and (now - last_emit["t"]) < 5.0:
                return
            last_emit["bucket"] = bucket
            pct = f"{ev.percent:5.1f}%"
        else:
            if (now - last_emit["t"]) < 5.0:
                return
            pct = "  n/a"
        last_emit["t"] = now
        rate = ""
        if ev.rate_bps and ev.rate_bps > 0:
            rate = f" @ {ev.rate_bps/1024/1024:.2f} MB/s"
        eta = ""
        if ev.eta_seconds is not None and ev.eta_seconds >= 0:
            eta = f" eta {ev.eta_seconds:.0f}s"
        total = "?" if ev.total_bytes is None else str(ev.total_bytes)
        print(f"[{ev.label}] downloading {pct} ({ev.bytes_downloaded}/{total}){rate}{eta}", flush=True)

    return _cb


def zip_contains_member(path: Path, member_name: str = "file_parameters.json") -> bool:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return member_name in set(zf.namelist())
    except Exception:
        return False


def _emit_progress(
    callback: ProgressCallback | None,
    *,
    label: str,
    phase: str,
    bytes_downloaded: int,
    total_bytes: int | None,
    start_time: float,
) -> None:
    if callback is None:
        return
    elapsed = max(1e-9, time.perf_counter() - start_time)
    rate_bps = bytes_downloaded / elapsed if bytes_downloaded > 0 else None
    percent = None if (total_bytes is None or total_bytes <= 0) else (100.0 * bytes_downloaded / total_bytes)
    eta = None
    if total_bytes is not None and rate_bps and rate_bps > 0:
        eta = max(0.0, (total_bytes - bytes_downloaded) / rate_bps)
    callback(
        DownloadProgressEvent(
            label=label,
            phase=phase,
            bytes_downloaded=int(bytes_downloaded),
            total_bytes=(None if total_bytes is None else int(total_bytes)),
            percent=percent,
            rate_bps=rate_bps,
            eta_seconds=eta,
        )
    )


def download_file_resumable(
    spec: DownloadSpec,
    *,
    overwrite_existing: bool = False,
    resume: bool = True,
    progress: bool = False,
    progress_callback: ProgressCallback | None = None,
    timeout_seconds: float | None = None,
    retries: int = 3,
    restart_on_no_range: bool = True,
) -> DownloadResult:
    dest = Path(spec.dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    label = spec.label or dest.name
    cb = progress_callback if progress_callback is not None else (_default_progress_logger() if progress else None)

    def _valid(path: Path) -> bool:
        if spec.integrity_check is None:
            return True
        return bool(spec.integrity_check(path))

    if dest.exists():
        if overwrite_existing:
            ts = int(time.time())
            dest.rename(dest.with_name(dest.name + f".forced.{ts}"))
        elif _valid(dest):
            return DownloadResult(
                status="skipped_existing",
                bytes_written=int(dest.stat().st_size),
                total_bytes=int(dest.stat().st_size),
                elapsed_seconds=0.0,
                resumed=False,
                range_supported=None,
                path=str(dest),
                reason="existing_valid_file",
            )
        else:
            ts = int(time.time())
            dest.rename(dest.with_name(dest.name + f".corrupt.{ts}"))

    last_exc: Exception | None = None
    for attempt in range(max(1, int(retries))):
        t0 = time.perf_counter()
        existing_size = int(part.stat().st_size) if (resume and part.exists()) else 0
        headers = dict(spec.headers or {})
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
        _emit_progress(cb, label=label, phase="starting", bytes_downloaded=existing_size, total_bytes=None, start_time=t0)
        try:
            resp = requests.get(
                spec.url,
                stream=True,
                headers=headers or None,
                cookies=dict(spec.cookies or {}) or None,
                timeout=timeout_seconds,
            )
            status = int(resp.status_code)
            if status == 416 and existing_size > 0:
                if part.exists():
                    part.unlink()
                raise requests.HTTPError(f"HTTP 416 for {spec.url}")
            if status not in (200, 206):
                raise requests.HTTPError(f"HTTP Error {status} for {spec.url}")

            requested_range = existing_size > 0
            range_supported = status == 206 if requested_range else None
            if requested_range and status == 200 and restart_on_no_range:
                if part.exists():
                    part.unlink()
                existing_size = 0
                headers = dict(spec.headers or {})
                resp.close()
                resp = requests.get(
                    spec.url,
                    stream=True,
                    headers=headers or None,
                    cookies=dict(spec.cookies or {}) or None,
                    timeout=timeout_seconds,
                )
                status = int(resp.status_code)
                if status != 200:
                    raise requests.HTTPError(f"HTTP Error {status} for {spec.url}")
                range_supported = False

            mode = "ab" if (existing_size > 0 and status == 206) else "wb"
            if mode == "wb" and part.exists() and existing_size == 0:
                part.unlink()
            bytes_written = int(existing_size if mode == "ab" else 0)
            total_bytes = _infer_total_bytes(resp, existing_size=existing_size if mode == "ab" else 0, resumed=(mode == "ab"))
            with open(part, mode) as fh:
                for chunk in resp.iter_content(spec.chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    bytes_written += len(chunk)
                    _emit_progress(cb, label=label, phase="downloading", bytes_downloaded=bytes_written, total_bytes=total_bytes, start_time=t0)
            resp.close()

            if total_bytes is not None and bytes_written != total_bytes:
                raise IOError(f"download_incomplete:{bytes_written}/{total_bytes}")

            _emit_progress(cb, label=label, phase="verifying", bytes_downloaded=bytes_written, total_bytes=total_bytes, start_time=t0)
            if not _valid(part):
                ts = int(time.time())
                part.rename(part.with_name(part.name + f".corrupt.{ts}"))
                raise IOError("integrity_check_failed")

            os.replace(part, dest)
            elapsed = float(time.perf_counter() - t0)
            _emit_progress(cb, label=label, phase="done", bytes_downloaded=bytes_written, total_bytes=total_bytes, start_time=t0)
            return DownloadResult(
                status=("resumed_and_completed" if (mode == "ab") else "downloaded"),
                bytes_written=int(bytes_written),
                total_bytes=(None if total_bytes is None else int(total_bytes)),
                elapsed_seconds=elapsed,
                resumed=(mode == "ab"),
                range_supported=range_supported,
                path=str(dest),
                reason="ok",
            )
        except Exception as exc:  # pragma: no cover - exercised in tests via controlled failures
            last_exc = exc
            if attempt >= max(1, int(retries)) - 1:
                break
            print(f"[{label}] download attempt {attempt + 1} failed: {type(exc).__name__}: {exc}; retrying...", flush=True)
            time.sleep(min(5.0, 0.5 * (attempt + 1)))

    elapsed = 0.0
    if part.exists():
        size = int(part.stat().st_size)
    else:
        size = 0
    return DownloadResult(
        status="failed",
        bytes_written=size,
        total_bytes=None,
        elapsed_seconds=elapsed,
        resumed=bool(part.exists()),
        range_supported=None,
        path=str(dest),
        reason=f"{type(last_exc).__name__}:{last_exc}" if last_exc is not None else "unknown_error",
    )


def download_urls_resumable(
    url_list: Iterable[str],
    storage_folder: str | Path,
    *,
    overwrite_existing: bool = False,
    downlog_handler: Any | None = None,
    access_cookie: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    resume: bool = True,
    progress: bool = False,
    chunk_size: int = 1024 * 1024,
    timeout_seconds: float | None = None,
    retries: int = 3,
    integrity_check: IntegrityCheck | None = None,
) -> Any | None:
    storage_dir = Path(storage_folder)
    storage_dir.mkdir(parents=True, exist_ok=True)
    for url in url_list:
        fname = filename_from_url(url)
        spec = DownloadSpec(
            url=url,
            dest_path=storage_dir / fname,
            label=fname,
            headers=headers,
            cookies=access_cookie,
            chunk_size=int(chunk_size),
            integrity_check=integrity_check,
        )
        res = download_file_resumable(
            spec,
            overwrite_existing=overwrite_existing,
            resume=resume,
            progress=progress,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        if res.status == "failed":
            raise RuntimeError(f"download_failed:{fname}:{res.reason}")
        if downlog_handler is not None:
            if res.status == "skipped_existing":
                downlog_handler._add_fileio(f"Skip download existing file {fname}")
            else:
                downlog_handler._add_fileio(f"Downloaded {url} to {fname}")
                downlog_handler.save()
    return downlog_handler
