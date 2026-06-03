"""Generic year-batch coverage utilities built on top of load_mrio()."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
import json
import time
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .data import load_mrio, load_mrio_matrices

ValidatorFn = Callable[[int, Any, pd.Series], Mapping[str, Any]]


@dataclass
class YearAttemptRecord:
    dataset: str
    year: int
    load_status: str
    load_reason: str
    calc_all_status: str
    calc_all_reason: str
    elapsed_seconds: float | None
    cache_hit: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    validator_statuses: dict[str, str] = field(default_factory=dict)
    validator_reasons: dict[str, str] = field(default_factory=dict)
    validator_payloads: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoverageManifest:
    dataset: str
    candidate_years: list[int]
    records: pd.DataFrame
    selection: dict[str, Any] | None = None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, pd.Series):
        return {str(k): _json_safe(v) for k, v in value.to_dict().items()}
    if isinstance(value, pd.Index):
        return [str(x) for x in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return {
            "n_rows": int(value.shape[0]),
            "n_cols": int(value.shape[1]),
            "columns": [str(c) for c in value.columns.tolist()],
        }
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _normalize_validator_result(raw: Mapping[str, Any] | None) -> tuple[str, str, Any]:
    if raw is None:
        return "ok", "ok", None
    status = str(raw.get("status", "ok"))
    reason = str(raw.get("reason", "ok"))
    payload = _json_safe(raw.get("payload")) if "payload" in raw else None
    return status, reason, payload



def _coverage_load_mrio(dataset: str, year: int, **kwargs):
    """Coverage loader dispatch. EXIOBASE uses the zip-stream matrix API, not full pymrio."""
    if str(dataset).strip().lower() == "exiobase":
        matrix_kwargs = {
            key: kwargs[key]
            for key in [
                "force_download",
                "data_root",
                "verbose",
                "allow_download_missing",
                "resume_downloads",
                "show_download_progress",
                "download_chunk_mb",
                "download_timeout_seconds",
                "download_retries",
                "delete_source_zip_after_parse",
            ]
            if key in kwargs
        }
        A, x, y, basis_index, metadata = load_mrio_matrices(int(year), calc_all=False, exio_system="pxp", **matrix_kwargs)
        mrio = SimpleNamespace(
            A=A,
            x=pd.Series(x).reindex(basis_index),
            Y=pd.DataFrame({"final_demand": pd.Series(y).reindex(basis_index)}, index=basis_index),
            y=pd.Series(y).reindex(basis_index),
            metadata=dict(metadata),
        )
        return mrio, mrio.y
    return load_mrio(dataset=dataset, year=int(year), **kwargs)

def attempt_mrio_year(
    dataset: str,
    year: int,
    *,
    force_download: bool = False,
    data_root: str | None = None,
    calc_all: bool = True,
    include_ghosh: bool = True,
    verbose: bool = False,
    validators: Mapping[str, ValidatorFn] | None = None,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    delete_source_zip_after_parse: bool = False,
) -> YearAttemptRecord:
    t0 = time.perf_counter()
    validator_statuses: dict[str, str] = {}
    validator_reasons: dict[str, str] = {}
    validator_payloads: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "force_download": bool(force_download),
        "data_root": str(data_root) if data_root is not None else None,
        "allow_download_missing": bool(allow_download_missing),
        "resume_downloads": bool(resume_downloads),
        "show_download_progress": bool(show_download_progress),
        "download_chunk_mb": int(download_chunk_mb),
        "download_timeout_seconds": (None if download_timeout_seconds is None else float(download_timeout_seconds)),
        "download_retries": int(download_retries),
        "delete_source_zip_after_parse": bool(delete_source_zip_after_parse),
    }

    try:
        mrio, y_full = _coverage_load_mrio(
            dataset=dataset,
            year=int(year),
            force_download=force_download,
            data_root=data_root,
            calc_all=calc_all,
            include_ghosh=include_ghosh,
            verbose=verbose,
            allow_download_missing=allow_download_missing,
            resume_downloads=resume_downloads,
            show_download_progress=show_download_progress,
            download_chunk_mb=download_chunk_mb,
            download_timeout_seconds=download_timeout_seconds,
            download_retries=download_retries,
            delete_source_zip_after_parse=delete_source_zip_after_parse,
        )
        metadata["mrio_type"] = type(mrio).__name__
        metadata["y_len"] = int(len(y_full)) if hasattr(y_full, "__len__") else None
        metadata["has_A"] = bool(getattr(mrio, "A", None) is not None)
        metadata["has_x"] = bool(getattr(mrio, "x", None) is not None)
        if validators:
            for vid, fn in validators.items():
                try:
                    status, reason, payload = _normalize_validator_result(fn(int(year), mrio, y_full))
                except Exception as exc:  # pragma: no cover - defensive
                    status = "fail"
                    reason = f"validator_error:{type(exc).__name__}:{exc}"
                    payload = None
                validator_statuses[str(vid)] = status
                validator_reasons[str(vid)] = reason
                validator_payloads[str(vid)] = payload
        elapsed = float(time.perf_counter() - t0)
        return YearAttemptRecord(
            dataset=str(dataset),
            year=int(year),
            load_status="ok",
            load_reason="ok",
            calc_all_status="ok" if calc_all else "not_attempted",
            calc_all_reason="ok" if calc_all else "calc_all_disabled",
            elapsed_seconds=elapsed,
            cache_hit=None,
            metadata=metadata,
            validator_statuses=validator_statuses,
            validator_reasons=validator_reasons,
            validator_payloads=validator_payloads,
        )
    except Exception as exc:
        elapsed = float(time.perf_counter() - t0)
        metadata["exception_type"] = type(exc).__name__
        metadata["exception_message"] = str(exc)
        # If load_mrio raised while calc_all=True, the exact stage may be load/parse or calc_all.
        # We conservatively classify as load_error unless future loaders expose staged diagnostics.
        return YearAttemptRecord(
            dataset=str(dataset),
            year=int(year),
            load_status="load_error",
            load_reason=f"{type(exc).__name__}:{exc}",
            calc_all_status="not_attempted",
            calc_all_reason="load_failed",
            elapsed_seconds=elapsed,
            cache_hit=None,
            metadata=metadata,
            validator_statuses=validator_statuses,
            validator_reasons=validator_reasons,
            validator_payloads=validator_payloads,
        )


def _record_to_row(rec: YearAttemptRecord) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": rec.dataset,
        "year": int(rec.year),
        "load_status": rec.load_status,
        "load_reason": rec.load_reason,
        "calc_all_status": rec.calc_all_status,
        "calc_all_reason": rec.calc_all_reason,
        "elapsed_seconds": rec.elapsed_seconds,
        "cache_hit": rec.cache_hit,
        "metadata_json": json.dumps(_json_safe(rec.metadata), sort_keys=True),
    }
    for vid in sorted(set(rec.validator_statuses) | set(rec.validator_reasons) | set(rec.validator_payloads)):
        row[f"{vid}__status"] = rec.validator_statuses.get(vid, "not_run")
        row[f"{vid}__reason"] = rec.validator_reasons.get(vid, "not_run")
        payload = rec.validator_payloads.get(vid, None)
        row[f"{vid}__payload"] = json.dumps(_json_safe(payload), sort_keys=True) if payload is not None else ""
    return row


def build_year_coverage_manifest(
    dataset: str,
    years: Sequence[int],
    *,
    force_download_missing: bool = True,
    data_root: str | None = None,
    calc_all: bool = True,
    include_ghosh: bool = True,
    verbose: bool = False,
    validators: Mapping[str, ValidatorFn] | None = None,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    delete_source_zip_after_parse: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in sorted(set(int(y) for y in years)):
        rec = attempt_mrio_year(
            dataset=dataset,
            year=year,
            force_download=force_download_missing,
            data_root=data_root,
            calc_all=calc_all,
            include_ghosh=include_ghosh,
            verbose=verbose,
            validators=validators,
            allow_download_missing=allow_download_missing,
            resume_downloads=resume_downloads,
            show_download_progress=show_download_progress,
            download_chunk_mb=download_chunk_mb,
            download_timeout_seconds=download_timeout_seconds,
            download_retries=download_retries,
            delete_source_zip_after_parse=delete_source_zip_after_parse,
        )
        rows.append(_record_to_row(rec))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "year",
                "load_status",
                "load_reason",
                "calc_all_status",
                "calc_all_reason",
                "elapsed_seconds",
                "cache_hit",
                "metadata_json",
            ]
        )
    return df.sort_values("year").reset_index(drop=True)


def _contiguous_runs(years: Sequence[int]) -> list[list[int]]:
    ys = sorted(set(int(y) for y in years))
    if not ys:
        return []
    runs: list[list[int]] = [[ys[0]]]
    for y in ys[1:]:
        if y == runs[-1][-1] + 1:
            runs[-1].append(y)
        else:
            runs.append([y])
    return runs


def select_contiguous_year_block(
    manifest_df: pd.DataFrame,
    *,
    required_status_columns: Sequence[str],
    min_years: int = 8,
    prefer: str = "latest",
    tie_breaker: str = "longest",
) -> dict[str, Any]:
    if manifest_df is None or manifest_df.empty:
        return {
            "selected_years": [],
            "year_start": None,
            "year_end": None,
            "n_years": 0,
            "selection_status": "empty_manifest",
            "selection_reason": "manifest_empty",
            "best_available_run": [],
        }
    df = manifest_df.copy()
    if "year" not in df.columns or "load_status" not in df.columns:
        missing = [c for c in ["year", "load_status"] if c not in df.columns]
        return {
            "selected_years": [],
            "year_start": None,
            "year_end": None,
            "n_years": 0,
            "selection_status": "invalid_manifest",
            "selection_reason": f"missing_columns:{'|'.join(missing)}",
            "best_available_run": [],
        }
    for col in required_status_columns:
        if col not in df.columns:
            return {
                "selected_years": [],
                "year_start": None,
                "year_end": None,
                "n_years": 0,
                "selection_status": "invalid_manifest",
                "selection_reason": f"missing_required_status_column:{col}",
                "best_available_run": [],
            }
    load_ok = df["load_status"].astype(str).eq("ok")
    extra_ok = pd.Series(True, index=df.index)
    for col in required_status_columns:
        extra_ok &= df[col].astype(str).eq("ok")
    eligible_years = df.loc[load_ok & extra_ok, "year"].astype(int).tolist()
    eligible_runs = _contiguous_runs(eligible_years)
    if not eligible_runs:
        return {
            "selected_years": [],
            "year_start": None,
            "year_end": None,
            "n_years": 0,
            "selection_status": "no_eligible_years",
            "selection_reason": "no_years_meet_status_requirements",
            "best_available_run": [],
            "eligible_years": [],
        }

    def _rank(run: list[int]) -> tuple[Any, ...]:
        end = run[-1]
        start = run[0]
        length = len(run)
        # deterministic ranking components (ascending for max() target)
        prefer_key = end if prefer == "latest" else -start
        tiebreak_key = length if tie_breaker == "longest" else end
        return (prefer_key, tiebreak_key, start)

    best_run = max(eligible_runs, key=lambda r: (len(r), r[-1], -r[0]))
    qualifying_runs = [r for r in eligible_runs if len(r) >= int(min_years)]
    if not qualifying_runs:
        return {
            "selected_years": [],
            "year_start": None,
            "year_end": None,
            "n_years": 0,
            "selection_status": "no_block_meets_min_years",
            "selection_reason": f"no_contiguous_run_meets_min_years:{int(min_years)}",
            "best_available_run": best_run,
            "best_available_run_start": best_run[0],
            "best_available_run_end": best_run[-1],
            "best_available_run_n_years": len(best_run),
            "eligible_years": eligible_years,
        }
    selected = max(qualifying_runs, key=_rank)
    return {
        "selected_years": selected,
        "year_start": selected[0],
        "year_end": selected[-1],
        "n_years": len(selected),
        "selection_status": "ok",
        "selection_reason": f"policy={prefer}_contiguous_block_tiebreak={tie_breaker}",
        "best_available_run": best_run,
        "eligible_years": eligible_years,
        "min_years": int(min_years),
        "required_status_columns": [str(c) for c in required_status_columns],
    }


def write_coverage_manifest(manifest_df: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(out, index=False)
    return out


def read_coverage_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))
