#!/usr/bin/env python3
"""Read-only magnitude trace for persisted MRIO environmental sidecars.

This diagnostic reuses the environmental-support census's source/provenance and
projection owners. It does not alter production semantics or canonical outputs.
"""
from __future__ import annotations

# environmental-magnitude-diagnostic-v1

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    root_text = str(import_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

from scripts.audit_environmental_support import (  # noqa: E402
    _coded_basis_and_output,
    _numeric_vector_csv,
    _resolve_scope,
    _scope_sector_sum,
    build_projection_from_split_artifacts,
    validate_split_provenance,
)

KEYS = ["sector", "extension", "stressor", "unit"]
DOM_COLUMNS = [
    *KEYS,
    "raw_global_F_total", "raw_global_abs_F_mass", "raw_scope_F_total",
    "raw_scope_abs_F_mass", "mapped_F", "mapped_x", "donor_intensity",
    "target_monetary_output", "implied_target_value", "observed_value",
    "observed_monetary_output", "observed_intensity", "value_abs_gap",
    "monetary_abs_gap", "intensity_abs_gap", "diagnostic_flags",
]
IMP_COLUMNS = [
    *KEYS,
    "raw_global_F_total", "raw_global_abs_F_mass", "origin_weighted_numerator",
    "net_import_use", "gross_abs_import_use", "cancellation_ratio",
    "n_external_origins", "n_nonzero_use_origins", "n_invalid_active_origins",
    "donor_import_intensity", "target_monetary_output", "implied_target_value",
    "observed_value", "observed_monetary_output", "observed_intensity",
    "value_abs_gap", "monetary_abs_gap", "intensity_abs_gap", "diagnostic_flags",
]
ORIGIN_COLUMNS = [
    *KEYS, "origin", "mapped_F", "mapped_x", "mapped_import_use",
    "origin_intensity", "environmental_contribution", "target_net_import_use",
    "target_gross_abs_import_use", "target_cancellation_ratio",
    "target_donor_import_intensity", "diagnostic_flags",
]
READER_COLUMNS = [
    "extension", "stressor", "unit", "robust_row_label", "match_state",
    "numeric_equal", "label_exact", "max_abs_diff", "max_scaled_diff",
    "production_total", "robust_total", "error",
]


def _write_csv(frame: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        pd.DataFrame(columns=pd.Index(list(columns), dtype="object")).to_csv(path, index=False)
        return
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Diagnostic CSV {path} is missing columns: {sorted(missing)!r}")
    frame.loc[:, list(columns)].to_csv(path, index=False)


def _read_target_sidecar(flow_dir: Path, role: str) -> pd.DataFrame:
    path = flow_dir / "environmental_extensions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "sector", "denominator_role", "extension", "stressor", "unit",
        "value", "monetary_output", "intensity",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)!r}")
    frame = frame.loc[frame["denominator_role"].astype(str).eq(role)].copy()
    if frame.duplicated(KEYS).any():
        sample = frame.loc[frame.duplicated(KEYS, keep=False), KEYS].head(10)
        raise ValueError(f"Duplicate target environmental keys: {sample.to_dict(orient='records')!r}")
    return frame.loc[:, KEYS + ["value", "monetary_output", "intensity"]].rename(
        columns={
            "value": "observed_value",
            "monetary_output": "observed_monetary_output",
            "intensity": "observed_intensity",
        }
    )


def _attach_observed(trace: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    out = trace.merge(observed, on=KEYS, how="left", validate="one_to_one")
    for column in ("observed_value", "observed_monetary_output", "observed_intensity"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["value_abs_gap"] = (out["implied_target_value"] - out["observed_value"]).abs()
    out["monetary_abs_gap"] = (
        out["target_monetary_output"] - out["observed_monetary_output"]
    ).abs()
    intensity_col = "donor_intensity" if "donor_intensity" in out.columns else "donor_import_intensity"
    out["intensity_abs_gap"] = (out[intensity_col] - out["observed_intensity"]).abs()
    return out


def _reader_comparison(
    *,
    extension: str,
    stressor: str,
    unit: str,
    production_values: pd.Series,
    robust_frame: pd.DataFrame | None,
    robust_error: str,
) -> dict[str, object]:
    base = {
        "extension": extension,
        "stressor": stressor,
        "unit": unit,
        "production_total": float(production_values.sum()),
    }
    if robust_frame is None:
        return {
            **base,
            "robust_row_label": "",
            "match_state": "ROBUST_READER_ERROR",
            "numeric_equal": False,
            "label_exact": False,
            "max_abs_diff": float("nan"),
            "max_scaled_diff": float("nan"),
            "robust_total": float("nan"),
            "error": robust_error,
        }
    labels = [str(value) for value in robust_frame.index]
    candidates = [label for label in labels if label == stressor]
    state = "EXACT"
    if len(candidates) != 1:
        candidates = [label for label in labels if label.casefold() == stressor.casefold()]
        state = "CASEFOLD_EXACT"
    if len(candidates) != 1:
        candidates = [
            label for label in labels
            if label.split(" | ", 1)[0].casefold() == stressor.casefold()
        ]
        state = "PREFIX_UNIQUE"
    if len(candidates) != 1:
        return {
            **base,
            "robust_row_label": "",
            "match_state": "NO_UNIQUE_ROBUST_ROW",
            "numeric_equal": False,
            "label_exact": False,
            "max_abs_diff": float("nan"),
            "max_scaled_diff": float("nan"),
            "robust_total": float("nan"),
            "error": "No unique robust-parser row matches the production stressor label.",
        }
    chosen = candidates[0]
    robust_row = robust_frame.loc[chosen]
    if isinstance(robust_row, pd.DataFrame):
        return {
            **base,
            "robust_row_label": chosen,
            "match_state": "DUPLICATE_ROBUST_ROW",
            "numeric_equal": False,
            "label_exact": chosen == stressor,
            "max_abs_diff": float("nan"),
            "max_scaled_diff": float("nan"),
            "robust_total": float("nan"),
            "error": "Robust parser produced duplicate matched rows.",
        }
    robust_series = cast(pd.Series, robust_row).astype("float64").reindex(production_values.index)
    if robust_series.isna().any():
        return {
            **base,
            "robust_row_label": chosen,
            "match_state": state,
            "numeric_equal": False,
            "label_exact": chosen == stressor,
            "max_abs_diff": float("nan"),
            "max_scaled_diff": float("nan"),
            "robust_total": float("nan"),
            "error": "Robust parser row does not cover the production source basis.",
        }
    prod = production_values.to_numpy(dtype="float64")
    robust = robust_series.to_numpy(dtype="float64")
    diff = np.abs(prod - robust)
    scale = np.maximum(1.0, np.maximum(np.abs(prod), np.abs(robust)))
    return {
        **base,
        "robust_row_label": chosen,
        "match_state": state,
        "numeric_equal": bool(np.allclose(prod, robust, rtol=1e-12, atol=0.0)),
        "label_exact": chosen == stressor,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "max_scaled_diff": float((diff / scale).max()) if diff.size else 0.0,
        "robust_total": float(robust.sum()),
        "error": "",
    }


def _collapse_import_origin_magnitudes(
    *,
    mapped_f: pd.DataFrame,
    mapped_x: pd.DataFrame,
    mapped_u: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not mapped_f.index.equals(mapped_x.index) or not mapped_f.index.equals(mapped_u.index):
        raise ValueError("Origin axes differ in import magnitude inputs.")
    if not mapped_f.columns.equals(mapped_x.columns) or not mapped_f.columns.equals(mapped_u.columns):
        raise ValueError("Target-sector axes differ in import magnitude inputs.")
    f = mapped_f.to_numpy(dtype="float64")
    x = mapped_x.to_numpy(dtype="float64")
    u = mapped_u.to_numpy(dtype="float64")
    if not np.isfinite(f).all() or not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError("Import magnitude primitives must be finite.")
    active = u != 0.0
    invalid_active = active & (x <= 0.0)
    coefficients = np.full_like(f, np.nan)
    positive_x = x > 0.0
    coefficients[positive_x] = f[positive_x] / x[positive_x]
    contributions = np.zeros_like(f)
    usable = active & positive_x
    contributions[usable] = coefficients[usable] * u[usable]

    net = u.sum(axis=0)
    gross = np.abs(u).sum(axis=0)
    invalid_count = invalid_active.sum(axis=0)
    numerator = contributions.sum(axis=0)
    numerator[invalid_count > 0] = np.nan
    cancellation = np.full(net.shape, np.nan)
    positive_gross = gross > 0.0
    cancellation[positive_gross] = np.abs(net[positive_gross]) / gross[positive_gross]
    intensity = np.full(net.shape, np.nan)
    usable_ratio = np.isfinite(numerator) & (net != 0.0)
    intensity[usable_ratio] = numerator[usable_ratio] / net[usable_ratio]

    summary = pd.DataFrame(
        {
            "sector": mapped_f.columns.astype(str),
            "origin_weighted_numerator": numerator,
            "net_import_use": net,
            "gross_abs_import_use": gross,
            "cancellation_ratio": cancellation,
            "n_external_origins": len(mapped_f.index),
            "n_nonzero_use_origins": active.sum(axis=0).astype(int),
            "n_invalid_active_origins": invalid_count.astype(int),
            "donor_import_intensity": intensity,
        }
    )
    return (
        summary,
        pd.DataFrame(coefficients, index=mapped_f.index, columns=mapped_f.columns),
        pd.DataFrame(contributions, index=mapped_f.index, columns=mapped_f.columns),
    )


def _flags(
    row: pd.Series,
    *,
    intensity_threshold: float,
    cancellation_threshold: float,
    import_row: bool,
) -> str:
    def as_float(value: object, default: float) -> float:
        return default if value is None else float(cast(Any, value))

    def as_int(value: object, default: int) -> int:
        return default if value is None else int(cast(Any, value))

    flags: list[str] = []
    intensity_key = "donor_import_intensity" if import_row else "donor_intensity"
    intensity = as_float(row.get(intensity_key), float("nan"))
    if math.isfinite(intensity) and abs(intensity) >= intensity_threshold:
        flags.append("EXTREME_ABS_INTENSITY")
    if import_row:
        gross = as_float(row.get("gross_abs_import_use"), 0.0)
        net = as_float(row.get("net_import_use"), float("nan"))
        cancellation = as_float(row.get("cancellation_ratio"), float("nan"))
        if gross > 0.0 and net == 0.0:
            flags.append("NET_USE_ZERO_WITH_GROSS_MASS")
        elif gross > 0.0 and math.isfinite(cancellation) and cancellation <= cancellation_threshold:
            flags.append("SIGNED_NET_CANCELLATION")
        if as_int(row.get("n_invalid_active_origins"), 0) > 0:
            flags.append("USED_ORIGIN_NONPOSITIVE_X")
    for gap_key, label in (
        ("value_abs_gap", "OBSERVED_VALUE_MISMATCH"),
        ("monetary_abs_gap", "OBSERVED_DENOMINATOR_MISMATCH"),
        ("intensity_abs_gap", "OBSERVED_INTENSITY_MISMATCH"),
    ):
        gap = as_float(row.get(gap_key), float("nan"))
        if math.isfinite(gap):
            reference_key = {
                "value_abs_gap": "observed_value",
                "monetary_abs_gap": "observed_monetary_output",
                "intensity_abs_gap": "observed_intensity",
            }[gap_key]
            reference = as_float(row.get(reference_key), float("nan"))
            if math.isfinite(reference) and gap > max(1e-9, 1e-9 * abs(reference)):
                flags.append(label)
    return ";".join(flags)


def _coded_import_use(source_zip: Path, scope: Any, system: str) -> pd.Series:
    from ecolecon_mrio.exiobase3 import (
        _code_index,
        _exio_label_to_code_lookup,
        read_exiobase3_scope_external_import_use_by_origin,
    )

    raw = read_exiobase3_scope_external_import_use_by_origin(source_zip=source_zip, scope=scope)
    regions = raw.index.get_level_values("region").astype(str)
    sectors = pd.Index(raw.index.get_level_values("sector").astype(str))
    codes = _code_index(sectors, label_to_code=_exio_label_to_code_lookup(system))
    return pd.Series(
        raw.to_numpy(dtype="float64"),
        index=pd.MultiIndex.from_arrays([regions, codes], names=["region", "sector"]),
        dtype="float64",
    )


def _resolve_source_zip(args: argparse.Namespace) -> Path:
    if args.source_zip:
        source = Path(args.source_zip).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        return source
    from ecolecon_mrio.data import ensure_exiobase_year_zip, resolve_exiobase_year_dir

    year_dir = resolve_exiobase_year_dir(int(args.year), data_root=args.data_root)
    return ensure_exiobase_year_zip(
        int(args.year),
        data_dir=year_dir,
        system=str(args.system),
        force_download=False,
        allow_download_missing=False,
        resume_downloads=True,
        show_download_progress=False,
        doi=str(args.doi),
        verbose=False,
    )


def run_trace(args: argparse.Namespace) -> dict[str, object]:
    from ecolecon_mrio.exiobase3_extensions import load_exiobase3_extension_rows_with_units
    from ecolecon_mrio.temporal import _read_exio_extension_table_from_zip

    source_zip = _resolve_source_zip(args)
    split_root = Path(args.split_root).resolve()
    dom_dir = split_root / "stk_flow=DOM"
    imp_dir = split_root / "stk_flow=IMP"
    if not dom_dir.is_dir() or not imp_dir.is_dir():
        raise FileNotFoundError("split-root must contain stk_flow=DOM and stk_flow=IMP")
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    coded_output, source_basis, source_regions, raw_sector_order = _coded_basis_and_output(
        source_zip, system=str(args.system)
    )
    coded_basis = cast(pd.MultiIndex, coded_output.index)
    scope_mapping = None if args.scope_mapping_csv is None else Path(args.scope_mapping_csv).resolve()
    scope = _resolve_scope(
        scope_code=str(args.scope_code),
        scope_mapping_csv=scope_mapping,
        source_regions=source_regions,
    )
    provenance = validate_split_provenance(
        split_root=split_root,
        source_zip=source_zip,
        scope_code=str(scope.code),
        scope_members=tuple(map(str, scope.members)),
        scope_mapping_csv=scope_mapping,
    )
    P_dom, _ = build_projection_from_split_artifacts(dom_dir, raw_exio_sectors=raw_sector_order)
    P_imp, _ = build_projection_from_split_artifacts(imp_dir, raw_exio_sectors=raw_sector_order)
    if not P_dom.columns.equals(P_imp.columns):
        raise ValueError("DOM and IMP persisted projections use different raw EXIO axes.")

    dom_observed = _read_target_sidecar(dom_dir, "target_domestic_output")
    imp_observed = _read_target_sidecar(imp_dir, "target_import_use")
    groups = pd.concat(
        [dom_observed[["extension", "stressor", "unit"]], imp_observed[["extension", "stressor", "unit"]]],
        ignore_index=True,
    ).drop_duplicates(ignore_index=True)
    dom_target = _numeric_vector_csv(dom_dir / "use_total_split.csv").reindex(P_dom.index)
    imp_target = _numeric_vector_csv(imp_dir / "use_total_split.csv").reindex(P_imp.index)
    if dom_target.isna().any() or imp_target.isna().any():
        raise ValueError("Target denominator vectors do not cover the persisted projection axes.")

    members = set(map(str, scope.members))
    sector_order = tuple(map(str, P_imp.columns))
    coded_use = _coded_import_use(source_zip, scope, str(args.system))
    origins = tuple(region for region in source_regions if region not in members)
    mapped_x_rows: list[pd.Series] = []
    mapped_u_rows: list[pd.Series] = []
    for origin in origins:
        x_origin = coded_output.xs(origin, level="region").reindex(sector_order)
        u_origin = coded_use.xs(origin, level="region").reindex(sector_order)
        if x_origin.isna().any() or u_origin.isna().any():
            raise ValueError(f"Origin denominator basis incomplete for {origin!r}.")
        mapped_x_rows.append(pd.Series(P_imp.to_numpy() @ x_origin.to_numpy(), index=P_imp.index, name=origin))
        mapped_u_rows.append(pd.Series(P_imp.to_numpy() @ u_origin.to_numpy(), index=P_imp.index, name=origin))
    mapped_x = pd.DataFrame(mapped_x_rows, columns=P_imp.index, dtype="float64")
    mapped_u = pd.DataFrame(mapped_u_rows, columns=P_imp.index, dtype="float64")
    mapped_x.index = pd.Index(origins, name="origin")
    mapped_u.index = pd.Index(origins, name="origin")

    dom_rows: list[pd.DataFrame] = []
    imp_rows: list[pd.DataFrame] = []
    origin_rows: list[dict[str, object]] = []
    reader_rows: list[dict[str, object]] = []
    source_errors: list[dict[str, str]] = []
    for extension, ext_groups in groups.groupby("extension", sort=True):
        extension_text = str(extension)
        try:
            production_rows = load_exiobase3_extension_rows_with_units(
                source_zip,
                extension=extension_text,
                kind="F",
                keywords=(),
                region=None,
                system=str(args.system),
                basis_index=source_basis,
            )
        except Exception as exc:
            source_errors.append({"extension": extension_text, "error": f"{type(exc).__name__}: {exc}"})
            continue
        production_by_stressor = {row.stressor: row for row in production_rows}
        robust_frame: pd.DataFrame | None = None
        robust_error = ""
        try:
            robust_frame = _read_exio_extension_table_from_zip(
                source_zip, f"{extension_text}.F", source_basis
            )
        except Exception as exc:
            robust_error = f"{type(exc).__name__}: {exc}"

        requested_groups = cast(pd.DataFrame, ext_groups.loc[:, ["stressor", "unit"]].drop_duplicates())
        for requested in requested_groups.to_dict(orient="records"):
            stressor = str(requested["stressor"])
            unit = str(requested["unit"])
            row = production_by_stressor.get(stressor)
            if row is None:
                source_errors.append(
                    {"extension": extension_text, "error": f"Missing production F row for stressor={stressor!r}."}
                )
                continue
            if str(row.unit) != unit:
                source_errors.append(
                    {
                        "extension": extension_text,
                        "error": (
                            f"Unit mismatch for stressor={stressor!r}: "
                            f"target_sidecar={unit!r}, source_unit={row.unit!r}."
                        ),
                    }
                )
                continue
            reader_rows.append(
                _reader_comparison(
                    extension=extension_text,
                    stressor=stressor,
                    unit=unit,
                    production_values=row.values,
                    robust_frame=robust_frame,
                    robust_error=robust_error,
                )
            )
            coded_f = pd.Series(row.values.to_numpy(dtype="float64"), index=coded_basis, dtype="float64")
            f_scope = _scope_sector_sum(
                row.values,
                coded_basis=coded_basis,
                members=tuple(map(str, scope.members)),
                sector_order=sector_order,
            )
            mapped_dom_f = pd.Series(P_dom.to_numpy() @ f_scope.to_numpy(), index=P_dom.index)
            x_scope = _scope_sector_sum(
                coded_output,
                coded_basis=coded_basis,
                members=tuple(map(str, scope.members)),
                sector_order=sector_order,
            )
            mapped_dom_x = pd.Series(P_dom.to_numpy() @ x_scope.to_numpy(), index=P_dom.index)
            dom = pd.DataFrame(
                {
                    "sector": P_dom.index.astype(str),
                    "extension": extension_text,
                    "stressor": stressor,
                    "unit": unit,
                    "raw_global_F_total": float(row.values.sum()),
                    "raw_global_abs_F_mass": float(np.abs(row.values.to_numpy()).sum()),
                    "raw_scope_F_total": float(f_scope.sum()),
                    "raw_scope_abs_F_mass": float(np.abs(f_scope.to_numpy()).sum()),
                    "mapped_F": mapped_dom_f.to_numpy(),
                    "mapped_x": mapped_dom_x.to_numpy(),
                }
            )
            dom["donor_intensity"] = np.where(dom["mapped_x"] > 0.0, dom["mapped_F"] / dom["mapped_x"], np.nan)
            dom["target_monetary_output"] = dom["sector"].map(
                lambda sector: dom_target.get(str(sector), np.nan)
            )
            dom["implied_target_value"] = dom["donor_intensity"] * dom["target_monetary_output"]
            dom = _attach_observed(dom, dom_observed)
            dom["diagnostic_flags"] = dom.apply(
                lambda item: _flags(
                    item,
                    intensity_threshold=float(args.intensity_threshold),
                    cancellation_threshold=float(args.cancellation_ratio_threshold),
                    import_row=False,
                ),
                axis=1,
            )
            dom_rows.append(dom)

            mapped_f_rows: list[pd.Series] = []
            for origin in origins:
                f_origin = coded_f.xs(origin, level="region").reindex(sector_order)
                mapped_f_rows.append(pd.Series(P_imp.to_numpy() @ f_origin.to_numpy(), index=P_imp.index, name=origin))
            mapped_f = pd.DataFrame(mapped_f_rows, columns=P_imp.index, dtype="float64")
            mapped_f.index = pd.Index(origins, name="origin")
            imp, coefficients, contributions = _collapse_import_origin_magnitudes(
                mapped_f=mapped_f, mapped_x=mapped_x, mapped_u=mapped_u
            )
            imp.insert(0, "unit", unit)
            imp.insert(0, "stressor", stressor)
            imp.insert(0, "extension", extension_text)
            imp["raw_global_F_total"] = float(row.values.sum())
            imp["raw_global_abs_F_mass"] = float(np.abs(row.values.to_numpy()).sum())
            imp["target_monetary_output"] = imp["sector"].map(
                lambda sector: imp_target.get(str(sector), np.nan)
            )
            imp["implied_target_value"] = imp["donor_import_intensity"] * imp["target_monetary_output"]
            imp = _attach_observed(imp, imp_observed)
            imp["diagnostic_flags"] = imp.apply(
                lambda item: _flags(
                    item,
                    intensity_threshold=float(args.intensity_threshold),
                    cancellation_threshold=float(args.cancellation_ratio_threshold),
                    import_row=True,
                ),
                axis=1,
            )
            imp_rows.append(imp)
            for summary in imp.loc[imp["diagnostic_flags"].astype(str).ne("")].to_dict(orient="records"):
                for origin in origins:
                    sector = str(summary["sector"])
                    origin_rows.append(
                        {
                            "sector": sector,
                            "extension": extension_text,
                            "stressor": stressor,
                            "unit": unit,
                            "origin": origin,
                            "mapped_F": float(mapped_f.loc[origin, sector]),
                            "mapped_x": float(mapped_x.loc[origin, sector]),
                            "mapped_import_use": float(mapped_u.loc[origin, sector]),
                            "origin_intensity": float(coefficients.loc[origin, sector]),
                            "environmental_contribution": float(contributions.loc[origin, sector]),
                            "target_net_import_use": float(cast(Any, summary["net_import_use"])),
                            "target_gross_abs_import_use": float(cast(Any, summary["gross_abs_import_use"])),
                            "target_cancellation_ratio": float(cast(Any, summary["cancellation_ratio"])),
                            "target_donor_import_intensity": float(cast(Any, summary["donor_import_intensity"])),
                            "diagnostic_flags": str(summary["diagnostic_flags"]),
                        }
                    )

    dom_trace = pd.concat(dom_rows, ignore_index=True) if dom_rows else pd.DataFrame()
    imp_trace = pd.concat(imp_rows, ignore_index=True) if imp_rows else pd.DataFrame()
    origin_trace = pd.DataFrame(origin_rows)
    reader_trace = pd.DataFrame(reader_rows)
    _write_csv(dom_trace, outdir / "production_magnitude_trace.csv", DOM_COLUMNS)
    _write_csv(imp_trace, outdir / "import_magnitude_trace.csv", IMP_COLUMNS)
    _write_csv(origin_trace, outdir / "import_origin_trace.csv", ORIGIN_COLUMNS)
    _write_csv(reader_trace, outdir / "source_reader_comparison.csv", READER_COLUMNS)
    pd.DataFrame(source_errors, columns=pd.Index(["extension", "error"], dtype="object")).to_csv(
        outdir / "source_errors.csv", index=False
    )

    reader_nonexact = int(
        ((~reader_trace["numeric_equal"].astype(bool)) | (~reader_trace["label_exact"].astype(bool))).sum()
    ) if not reader_trace.empty else 0
    summary: dict[str, object] = {
        "source_zip": str(source_zip),
        "provenance": provenance,
        "scope_code": str(scope.code),
        "scope_members": list(map(str, scope.members)),
        "selected_extension_stressor_groups": int(len(groups)),
        "source_error_count": len(source_errors),
        "reader_nonexact_rows": reader_nonexact,
        "dom_rows": int(len(dom_trace)),
        "dom_flagged_rows": int(dom_trace["diagnostic_flags"].astype(str).ne("").sum()) if not dom_trace.empty else 0,
        "imp_rows": int(len(imp_trace)),
        "imp_flagged_rows": int(imp_trace["diagnostic_flags"].astype(str).ne("").sum()) if not imp_trace.empty else 0,
        "imp_signed_cancellation_rows": int(
            imp_trace["diagnostic_flags"].astype(str).str.contains("SIGNED_NET_CANCELLATION", regex=False).sum()
        ) if not imp_trace.empty else 0,
        "thresholds_are_report_only": True,
        "absolute_intensity_report_threshold": float(args.intensity_threshold),
        "signed_net_cancellation_report_ratio": float(args.cancellation_ratio_threshold),
    }
    (outdir / "magnitude_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trace implausible persisted DOM/IMP environmental magnitudes back through "
            "the exact EXIO source ZIP, persisted product projection, origin-specific import use, "
            "and target retargeting denominator. The command is read-only."
        )
    )
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--source-zip", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--doi", default="10.5281/zenodo.20051562")
    parser.add_argument("--system", default="pxp")
    parser.add_argument("--scope-code", required=True)
    parser.add_argument("--scope-mapping-csv", default=None)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--intensity-threshold",
        type=float,
        default=1e9,
        help="Report-only absolute intensity threshold (default: 1e9).",
    )
    parser.add_argument(
        "--cancellation-ratio-threshold",
        type=float,
        default=1e-10,
        help="Report-only threshold for abs(net import use)/gross abs import use (default: 1e-10).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.intensity_threshold) or args.intensity_threshold <= 0.0:
        raise ValueError("--intensity-threshold must be finite and positive.")
    if (
        not math.isfinite(args.cancellation_ratio_threshold)
        or args.cancellation_ratio_threshold < 0.0
        or args.cancellation_ratio_threshold > 1.0
    ):
        raise ValueError("--cancellation-ratio-threshold must lie in [0, 1].")
    run_trace(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
