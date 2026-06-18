#!/usr/bin/env python3

from __future__ import annotations

import json
import time
import re
from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import pandas as pd

from ecolecon_mrio.data import DEFAULT_EXIOBASE3_DOI, default_data_root
from ecolecon_mrio.exio_share_split import (
    load_local_iot,
    split_iot_using_exiobase_shares,
    write_split_report_xlsx,
    map_exio_value_per_physical_input,
    map_exio_emission_intensity,
)
from ecolecon_mrio.exiobase3 import ExioRegionSlice, load_exiobase3_region_slice
from ecolecon_mrio.exiobase3_extensions import (
    extract_emissions_and_intensity as extract_exio_emissions_and_intensity,
    extract_final_demand_emissions_and_intensity as extract_exio_final_demand_emissions_and_intensity,
)
from ecolecon_mrio.eurostat_iot import (
    ParsedEurostatIOT,
    build_eurostat_product_use_reference,
    load_eurostat_iot,
    load_eurostat_iot_variants,
)
from ecolecon_mrio.canonical_mapping_csv import read_canonical_split_mapping
from ecolecon_mrio.mapping_csv import (
    flatten_exio_sectors_from_mapping,
    load_weighted_mapping_frame,
    read_split_mapping,
    read_weighted_mapping,
    validate_mapping_weights,
)


DEFAULT_PATTERNS_19: tuple[str, ...] = ("coke", "petroleum", "refined")
DEFAULT_PATTERNS_35: tuple[str, ...] = ("electricity", "gas", "steam", "hot water", "air conditioning")
DEFAULT_ELECTRICITY_BY_SOURCE_PATTERNS: tuple[str, ...] = (
    "electricity from",
    "electricity by",
    "nuclear",
    "hydro",
    "wind",
    "solar",
    "biomass",
    "geothermal",
)


def _vprint(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


class _Timer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0


def _is_canonical_mapping_csv(path: str | Path) -> bool:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return False
    required = {"source_code", "source_label", "target_code", "target_label", "weight", "enabled", "notes"}
    return required.issubset(set(header.columns))


def _canonical_source_dictionary_path(mapping_csv: str | Path, classification: str) -> Path:
    base = Path(mapping_csv).resolve().parent
    if classification == "industry":
        return base / "exiobase_industries_dictionary.csv"
    return base / "exiobase_products_dictionary.csv"


def _canonical_target_dictionary_path(mapping_csv: str | Path, classification: str) -> Path:
    base = Path(mapping_csv).resolve().parent
    if classification == "industry":
        return base / "eurostat_nace_rev_2_1_industries_dictionary.csv"
    return base / "eurostat_cpa65_products_dictionary.csv"


def _write_text(path: Path, content: str, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_df_csv(df: pd.DataFrame, path: Path, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_series_csv(series: pd.Series, path: Path, *, verbose: bool, header: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, header=header)
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_rows_csv(df: pd.DataFrame, path: Path, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _vprint(verbose, f"Wrote: {path.resolve()}")


SPLIT_OUTPUT_NAME_ALIASES: tuple[tuple[str, str], ...] = (
    ("Z_split.csv", "Z_split_balanced.csv"),
    ("Y_split.csv", "Y_split_balanced.csv"),
    ("x_split.csv", "x_split_balanced.csv"),
    ("A_split.csv", "A_split_balanced.csv"),
)


def _split_info_payload(result, *, variant_dim: str | None = None, variant_value: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "method": result.balancing_method,
        "is_ras_balanced": False,
        "is_reconstruction_validated": True,
        "iterations": result.balancing_stats.iterations,
        "max_abs_row_gap": result.balancing_stats.max_abs_row_gap,
        "max_abs_col_gap": result.balancing_stats.max_abs_col_gap,
        "fallback_count": result.fallback_count,
        "has_fallback_diagnostics": result.fallback_diagnostics is not None and not result.fallback_diagnostics.empty,
    }
    if variant_dim is not None:
        payload["variant_dim"] = variant_dim
    if variant_value is not None:
        payload["variant_value"] = variant_value
    return payload


def _legacy_split_info_payload(split_info: dict[str, object]) -> dict[str, object]:
    legacy_info = dict(split_info)
    legacy_info.update({"deprecated_name": True, "actual_method": split_info["method"]})
    return legacy_info


def write_split_core_outputs(res, outdir: Path, *, verbose: bool, extra_info: dict | None = None) -> None:
    for primary_name, legacy_name in SPLIT_OUTPUT_NAME_ALIASES:
        if primary_name.startswith("Z_"):
            _write_df_csv(res.Z, outdir / primary_name, verbose=verbose)
            _write_df_csv(res.Z, outdir / legacy_name, verbose=verbose)
        elif primary_name.startswith("Y_"):
            _write_df_csv(res.Y, outdir / primary_name, verbose=verbose)
            _write_df_csv(res.Y, outdir / legacy_name, verbose=verbose)
        elif primary_name.startswith("x_"):
            _write_series_csv(res.x, outdir / primary_name, verbose=verbose, header=True)
            _write_series_csv(res.x, outdir / legacy_name, verbose=verbose, header=True)
        else:
            _write_df_csv(res.A, outdir / primary_name, verbose=verbose)
            _write_df_csv(res.A, outdir / legacy_name, verbose=verbose)

    _write_text(
        outdir / "exio_selected_sectors.json",
        json.dumps(res.exio_selected_sectors, indent=2),
        verbose=verbose,
    )
    split_info = _split_info_payload(res, **(extra_info or {}))
    _write_text(outdir / "split_info.json", json.dumps(split_info, indent=2), verbose=verbose)
    legacy_info = _legacy_split_info_payload(split_info)
    _write_text(outdir / "balance_info.json", json.dumps(legacy_info, indent=2), verbose=verbose)
    _write_text(outdir / "ras_info.json", json.dumps(legacy_info, indent=2), verbose=verbose)
    if res.fallback_diagnostics is not None and not res.fallback_diagnostics.empty:
        _write_df_csv(res.fallback_diagnostics, outdir / "fallback_diagnostics.csv", verbose=verbose)


def _read_exio_to_eurostat_mapping_with_notes(
    path: str | Path,
    notes_col: str = "notes",
    *,
    target_output_by_code: object | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    df = load_weighted_mapping_frame(path)
    mapping = read_weighted_mapping(path, target_output_by_code=target_output_by_code)
    validate_mapping_weights(mapping)

    notes_by_sector: dict[str, list[str]] = {}
    if notes_col in df.columns:
        for _, row in df.iterrows():
            agg = str(row["aggregate_sector"])
            note_val = str(row.get(notes_col, "")).strip()
            if note_val:
                notes_by_sector.setdefault(agg, []).append(note_val)

    condensed_notes: dict[str, str] = {}
    for agg, notes in notes_by_sector.items():
        uniq = []
        seen = set()
        for n in notes:
            if n not in seen:
                uniq.append(n)
                seen.add(n)
        condensed_notes[agg] = " | ".join(uniq)

    return mapping, condensed_notes



def _eurostat_output_reference_from_frames(
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
) -> pd.Series:
    """Build the Eurostat product-use output reference from in-memory Z/Y frames."""
    parsed = ParsedEurostatIOT(Z=eurostat_Z, Y=eurostat_Y, meta={})
    reference_raw = build_eurostat_product_use_reference(parsed)
    reference = cast(pd.Series, pd.to_numeric(reference_raw, errors="coerce")).fillna(0.0).astype("float64")
    reference.index = reference.index.astype(str)
    return reference

def _assert_mapping_integrity(
    *,
    sector_mapping: dict[str, dict[str, float]],
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    required_target_codes: Sequence[str] | None = None,
    label: str,
) -> None:
    euro_x = _eurostat_output_reference_from_frames(eurostat_Z, eurostat_Y)
    if required_target_codes is None:
        required_target_codes = tuple(map(str, euro_x.index))
    required_codes = [str(code) for code in required_target_codes if str(code).strip() != ""]
    missing = sorted([code for code in required_codes if code not in sector_mapping])
    if missing:
        show = ", ".join(missing[:10])
        raise SystemExit(
            "Mapping integrity check failed: missing required target codes in mapping "
            f"({label}). Examples: {show}"
        )
    try:
        validate_mapping_weights(sector_mapping)
    except ValueError as exc:
        raise SystemExit(f"Mapping integrity check failed ({label}): {exc}") from exc


def _build_output_consistency_frame(
    *,
    sector_mapping: dict[str, dict[str, float]],
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    exio_slice: object,
) -> tuple[pd.DataFrame, float, float]:
    euro_x = _eurostat_output_reference_from_frames(eurostat_Z, eurostat_Y)

    exio_raw = getattr(exio_slice, "x_region")
    exio_out = cast(pd.Series, pd.to_numeric(exio_raw, errors="coerce")).fillna(0.0).astype("float64")
    exio_out.index = exio_out.index.astype(str)

    mapped_rows: list[dict[str, float | str | bool]] = []
    for cpa, euro_val in euro_x.items():
        total = 0.0
        for exio, weight in sector_mapping.get(str(cpa), {}).items():
            if exio in exio_out.index:
                total += float(exio_out.get(exio, 0.0)) * float(weight)
        mapped_rows.append(
            {
                "cpa": str(cpa),
                "eurostat_output": float(euro_val),
                "exio_mapped_output": float(total),
            }
        )

    frame = pd.DataFrame(mapped_rows)
    exio_total = float(frame["exio_mapped_output"].sum())
    euro_total = float(frame["eurostat_output"].sum())
    if np.isfinite(exio_total) and exio_total > 0 and np.isfinite(euro_total) and euro_total > 0:
        scale = euro_total / exio_total
    else:
        scale = np.nan

    frame["exio_scale_to_eurostat_total"] = scale
    frame["exio_scaled_output"] = frame["exio_mapped_output"] * scale
    frame["abs_diff_scaled"] = (frame["exio_scaled_output"] - frame["eurostat_output"]).abs()
    frame["rel_diff_scaled"] = np.where(
        frame["eurostat_output"].abs() > 0,
        frame["abs_diff_scaled"] / frame["eurostat_output"].abs(),
        np.nan,
    )
    return frame, exio_total, euro_total


def _assert_output_consistency(
    *,
    sector_mapping: dict[str, dict[str, float]],
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    exio_slice: object,
    rtol: float,
    atol: float,
    label: str,
) -> None:
    _, exio_total, euro_total = _build_output_consistency_frame(
        sector_mapping=sector_mapping,
        eurostat_Z=eurostat_Z,
        eurostat_Y=eurostat_Y,
        exio_slice=exio_slice,
    )
    if not np.isfinite(exio_total) or not np.isfinite(euro_total) or exio_total <= 0 or euro_total <= 0:
        raise SystemExit(
            "Mapping consistency check failed: invalid total outputs for comparison "
            f"({label}). EXIO total={exio_total}, Eurostat total={euro_total}."
        )

    diff_total = abs(exio_total - euro_total)
    tol_total = max(atol, rtol * abs(euro_total))
    if diff_total > tol_total:
        raise SystemExit(
            "Mapping consistency check failed: EXIO-mapped total output does not match the Eurostat "
            f"total output reference within tolerances (rtol={rtol}, atol={atol}) for {label}. "
            f"EXIO total={exio_total:.6g}, Eurostat total={euro_total:.6g}, diff={diff_total:.6g}, tol={tol_total:.6g}."
        )


def _extract_year(value: object) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, int):
        return int(value)
    s = str(value).strip()
    if s.isdigit() and len(s) == 4:
        return int(s)
    m = re.search(r"(\\d{4})", s)
    if m:
        return int(m.group(1))
    return None


def _should_check_output(flow_value: object, eurostat_year: int | None, exio_year: int) -> bool:
    if str(flow_value).strip().upper() != "TOTAL":
        return False
    if eurostat_year is None:
        return False
    return int(exio_year) == int(eurostat_year)

def _aggregate_exio_emissions_to_eurostat(
    exio_emissions: pd.DataFrame,
    sector_to_exio_sectors: dict[str, list[str]] | dict[str, dict[str, list[str]]] | dict[str, dict[str, float]],
) -> pd.DataFrame:
    from ecolecon_mrio.sector_intensity import aggregate_intensity_by_mapping

    return aggregate_intensity_by_mapping(
        exio_emissions,
        sector_to_exio_sectors,
        group_cols=("extension", "stressor", "unit"),
        value_col="emission",
        output_col="monetary_output",
        intensity_col="emission_intensity",
        method_label="ratio-of-sums (EXIO)",
        include_exio_list=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    def _normalize_stk_flow(v: str) -> str:
        # Eurostat uses "TOTAL" (not "TOT") for stk_flow.
        if str(v).upper() == "TOT":
            return "TOTAL"
        return str(v)

    def _normalize_eurostat_filters(filters: dict) -> dict:
        if args.eurostat_stk_flow_dim == "stk_flow" and "stk_flow" in filters:
            cur = filters["stk_flow"]
            if isinstance(cur, (list, tuple)):
                filters["stk_flow"] = [_normalize_stk_flow(x) for x in cur]
            else:
                filters["stk_flow"] = _normalize_stk_flow(cur)
        return filters

    def _exio_flow_for_eurostat_flow_value(flow_value: str) -> str:
        if args.exio_share_flow != "auto":
            return str(args.exio_share_flow)
        if args.eurostat_stk_flow_dim == "stk_flow":
            f = str(flow_value).upper()
            if f == "DOM":
                return "domestic"
            if f == "IMP":
                return "imports"
            if f in {"TOTAL", "TOT"}:
                return "total"
        return "domestic"

    p = argparse.ArgumentParser(
        description="Split Eurostat AT IOT energy sectors (NACE 19/35) using EXIOBASE shares and RAS/GRAS-balance."
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Base folder for cached datasets (defaults to repo-root `data/`, overridable via env var MRIO_DATA_ROOT).",
    )
    p.add_argument("--verbose", action="store_true", help="Print progress and full paths for outputs.")
    p.add_argument(
        "--classification",
        choices=["product", "industry"],
        default="product",
        help="Choose the source/target dictionary pair for the canonical mapping CSV.",
    )

    euro = p.add_argument_group("Eurostat input")
    euro.add_argument("--eurostat-z", help="Path to Eurostat Z (square).")
    euro.add_argument("--eurostat-y", help="Path to Eurostat Y (same index as Z).")
    euro.add_argument("--eurostat-dataset", help="Eurostat dataset code (e.g. naio_10_cp1700).")
    euro.add_argument("--eurostat-filters", help="JSON dict of Eurostat dimension filters.")
    euro.add_argument("--eurostat-row-dim", help="Row dimension id in the Eurostat payload.")
    euro.add_argument("--eurostat-col-dim", help="Column dimension id in the Eurostat payload.")
    euro.add_argument("--eurostat-force-download", action="store_true")
    euro.add_argument("--eurostat-force-reparse", action="store_true")
    euro.add_argument(
        "--eurostat-stk-flows",
        help="JSON list of stk_flow values to run in one go (e.g. '[\"DOM\",\"IMP\"]'). Requires --eurostat-dataset mode.",
    )
    euro.add_argument("--eurostat-stk-flow-dim", default="stk_flow")

    p.add_argument("--exio-year", type=int, default=2020)
    p.add_argument("--exio-region", default="AT")
    p.add_argument(
        "--exio-doi",
        default=DEFAULT_EXIOBASE3_DOI,
        help="EXIOBASE Zenodo DOI or doi.org URL used for source zip discovery/download.",
    )
    p.add_argument(
        "--exio-share-flow",
        default="auto",
        choices=["auto", "domestic", "imports", "total"],
        help="Which EXIO use-slice to derive split shares from. "
        "auto maps Eurostat stk_flow DOM/IMP/TOTAL -> EXIO domestic/imports/total.",
    )
    p.add_argument(
        "--balancing-scope",
        default="local",
        choices=["local", "global"],
        help="How to apply RAS/GRAS balancing after splitting. "
        "local keeps the untouched parts of the IO table fixed and only balances cells that touch the new subsectors; "
        "global allows all cells to change.",
    )
    p.add_argument(
        "--no-mapping-consistency-check",
        action="store_true",
        help="Disable consistency checks between mapped EXIO outputs and Eurostat outputs.",
    )
    p.add_argument(
        "--mapping-consistency-rtol",
        type=float,
        default=0.2,
        help="Relative tolerance for mapping consistency checks (scaled EXIO vs Eurostat output).",
    )
    p.add_argument(
        "--mapping-consistency-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for mapping consistency checks.",
    )

    p.add_argument(
        "--dump-exio-sectors-csv",
        default=None,
        help="Write the available EXIO sector labels for the selected region slice to CSV and exit.",
    )
    p.add_argument(
        "--exio-mapping-csv",
        default=None,
        help="Canonical annotated split mapping CSV. When provided, pattern matching is disabled (recommended).",
    )
    p.add_argument(
        "--require-mapping-coverage",
        action="store_true",
        help=(
            "Require a constituent coverage check before splitting. When enabled, the mapping CSV must list "
            "the full constituent set for each aggregate (including any disabled rows). The split will error "
            "if the enabled subset covers less than (1 - --coverage-tol) of the constituent total, as measured "
            "in the selected EXIO flow."
        ),
    )
    p.add_argument(
        "--coverage-tol",
        type=float,
        default=0.0,
        help=(
            "Tolerance for the coverage check (default 0.0). Example: 1e-6 allows up to 0.0001%% of EXIO Z+Y row totals "
            "to be omitted/disabled."
        ),
    )
    p.add_argument(
        "--allow-normalized-exio-match",
        action="store_true",
        help="Allow whitespace/case normalized matching of EXIO sector labels (default: strict exact match).",
    )
    p.add_argument(
        "--write-mapping-template-csv",
        default=None,
        help="Write a mapping CSV template generated by the current pattern selection and exit.",
    )

    # Energy-specific defaults belong to the CLI (module code is generic).
    p.add_argument("--sector-19", default="C19", help="Aggregate sector code for refined products.")
    p.add_argument("--sector-35", default="D35", help="Aggregate sector code for electricity/gas/steam.")
    p.add_argument("--patterns-19", default=None, help="JSON list of EXIO patterns for sector-19 subsectors.")
    p.add_argument("--patterns-35", default=None, help="JSON list of EXIO patterns for sector-35 subsectors.")
    p.add_argument("--electricity-by-source", action="store_true", help="Add more granular electricity patterns.")

    p.add_argument(
        "--no-exio-quantity-price",
        action="store_true",
        help="Skip extracting physical quantities and value-per-physical-input proxies from EXIO extensions.",
    )
    p.add_argument(
        "--qp-extensions",
        default='["energy"]',
        help='JSON list of EXIO extension names to read for quantities (default: ["energy"]).',
    )
    p.add_argument(
        "--qp-stressor-keywords",
        default='["Energy use -"]',
        help='JSON list of stressor substring keywords to select quantity rows (default: ["Energy use -"]).',
    )

    p.add_argument(
        "--no-exio-co2-intensity",
        action="store_true",
        help="Skip extracting direct CO2 intensity (emissions per monetary output) from EXIO air_emissions extension.",
    )
    p.add_argument(
        "--co2-extension",
        default="air_emissions",
        help="EXIO extension to read for CO2 (default: air_emissions).",
    )
    p.add_argument(
        "--co2-stressor-keywords",
        default='["CO2 - combustion - air"]',
        help='JSON list of stressor substring keywords to select CO2 rows (default: ["CO2 - combustion - air"]).',
    )
    p.add_argument(
        "--exio-to-eurostat-mapping-csv",
        default=None,
        help=(
            "Optional canonical annotated weighted mapping CSV to map EXIO CO2 intensities to Eurostat sectors. "
            "If provided, writes eurostat_sector_co2_intensity.csv."
        ),
    )

    p.add_argument("--outdir", default=None, help="Output directory (required unless using a dump/template mode).")
    p.add_argument("--no-xlsx-report", action="store_true", help="Skip writing the XLSX report.")
    args = p.parse_args(argv)

    data_root = Path(args.data_root) if args.data_root is not None else default_data_root()
    eurostat_storage = data_root / "eurostat"
    _vprint(args.verbose, f"Using data root: {data_root.resolve()}")

    if (args.eurostat_z or args.eurostat_y) and args.eurostat_dataset:
        raise SystemExit("Provide either --eurostat-z/--eurostat-y OR --eurostat-dataset (not both).")

    patterns_19 = DEFAULT_PATTERNS_19 if args.patterns_19 is None else tuple(json.loads(args.patterns_19))
    patterns_35 = DEFAULT_PATTERNS_35 if args.patterns_35 is None else tuple(json.loads(args.patterns_35))
    if args.electricity_by_source:
        patterns_35 = tuple(patterns_35) + DEFAULT_ELECTRICITY_BY_SOURCE_PATTERNS
    sector_patterns = {str(args.sector_19): patterns_19, str(args.sector_35): patterns_35}

    if args.outdir is None and not (
        args.dump_exio_sectors_csv is not None or args.write_mapping_template_csv is not None
    ):
        raise SystemExit("Provide --outdir (or use --dump-exio-sectors-csv / --write-mapping-template-csv).")

    outdir = Path(args.outdir) if args.outdir is not None else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    extract_qp = not args.no_exio_quantity_price
    qp_extensions = tuple(json.loads(args.qp_extensions))
    qp_keywords = tuple(json.loads(args.qp_stressor_keywords))
    extract_co2 = not args.no_exio_co2_intensity
    co2_extension = str(args.co2_extension)
    co2_keywords = tuple(json.loads(args.co2_stressor_keywords))
    full_sector_mapping: dict[str, dict[str, float]] | None = None
    full_sector_notes: dict[str, str] | None = None
    full_sector_mapping_csv: str | None = args.exio_to_eurostat_mapping_csv
    qp_cache_dir = (
        data_root / f"exiobase3_{args.exio_year}" / "slices" / args.exio_region / "pxp" / "extensions"
    )

    if args.dump_exio_sectors_csv is not None or args.write_mapping_template_csv is not None:
        _vprint(args.verbose, "Calling: load_exiobase3_region_slice(...)")
        t = _Timer()
        exio_slice = load_exiobase3_region_slice(
            year=args.exio_year,
            region=args.exio_region,
            system="pxp",
            flow="domestic",
            cache_root=data_root,
            doi=args.exio_doi,
            verbose=args.verbose,
        )
        _vprint(args.verbose, f"Returned: load_exiobase3_region_slice ({t.elapsed_s():.2f}s)")

        if args.dump_exio_sectors_csv is not None:
            out = Path(args.dump_exio_sectors_csv)
            _write_rows_csv(
                pd.DataFrame({"exio_sector": list(map(str, exio_slice.sectors))}),
                out,
                verbose=args.verbose,
            )
            return 0

        # Template from pattern selection (not robust; intended to be edited into an explicit mapping).
        assert args.write_mapping_template_csv is not None
        rows: list[dict[str, str]] = []
        for agg, pats in sector_patterns.items():
            matched = []
            for s in map(str, exio_slice.sectors):
                s_l = s.casefold()
                if any(p.casefold() in s_l for p in pats):
                    matched.append(s)
            for s in matched:
                rows.append(
                    {
                        "aggregate_sector": str(agg),
                        "subsector": str(s),
                        "exio_sector": str(s),
                        "enabled": "1",
                    }
                )
        out = Path(args.write_mapping_template_csv)
        _write_rows_csv(pd.DataFrame(rows), out, verbose=args.verbose)
        return 0

    sector_to_exio_sectors = None
    coverage_expected_exio = None
    if args.exio_mapping_csv is not None:
        if _is_canonical_mapping_csv(args.exio_mapping_csv):
            _vprint(args.verbose, "Calling: read_canonical_split_mapping(...)")
            t = _Timer()
            sector_to_exio_sectors = read_canonical_split_mapping(
                args.exio_mapping_csv,
                _canonical_source_dictionary_path(args.exio_mapping_csv, args.classification),
                _canonical_target_dictionary_path(args.exio_mapping_csv, args.classification),
            )
            _vprint(args.verbose, f"Returned: read_canonical_split_mapping ({t.elapsed_s():.2f}s)")
        else:
            _vprint(args.verbose, "Calling: read_split_mapping(...)")
            t = _Timer()
            sector_to_exio_sectors = read_split_mapping(
                args.exio_mapping_csv,
                exio_to_eurostat_mapping_csv=args.exio_to_eurostat_mapping_csv,
            )
            _vprint(args.verbose, f"Returned: read_split_mapping ({t.elapsed_s():.2f}s)")
        if args.verbose and (args.patterns_19 is not None or args.patterns_35 is not None or args.electricity_by_source):
            _vprint(args.verbose, "NOTE: --exio-mapping-csv provided; ignoring pattern selection flags.")
        if args.require_mapping_coverage:
            def _flatten(values: object) -> list[str]:
                if isinstance(values, dict):
                    out: list[str] = []
                    for inner in values.values():
                        if isinstance(inner, dict):
                            out.extend(list(map(str, inner.keys())))
                        else:
                            out.extend(list(map(str, inner)))
                    return list(dict.fromkeys(out))
                if isinstance(values, IterableABC) and not isinstance(values, (str, bytes)):
                    return list(map(str, values))
                raise TypeError(f"Expected iterable EXIO members, got {type(values).__name__}")

            coverage_expected_exio = {
                str(aggregate): _flatten(members)
                for aggregate, members in (sector_to_exio_sectors or {}).items()
            }
    report_aggregates = (
        tuple(sector_to_exio_sectors.keys())
        if sector_to_exio_sectors is not None
        else (args.sector_19, args.sector_35)
    )

    base_filters: dict[str, object] = {}
    eurostat_year = None
    if args.eurostat_dataset:
        if not (args.eurostat_filters and args.eurostat_row_dim and args.eurostat_col_dim):
            raise SystemExit(
                "When using --eurostat-dataset, also provide --eurostat-filters, --eurostat-row-dim, --eurostat-col-dim."
            )
        base_filters = _normalize_eurostat_filters(json.loads(args.eurostat_filters))
        time_val = base_filters.get("time")
        if isinstance(time_val, (list, tuple)) and time_val:
            time_val = time_val[0]
        eurostat_year = _extract_year(time_val)

        if args.eurostat_stk_flows is not None:
            flows = list(map(str, json.loads(args.eurostat_stk_flows)))
            if args.eurostat_stk_flow_dim == "stk_flow":
                flows = [_normalize_stk_flow(f) for f in flows]
            _vprint(args.verbose, "Calling: load_eurostat_iot_variants(...)")
            t = _Timer()
            variants = load_eurostat_iot_variants(
                dataset=args.eurostat_dataset,
                base_filters=base_filters,
                variant_dim=args.eurostat_stk_flow_dim,
                variant_values=flows,
                row_dim=args.eurostat_row_dim,
                col_dim=args.eurostat_col_dim,
                storage_folder=eurostat_storage,
                force_download=args.eurostat_force_download,
                force_reparse=args.eurostat_force_reparse,
            )
            _vprint(args.verbose, f"Returned: load_eurostat_iot_variants ({t.elapsed_s():.2f}s)")

            if full_sector_mapping_csv and full_sector_mapping is None:
                total_variant = variants.get("TOTAL")
                if total_variant is None:
                    total_variant = next(iter(variants.values()))
                total_reference = build_eurostat_product_use_reference(total_variant)
                full_sector_mapping, full_sector_notes = _read_exio_to_eurostat_mapping_with_notes(
                    full_sector_mapping_csv,
                    target_output_by_code=total_reference,
                )

            exio_slices: dict[str, ExioRegionSlice] = {}
            for flow in flows:
                exio_flow = _exio_flow_for_eurostat_flow_value(flow)
                if exio_flow in exio_slices:
                    continue
                _vprint(args.verbose, f"Calling: load_exiobase3_region_slice(..., flow={exio_flow})")
                t = _Timer()
                exio_slices[exio_flow] = load_exiobase3_region_slice(
                    year=args.exio_year,
                    region=args.exio_region,
                    system="pxp",
                    flow=exio_flow,
                    cache_root=data_root,
                    doi=args.exio_doi,
                    verbose=args.verbose,
                )
                _vprint(args.verbose, f"Returned: load_exiobase3_region_slice ({t.elapsed_s():.2f}s)")

            assert outdir is not None
            for flow, parsed in variants.items():
                flow_dir = outdir / f"{args.eurostat_stk_flow_dim}={flow}"
                flow_dir.mkdir(parents=True, exist_ok=True)

                _vprint(args.verbose, f"Calling: split_iot_using_exiobase_split_profiles(...) [{flow}]")
                t = _Timer()
                exio_flow = _exio_flow_for_eurostat_flow_value(flow)
                exio_slice = exio_slices[exio_flow]
                if full_sector_mapping is not None and not args.no_mapping_consistency_check:
                    label = f"{args.eurostat_stk_flow_dim}={flow}"
                    _assert_mapping_integrity(
                        sector_mapping=full_sector_mapping,
                        eurostat_Z=parsed.Z,
                        eurostat_Y=parsed.Y,
                        required_target_codes=report_aggregates,
                        label=label,
                    )
                    if _should_check_output(flow, eurostat_year, args.exio_year):
                        output_consistency, _, _ = _build_output_consistency_frame(
                            sector_mapping=full_sector_mapping,
                            eurostat_Z=parsed.Z,
                            eurostat_Y=parsed.Y,
                            exio_slice=exio_slice,
                        )
                        out = flow_dir / "mapped_output_consistency.csv"
                        output_consistency.to_csv(out, index=False)
                        _vprint(args.verbose, f"Wrote: {out.resolve()}")
                        _assert_output_consistency(
                            sector_mapping=full_sector_mapping,
                            eurostat_Z=parsed.Z,
                            eurostat_Y=parsed.Y,
                            exio_slice=exio_slice,
                            rtol=float(args.mapping_consistency_rtol),
                            atol=float(args.mapping_consistency_atol),
                            label=label,
                        )
                    else:
                        _vprint(
                            args.verbose,
                            f"Skipping output consistency check for {label} "
                            f"(flow != TOTAL or year mismatch: Eurostat={eurostat_year}, EXIO={args.exio_year}).",
                        )
                res = split_iot_using_exiobase_shares(
                    eurostat_Z=parsed.Z,
                    eurostat_Y=parsed.Y,
                    exio_year=args.exio_year,
                    exio_region=args.exio_region,
                    exio_slice=exio_slice,
                    sector_to_exio_sectors=sector_to_exio_sectors,
                    sector_patterns=None if sector_to_exio_sectors is not None else sector_patterns,
                    exio_cache_root=data_root,
                    balancing_scope=args.balancing_scope,
                    eurostat_sector_mapping=full_sector_mapping,
                    coverage_expected_exio_sectors=coverage_expected_exio,
                    coverage_tol=args.coverage_tol,
                    extract_quantity_and_price=extract_qp,
                    quantity_price_extensions=qp_extensions if extract_qp else None,
                    quantity_price_stressor_keywords=qp_keywords if extract_qp else None,
                    quantity_price_cache_dir=qp_cache_dir if extract_qp else None,
                )
                _vprint(args.verbose, f"Returned: split_iot_using_exiobase_shares ({t.elapsed_s():.2f}s)")

                write_split_core_outputs(
                    res,
                    flow_dir,
                    verbose=args.verbose,
                    extra_info={
                        "variant_dim": args.eurostat_stk_flow_dim,
                        "variant_value": flow,
                        "exio_doi": args.exio_doi,
                    },
                )
                if res.exio_quantity_and_price is not None:
                    out = flow_dir / "exio_quantity_and_price.csv"
                    res.exio_quantity_and_price.to_csv(out, index=False)
                    _vprint(args.verbose, f"Wrote: {out.resolve()}")

                    # Compute and output disaggregated sector values per physical input
                    mapping_for_values = sector_to_exio_sectors or res.exio_selected_sectors
                    disagg_values = map_exio_value_per_physical_input(
                        exio_quantity_and_price=res.exio_quantity_and_price,
                        sector_to_exio_sectors=mapping_for_values,
                        split_profiles=res.split_profiles,
                    )
                    if disagg_values is not None:
                        out = flow_dir / "disaggregated_sector_value_per_physical_input.csv"
                        disagg_values.to_csv(out, index=False)
                        _vprint(args.verbose, f"Wrote: {out.resolve()}")

                # Compute and output disaggregated sector CO2 intensities (direct emissions per monetary output)
                if extract_co2:
                    exio_sectors: list[str] = []
                    for agg in res.split_profiles.keys():
                        exio_sectors.extend(res.exio_selected_sectors.get(str(agg), []))
                    exio_sectors = list(dict.fromkeys(map(str, exio_sectors)))

                    exio_out = exio_slice.x_region.astype("float64")

                    # Final-demand monetary totals by EXIO category (EUR)
                    y_by_category = exio_slice.Y_region.sum(axis=0)
                    if isinstance(y_by_category, pd.DataFrame):
                        y_by_category = y_by_category.iloc[:, 0]
                    y_by_category = y_by_category.astype("float64")

                    exio_co2 = extract_exio_emissions_and_intensity(
                        container=exio_slice.source_zip,
                        region=exio_slice.region,
                        sectors=exio_sectors,
                        monetary_output=exio_out,
                        extension=co2_extension,
                        stressor_keywords=co2_keywords,
                        cache_dir=qp_cache_dir,
                    )
                    if exio_co2 is not None:
                        out = flow_dir / "exio_co2_intensity.csv"
                        exio_co2.to_csv(out, index=False)
                        _vprint(args.verbose, f"Wrote: {out.resolve()}")

                        mapping_for_values = sector_to_exio_sectors or res.exio_selected_sectors
                        disagg_co2 = map_exio_emission_intensity(
                            exio_emissions_and_intensity=exio_co2,
                            sector_to_exio_sectors=mapping_for_values,
                            split_profiles=res.split_profiles,
                        )
                        if disagg_co2 is not None:
                            out = flow_dir / "disaggregated_sector_co2_intensity.csv"
                            disagg_co2.to_csv(out, index=False)
                            _vprint(args.verbose, f"Wrote: {out.resolve()}")

                    # Also extract direct final-demand CO2 (F_Y), so users can report F_Y (or intensity * y)
                    exio_fd_co2 = extract_exio_final_demand_emissions_and_intensity(
                        container=exio_slice.source_zip,
                        region=exio_slice.region,
                        monetary_final_demand=y_by_category,
                        extension=co2_extension,
                        stressor_keywords=co2_keywords,
                        cache_dir=qp_cache_dir,
                    )
                    if exio_fd_co2 is not None:
                        out = flow_dir / "exio_final_demand_co2_intensity.csv"
                        exio_fd_co2.to_csv(out, index=False)
                        _vprint(args.verbose, f"Wrote: {out.resolve()}")

                if extract_co2 and full_sector_mapping is not None:
                    exio_sectors_full = flatten_exio_sectors_from_mapping(full_sector_mapping)

                    available = set(map(str, exio_slice.sectors))
                    exio_sectors_full = [s for s in exio_sectors_full if s in available]

                    exio_out_full = exio_slice.x_region.astype("float64")
                    exio_co2_full = extract_exio_emissions_and_intensity(
                        container=exio_slice.source_zip,
                        region=exio_slice.region,
                        sectors=exio_sectors_full,
                        monetary_output=exio_out_full,
                        extension=co2_extension,
                        stressor_keywords=co2_keywords,
                        cache_dir=qp_cache_dir,
                    )
                    if exio_co2_full is not None:
                        eurostat_co2 = _aggregate_exio_emissions_to_eurostat(
                            exio_co2_full, full_sector_mapping
                        )
                        if full_sector_notes is not None and not eurostat_co2.empty:
                            eurostat_co2 = eurostat_co2.copy()
                            eurostat_co2["mapping_notes"] = [
                                full_sector_notes.get(str(sector), "")
                                for sector in eurostat_co2["aggregated_sector"].astype(str)
                            ]
                        if not eurostat_co2.empty:
                            out = flow_dir / "eurostat_sector_co2_intensity.csv"
                            eurostat_co2.to_csv(out, index=False)
                            _vprint(args.verbose, f"Wrote: {out.resolve()}")
                if not args.no_xlsx_report:
                    out = flow_dir / "energy_split_report.xlsx"
                    write_split_report_xlsx(
                        out,
                        result=res,
                        aggregates=report_aggregates,
                    )
                    _vprint(args.verbose, f"Wrote: {out.resolve()}")
            return 0

        _vprint(args.verbose, "Calling: load_eurostat_iot(...)")
        t = _Timer()
        parsed = load_eurostat_iot(
            dataset=args.eurostat_dataset,
            filters=base_filters,
            row_dim=args.eurostat_row_dim,
            col_dim=args.eurostat_col_dim,
            storage_folder=eurostat_storage,
            force_download=args.eurostat_force_download,
            force_reparse=args.eurostat_force_reparse,
        )
        _vprint(args.verbose, f"Returned: load_eurostat_iot ({t.elapsed_s():.2f}s)")
        Z, Y = parsed.Z, parsed.Y
        exio_flow_single = _exio_flow_for_eurostat_flow_value(
            str(base_filters.get(args.eurostat_stk_flow_dim, "DOM"))
        )
    else:
        if not (args.eurostat_z and args.eurostat_y):
            raise SystemExit("Provide --eurostat-z and --eurostat-y, or use --eurostat-dataset mode.")
        _vprint(args.verbose, "Calling: load_local_iot(...)")
        t = _Timer()
        Z, Y = load_local_iot(args.eurostat_z, args.eurostat_y)
        _vprint(args.verbose, f"Returned: load_local_iot ({t.elapsed_s():.2f}s)")
        exio_flow_single = "domestic" if args.exio_share_flow == "auto" else str(args.exio_share_flow)

    _vprint(args.verbose, "Calling: split_iot_using_exiobase_shares(...)")
    t = _Timer()
    res = split_iot_using_exiobase_shares(
        eurostat_Z=Z,
        eurostat_Y=Y,
        exio_year=args.exio_year,
        exio_region=args.exio_region,
        exio_slice=load_exiobase3_region_slice(
            year=args.exio_year,
            region=args.exio_region,
            system="pxp",
            flow=exio_flow_single,
            cache_root=data_root,
            doi=args.exio_doi,
            verbose=args.verbose,
        ),
        sector_to_exio_sectors=sector_to_exio_sectors,
        sector_patterns=None if sector_to_exio_sectors is not None else sector_patterns,
        exio_cache_root=data_root,
        balancing_scope=args.balancing_scope,
        eurostat_sector_mapping=full_sector_mapping,
        coverage_expected_exio_sectors=coverage_expected_exio,
        coverage_tol=args.coverage_tol,
        extract_quantity_and_price=extract_qp,
        quantity_price_extensions=qp_extensions if extract_qp else None,
        quantity_price_stressor_keywords=qp_keywords if extract_qp else None,
        quantity_price_cache_dir=qp_cache_dir if extract_qp else None,
    )
    _vprint(args.verbose, f"Returned: split_iot_using_exiobase_shares ({t.elapsed_s():.2f}s)")

    assert outdir is not None
    write_split_core_outputs(
        res,
        outdir,
        verbose=args.verbose,
        extra_info={"exio_doi": args.exio_doi},
    )
    if res.exio_quantity_and_price is not None:
        out = outdir / "exio_quantity_and_price_proxies.csv"
        res.exio_quantity_and_price.to_csv(out, index=False)
        _vprint(args.verbose, f"Wrote: {out.resolve()}")

        # Compute and output disaggregated sector values per physical input
        mapping_for_values = sector_to_exio_sectors or res.exio_selected_sectors
        disagg_values = map_exio_value_per_physical_input(
            exio_quantity_and_price=res.exio_quantity_and_price,
            sector_to_exio_sectors=mapping_for_values,
            split_profiles=res.split_profiles,
        )
        if disagg_values is not None:
            out = outdir / "disaggregated_sector_value_per_physical_input.csv"
            disagg_values.to_csv(out, index=False)
            _vprint(args.verbose, f"Wrote: {out.resolve()}")

    # Compute and output disaggregated sector CO2 intensities (direct emissions per monetary output)
    exio_slice_single = None
    if full_sector_mapping is not None and not args.no_mapping_consistency_check:
        exio_slice_single = load_exiobase3_region_slice(
            year=args.exio_year,
            region=args.exio_region,
            system="pxp",
            flow=exio_flow_single,
            cache_root=data_root,
            doi=args.exio_doi,
            verbose=args.verbose,
        )
        _assert_mapping_integrity(
            sector_mapping=full_sector_mapping,
            eurostat_Z=Z,
            eurostat_Y=Y,
            required_target_codes=report_aggregates,
            label="single",
        )
        flow_value_single = None
        if args.eurostat_dataset:
            flow_value_single = base_filters.get(args.eurostat_stk_flow_dim, "DOM")
        if _should_check_output(flow_value_single, eurostat_year, args.exio_year):
            output_consistency, _, _ = _build_output_consistency_frame(
                sector_mapping=full_sector_mapping,
                eurostat_Z=Z,
                eurostat_Y=Y,
                exio_slice=exio_slice_single,
            )
            out = outdir / "mapped_output_consistency.csv"
            output_consistency.to_csv(out, index=False)
            _vprint(args.verbose, f"Wrote: {out.resolve()}")
            _assert_output_consistency(
                sector_mapping=full_sector_mapping,
                eurostat_Z=Z,
                eurostat_Y=Y,
                exio_slice=exio_slice_single,
                rtol=float(args.mapping_consistency_rtol),
                atol=float(args.mapping_consistency_atol),
                label="single",
            )
        else:
            _vprint(
                args.verbose,
                "Skipping output consistency check for single flow "
                f"(flow != TOTAL or year mismatch: Eurostat={eurostat_year}, EXIO={args.exio_year}).",
            )
    if extract_co2:
        exio_sectors: list[str] = []
        for agg in res.split_profiles.keys():
            exio_sectors.extend(res.exio_selected_sectors.get(str(agg), []))
        exio_sectors = list(dict.fromkeys(map(str, exio_sectors)))

        if exio_slice_single is None:
            exio_slice_single = load_exiobase3_region_slice(
                year=args.exio_year,
                region=args.exio_region,
                system="pxp",
                flow=exio_flow_single,
                cache_root=data_root,
                doi=args.exio_doi,
                verbose=args.verbose,
            )
        exio_out = exio_slice_single.x_region.astype("float64")
        exio_co2 = extract_exio_emissions_and_intensity(
            container=exio_slice_single.source_zip,
            region=exio_slice_single.region,
            sectors=exio_sectors,
            monetary_output=exio_out,
            extension=co2_extension,
            stressor_keywords=co2_keywords,
            cache_dir=qp_cache_dir,
        )
        if exio_co2 is not None:
            out = outdir / "exio_co2_intensity.csv"
            exio_co2.to_csv(out, index=False)
            _vprint(args.verbose, f"Wrote: {out.resolve()}")

            mapping_for_values = sector_to_exio_sectors or res.exio_selected_sectors
            disagg_co2 = map_exio_emission_intensity(
                exio_emissions_and_intensity=exio_co2,
                sector_to_exio_sectors=mapping_for_values,
                split_profiles=res.split_profiles,
            )
            if disagg_co2 is not None:
                out = outdir / "disaggregated_sector_co2_intensity.csv"
                disagg_co2.to_csv(out, index=False)
                _vprint(args.verbose, f"Wrote: {out.resolve()}")

    if extract_co2 and full_sector_mapping is not None:
        exio_sectors_full = flatten_exio_sectors_from_mapping(full_sector_mapping)

        if exio_slice_single is None:
            exio_slice_single = load_exiobase3_region_slice(
                year=args.exio_year,
                region=args.exio_region,
                system="pxp",
                flow=exio_flow_single,
                cache_root=data_root,
                doi=args.exio_doi,
                verbose=args.verbose,
            )
        available = set(map(str, exio_slice_single.sectors))
        exio_sectors_full = [s for s in exio_sectors_full if s in available]

        exio_out_full = exio_slice_single.x_region.astype("float64")
        exio_co2_full = extract_exio_emissions_and_intensity(
            container=exio_slice_single.source_zip,
            region=exio_slice_single.region,
            sectors=exio_sectors_full,
            monetary_output=exio_out_full,
            extension=co2_extension,
            stressor_keywords=co2_keywords,
            cache_dir=qp_cache_dir,
        )
        if exio_co2_full is not None:
            eurostat_co2 = _aggregate_exio_emissions_to_eurostat(exio_co2_full, full_sector_mapping)
            if full_sector_notes is not None and not eurostat_co2.empty:
                eurostat_co2 = eurostat_co2.copy()
                eurostat_co2["mapping_notes"] = eurostat_co2["aggregated_sector"].map(
                    full_sector_notes
                ).fillna("")
            if not eurostat_co2.empty:
                out = outdir / "eurostat_sector_co2_intensity.csv"
                eurostat_co2.to_csv(out, index=False)
                _vprint(args.verbose, f"Wrote: {out.resolve()}")
    if not args.no_xlsx_report:
        out = outdir / "energy_split_report.xlsx"
        write_split_report_xlsx(out, result=res, aggregates=report_aggregates)
        _vprint(args.verbose, f"Wrote: {out.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
