#!/usr/bin/env python3
"""Compose a direct source->target mapping through a common intermediate.

Default Route-B use:

    Eurostat Energy Balance row -> EXIOBASE3 product -> Eurostat CPA target

The generated output is a direct canonical category-target mapping, so runtime
pipelines can use the direct EB->CPA map without carrying the EXIO intermediate.
The intermediate edge table is written separately for audit and reproducibility.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_LEFT_CSV: Final[Path] = (
    REPO_ROOT
    / "data"
    / "exiobase_concordance_sources"
    / "derived"
    / "eurostat_energy_balance_to_exiobase3"
    / "eb_to_exio3_products_collapsed.csv"
)
DEFAULT_RIGHT_CSV: Final[Path] = (
    REPO_ROOT
    / "scripts"
    / "mappings"
    / "generated"
    / "exio_to_eurostat_cpa65_products_from_bridge_nsi_at.csv"
)
DEFAULT_OUT_DIR: Final[Path] = (
    REPO_ROOT
    / "scripts"
    / "mappings"
    / "generated"
    / "energy_balance_to_cpa_from_common_intermediate"
)

CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "order",
    "kind",
    "source_dataset",
    "source_system",
    "source_version",
    "source_level",
    "target_dataset",
    "target_system",
    "target_version",
    "target_level",
    "source_code",
    "source_label",
    "target_code",
    "target_label",
    "enabled",
    "provenance",
    "certainty",
    "allocation_rule",
    "source_name",
    "official_source",
    "notes",
    "source_balance_role",
    "target_sector_role",
    "include_in_direct_energy",
)

SOURCE_ROLES_TO_TARGET_ROLES: Final[dict[str, str]] = {
    "final_use": "end_use_sector",
    "non_energy_use": "end_use_sector",
    "transformation_input": "transformation_sector",
    "transformation_output": "transformation_sector",
    "energy_sector_own_use": "transformation_sector",
}


@dataclass(frozen=True)
class ComposeResult:
    mapping: pd.DataFrame
    edges: pd.DataFrame
    diagnostics: pd.DataFrame
    coverage: dict[str, object]


def clean_text(value: object) -> str:
    """Normalize text for stable joins and CSV outputs."""

    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).strip().split())


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV as strings with blanks preserved."""

    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def parse_filter_values(values: Sequence[str] | None) -> set[str]:
    """Parse repeated comma/semicolon-separated CLI filter values."""

    out: set[str] = set()
    for value in values or ():
        for token in str(value).replace(";", ",").split(","):
            cleaned = clean_text(token)
            if cleaned:
                out.add(cleaned)
    return out


def require_columns(frame: pd.DataFrame, *, path: Path, columns: Sequence[str]) -> None:
    """Validate that a frame has all required columns."""

    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def prepare_left(
    frame: pd.DataFrame,
    *,
    path: Path,
    source_code_col: str,
    source_label_col: str,
    intermediate_code_col: str,
    intermediate_label_col: str,
    source_balance_role_col: str,
    include_col: str,
    source_role_filter: set[str] | None = None,
    include_direct_energy_only: bool = False,
) -> pd.DataFrame:
    """Normalize the left source->intermediate mapping side."""

    require_columns(
        frame,
        path=path,
        columns=[source_code_col, source_label_col, intermediate_code_col, intermediate_label_col],
    )
    out = pd.DataFrame(
        {
            "left_source_code": frame[source_code_col].map(clean_text),
            "left_source_label": frame[source_label_col].map(clean_text),
            "intermediate_code": frame[intermediate_code_col].map(clean_text),
            "intermediate_label": frame[intermediate_label_col].map(clean_text),
            "source_balance_role": frame[source_balance_role_col].map(clean_text) if source_balance_role_col in frame.columns else "",
            "include_in_direct_energy": frame[include_col].map(clean_text) if include_col in frame.columns else "1",
        }
    )
    out = out.loc[(out["left_source_code"] != "") & (out["intermediate_code"] != "")].copy()
    if source_role_filter:
        out = out.loc[out["source_balance_role"].isin(source_role_filter)].copy()
    if include_direct_energy_only:
        include_norm = out["include_in_direct_energy"].astype(str).str.strip().str.lower()
        out = out.loc[include_norm.isin({"1", "true", "t", "yes", "y", "on"})].copy()
    return out.drop_duplicates().copy()


def prepare_right(
    frame: pd.DataFrame,
    *,
    path: Path,
    intermediate_code_col: str,
    intermediate_label_col: str,
    target_code_col: str,
    target_label_col: str,
    enabled_col: str,
) -> pd.DataFrame:
    """Normalize the right intermediate->target mapping side."""

    require_columns(frame, path=path, columns=[intermediate_code_col, target_code_col])
    enabled = pd.Series(True, index=frame.index)
    if enabled_col in frame.columns:
        enabled = frame[enabled_col].astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y", "on"})
    out = pd.DataFrame(
        {
            "intermediate_code": frame[intermediate_code_col].map(clean_text),
            "intermediate_label_right": frame[intermediate_label_col].map(clean_text) if intermediate_label_col in frame.columns else "",
            "target_code": frame[target_code_col].map(clean_text),
            "target_label": frame[target_label_col].map(clean_text) if target_label_col in frame.columns else "",
        }
    )
    out = out.loc[enabled & (out["intermediate_code"] != "") & (out["target_code"] != "")].drop_duplicates().copy()
    return out


def infer_target_sector_role(source_balance_role: str, target_code: str) -> str:
    """Infer the direct-energy category-target role used by mapping_csv.py."""

    if target_code == "Households" or target_code == "CPA_T":
        return "households"
    return SOURCE_ROLES_TO_TARGET_ROLES.get(source_balance_role, "end_use_sector")


def compose_common_intermediate_mapping(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    target_version: str,
    source_name: str,
    official_source: str,
) -> ComposeResult:
    """Compose left source->intermediate and right intermediate->target edges."""

    edges = left.merge(right, on="intermediate_code", how="left", validate="many_to_many")
    missing = edges.loc[edges["target_code"].astype(str).str.strip().eq("")].copy()
    diagnostics = pd.DataFrame(
        {
            "diagnostic_kind": ["intermediate_without_target"] * len(missing),
            "source_code": missing["left_source_code"].astype(str).tolist(),
            "intermediate_code": missing["intermediate_code"].astype(str).tolist(),
            "message": ["No right-side target row matched this intermediate code."] * len(missing),
        }
    )
    matched = edges.loc[edges["target_code"].astype(str).str.strip().ne("")].copy()
    if matched.empty:
        mapping = pd.DataFrame(columns=CANONICAL_COLUMNS)
    else:
        grouped = matched.groupby(
            ["left_source_code", "left_source_label", "source_balance_role", "include_in_direct_energy", "target_code", "target_label"],
            as_index=False,
            dropna=False,
        ).agg(
            intermediate_codes=("intermediate_code", lambda values: ";".join(sorted(set(map(str, values))))),
            intermediate_labels=("intermediate_label", lambda values: " | ".join(sorted({clean_text(value) for value in values if clean_text(value)}))),
        )
        target_counts = grouped.groupby("left_source_code")["target_code"].transform("nunique").astype(int)
        grouped["n_targets_for_source"] = target_counts
        rows: list[dict[str, object]] = []
        for order, row in enumerate(grouped.sort_values(["left_source_code", "target_code"]).itertuples(index=False), start=1):
            multi = int(row.n_targets_for_source) > 1
            allocation_rule = "split-by-energy-product-prd-use" if multi else "fixed-share"
            source_role = clean_text(row.source_balance_role)
            target_code = clean_text(row.target_code)
            rows.append(
                {
                    "order": order,
                    "kind": "category-target",
                    "source_dataset": "Eurostat_Energy_Balances",
                    "source_system": "nrg_bal",
                    "source_version": "latest",
                    "source_level": "balance-flow",
                    "target_dataset": "Eurostat_IO",
                    "target_system": "pxp",
                    "target_version": target_version,
                    "target_level": "CPA_product",
                    "source_code": row.left_source_code,
                    "source_label": row.left_source_label,
                    "target_code": target_code,
                    "target_label": row.target_label or target_code,
                    "enabled": "1",
                    "provenance": "common-intermediate-composition",
                    "certainty": "generated-relational-concordance",
                    "allocation_rule": allocation_rule,
                    "source_name": source_name,
                    "official_source": official_source,
                    "notes": (
                        f"Composed through intermediate code(s) {row.intermediate_codes}. "
                        f"Intermediate labels: {row.intermediate_labels}. "
                        "Many-target rows are intentionally unweighted here; the existing EB pipeline resolves shares from the selected Eurostat target-side value reference."
                    ),
                    "source_balance_role": source_role,
                    "target_sector_role": infer_target_sector_role(source_role, target_code),
                    "include_in_direct_energy": row.include_in_direct_energy or "1",
                }
            )
        mapping = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)

    coverage = {
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "edge_rows": int(len(matched)),
        "mapping_rows": int(len(mapping)),
        "source_rows": int(mapping["source_code"].nunique()) if not mapping.empty else 0,
        "target_rows": int(mapping["target_code"].nunique()) if not mapping.empty else 0,
        "diagnostic_rows": int(len(diagnostics)),
        "multi_target_sources": int(mapping.loc[mapping["allocation_rule"].eq("split-by-energy-product-prd-use"), "source_code"].nunique()) if not mapping.empty else 0,
    }
    return ComposeResult(mapping=mapping, edges=matched, diagnostics=diagnostics, coverage=coverage)


def write_outputs(result: ComposeResult, *, out_csv: Path, edge_out_csv: Path, diagnostics_csv: Path, manifest_json: Path) -> dict[str, object]:
    """Write direct mapping and audit outputs."""

    for path in (out_csv, edge_out_csv, diagnostics_csv, manifest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    result.mapping.to_csv(out_csv, index=False)
    result.edges.to_csv(edge_out_csv, index=False)
    result.diagnostics.to_csv(diagnostics_csv, index=False)
    manifest = {
        "mapping_name": "source_to_target_via_common_intermediate",
        "created_utc": datetime.now(UTC).isoformat(),
        "created_by_script": "scripts/mappings/compose_mapping_via_common_intermediate.py",
        "method_notes": [
            "Composes left source->intermediate and right intermediate->target relations into a direct source->target mapping.",
            "Default Route-B use is Eurostat Energy Balance -> EXIOBASE3 -> Eurostat CPA.",
            "The direct mapping can be consumed by the existing EB mapping APIs; intermediate edges remain in the audit CSV.",
            "Many-target rows retain allocation_rule=split-by-energy-product-prd-use so runtime Eurostat target-side values resolve weights.",
        ],
        "coverage": result.coverage,
        "output_files": {
            "mapping": out_csv.name,
            "edge_audit": edge_out_csv.name,
            "diagnostics": diagnostics_csv.name,
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-csv", type=Path, default=DEFAULT_LEFT_CSV)
    parser.add_argument("--right-csv", type=Path, default=DEFAULT_RIGHT_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_DIR / "energy_balance_to_cpa_products_from_common_intermediate.csv")
    parser.add_argument("--edge-out-csv", type=Path, default=DEFAULT_OUT_DIR / "energy_balance_to_cpa_products_from_common_intermediate_edges.csv")
    parser.add_argument("--diagnostics-csv", type=Path, default=DEFAULT_OUT_DIR / "energy_balance_to_cpa_products_from_common_intermediate_diagnostics.csv")
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_OUT_DIR / "energy_balance_to_cpa_products_from_common_intermediate_manifest.json")
    parser.add_argument("--left-source-code-col", default="source_code")
    parser.add_argument("--left-source-label-col", default="source_label")
    parser.add_argument("--left-intermediate-code-col", default="exio3_product_code")
    parser.add_argument("--left-intermediate-label-col", default="exio3_product_name")
    parser.add_argument("--left-source-balance-role-col", default="source_balance_role")
    parser.add_argument("--left-include-col", default="include_in_direct_energy")
    parser.add_argument(
        "--source-balance-role-filter",
        action="append",
        default=None,
        help=(
            "Restrict left-side source rows to these source_balance_role values. "
            "May be repeated or comma/semicolon-separated. For the direct EB workbook path, use final_use."
        ),
    )
    parser.add_argument(
        "--include-direct-energy-only",
        action="store_true",
        help="Keep only left-side rows whose include_in_direct_energy column is truthy.",
    )
    parser.add_argument("--right-intermediate-code-col", default="source_code")
    parser.add_argument("--right-intermediate-label-col", default="source_label")
    parser.add_argument("--right-target-code-col", default="target_code")
    parser.add_argument("--right-target-label-col", default="target_label")
    parser.add_argument("--right-enabled-col", default="enabled")
    parser.add_argument("--target-version", default="CPA_65")
    parser.add_argument("--source-name", default="EB->EXIO3 plus EXIO3->CPA common-intermediate composition")
    parser.add_argument("--official-source", default="generated common-intermediate concordance")
    parser.add_argument("--allow-diagnostics", action="store_true", help="Write outputs even if unmatched intermediate diagnostics are present.")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the common-intermediate mapping composer."""

    args = parse_args(argv)
    left_raw = read_csv(args.left_csv)
    right_raw = read_csv(args.right_csv)
    left = prepare_left(
        left_raw,
        path=args.left_csv,
        source_code_col=args.left_source_code_col,
        source_label_col=args.left_source_label_col,
        intermediate_code_col=args.left_intermediate_code_col,
        intermediate_label_col=args.left_intermediate_label_col,
        source_balance_role_col=args.left_source_balance_role_col,
        include_col=args.left_include_col,
        source_role_filter=parse_filter_values(args.source_balance_role_filter),
        include_direct_energy_only=bool(args.include_direct_energy_only),
    )
    right = prepare_right(
        right_raw,
        path=args.right_csv,
        intermediate_code_col=args.right_intermediate_code_col,
        intermediate_label_col=args.right_intermediate_label_col,
        target_code_col=args.right_target_code_col,
        target_label_col=args.right_target_label_col,
        enabled_col=args.right_enabled_col,
    )
    result = compose_common_intermediate_mapping(
        left,
        right,
        target_version=args.target_version,
        source_name=args.source_name,
        official_source=args.official_source,
    )
    if result.coverage["diagnostic_rows"] and not args.allow_diagnostics:
        if not args.check_only:
            write_outputs(result, out_csv=args.out_csv, edge_out_csv=args.edge_out_csv, diagnostics_csv=args.diagnostics_csv, manifest_json=args.manifest_json)
        print(json.dumps({"coverage": result.coverage, "wrote_outputs": not args.check_only}, indent=2, sort_keys=True))
        raise SystemExit("Common-intermediate composition has diagnostics; inspect diagnostics or pass --allow-diagnostics.")
    if not args.check_only:
        write_outputs(result, out_csv=args.out_csv, edge_out_csv=args.edge_out_csv, diagnostics_csv=args.diagnostics_csv, manifest_json=args.manifest_json)
    print(json.dumps({"coverage": result.coverage, "wrote_outputs": not args.check_only}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
