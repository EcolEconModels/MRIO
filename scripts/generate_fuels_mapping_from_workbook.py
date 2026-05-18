#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ecolecon_mrio.mapping_csv import MAPPING_CSV_COLUMNS, load_weighted_mapping_frame


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().casefold().split())


def _boolish_enabled(value: object) -> bool:
    text = str(value).strip().casefold()
    if text in {"", "1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Unsupported enabled value: {value!r}")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return slug.upper() or "GROUP"


def _ensure_mapping_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in MAPPING_CSV_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    extras = [column for column in out.columns if column not in MAPPING_CSV_COLUMNS]
    return out.loc[:, [*MAPPING_CSV_COLUMNS, *extras]]


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _resolve_cpa_aliases(concordance: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    target_info = (
        concordance[["target_key", "target_code", "target_label"]]
        .drop_duplicates()
        .rename(columns={"target_key": "sector_code", "target_label": "sector_label"})
    )
    aliases: dict[str, str] = {}
    labels: dict[str, str] = {}
    for row in target_info.itertuples(index=False):
        code = str(row.sector_code).strip()
        label = str(row.sector_label).strip() or code
        labels[code] = label
        for alias in {code, str(row.target_code).strip(), label}:
            norm = _normalize_text(alias)
            if norm:
                aliases.setdefault(norm, code)
    return aliases, labels


def _resolve_cpa_code(value: str, aliases: dict[str, str], labels: dict[str, str]) -> str:
    norm = _normalize_text(value)
    if not norm:
        return ""
    if norm in aliases:
        return aliases[norm]

    code_hits = [code for code in labels if _normalize_text(code) and _normalize_text(code) in norm]
    if len(code_hits) == 1:
        return code_hits[0]

    label_hits = [code for code, label in labels.items() if _normalize_text(label) and _normalize_text(label) in norm]
    if len(label_hits) == 1:
        return label_hits[0]
    return ""


def _is_header_like(source_value: str, disagg_value: str, target_value: str) -> bool:
    source_norm = _normalize_text(source_value)
    disagg_norm = _normalize_text(disagg_value)
    target_norm = _normalize_text(target_value)
    return (
        ("eurostat" in source_norm or "source" in source_norm or "cpa" in source_norm)
        and ("exio" in disagg_norm or "disagg" in disagg_norm or "subsector" in disagg_norm)
        and ("reagg" in target_norm or "target" in target_norm or "aggreg" in target_norm)
    )


def _load_workbook_sheet(
    workbook_path: Path,
    *,
    sheet_name: str,
    source_col: int = 0,
    disagg_col: int = 1,
    target_col: int = 2,
) -> pd.DataFrame:
    raw = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)
    max_col = max(source_col, disagg_col, target_col)
    if raw.shape[1] <= max_col:
        raise ValueError(
            f"Worksheet {sheet_name!r} has only {raw.shape[1]} column(s); need at least {max_col + 1}."
        )

    frame = raw.iloc[:, [source_col, disagg_col, target_col]].copy()
    frame.columns = ["sheet_source_sector", "sheet_disaggregated_sector", "sheet_reaggregation_target"]
    frame = frame.fillna("")
    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip()
    frame = frame[(frame != "").any(axis=1)].copy()

    if not frame.empty:
        first = frame.iloc[0]
        if _is_header_like(
            str(first["sheet_source_sector"]),
            str(first["sheet_disaggregated_sector"]),
            str(first["sheet_reaggregation_target"]),
        ):
            frame = frame.iloc[1:].copy()

    frame["sheet_source_sector"] = frame["sheet_source_sector"].replace("", pd.NA).ffill().fillna("")
    frame = frame[
        (frame["sheet_source_sector"] != "")
        & (frame["sheet_disaggregated_sector"] != "")
        & (frame["sheet_reaggregation_target"] != "")
    ].copy()
    frame.reset_index(drop=False, inplace=True)
    frame.rename(columns={"index": "workbook_row_zero_based"}, inplace=True)
    frame["workbook_row_one_based"] = frame["workbook_row_zero_based"] + 1
    return frame


def _build_subsector_aliases(group: pd.DataFrame) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for row in group.itertuples(index=False):
        source_key = str(row.source_key).strip()
        source_label = str(row.source_label).strip() or source_key
        source_code = str(row.source_code).strip()
        for alias in {source_key, source_label, source_code}:
            norm = _normalize_text(alias)
            if norm:
                aliases.setdefault(norm, source_key)
    return aliases


def _build_target_key(
    *,
    source_sector_code: str,
    source_sector_label: str,
    target_label: str,
    member_labels: list[str],
) -> tuple[str, str]:
    target_norm = _normalize_text(target_label)
    member_norms = {_normalize_text(value) for value in member_labels}
    source_label_norm = _normalize_text(source_sector_label)
    source_code_norm = _normalize_text(source_sector_code)

    if len(member_labels) == 1 and target_norm in member_norms:
        member = member_labels[0]
        return f"{source_sector_code}::{member}", f"{source_sector_code}: {member}"

    if target_norm in {source_label_norm, source_code_norm, f"{source_label_norm} remaining", f"remaining {source_label_norm}"}:
        return f"{source_sector_code}_remaining", f"{source_sector_code}: {source_sector_label}_remaining"

    return f"{source_sector_code}_{_slugify(target_label)}", f"{source_sector_code}: {target_label}"


def build_fuels_mappings(
    *,
    workbook_path: Path,
    sheet_name: str,
    concordance_path: Path,
    outdir: Path,
    output_prefix: str = "fuels",
    split_mapping_name: str | None = None,
    identity_mapping_name: str | None = None,
    reaggregation_mapping_name: str | None = None,
) -> dict[str, Path]:
    prefix = _slugify(output_prefix).lower()
    split_mapping_name = split_mapping_name or f"exio_{prefix}_split_from_workbook.csv"
    identity_mapping_name = identity_mapping_name or f"{prefix}_identity_from_workbook.csv"
    reaggregation_mapping_name = reaggregation_mapping_name or f"{prefix}_reaggregation_from_workbook.csv"
    workbook_rows = _load_workbook_sheet(workbook_path, sheet_name=sheet_name)
    concordance = _ensure_mapping_columns(load_weighted_mapping_frame(concordance_path)).copy()
    concordance["enabled_bool"] = concordance["enabled"].map(_boolish_enabled)

    cpa_aliases, cpa_labels = _resolve_cpa_aliases(concordance)
    workbook_rows["resolved_source_sector_code"] = workbook_rows["sheet_source_sector"].map(
        lambda value: _resolve_cpa_code(str(value), cpa_aliases, cpa_labels)
    )
    workbook_rows["resolved_source_sector_label"] = workbook_rows["resolved_source_sector_code"].map(
        lambda code: cpa_labels.get(str(code), "")
    )

    selected_sector_codes = sorted(
        {str(value) for value in workbook_rows["resolved_source_sector_code"] if str(value).strip() != ""}
    )
    selected_concordance = concordance[concordance["target_key"].astype(str).isin(selected_sector_codes)].copy()
    selected_concordance["split_subsector_label"] = selected_concordance["source_label"].astype(str).str.strip()

    split_rows = selected_concordance.copy()
    split_rows["kind"] = "split"
    split_rows["subtarget_key"] = split_rows["split_subsector_label"]
    split_rows["weight"] = ""
    split_rows["share"] = ""
    split_rows["transformation_rule"] = "workbook-selected-concordance-split"
    split_rows["source_name"] = split_mapping_name
    split_rows["notes"] = split_rows.apply(
        lambda row: (
            f"Workbook-driven fuels split row for {row['target_key']} ({row['target_label']}). "
            f"Constituent EXIO sector retained from exio_to_eurostat_cpa.csv: {row['source_key']}."
        ),
        axis=1,
    )
    split_rows = _ensure_mapping_columns(split_rows)

    resolved_sheet_rows: list[dict[str, object]] = []
    unresolved_sheet_rows: list[dict[str, object]] = []
    for source_sector_code, group in workbook_rows.groupby("resolved_source_sector_code", dropna=False, sort=False):
        source_sector_code = str(source_sector_code).strip()
        if not source_sector_code:
            for row in group.itertuples(index=False):
                unresolved_sheet_rows.append(
                    {
                        "workbook_row_one_based": int(row.workbook_row_one_based),
                        "sheet_source_sector": str(row.sheet_source_sector),
                        "sheet_disaggregated_sector": str(row.sheet_disaggregated_sector),
                        "sheet_reaggregation_target": str(row.sheet_reaggregation_target),
                        "status": "unresolved_source_sector",
                    }
                )
            continue

        concordance_group = selected_concordance[selected_concordance["target_key"].astype(str) == source_sector_code].copy()
        subsector_aliases = _build_subsector_aliases(concordance_group)
        for row in group.itertuples(index=False):
            disagg_value = str(row.sheet_disaggregated_sector).strip()
            resolved_subsector = subsector_aliases.get(_normalize_text(disagg_value), "")
            if not resolved_subsector:
                unresolved_sheet_rows.append(
                    {
                        "workbook_row_one_based": int(row.workbook_row_one_based),
                        "sheet_source_sector": str(row.sheet_source_sector),
                        "resolved_source_sector_code": source_sector_code,
                        "sheet_disaggregated_sector": disagg_value,
                        "sheet_reaggregation_target": str(row.sheet_reaggregation_target),
                        "status": "unresolved_disaggregated_sector",
                    }
                )
                continue
            resolved_sheet_rows.append(
                {
                    "workbook_row_one_based": int(row.workbook_row_one_based),
                    "sheet_source_sector": str(row.sheet_source_sector),
                    "resolved_source_sector_code": source_sector_code,
                    "resolved_source_sector_label": cpa_labels.get(source_sector_code, source_sector_code),
                    "sheet_disaggregated_sector": disagg_value,
                    "resolved_subsector_label": resolved_subsector,
                    "from_sector": f"{source_sector_code}::{resolved_subsector}",
                    "sheet_reaggregation_target": str(row.sheet_reaggregation_target).strip(),
                    "status": "resolved",
                }
            )

    resolved_sheet_frame = pd.DataFrame(resolved_sheet_rows)
    unresolved_sheet_frame = pd.DataFrame(unresolved_sheet_rows)

    enabled_split_rows = split_rows[split_rows["enabled"].map(_boolish_enabled)].copy()
    enabled_from_sectors = {
        f"{row.target_key}::{row.subtarget_key}" for row in enabled_split_rows.itertuples(index=False)
    }
    mapped_from_sectors = set(resolved_sheet_frame.get("from_sector", pd.Series(dtype=str)).astype(str))
    uncovered_from_sectors = sorted(enabled_from_sectors - mapped_from_sectors)

    if len(unresolved_sheet_frame) > 0 or len(uncovered_from_sectors) > 0:
        unresolved_examples = unresolved_sheet_frame.head(10).to_dict(orient="records")
        uncovered_examples = uncovered_from_sectors[:10]
        raise ValueError(
            "Workbook-driven fuels mapping validation failed: "
            f"{len(unresolved_sheet_frame)} unresolved workbook row(s), "
            f"{len(uncovered_from_sectors)} uncovered split subsector(s). "
            f"Unresolved examples: {unresolved_examples}. "
            f"Uncovered examples: {uncovered_examples}."
        )

    identity_rows: list[dict[str, object]] = []
    for row in enabled_split_rows.itertuples(index=False):
        from_sector = f"{row.target_key}::{row.subtarget_key}"
        identity_rows.append(
            {
                "kind": "aggregate",
                "source_dataset": "Eurostat_IO",
                "source_system": "pxp",
                "source_version": "CPA_65",
                "source_level": "product",
                "target_dataset": "Eurostat_IO",
                "target_system": "pxp",
                "target_version": "fuels_identity",
                "target_level": "aggregate",
                "source_key": from_sector,
                "source_code": "",
                "source_label": from_sector,
                "target_key": from_sector,
                "target_code": from_sector,
                "target_label": from_sector,
                "subtarget_key": "",
                "weight": "",
                "share": "",
                "enabled": "1",
                "provenance": "workbook-derived",
                "certainty": "identity",
                "transformation_rule": "identity-aggregation",
                "source_name": identity_mapping_name,
                "official_source": "",
                "notes": f"Identity aggregation row for split subsector {from_sector}.",
                "source_balance_role": "",
                "target_sector_role": "",
                "include_in_direct_energy": "",
            }
        )
    identity_frame = _ensure_mapping_columns(pd.DataFrame(identity_rows))

    reaggregation_rows: list[dict[str, object]] = []
    if not resolved_sheet_frame.empty:
        grouped = resolved_sheet_frame.groupby(
            ["resolved_source_sector_code", "sheet_reaggregation_target"], sort=False
        )
        target_key_lookup: dict[tuple[str, str], tuple[str, str]] = {}
        for (source_sector_code, target_label), group in grouped:
            source_sector_code = str(source_sector_code)
            source_sector_label = cpa_labels.get(source_sector_code, source_sector_code)
            member_labels = sorted(group["resolved_subsector_label"].astype(str).unique().tolist())
            target_key_lookup[(source_sector_code, str(target_label))] = _build_target_key(
                source_sector_code=source_sector_code,
                source_sector_label=source_sector_label,
                target_label=str(target_label),
                member_labels=member_labels,
            )
        for row in resolved_sheet_frame.itertuples(index=False):
            target_key, target_label = target_key_lookup[
                (str(row.resolved_source_sector_code), str(row.sheet_reaggregation_target))
            ]
            reaggregation_rows.append(
                {
                    "kind": "aggregate",
                    "source_dataset": "Eurostat_IO",
                    "source_system": "pxp",
                    "source_version": "CPA_65",
                    "source_level": "product",
                    "target_dataset": "Eurostat_IO",
                    "target_system": "pxp",
                    "target_version": prefix,
                    "target_level": "aggregate",
                    "source_key": str(row.from_sector),
                    "source_code": "",
                    "source_label": str(row.from_sector),
                    "target_key": target_key,
                    "target_code": target_key,
                    "target_label": target_label,
                    "subtarget_key": "",
                    "weight": "",
                    "share": "",
                    "enabled": "1",
                    "provenance": "workbook-derived",
                    "certainty": "sheet-reaggregation",
                    "transformation_rule": "workbook-reaggregation",
                    "source_name": reaggregation_mapping_name,
                    "official_source": "",
                    "notes": (
                        f"Workbook-driven fuels reaggregation row from {row.from_sector} to {target_label}."
                    ),
                    "source_balance_role": "",
                    "target_sector_role": "",
                    "include_in_direct_energy": "",
                }
            )
    reaggregation_frame = _ensure_mapping_columns(
        pd.DataFrame(reaggregation_rows).drop_duplicates(subset=["source_key", "target_key"], keep="first")
    )

    outdir.mkdir(parents=True, exist_ok=True)
    split_path = outdir / split_mapping_name
    identity_path = outdir / identity_mapping_name
    reaggregation_path = outdir / reaggregation_mapping_name

    _write_csv(split_path, split_rows)
    _write_csv(identity_path, identity_frame)
    _write_csv(reaggregation_path, reaggregation_frame)

    return {
        "split_mapping": split_path,
        "identity_mapping": identity_path,
        "reaggregation_mapping": reaggregation_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate workbook-driven fuels split and reaggregation mapping CSVs."
    )
    parser.add_argument("--workbook", required=True, help="Path to the saved workbook copy.")
    parser.add_argument(
        "--sheet-name",
        default="Step2_Reaggregation",
        help="Worksheet name containing source/disaggregation/reaggregation columns.",
    )
    parser.add_argument(
        "--concordance-csv",
        default="scripts/mappings/exio_to_eurostat_cpa.csv",
        help="Canonical EXIO-to-Eurostat concordance CSV.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Folder where the generated mapping CSVs will be written.",
    )
    parser.add_argument(
        "--output-prefix",
        default="fuels",
        help="Prefix used for generated mapping filenames and target_version metadata.",
    )
    args = parser.parse_args(argv)

    outputs = build_fuels_mappings(
        workbook_path=Path(args.workbook),
        sheet_name=str(args.sheet_name),
        concordance_path=Path(args.concordance_csv),
        outdir=Path(args.outdir),
        output_prefix=str(args.output_prefix),
    )

    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())