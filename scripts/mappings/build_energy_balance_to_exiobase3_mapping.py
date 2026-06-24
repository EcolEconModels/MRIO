#!/usr/bin/env python3
"""Generate Eurostat Energy Balance row to EXIOBASE3 product concordance.

The generated concordance is independent of the repo-authored manual
Energy-Balance-to-CPA mapping.  It consumes only:

* a small definition-level rule table that maps Eurostat Energy Balance rows to
  NACE Rev. 2 selectors or explicit non-production diagnostics; and
* the local EXIOBASE/XIO concordance archive.

The EXIO target links are generated through this ZIP chain:

    Energy Balance row -> NACE selector -> EXIOBASE2 product -> EXIOBASE3 product

Rows are relational and unweighted.  They are suitable for audit and for later
runtime weighting; they are not empirical allocation shares.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CONCORDANCES_ZIP: Final[Path] = REPO_ROOT / "data" / "exiobase_concordance_sources" / "exiobase_concordances.zip"
DEFAULT_RULES_CSV: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "energy_balance_sector_rules_for_exiobase3.csv"
DEFAULT_OUT_DIR: Final[Path] = REPO_ROOT / "data" / "exiobase_concordance_sources" / "derived" / "eurostat_energy_balance_to_exiobase3"

NACE_EXIO20_SUFFIX: Final[str] = "NACE2_EXIOBASE20p_list.csv"
EXIO20_EXIO30_SUFFIX: Final[str] = "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
SOURCE_CHAIN: Final[str] = "EB definition rule -> NACE Rev.2 selector -> EXIOBASE2 product -> EXIOBASE3 product"

RULE_COLUMNS: Final[tuple[str, ...]] = (
    "order",
    "enabled",
    "source_dataset",
    "source_system",
    "source_version",
    "source_level",
    "source_code",
    "source_label",
    "source_balance_role",
    "mapping_basis",
    "selector_codes",
    "selector_label",
    "relation_type",
    "allocation_rule",
    "certainty",
    "official_source",
    "include_in_direct_energy",
    "notes",
)

LONG_COLUMNS: Final[tuple[str, ...]] = (
    "order",
    "source_dataset",
    "source_system",
    "source_version",
    "source_level",
    "source_code",
    "source_label",
    "source_balance_role",
    "mapping_basis",
    "selector_codes",
    "selector_label",
    "relation_type",
    "allocation_rule",
    "certainty",
    "official_source",
    "include_in_direct_energy",
    "nace2_code",
    "nace2_division",
    "source_exio_code",
    "source_exio20_product_name",
    "bridge_key_kind",
    "exio20_product_code",
    "exio20_product_name",
    "exio3_product_code",
    "exio3_product_name",
    "source_chain",
    "notes",
)

DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "source_code",
    "source_label",
    "mapping_basis",
    "selector_codes",
    "diagnostic_kind",
    "diagnostic_message",
    "certainty",
    "notes",
)

SUPPORTED_MAPPING_BASES: Final[set[str]] = {"nace2_division", "nace2_exact_or_prefix", "no_exio_production_sector"}


@dataclass(frozen=True)
class BuildResult:
    """Generated mapping tables and coverage metadata."""

    long_links: pd.DataFrame
    collapsed: pd.DataFrame
    diagnostics: pd.DataFrame
    coverage: dict[str, object]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest for bytes read from inside the ZIP archive."""

    return hashlib.sha256(data).hexdigest()


def clean_text(value: object) -> str:
    """Normalize text fields used as stable CSV keys."""

    # Avoid pandas.isna here: for object-typed values it may return an array-like
    # object, which is both a runtime ambiguity and a Pyright/Pylance boundary.
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return " ".join(str(value).strip().split())


def clean_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a stripped string Series from a pandas column."""

    series = cast(pd.Series, frame[column])
    return cast(pd.Series, series.map(clean_text))


def normalize_nace_code(value: object) -> str:
    """Normalize a NACE code while preserving decimals for prefix selection."""

    text = clean_text(value).replace(",", ".")
    if text == "":
        return ""
    match = re.match(r"^(\d{1,2})(?:\.(\d+))?$", text)
    if not match:
        return text
    division = f"{int(match.group(1)):02d}"
    suffix = match.group(2)
    return division if suffix is None else f"{division}.{suffix}"


def nace2_to_division(nace2_code: object) -> str:
    """Infer a two-digit NACE Rev. 2 division from a detailed NACE code."""

    text = normalize_nace_code(nace2_code)
    match = re.match(r"^(\d{2})(?:\.|$)", text)
    if not match:
        raise ValueError(f"Cannot infer NACE division from code {text!r}")
    return match.group(1)


def split_selector_tokens(selector_codes: object) -> list[str]:
    """Split semicolon-separated selector codes and remove blank tokens."""

    text = clean_text(selector_codes)
    if text == "":
        return []
    return [token.strip() for token in text.split(";") if token.strip()]


def expand_division_selectors(selector_codes: object) -> set[str]:
    """Expand selectors such as ``05-09;24`` into two-digit NACE divisions."""

    divisions: set[str] = set()
    for token in split_selector_tokens(selector_codes):
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"Bad descending NACE selector range: {token!r}")
            divisions.update(f"{value:02d}" for value in range(start, end + 1))
        else:
            divisions.add(f"{int(token):02d}")
    return divisions


def expand_prefix_selectors(selector_codes: object) -> tuple[str, ...]:
    """Normalize exact/prefix NACE selectors such as ``49.1;49.2``."""

    return tuple(normalize_nace_code(token) for token in split_selector_tokens(selector_codes))


def find_member_by_suffix(zf: zipfile.ZipFile, suffix: str) -> str:
    """Find one ZIP member whose normalized path ends with ``suffix``."""

    matches = [name for name in zf.namelist() if name.replace("\\\\", "/").endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"No ZIP member ending with {suffix!r}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple ZIP members ending with {suffix!r}: {matches}")
    return matches[0]


def read_nace_to_exio20(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    """Read the ZIP NACE Rev. 2 to EXIOBASE2 product correspondence."""

    with zf.open(member) as handle:
        frame = pd.read_csv(handle, dtype=str, encoding="utf-8-sig").fillna("")
    required = {"NACE2", "EXIOBASE name", "EXIOBASE code"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{member} missing required columns: {missing}")
    out = frame.rename(
        columns={
            "NACE2": "nace2_code",
            "EXIOBASE name": "source_exio20_product_name",
            "EXIOBASE code": "source_exio_code",
        }
    ).copy()
    out["nace2_code"] = clean_series(out, "nace2_code").map(normalize_nace_code)
    out["nace2_division"] = out["nace2_code"].map(nace2_to_division)
    out["source_exio_code"] = clean_series(out, "source_exio_code")
    out["source_exio20_product_name"] = clean_series(out, "source_exio20_product_name")
    return out.loc[:, ["nace2_code", "nace2_division", "source_exio_code", "source_exio20_product_name"]].drop_duplicates()


def read_exio20_to_exio30_bridge(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    """Read the ZIP EXIOBASE2 product to EXIOBASE3 product bridge."""

    data = zf.read(member)
    frame = pd.read_excel(io.BytesIO(data), dtype=str).fillna("")
    required = {"EXIOBASE 2 Name", "EXIOBASE 2 code", "EXIOBASE 3 name", "EXIOBASE 3 code"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{member} missing required columns: {missing}")
    renamed = frame.rename(
        columns={
            "EXIOBASE 2 Name": "exio20_product_name",
            "EXIOBASE 2 code": "exio20_product_code",
            "EXIOBASE 3 name": "exio3_product_name",
            "EXIOBASE 3 code": "exio3_product_code",
        }
    ).copy()
    base = cast(
        pd.DataFrame,
        renamed.loc[:, ["exio20_product_name", "exio20_product_code", "exio3_product_name", "exio3_product_code"]].copy(),
    )
    for column in base.columns:
        base[str(column)] = clean_series(base, str(column))

    records: list[dict[str, str]] = []
    for row in base.itertuples(index=False):
        rec = {
            "exio20_product_name": clean_text(getattr(row, "exio20_product_name")),
            "exio20_product_code": clean_text(getattr(row, "exio20_product_code")),
            "exio3_product_name": clean_text(getattr(row, "exio3_product_name")),
            "exio3_product_code": clean_text(getattr(row, "exio3_product_code")),
        }
        if rec["exio20_product_code"]:
            records.append({"source_exio_code": rec["exio20_product_code"], "bridge_key_kind": "exio20_code", **rec})
        if rec["exio3_product_code"] and rec["exio3_product_code"] != rec["exio20_product_code"]:
            records.append({"source_exio_code": rec["exio3_product_code"], "bridge_key_kind": "exio3_code_alias", **rec})
    return pd.DataFrame.from_records(records).drop_duplicates()


def read_rules(path: Path) -> pd.DataFrame:
    """Read and validate the definition-level Energy Balance selector rules."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(set(RULE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Rules CSV {path} missing required columns: {missing}")
    frame = frame.loc[:, list(RULE_COLUMNS)].copy()
    for column in frame.columns:
        frame[column] = clean_series(frame, str(column))
    frame = frame.loc[frame["enabled"].str.lower().isin(("1", "true", "yes", "y"))].copy()
    unknown_bases = sorted(set(frame["mapping_basis"]) - SUPPORTED_MAPPING_BASES)
    if unknown_bases:
        raise ValueError(f"Rules CSV has unsupported mapping_basis values: {unknown_bases}")
    duplicate_codes = sorted(frame.loc[frame["source_code"].duplicated(), "source_code"].unique().tolist())
    if duplicate_codes:
        raise ValueError(f"Rules CSV has duplicate enabled source_code rows: {duplicate_codes}")
    return frame


def select_nace_rows(nace: pd.DataFrame, rule: pd.Series) -> pd.DataFrame:
    """Return NACE->EXIO rows selected by one Energy Balance definition rule."""

    basis = clean_text(rule["mapping_basis"])
    selectors = clean_text(rule["selector_codes"])
    if basis == "no_exio_production_sector":
        return nace.iloc[0:0].copy()
    if basis == "nace2_division":
        divisions = expand_division_selectors(selectors)
        return cast(pd.DataFrame, nace.loc[nace["nace2_division"].isin(sorted(divisions))].copy())
    if basis == "nace2_exact_or_prefix":
        prefixes = expand_prefix_selectors(selectors)
        if not prefixes:
            return nace.iloc[0:0].copy()
        codes = nace["nace2_code"].astype(str)
        mask = pd.Series(False, index=nace.index)
        for prefix in prefixes:
            mask = mask | codes.eq(prefix) | codes.str.startswith(prefix)
        return cast(pd.DataFrame, nace.loc[mask].copy())
    raise ValueError(f"Unsupported mapping_basis: {basis!r}")


def build_mapping(concordances_zip: Path, rules_csv: Path) -> BuildResult:
    """Build long and collapsed EB-to-EXIO3 product concordance tables."""

    if not concordances_zip.exists():
        raise FileNotFoundError(f"Missing concordance archive: {concordances_zip}")
    if not rules_csv.exists():
        raise FileNotFoundError(f"Missing Energy Balance rule CSV: {rules_csv}")

    with zipfile.ZipFile(concordances_zip) as zf:
        nace_member = find_member_by_suffix(zf, NACE_EXIO20_SUFFIX)
        bridge_member = find_member_by_suffix(zf, EXIO20_EXIO30_SUFFIX)
        nace_bytes = zf.read(nace_member)
        bridge_bytes = zf.read(bridge_member)
        nace = read_nace_to_exio20(zf, nace_member)
        bridge = read_exio20_to_exio30_bridge(zf, bridge_member)

    rules = read_rules(rules_csv)
    long_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []

    for _, rule in rules.sort_values("order", key=lambda series: series.astype(int)).iterrows():
        selected = select_nace_rows(nace, rule)
        if selected.empty:
            kind = "expected_no_exio_target" if rule["mapping_basis"] == "no_exio_production_sector" else "no_nace_selector_match"
            message = (
                "Rule intentionally has no EXIO production-sector target."
                if kind == "expected_no_exio_target"
                else "No NACE rows matched the rule selector."
            )
            diagnostics.append(
                {
                    "source_code": rule["source_code"],
                    "source_label": rule["source_label"],
                    "mapping_basis": rule["mapping_basis"],
                    "selector_codes": rule["selector_codes"],
                    "diagnostic_kind": kind,
                    "diagnostic_message": message,
                    "certainty": rule["certainty"],
                    "notes": rule["notes"],
                }
            )
            continue

        merged = selected.merge(bridge, on="source_exio_code", how="left", validate="many_to_many")
        unmatched = cast(pd.DataFrame, merged.loc[merged["exio3_product_code"].isna() | merged["exio3_product_code"].astype(str).eq("")].copy())
        if not unmatched.empty:
            diagnostics.append(
                {
                    "source_code": rule["source_code"],
                    "source_label": rule["source_label"],
                    "mapping_basis": rule["mapping_basis"],
                    "selector_codes": rule["selector_codes"],
                    "diagnostic_kind": "unmatched_exio20_bridge",
                    "diagnostic_message": ";".join(sorted(set(unmatched["source_exio_code"].astype(str).tolist()))),
                    "certainty": rule["certainty"],
                    "notes": rule["notes"],
                }
            )
        matched = cast(pd.DataFrame, merged.loc[merged["exio3_product_code"].notna() & merged["exio3_product_code"].astype(str).ne("")].copy())
        if matched.empty:
            continue
        for column in RULE_COLUMNS:
            if column not in {"enabled"}:
                matched[column] = rule[column]
        matched["source_chain"] = SOURCE_CHAIN
        long_frames.append(matched.loc[:, list(LONG_COLUMNS)].copy())

    long_links = pd.concat(long_frames, ignore_index=True).drop_duplicates() if long_frames else pd.DataFrame(columns=list(LONG_COLUMNS))
    diagnostics_df = pd.DataFrame.from_records(diagnostics, columns=list(DIAGNOSTIC_COLUMNS)).fillna("")

    if long_links.empty:
        collapsed = pd.DataFrame(
            columns=[
                "source_code",
                "source_label",
                "source_balance_role",
                "mapping_basis",
                "selector_codes",
                "exio3_product_code",
                "exio3_product_name",
                "n_source_nace2_codes",
                "source_nace2_codes",
                "source_nace2_divisions",
                "source_exio_codes",
                "bridge_key_kinds",
                "n_exio3_targets_for_eb_row",
                "is_multi_target",
                "allocation_rule",
                "certainty",
                "relation_type",
                "weight_status",
                "include_in_direct_energy",
                "notes",
            ]
        )
    else:
        grouped = long_links.groupby(
            [
                "source_code",
                "source_label",
                "source_balance_role",
                "mapping_basis",
                "selector_codes",
                "exio3_product_code",
                "exio3_product_name",
                "allocation_rule",
                "certainty",
                "relation_type",
                "include_in_direct_energy",
                "notes",
            ],
            dropna=False,
            sort=True,
        )
        collapsed = grouped.agg(
            n_source_nace2_codes=("nace2_code", "nunique"),
            source_nace2_codes=("nace2_code", lambda values: ";".join(sorted(set(map(str, values))))),
            source_nace2_divisions=("nace2_division", lambda values: ";".join(sorted(set(map(str, values))))),
            source_exio_codes=("source_exio_code", lambda values: ";".join(sorted(set(map(str, values))))),
            bridge_key_kinds=("bridge_key_kind", lambda values: ";".join(sorted(set(map(str, values))))),
        ).reset_index()
        target_counts = collapsed.groupby("source_code")["exio3_product_code"].transform("nunique")
        collapsed["n_exio3_targets_for_eb_row"] = target_counts.astype(int)
        collapsed["is_multi_target"] = collapsed["n_exio3_targets_for_eb_row"] > 1
        collapsed["weight_status"] = "unweighted_relational_not_empirical"
        ordered_cols = [
            "source_code",
            "source_label",
            "source_balance_role",
            "mapping_basis",
            "selector_codes",
            "exio3_product_code",
            "exio3_product_name",
            "n_source_nace2_codes",
            "source_nace2_codes",
            "source_nace2_divisions",
            "source_exio_codes",
            "bridge_key_kinds",
            "n_exio3_targets_for_eb_row",
            "is_multi_target",
            "allocation_rule",
            "certainty",
            "relation_type",
            "weight_status",
            "include_in_direct_energy",
            "notes",
        ]
        collapsed = collapsed.loc[:, ordered_cols].sort_values(["source_code", "exio3_product_code"]).reset_index(drop=True)

    coverage = {
        "source_archive_sha256": sha256_file(concordances_zip),
        "rules_csv_sha256": sha256_file(rules_csv),
        "source_files": [
            {
                "path_in_zip": nace_member,
                "sha256": sha256_bytes(nace_bytes),
                "role": "NACE Rev.2 class to EXIOBASE2 product links",
            },
            {
                "path_in_zip": bridge_member,
                "sha256": sha256_bytes(bridge_bytes),
                "role": "EXIOBASE2 product to EXIOBASE3 product bridge",
            },
        ],
        "enabled_rule_rows": int(len(rules)),
        "mapped_eb_rows": int(collapsed["source_code"].nunique()) if not collapsed.empty else 0,
        "diagnostic_rows": int(len(diagnostics_df)),
        "long_link_rows": int(len(long_links)),
        "collapsed_rows": int(len(collapsed)),
        "distinct_exio3_products": int(collapsed["exio3_product_code"].nunique()) if not collapsed.empty else 0,
        "multi_target_eb_rows": int(collapsed.loc[collapsed["is_multi_target"], "source_code"].nunique()) if not collapsed.empty else 0,
    }
    return BuildResult(long_links=long_links, collapsed=collapsed, diagnostics=diagnostics_df, coverage=coverage)


def write_readme(out_dir: Path) -> None:
    """Write a compact README next to generated concordance outputs."""

    text = """# Eurostat Energy Balances to EXIOBASE3 product concordance

This directory is generated by `scripts/mappings/build_energy_balance_to_exiobase3_mapping.py`.

The generator does not consume the repo-authored manual Energy-Balance-to-CPA mapping. It uses:

1. `scripts/mappings/energy_balance_sector_rules_for_exiobase3.csv`, a compact rule file mapping Energy Balance rows to NACE selectors or explicit non-production diagnostics; and
2. the local EXIOBASE/XIO concordance archive at `data/exiobase_concordance_sources/exiobase_concordances.zip`.

Generated links are relational and unweighted. They are not empirical allocation shares. Use the diagnostics file to review households, residual sectors, and any unmatched ZIP bridge cases before numerical use.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_outputs(result: BuildResult, *, concordances_zip: Path, rules_csv: Path, out_dir: Path) -> dict[str, object]:
    """Write generated CSV outputs and a source manifest."""

    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "eb_to_exio3_products_long.csv"
    collapsed_path = out_dir / "eb_to_exio3_products_collapsed.csv"
    diagnostics_path = out_dir / "eb_unmatched_or_ambiguous.csv"
    manifest_path = out_dir / "manifest.json"

    result.long_links.to_csv(long_path, index=False)
    result.collapsed.to_csv(collapsed_path, index=False)
    result.diagnostics.to_csv(diagnostics_path, index=False)

    manifest: dict[str, object] = {
        "mapping_name": "eurostat_energy_balance_to_exiobase3_products",
        "created_utc": datetime.now(UTC).isoformat(),
        "created_by_script": "scripts/mappings/build_energy_balance_to_exiobase3_mapping.py",
        "source_archive": {
            "path": str(concordances_zip),
            "sha256": result.coverage["source_archive_sha256"],
        },
        "rules_csv": {
            "path": str(rules_csv),
            "sha256": result.coverage["rules_csv_sha256"],
        },
        "source_files": result.coverage["source_files"],
        "method_notes": [
            "Generated independently from the repo-authored manual Energy-Balance-to-CPA mapping.",
            "Energy Balance source rows are mapped only to definition-level NACE selectors or explicit diagnostics.",
            "EXIO targets are generated via NACE Rev.2 -> EXIOBASE2 product -> EXIOBASE3 product ZIP concordance links.",
            "Outputs are unweighted relational links, not empirical allocation shares.",
        ],
        "coverage": {key: value for key, value in result.coverage.items() if key != "source_files"},
        "output_files": {
            "long_links": long_path.name,
            "collapsed": collapsed_path.name,
            "diagnostics": diagnostics_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_dir)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the generator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concordances-zip", type=Path, default=DEFAULT_CONCORDANCES_ZIP)
    parser.add_argument("--rules-csv", "--rules", dest="rules_csv", type=Path, default=DEFAULT_RULES_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check-only", action="store_true", help="Build in memory and print coverage without writing generated outputs.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the mapping generator."""

    args = parse_args(argv)
    result = build_mapping(args.concordances_zip, args.rules_csv)
    manifest = None
    if not args.check_only:
        manifest = write_outputs(result, concordances_zip=args.concordances_zip, rules_csv=args.rules_csv, out_dir=args.out_dir)
    print(json.dumps({"coverage": result.coverage, "out_dir": str(args.out_dir), "wrote_outputs": not args.check_only}, indent=2, sort_keys=True))
    if manifest is not None and not result.collapsed.empty:
        print(f"Wrote generated concordance manifest for {manifest['mapping_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
