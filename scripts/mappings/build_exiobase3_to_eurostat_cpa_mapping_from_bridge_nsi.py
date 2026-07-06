#!/usr/bin/env python3
"""Build EXIOBASE3 -> Eurostat CPA mappings from official concordances.

This replaces the earlier Bridge_NSI_AT label-derived fallback.  The generator
uses only source concordance files under ``data/exiobase_concordance_sources``
and ``data/eurostat_concordance_sources``:

* EXIOBASE20p -> ESA59p aggregation, with old CPA/NACE-style division codes;
* EXIOBASE2 product -> EXIOBASE3 product bridge; and
* RAMON/Eurostat CPA 2002 -> CPA 2008 -> CPA 2.1 correspondence tables.

The final CPA64/CPA65 targets are generated from two-digit CPA 2.1 divisions and
the committed curated target-division bridge CSVs.  CPA65 splits division 68 into
the repo's two real-estate targets through that bridge; dictionary descriptions are
not parsed for machine topology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_EXIOBASE_SOURCE_ROOT: Final[Path] = REPO_ROOT / "data" / "exiobase_concordance_sources"
DEFAULT_EUROSTAT_SOURCE_ROOT: Final[Path] = REPO_ROOT / "data" / "eurostat_concordance_sources"
DEFAULT_CONCORDANCES_ZIP: Final[Path] = DEFAULT_EXIOBASE_SOURCE_ROOT / "exiobase_concordances.zip"
DEFAULT_CPA2002_CPA2008: Final[Path] = DEFAULT_EUROSTAT_SOURCE_ROOT / "CPA 2002 - CPA 2008 (CSV format).csv"
DEFAULT_CPA2008_CPA21: Final[Path] = DEFAULT_EUROSTAT_SOURCE_ROOT / "CPA 2008 - CPA 2.1.csv"
DEFAULT_OUT_DIR: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "generated"
DEFAULT_CPA64_DICTIONARY: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "eurostat_cpa64_products_dictionary.csv"
DEFAULT_CPA65_DICTIONARY: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "eurostat_cpa65_products_dictionary.csv"
DEFAULT_CPA64_DIVISION_MAP: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "eurostat_cpa64_to_cpa21_divisions.csv"
DEFAULT_CPA65_DIVISION_MAP: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "eurostat_cpa65_to_cpa21_divisions.csv"
DEFAULT_FIRST_STEP_CONCILIATION: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "exiobase2_old_cpa2002_first_step_conciliation.csv"
DEFAULT_PRODUCT_ANCHOR_CONCILIATION: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "exiobase2_cpa2002_product_anchor_conciliation.csv"

ESA59_SUFFIX: Final[str] = "EXIOBASE20p_ESA59p.txt"
EXIO20_EXIO30_SUFFIX: Final[str] = "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
CPA2002_EXIOBASE_SUFFIX: Final[str] = "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
ESA59_MEMBER: Final[str] = "Concordances/Different aggregations/EXIOBASE20p_ESA59p.txt"
EXIO20_EXIO30_MEMBER: Final[str] = "Concordances/EXIOBASE/EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
CPA2002_EXIOBASE_MEMBER: Final[str] = "Concordances/Archive/International classifications/CPA2002_6digit_EXIOBASE_FINAL.xlsx"

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


@dataclass(frozen=True)
class BuildResult:
    """Generated mapping tables and coverage metadata."""

    mapping: pd.DataFrame
    edges: pd.DataFrame
    diagnostics: pd.DataFrame
    coverage: dict[str, object]


@dataclass(frozen=True)
class CorrespondenceLevelAudit:
    """Original-level source/target universes derived from one correspondence file."""

    name: str
    edges: pd.DataFrame
    source_level_counts: dict[str, int]
    target_level_counts: dict[str, int]
    edge_kind_counts: dict[str, int]
    mixed_level_edges: pd.DataFrame


@dataclass(frozen=True)
class ProductPathResult:
    """Level-aware product-composition paths and diagnostics."""

    paths: pd.DataFrame
    diagnostics: pd.DataFrame


def clean_text(value: object) -> str:
    """Normalize text for stable keys."""

    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return " ".join(str(value).strip().split())


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a source file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest for an in-memory source payload."""

    return hashlib.sha256(data).hexdigest()


def find_zip_member_by_suffix(concordances_zip: Path, member_suffix: str) -> str:
    """Find exactly one member in the EXIOBASE concordance zip by path suffix."""

    normalized_suffix = member_suffix.replace("\\", "/")
    if not concordances_zip.exists():
        raise FileNotFoundError(f"Missing EXIOBASE concordance zip: {concordances_zip}")
    with zipfile.ZipFile(concordances_zip) as zf:
        matches = [
            name
            for name in zf.namelist()
            if name.replace("\\", "/").endswith(normalized_suffix)
        ]
    unique = sorted(set(matches))
    if not unique:
        raise FileNotFoundError(f"No zip member ending with {member_suffix!r} in {concordances_zip}")
    if len(unique) > 1:
        rendered = "\n  ".join(unique)
        raise RuntimeError(f"Multiple zip members matched {member_suffix!r}:\n  {rendered}")
    return unique[0]


def read_zip_member_bytes(concordances_zip: Path, member_suffix: str) -> tuple[bytes, str]:
    """Read one source member from the EXIOBASE concordance zip."""

    member = find_zip_member_by_suffix(concordances_zip, member_suffix)
    with zipfile.ZipFile(concordances_zip) as zf:
        return zf.read(member), member


def canonical_code(value: object) -> str:
    """Return a compact classification code key, preserving only letters/digits."""

    text = clean_text(value).upper()
    text = text.replace("CPA", "").replace("NACE", "")
    return re.sub(r"[^A-Z0-9]+", "", text)


def cpa_division_from_code(value: object) -> str:
    """Return the two-digit CPA division from a CPA code-like value."""

    code = canonical_code(value)
    match = re.search(r"(\d{2})", code)
    if not match:
        return ""
    return match.group(1)


def classify_correspondence_code(value: object) -> dict[str, str]:
    """Classify a CPA correspondence endpoint without collapsing its level.

    RAMON correspondence files contain mixed hierarchy levels, including section
    letters (``A``), divisions (``01``), detailed product codes (``01.11.11``),
    and occasional broad-to-detailed rows such as ``C -> 09.90.19``.  This helper
    preserves that source evidence for Phase 2 composition instead of forcing all
    rows into two-digit divisions.
    """

    raw = clean_text(value).strip('"')
    code = canonical_code(raw)
    if not raw or not code:
        return {"raw": raw, "code": code, "level": "blank", "division": "", "display_code": raw}
    if code in {"SOURCE", "TARGET", "FROM", "TO"}:
        return {"raw": raw, "code": code, "level": "header", "division": "", "display_code": raw}
    if re.fullmatch(r"[A-Z]", code):
        return {"raw": raw, "code": code, "level": "section", "division": "", "display_code": raw}
    if re.fullmatch(r"\d+", code):
        level_by_length = {
            2: "division",
            3: "group",
            4: "class",
            5: "category",
        }
        level = level_by_length.get(len(code), "subcategory" if len(code) >= 6 else "numeric_unknown")
        division = code[:2] if len(code) >= 2 else ""
        return {"raw": raw, "code": code, "level": level, "division": division, "display_code": raw or code}
    division = cpa_division_from_code(code)
    if division:
        return {"raw": raw, "code": code, "level": "aggregate", "division": division, "display_code": raw or code}
    return {"raw": raw, "code": code, "level": "non_code_label", "division": "", "display_code": raw}


def correspondence_edge_kind(source_level: str, target_level: str) -> str:
    """Return a stable level-aware edge-kind label for diagnostics."""

    if source_level in {"blank", "header", "non_code_label"} or target_level in {"blank", "header", "non_code_label"}:
        return "non_mapping_row"
    return f"{source_level}_to_{target_level}"


def is_prefix_related(left: str, right: str) -> bool:
    """Return True if two compact classification keys match by hierarchy prefix."""

    return bool(left and right and (left == right or left.startswith(right) or right.startswith(left)))


def read_esa59_mapping(path: Path) -> pd.DataFrame:
    """Read EXIOBASE2 product -> ESA59 old-product-group mapping.

    The upstream EXIOBASE ``EXIOBASE20p_ESA59p.txt`` source is a tab-separated
    0/1 matrix: rows are EXIOBASE2 product names, columns are ESA59 product
    groups, and active cells indicate membership.  The parser converts that
    matrix to the repository's canonical edge-list representation internally.
    No Layer-2 derived concordance file is written.  A legacy line-oriented
    fallback is kept only for small synthetic tests and possible upstream format
    variants.
    """

    def parse_group_header(value: object) -> tuple[str, str]:
        header = clean_text(value).strip('"')
        match = re.search(r"\((\d{2})\)\s*$", header)
        if not match:
            return "", header
        return match.group(1), header[: match.start()].strip()

    def is_active_matrix_cell(value: object) -> bool:
        text = clean_text(value).strip('"')
        if not text:
            return False
        try:
            return float(text) != 0.0
        except ValueError:
            return text.casefold() in {"x", "true", "t", "yes", "y"}

    def parse_matrix_format() -> pd.DataFrame:
        try:
            raw = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False, engine="python").fillna("")
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            return pd.DataFrame()
        if raw.shape[0] < 2 or raw.shape[1] < 2:
            return pd.DataFrame()
        headers: list[tuple[int, str, str]] = []
        for col in range(1, raw.shape[1]):
            code, label = parse_group_header(raw.iat[0, col])
            if code and label:
                headers.append((col, code, label))
        if not headers:
            return pd.DataFrame()
        records: list[dict[str, str]] = []
        for row_idx in range(1, raw.shape[0]):
            product_name = clean_text(raw.iat[row_idx, 0])
            if not product_name:
                continue
            for col, group_code, group_label in headers:
                if col < raw.shape[1] and is_active_matrix_cell(raw.iat[row_idx, col]):
                    records.append(
                        {
                            "exio20_product_code": "",
                            "exio20_product_name": product_name,
                            "esa59_product_code": group_code,
                            "esa59_product_label": group_label,
                        }
                    )
        if not records:
            return pd.DataFrame()
        return pd.DataFrame.from_records(records).drop_duplicates().reset_index(drop=True)

    matrix = parse_matrix_format()
    if not matrix.empty:
        return matrix

    group_code = ""
    group_label = ""
    records: list[dict[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        header_match = re.search(r"\((\d{2})\)\s*$", line)
        product_code_match = re.match(r"^([pi]\d[\w.\-]*)\b\s*(.*)$", line, flags=re.IGNORECASE)
        if header_match and not product_code_match:
            group_code = header_match.group(1)
            group_label = line[: header_match.start()].strip().strip('"')
            continue
        parts = [part.strip() for part in re.split(r"\t+|\s{2,}|,", line, maxsplit=2) if part.strip()]
        exio_code = ""
        exio_label = line.strip('"')
        if product_code_match:
            exio_code = product_code_match.group(1)
            exio_label = clean_text(product_code_match.group(2)) or exio_code
        elif len(parts) >= 2 and re.match(r"^[pi]\d", parts[0], flags=re.IGNORECASE):
            exio_code = parts[0]
            exio_label = parts[1]
        if group_code and (exio_code or exio_label):
            records.append(
                {
                    "exio20_product_code": clean_text(exio_code),
                    "exio20_product_name": clean_text(exio_label),
                    "esa59_product_code": group_code,
                    "esa59_product_label": clean_text(group_label),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"No ESA59 product rows parsed from {path}")
    return frame.drop_duplicates().reset_index(drop=True)


def read_exio20_to_exio30_bridge(path: Path) -> pd.DataFrame:
    """Read EXIOBASE2 product -> EXIOBASE3 product bridge."""

    frame = pd.read_excel(path, dtype=str).fillna("")
    required = {"EXIOBASE 2 Name", "EXIOBASE 2 code", "EXIOBASE 3 name", "EXIOBASE 3 code"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = frame.rename(
        columns={
            "EXIOBASE 2 Name": "exio20_product_name",
            "EXIOBASE 2 code": "exio20_product_code",
            "EXIOBASE 3 name": "exio3_product_name",
            "EXIOBASE 3 code": "exio3_product_code",
        }
    ).copy()
    for column in ["exio20_product_name", "exio20_product_code", "exio3_product_name", "exio3_product_code"]:
        out[column] = out[column].map(clean_text)
    out["exio20_product_name_norm"] = out["exio20_product_name"].map(lambda value: re.sub(r"\W+", "", value.casefold()))
    return out.loc[:, ["exio20_product_name", "exio20_product_name_norm", "exio20_product_code", "exio3_product_name", "exio3_product_code"]].drop_duplicates()


def read_cpa2002_exiobase_mapping(path: Path) -> pd.DataFrame:
    """Read the archived direct CPA2002 6-digit -> EXIOBASE2 product mapping.

    Economics purpose: this workbook is an independent product-classification
    concordance for the first EXIOBASE2-product classification step. It is not an
    allocation-share table. Some rows point to several EXIOBASE2 products with
    semicolon-separated codes, so the parser expands those cells before any
    coverage or consistency check.
    """

    frame = pd.read_excel(path, dtype=str).fillna("")
    required = {"CPA 2002", "CPA 2002.1", "EXIOBASE_2.0"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    # Treat the archived direct CPA2002 workbook as a real mapping table:
    # every CPA row used by the workbook must name at least one EXIOBASE2
    # product, and every EXIOBASE2 cell must have a CPA code.  This catches
    # one-sided rows before the parser expands semicolon-delimited targets.
    assert_raw_edge_table_complete(
        step_name=f"CPA2002 direct EXIOBASE mapping [{path}]",
        frame=frame,
        source_col="CPA 2002.1",
        target_col="EXIOBASE_2.0",
    )

    rows: list[dict[str, str]] = []
    bad_code_rows: list[dict[str, str]] = []
    for _, source_row in frame.iterrows():
        cpa_label = clean_text(source_row.get("CPA 2002", ""))
        cpa_code = clean_text(source_row.get("CPA 2002.1", ""))
        compact_code = clean_text(source_row.get("CPA 2002.2", ""))
        division = cpa_division_from_code(cpa_code)
        if cpa_code and not division:
            bad_code_rows.append({"cpa2002_code": cpa_code, "cpa2002_label": cpa_label, "exio20_cell": clean_text(source_row.get("EXIOBASE_2.0", ""))})
            continue
        if not division:
            continue
        # The source workbook uses semicolon-separated EXIOBASE2 codes for some
        # many-product CPA rows. Split only on explicit source delimiters; do not
        # infer extra product codes from labels.
        for part in clean_text(source_row.get("EXIOBASE_2.0", "")).split(";"):
            exio_code = clean_text(part)
            if not exio_code:
                continue
            rows.append(
                {
                    "exio20_product_code": exio_code,
                    "cpa2002_code": cpa_code,
                    "cpa2002_compact_code": compact_code,
                    "cpa2002_label": cpa_label,
                    "cpa2002_division": division,
                }
            )
    _print_mapping_rows("CPA2002 direct EXIOBASE rows with unparsable CPA code:", bad_code_rows)
    if bad_code_rows:
        raise ValueError(f"CPA2002 direct EXIOBASE mapping {path} contains nonblank CPA codes without a two-digit division.")

    out = pd.DataFrame.from_records(
        rows,
        columns=[
            "exio20_product_code",
            "cpa2002_code",
            "cpa2002_compact_code",
            "cpa2002_label",
            "cpa2002_division",
        ],
    ).drop_duplicates()
    if out.empty:
        raise ValueError(f"No usable CPA2002->EXIOBASE2 rows parsed from {path}")
    return out.reset_index(drop=True)


def _mapping_label(code: str, label: str) -> str:
    """Render a compact code+label cell for diagnostics."""

    return f"{code}: {label}" if label else code


def _print_mapping_rows(title: str, rows: list[dict[str, str]]) -> None:
    """Print coverage rows in a stable human-readable form."""

    print(title)
    if not rows:
        print("  none")
        return
    for row in rows:
        rendered = "; ".join(f"{key}={value}" for key, value in row.items() if value)
        print(f"  - {rendered}")


def _code_set(values: Iterable[object]) -> set[str]:
    """Normalize a sequence of classification codes into a non-empty-code set."""

    return {clean_text(value) for value in values if clean_text(value)}


def assert_mapping_step_coverage(
    *,
    step_name: str,
    required_left_codes: Iterable[str],
    mapped_left_codes: Iterable[str],
    required_right_codes: Iterable[str],
    mapped_right_codes: Iterable[str],
    left_labels: dict[str, str] | None = None,
    right_labels: dict[str, str] | None = None,
) -> None:
    """Print and enforce both-side coverage for a declared mapping universe.

    Economics purpose: a concordance step is only valid over the code universe it
    claims to classify. If a source-side code has no target, output disappears; if
    a required target-side code has no source, the target classification is not
    actually reachable. The universes are passed in from source data or adjacent
    official mappings, never hard-coded into this helper.
    """

    left_label_map = left_labels or {}
    right_label_map = right_labels or {}
    required_left = _code_set(required_left_codes)
    mapped_left = _code_set(mapped_left_codes)
    required_right = _code_set(required_right_codes)
    mapped_right = _code_set(mapped_right_codes)
    missing_left = sorted(required_left - mapped_left)
    missing_right = sorted(required_right - mapped_right)
    print(
        f"Mapping-step coverage [{step_name}]: "
        f"required_left={len(required_left)} mapped_left={len(mapped_left & required_left)} "
        f"required_right={len(required_right)} mapped_right={len(mapped_right & required_right)}"
    )
    _print_mapping_rows(
        f"Mapping-step missing left codes [{step_name}]:",
        [{"code": code, "label": left_label_map.get(code, "")} for code in missing_left],
    )
    _print_mapping_rows(
        f"Mapping-step missing right codes [{step_name}]:",
        [{"code": code, "label": right_label_map.get(code, "")} for code in missing_right],
    )
    if missing_left or missing_right:
        raise ValueError(f"Mapping step {step_name!r} has unmapped required codes; see printed coverage rows above.")


def assert_raw_edge_table_complete(
    *,
    step_name: str,
    frame: pd.DataFrame,
    source_col: str,
    target_col: str,
) -> None:
    """Fail if a raw correspondence table has one-sided rows before filtering.

    Software purpose: the old implementation silently dropped rows where one side
    of a RAMON correspondence was blank. Dropping a row is itself a mapping
    decision, so the generator now reports those rows and stops instead.
    """

    raw_source = frame[source_col].map(canonical_code)
    raw_target = frame[target_col].map(canonical_code)
    missing_source = [
        {"source": clean_text(source), "target": clean_text(target)}
        for source, target in zip(raw_source.tolist(), raw_target.tolist(), strict=False)
        if not clean_text(source) and clean_text(target)
    ]
    missing_target = [
        {"source": clean_text(source), "target": clean_text(target)}
        for source, target in zip(raw_source.tolist(), raw_target.tolist(), strict=False)
        if clean_text(source) and not clean_text(target)
    ]
    print(
        f"Raw mapping-table completeness [{step_name}]: "
        f"rows={len(frame)} source_codes={len(_code_set(raw_source.tolist()))} target_codes={len(_code_set(raw_target.tolist()))}"
    )
    _print_mapping_rows(f"Raw mapping rows with missing source [{step_name}]:", missing_source)
    _print_mapping_rows(f"Raw mapping rows with missing target [{step_name}]:", missing_target)
    if missing_source or missing_target:
        raise ValueError(f"Raw mapping table {step_name!r} contains one-sided correspondence rows.")


def _prefix_related_any(code: str, candidates: Iterable[str]) -> bool:
    """Return True when a code can be consumed by at least one hierarchical peer."""

    return any(is_prefix_related(clean_text(code), clean_text(candidate)) for candidate in candidates)


def assert_active_targets_consumable(
    *,
    boundary_name: str,
    produced_right_codes: Iterable[str],
    next_left_codes: Iterable[str],
    produced_labels: dict[str, str] | None = None,
) -> None:
    """Ensure target codes produced by one active step are accepted downstream.

    This is the cross-step no-loss invariant. It complements full table
    completeness: an official mapping table can be internally complete while still
    producing active targets that the next official table cannot consume.
    """

    labels = produced_labels or {}
    produced = _code_set(produced_right_codes)
    next_left = _code_set(next_left_codes)
    unconsumed = sorted(code for code in produced if not _prefix_related_any(code, next_left))
    print(
        f"Cross-step no-loss [{boundary_name}]: "
        f"produced_right={len(produced)} next_left_available={len(next_left)} unconsumed={len(unconsumed)}"
    )
    _print_mapping_rows(
        f"Cross-step unconsumed produced codes [{boundary_name}]:",
        [{"code": code, "label": labels.get(code, "")} for code in unconsumed],
    )
    if unconsumed:
        raise ValueError(f"Cross-step boundary {boundary_name!r} loses produced mapping codes.")


def _join_esa59_to_exio_bridge(esa: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    """Attach EXIOBASE2/3 codes to ESA59 rows using code first, then name.

    The official ESA59 file is normally a matrix keyed by EXIOBASE2 product name
    only. The EXIOBASE2->3 bridge supplies stable product codes. Keeping this as
    one explicit boundary prevents later mapping steps from relying on implicit row
    position or on labels after a code has become available.
    """

    esa = esa.copy()
    esa["exio20_product_name_norm"] = esa["exio20_product_name"].map(lambda value: re.sub(r"\W+", "", value.casefold()))
    with_code = esa.loc[esa["exio20_product_code"].astype(str).str.strip().ne("")].merge(
        bridge,
        on="exio20_product_code",
        how="left",
        validate="many_to_many",
    )
    without_code = esa.loc[esa["exio20_product_code"].astype(str).str.strip().eq("")].merge(
        bridge,
        on="exio20_product_name_norm",
        how="left",
        validate="many_to_many",
        suffixes=("", "_bridge"),
    )
    return pd.concat([with_code, without_code], ignore_index=True, sort=False).fillna("")


def _direct_cpa2002_summary(direct_cpa2002: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Collapse direct CPA2002 rows to EXIOBASE2-code diagnostics."""

    summary: dict[str, dict[str, object]] = {}
    if direct_cpa2002.empty:
        return summary
    for row in direct_cpa2002.itertuples(index=False):
        exio_code = clean_text(getattr(row, "exio20_product_code"))
        division = clean_text(getattr(row, "cpa2002_division"))
        cpa_code = clean_text(getattr(row, "cpa2002_code"))
        cpa_label = clean_text(getattr(row, "cpa2002_label"))
        if not exio_code or not division:
            continue
        entry = summary.setdefault(exio_code, {"divisions": set(), "labels_by_division": {}})
        cast(set[str], entry["divisions"]).add(division)
        labels_by_division = cast(dict[str, set[str]], entry["labels_by_division"])
        labels_by_division.setdefault(division, set()).add(_mapping_label(cpa_code, cpa_label))
    return summary


CONCILIATION_COLUMNS: Final[set[str]] = {
    "exio20_product_code",
    "exio20_product_label",
    "chosen_old_cpa2002_divisions",
    "chosen_old_cpa2002_labels",
    "conciliation_basis",
    "esa59_old_cpa2002_divisions",
    "direct_old_cpa2002_divisions",
    "notes",
}


def _split_semicolon_codes(value: object) -> list[str]:
    """Split a curated semicolon code cell without inventing classifications."""

    return [clean_text(part) for part in clean_text(value).split(";") if clean_text(part)]




PRODUCT_ANCHOR_COLUMNS: Final[set[str]] = {
    "exio20_product_code",
    "exio20_product_label",
    "esa59_old_cpa2002_division",
    "esa59_old_cpa2002_label",
    "chosen_cpa2002_code",
    "chosen_cpa2002_label",
    "chosen_cpa2002_level",
    "anchor_basis",
    "notes",
}


def read_product_anchor_conciliation_map(path: Path | None) -> pd.DataFrame:
    """Read curated detailed CPA2002 anchors for ESA-only broad products."""

    columns = [
        "exio20_product_code",
        "exio20_product_label",
        "esa59_old_cpa2002_division",
        "esa59_old_cpa2002_label",
        "chosen_cpa2002_code",
        "chosen_cpa2002_label",
        "chosen_cpa2002_level",
        "anchor_basis",
        "notes",
        "chosen_cpa2002_normalized_code",
        "chosen_cpa2002_display_code",
        "chosen_cpa2002_division",
    ]
    if path is None:
        return pd.DataFrame(columns=columns)
    if not path.exists():
        raise FileNotFoundError(f"Missing product-anchor conciliation map: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    missing = sorted(PRODUCT_ANCHOR_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Product-anchor conciliation map {path} missing columns: {missing}")
    rows: list[dict[str, str]] = []
    bad_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in frame.itertuples(index=False):
        exio_code = clean_text(getattr(row, "exio20_product_code"))
        esa_division = clean_text(getattr(row, "esa59_old_cpa2002_division"))
        chosen_raw = clean_text(getattr(row, "chosen_cpa2002_code"))
        basis = clean_text(getattr(row, "anchor_basis"))
        endpoint = classify_correspondence_code(chosen_raw)
        chosen_division = clean_text(endpoint["division"])
        key = (exio_code, esa_division, clean_text(endpoint["code"]))
        if not exio_code or not esa_division or not chosen_raw or not basis:
            bad_rows.append({"exio20_product_code": exio_code, "chosen_cpa2002_code": chosen_raw, "message": "blank product code, ESA division, chosen CPA2002 code, or basis"})
            continue
        if endpoint["level"] in {"blank", "header", "section", "non_code_label"} or not chosen_division:
            bad_rows.append({"exio20_product_code": exio_code, "chosen_cpa2002_code": chosen_raw, "message": "chosen CPA2002 anchor is not a numeric CPA product code"})
            continue
        if chosen_division != esa_division:
            bad_rows.append({"exio20_product_code": exio_code, "chosen_cpa2002_code": chosen_raw, "esa59_old_cpa2002_division": esa_division, "message": "chosen CPA2002 anchor is outside the ESA59 old-CPA division it is supposed to refine"})
            continue
        if key in seen:
            bad_rows.append({"exio20_product_code": exio_code, "chosen_cpa2002_code": chosen_raw, "message": "duplicate product-anchor row"})
            continue
        seen.add(key)
        rows.append(
            {
                "exio20_product_code": exio_code,
                "exio20_product_label": clean_text(getattr(row, "exio20_product_label")),
                "esa59_old_cpa2002_division": esa_division,
                "esa59_old_cpa2002_label": clean_text(getattr(row, "esa59_old_cpa2002_label")),
                "chosen_cpa2002_code": chosen_raw,
                "chosen_cpa2002_label": clean_text(getattr(row, "chosen_cpa2002_label")),
                "chosen_cpa2002_level": clean_text(endpoint["level"]),
                "anchor_basis": basis,
                "notes": clean_text(getattr(row, "notes")),
                "chosen_cpa2002_normalized_code": clean_text(endpoint["code"]),
                "chosen_cpa2002_display_code": clean_text(endpoint["display_code"]),
                "chosen_cpa2002_division": chosen_division,
            }
        )
    _print_mapping_rows("Product-anchor conciliation map invalid rows:", bad_rows)
    if bad_rows:
        raise ValueError(f"Invalid product-anchor conciliation map {path}; see printed rows above.")
    out = pd.DataFrame.from_records(rows, columns=columns).drop_duplicates().reset_index(drop=True)
    print(f"Read product-anchor conciliation map {path}: rows={len(out)}")
    return out


def read_first_step_conciliation_map(path: Path | None) -> pd.DataFrame:
    """Read the curated EXIOBASE2 old-CPA first-step conflict conciliation map.

    Economics purpose: the two official first-step sources sometimes disagree for
    the same EXIOBASE2 product.  The generator must not hard-code which source wins
    for those cases.  This small curated table records the explicit product-level
    decision, the rejected source alternatives, and the economic rationale.
    """

    columns = [
        "exio20_product_code",
        "exio20_product_label",
        "chosen_old_cpa2002_divisions",
        "chosen_old_cpa2002_labels",
        "conciliation_basis",
        "esa59_old_cpa2002_divisions",
        "direct_old_cpa2002_divisions",
        "notes",
    ]
    if path is None:
        return pd.DataFrame(columns=columns)
    if not path.exists():
        raise FileNotFoundError(f"Missing first-step conciliation map: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    missing = sorted(CONCILIATION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"First-step conciliation map {path} missing columns: {missing}")
    rows: list[dict[str, str]] = []
    bad_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in frame.itertuples(index=False):
        code = clean_text(getattr(row, "exio20_product_code"))
        chosen = _split_semicolon_codes(getattr(row, "chosen_old_cpa2002_divisions"))
        basis = clean_text(getattr(row, "conciliation_basis"))
        if not code or not chosen or not basis:
            bad_rows.append({"exio20_product_code": code, "message": "blank product code, chosen divisions, or basis"})
            continue
        if code in seen:
            bad_rows.append({"exio20_product_code": code, "message": "duplicate conciliation row"})
            continue
        seen.add(code)
        rows.append({column: clean_text(getattr(row, column)) for column in columns})
    _print_mapping_rows("First-step conciliation map invalid rows:", bad_rows)
    if bad_rows:
        raise ValueError(f"Invalid first-step conciliation map {path}; see printed rows above.")
    out = pd.DataFrame.from_records(rows, columns=columns).drop_duplicates().reset_index(drop=True)
    print(f"Read first-step conciliation map {path}: rows={len(out)}")
    return out


def _first_step_conciliation_by_code(conciliation: pd.DataFrame | None, *, bridge_codes: set[str]) -> dict[str, dict[str, object]]:
    """Return curated conflict decisions for EXIOBASE2 codes in this bridge."""

    decisions: dict[str, dict[str, object]] = {}
    if conciliation is None or conciliation.empty:
        return decisions
    outside_bridge_rows: list[dict[str, str]] = []
    bad_rows: list[dict[str, str]] = []
    for row in conciliation.itertuples(index=False):
        code = clean_text(getattr(row, "exio20_product_code"))
        if not code:
            continue
        if code not in bridge_codes:
            # The default curated conciliation file is for the full official
            # EXIOBASE2 bridge, while unit tests and ad-hoc calls often use small
            # bridge subsets.  Rows outside the current subset are therefore not
            # stale for this run; print them for auditability and ignore them.
            outside_bridge_rows.append({"exio20_product_code": code, "message": "conciliation product code not present in this EXIOBASE2 bridge subset; ignored"})
            continue
        divisions_ordered = _split_semicolon_codes(getattr(row, "chosen_old_cpa2002_divisions"))
        labels_ordered = _split_semicolon_codes(getattr(row, "chosen_old_cpa2002_labels"))
        if labels_ordered and len(labels_ordered) != len(divisions_ordered):
            bad_rows.append(
                {
                    "exio20_product_code": code,
                    "chosen_old_cpa2002_divisions": ";".join(divisions_ordered),
                    "message": "chosen division/label counts differ",
                }
            )
            continue
        divisions = set(divisions_ordered)
        label_by_division = {
            division: labels_ordered[index] if index < len(labels_ordered) else ""
            for index, division in enumerate(divisions_ordered)
        }
        decisions[code] = {
            "divisions": divisions,
            "divisions_ordered": divisions_ordered,
            "label_by_division": label_by_division,
            "basis": clean_text(getattr(row, "conciliation_basis")),
            "notes": clean_text(getattr(row, "notes")),
            "esa59_old_cpa2002_divisions": set(_split_semicolon_codes(getattr(row, "esa59_old_cpa2002_divisions"))),
            "direct_old_cpa2002_divisions": set(_split_semicolon_codes(getattr(row, "direct_old_cpa2002_divisions"))),
        }
    _print_mapping_rows("First-step conciliation rows outside current EXIOBASE2 bridge subset; ignored:", outside_bridge_rows)
    _print_mapping_rows("First-step conciliation rows with malformed labels:", bad_rows)
    if bad_rows:
        raise ValueError("Invalid first-step conciliation map; see printed rows above.")
    return decisions

def build_closed_first_step_edges(esa: pd.DataFrame, bridge: pd.DataFrame, direct_cpa2002: pd.DataFrame, conciliation: pd.DataFrame | None = None) -> pd.DataFrame:
    """Close and validate the two EXIOBASE2->old-CPA first-step mappings.

    Economics purpose: ESA59 is the aggregation path used by the generator, while
    CPA2002_6digit_EXIOBASE_FINAL.xlsx is a direct product concordance. Product
    and codomain gaps in one source are warnings when the closed union remains
    complete. Product-level disagreements are different: they require an explicit
    curated conciliation row because choosing either source changes the economic
    product boundary passed into the CPA2002->CPA2008 chain.
    """

    joined = _join_esa59_to_exio_bridge(esa, bridge)
    bridge_codes = {clean_text(value) for value in bridge["exio20_product_code"].astype(str) if clean_text(value)}
    bridge_labels = {
        clean_text(getattr(row, "exio20_product_code")): {
            "exio20_product_name": clean_text(getattr(row, "exio20_product_name")),
            "exio3_product_code": clean_text(getattr(row, "exio3_product_code")),
            "exio3_product_name": clean_text(getattr(row, "exio3_product_name")),
        }
        for row in bridge.itertuples(index=False)
        if clean_text(getattr(row, "exio20_product_code"))
    }

    esa_by_code: dict[str, set[tuple[str, str]]] = {}
    esa_without_bridge: list[dict[str, str]] = []
    for row in joined.itertuples(index=False):
        exio_code = clean_text(getattr(row, "exio20_product_code", "")) or clean_text(getattr(row, "exio20_product_code_bridge", ""))
        esa_code = clean_text(getattr(row, "esa59_product_code", ""))
        esa_label = clean_text(getattr(row, "esa59_product_label", ""))
        if not exio_code:
            esa_without_bridge.append({"esa_code": esa_code, "esa_label": esa_label, "exio20_name": clean_text(getattr(row, "exio20_product_name", ""))})
            continue
        if esa_code:
            esa_by_code.setdefault(exio_code, set()).add((esa_code, esa_label))

    direct_by_code = _direct_cpa2002_summary(direct_cpa2002)
    direct_codes = set(direct_by_code)
    esa_codes = set(esa_by_code)
    esa_division_labels: dict[str, str] = {}
    for values in esa_by_code.values():
        for division, label in values:
            esa_division_labels.setdefault(division, label)
    direct_division_labels: dict[str, str] = {}
    for entry in direct_by_code.values():
        labels_by_division = cast(dict[str, set[str]], entry.get("labels_by_division", {}))
        for division, examples in labels_by_division.items():
            direct_division_labels.setdefault(division, " | ".join(sorted(examples)[:6]))
    old_division_universe = set(esa_division_labels) | set(direct_division_labels)
    conciliation_by_code = _first_step_conciliation_by_code(conciliation, bridge_codes=bridge_codes)

    # Product-side gaps are printed and allowed only when the peer official source
    # closes them. This is the intentional two-source completion step.
    missing_direct_rows: list[dict[str, str]] = []
    for code in sorted(bridge_codes - direct_codes):
        info = bridge_labels[code]
        esa_targets = sorted(esa_by_code.get(code, set()))
        missing_direct_rows.append(
            {
                "exio20_code": code,
                "exio20_label": info["exio20_product_name"],
                "filled_from_esa59_old_cpa": " | ".join(_mapping_label(target_code, target_label) for target_code, target_label in esa_targets),
            }
        )

    missing_esa_rows: list[dict[str, str]] = []
    for code in sorted(bridge_codes - esa_codes):
        info = bridge_labels[code]
        direct_entry = direct_by_code.get(code, {"divisions": set(), "labels_by_division": {}})
        labels_by_division = cast(dict[str, set[str]], direct_entry.get("labels_by_division", {}))
        direct_targets = []
        for division in sorted(cast(set[str], direct_entry.get("divisions", set()))):
            examples = " | ".join(sorted(labels_by_division.get(division, set()))[:6])
            direct_targets.append(_mapping_label(division, examples))
        missing_esa_rows.append(
            {
                "exio20_code": code,
                "exio20_label": info["exio20_product_name"],
                "filled_from_direct_cpa2002_old_cpa": " || ".join(direct_targets),
            }
        )

    direct_extra_rows = [
        {"exio20_code": code, "message": "Direct CPA2002 workbook refers to an EXIOBASE2 code absent from the EXIOBASE2->3 bridge."}
        for code in sorted(direct_codes - bridge_codes)
    ]
    unclosed_rows = [
        {"exio20_code": code, "exio20_label": bridge_labels[code]["exio20_product_name"]}
        for code in sorted(bridge_codes - (esa_codes | direct_codes))
    ]

    esa_missing_division_rows = [
        {"old_cpa2002_division": code, "direct_cpa2002_label_examples": direct_division_labels.get(code, "")}
        for code in sorted(old_division_universe - set(esa_division_labels))
    ]
    direct_missing_division_rows = [
        {"old_cpa2002_division": code, "esa59_label": esa_division_labels.get(code, "")}
        for code in sorted(old_division_universe - set(direct_division_labels))
    ]

    mismatch_rows: list[dict[str, str]] = []
    for code in sorted(bridge_codes & esa_codes & direct_codes):
        esa_targets = esa_by_code[code]
        esa_divisions = {target_code for target_code, _ in esa_targets}
        direct_entry = direct_by_code[code]
        direct_divisions = cast(set[str], direct_entry["divisions"])
        if esa_divisions == direct_divisions:
            continue
        labels_by_division = cast(dict[str, set[str]], direct_entry["labels_by_division"])
        direct_rendered = []
        for division in sorted(direct_divisions):
            examples = " | ".join(sorted(labels_by_division.get(division, set()))[:8])
            direct_rendered.append(_mapping_label(division, examples))
        info = bridge_labels[code]
        mismatch_rows.append(
            {
                "exio20_code": code,
                "exio20_label": info["exio20_product_name"],
                "esa59_old_cpa": " | ".join(_mapping_label(target_code, target_label) for target_code, target_label in sorted(esa_targets)),
                "direct_cpa2002_old_cpa": " || ".join(direct_rendered),
            }
        )

    closed_codes = bridge_codes & (esa_codes | direct_codes)
    print(
        "First-step EXIOBASE2 coverage: "
        f"bridge={len(bridge_codes)} ESA59_products={len(esa_codes)} direct_CPA2002_products={len(direct_codes)} "
        f"closed_products={len(closed_codes)} ESA59_old_divisions={len(esa_division_labels)} "
        f"direct_old_divisions={len(direct_division_labels)} old_division_union={len(old_division_universe)}"
    )
    _print_mapping_rows("First-step direct CPA2002 missing EXIOBASE2 products; filled from ESA59 where possible:", missing_direct_rows)
    _print_mapping_rows("First-step ESA59 missing EXIOBASE2 products; filled from direct CPA2002 where possible:", missing_esa_rows)
    _print_mapping_rows("First-step direct CPA2002 EXIOBASE2 codes absent from bridge:", direct_extra_rows)
    _print_mapping_rows("First-step EXIOBASE2 products still unmapped after closure:", unclosed_rows)
    _print_mapping_rows("First-step ESA59 missing old CPA2002 two-digit codes present in direct CPA2002:", esa_missing_division_rows)
    _print_mapping_rows("First-step direct CPA2002 missing old CPA2002 two-digit codes present in ESA59:", direct_missing_division_rows)
    _print_mapping_rows("First-step ESA59/direct CPA2002 product-level two-digit inconsistencies:", mismatch_rows)

    # Source-specific gaps are warnings: the economic requirement is that the
    # closed union of official first-step sources preserves every EXIOBASE2 product
    # and every old-CPA division before downstream correspondence is composed.
    # Actual product-level disagreements still require a curated row; otherwise the
    # generator would be silently choosing between incompatible product boundaries.
    mismatch_codes = {row["exio20_code"] for row in mismatch_rows}
    resolved_mismatch_rows: list[dict[str, str]] = []
    unresolved_mismatch_rows: list[dict[str, str]] = []
    bad_conciliation_rows: list[dict[str, str]] = []
    for code in sorted(mismatch_codes):
        info = bridge_labels[code]
        resolution = conciliation_by_code.get(code)
        esa_divisions = {target_code for target_code, _ in esa_by_code.get(code, set())}
        direct_entry = direct_by_code.get(code, {"divisions": set(), "labels_by_division": {}})
        direct_divisions = cast(set[str], direct_entry.get("divisions", set()))
        observed_union = esa_divisions | direct_divisions
        if resolution is None:
            unresolved_mismatch_rows.append(
                {
                    "exio20_code": code,
                    "exio20_label": info["exio20_product_name"],
                    "message": "missing curated conciliation row for ESA59/direct CPA2002 conflict",
                }
            )
            continue
        chosen = cast(set[str], resolution["divisions"])
        recorded_esa = cast(set[str], resolution.get("esa59_old_cpa2002_divisions", set()))
        recorded_direct = cast(set[str], resolution.get("direct_old_cpa2002_divisions", set()))
        invented = sorted(chosen - observed_union)
        evidence_mismatch = recorded_esa != esa_divisions or recorded_direct != direct_divisions
        if invented or evidence_mismatch:
            row = {
                "exio20_code": code,
                "exio20_label": info["exio20_product_name"],
                "invented_old_cpa2002_divisions": ";".join(invented),
                "observed_old_cpa2002_divisions": ";".join(sorted(observed_union)),
            }
            if evidence_mismatch:
                row.update(
                    {
                        "recorded_esa59_old_cpa2002_divisions": ";".join(sorted(recorded_esa)),
                        "actual_esa59_old_cpa2002_divisions": ";".join(sorted(esa_divisions)),
                        "recorded_direct_old_cpa2002_divisions": ";".join(sorted(recorded_direct)),
                        "actual_direct_old_cpa2002_divisions": ";".join(sorted(direct_divisions)),
                    }
                )
            bad_conciliation_rows.append(row)
            continue
        resolved_mismatch_rows.append(
            {
                "exio20_code": code,
                "exio20_label": info["exio20_product_name"],
                "chosen_old_cpa2002_divisions": ";".join(sorted(chosen)),
                "conciliation_basis": clean_text(resolution.get("basis", "")),
            }
        )

    stale_conciliation_rows = [
        {
            "exio20_code": code,
            "exio20_label": bridge_labels[code]["exio20_product_name"],
            "message": "conciliation row applies to this bridge product but no current ESA59/direct conflict exists",
        }
        for code in sorted((set(conciliation_by_code) & bridge_codes) - mismatch_codes)
    ]
    _print_mapping_rows("First-step ESA59/direct CPA2002 product-level conflicts resolved by curated conciliation:", resolved_mismatch_rows)
    _print_mapping_rows("First-step ESA59/direct CPA2002 conflicts missing curated conciliation:", unresolved_mismatch_rows)
    _print_mapping_rows("First-step curated conciliation rows with non-observed old CPA divisions:", bad_conciliation_rows)
    _print_mapping_rows("First-step stale curated conciliation rows:", stale_conciliation_rows)

    hard_errors = direct_extra_rows + unclosed_rows + esa_without_bridge + unresolved_mismatch_rows + bad_conciliation_rows + stale_conciliation_rows
    if hard_errors:
        raise ValueError("First-step EXIOBASE2->old-CPA two-source closure failed; see printed coverage/inconsistency rows above.")

    records: list[dict[str, str]] = []
    for exio_code in sorted(bridge_codes):
        info = bridge_labels[exio_code]
        if exio_code in mismatch_codes:
            resolution = conciliation_by_code[exio_code]
            label_by_division = cast(dict[str, str], resolution.get("label_by_division", {}))
            targets = [(division, label_by_division.get(division, "")) for division in sorted(cast(set[str], resolution["divisions"]))]
            source = "curated_first_step_conciliation"
        else:
            targets = sorted(esa_by_code.get(exio_code, set()))
            source = "ESA59_EXIOBASE20p_ESA59p"
            if not targets:
                direct_entry = direct_by_code.get(exio_code, {"divisions": set(), "labels_by_division": {}})
                labels_by_division = cast(dict[str, set[str]], direct_entry.get("labels_by_division", {}))
                targets = [
                    (division, "CPA2002 direct fallback: " + " | ".join(sorted(labels_by_division.get(division, set()))[:8]))
                    for division in sorted(cast(set[str], direct_entry.get("divisions", set())))
                ]
                source = "CPA2002_6digit_EXIOBASE_FINAL_fallback"
        for old_code, old_label in targets:
            records.append(
                {
                    "exio20_product_code": exio_code,
                    "exio20_product_name": info["exio20_product_name"],
                    "exio3_product_code": info["exio3_product_code"],
                    "exio3_product_name": info["exio3_product_name"],
                    "esa59_product_code": old_code,
                    "esa59_product_label": old_label,
                    "first_step_source": source,
                }
            )
    closed = pd.DataFrame.from_records(records).drop_duplicates().reset_index(drop=True)
    assert_mapping_step_coverage(
        step_name="closed EXIOBASE2 product -> old CPA2002 division",
        required_left_codes=bridge_codes,
        mapped_left_codes=closed["exio20_product_code"].astype(str).tolist(),
        required_right_codes=old_division_universe,
        mapped_right_codes=closed["esa59_product_code"].astype(str).tolist(),
        left_labels={code: info["exio20_product_name"] for code, info in bridge_labels.items()},
        right_labels={**esa_division_labels, **direct_division_labels},
    )
    return closed


def summarize_exio20_to_exio30_bridge_multiplicity(bridge: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    """Return EXIOBASE2->3 bridge rows that are not one-to-one."""

    exio20_to_3: dict[tuple[str, str], set[tuple[str, str]]] = {}
    exio3_to_20: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in bridge.itertuples(index=False):
        exio20 = (clean_text(getattr(row, "exio20_product_code")), clean_text(getattr(row, "exio20_product_name")))
        exio3 = (clean_text(getattr(row, "exio3_product_code")), clean_text(getattr(row, "exio3_product_name")))
        if not exio20[0] or not exio3[0]:
            continue
        exio20_to_3.setdefault(exio20, set()).add(exio3)
        exio3_to_20.setdefault(exio3, set()).add(exio20)

    one_to_many = [
        {
            "exio20_product_code": code,
            "exio20_product_name": name,
            "exio3_targets": ";".join(f"{target_code}:{target_name}" for target_code, target_name in sorted(targets)),
        }
        for (code, name), targets in sorted(exio20_to_3.items())
        if len(targets) > 1
    ]
    many_to_one = [
        {
            "exio3_product_code": code,
            "exio3_product_name": name,
            "exio20_sources": ";".join(f"{source_code}:{source_name}" for source_code, source_name in sorted(sources)),
        }
        for (code, name), sources in sorted(exio3_to_20.items())
        if len(sources) > 1
    ]
    return {"exio20_to_many_exio3": one_to_many, "many_exio20_to_exio3": many_to_one}


def print_exio20_to_exio30_bridge_multiplicity(bridge: pd.DataFrame) -> None:
    """Print non-one-to-one EXIOBASE2<->EXIOBASE3 bridge rows for CLI review."""

    report = summarize_exio20_to_exio30_bridge_multiplicity(bridge)
    if not report["exio20_to_many_exio3"] and not report["many_exio20_to_exio3"]:
        print("EXIOBASE2->3 bridge multiplicity: all mapped product codes are one-to-one.")
        return
    print("EXIOBASE2->3 bridge multiplicity: non-one-to-one mappings found:")
    print(json.dumps(report, indent=2, sort_keys=True))


def read_any_table(path: Path) -> pd.DataFrame:
    """Read a CSV/XLSX/XML correspondence table as strings."""

    suffix = path.suffix.casefold()
    if suffix in {".csv", ".txt", ".tsv"}:
        try:
            return pd.read_csv(path, dtype=str, sep=None, engine="python", keep_default_na=False).fillna("")
        except UnicodeDecodeError:
            return pd.read_csv(path, dtype=str, sep=None, engine="python", encoding="latin-1", keep_default_na=False).fillna("")
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, dtype=str, sheet_name=None).values()
        return pd.concat([sheet.fillna("") for sheet in sheets], ignore_index=True, sort=False).fillna("")
    if suffix == ".xml":
        try:
            frame = pd.read_xml(path).fillna("")
            if not frame.empty:
                return frame.astype(str)
        except Exception:
            pass
        root = ET.parse(path).getroot()
        rows: list[dict[str, str]] = []
        for element in root.iter():
            attrs = {key: clean_text(value) for key, value in element.attrib.items()}
            if len(attrs) >= 2:
                rows.append(attrs)
        return pd.DataFrame.from_records(rows).fillna("")
    raise ValueError(f"Unsupported correspondence file type: {path}")


def column_score(name: str, tokens: Iterable[str]) -> int:
    """Score a correspondence-table column name for fuzzy source/target role."""

    norm = re.sub(r"[^a-z0-9]+", "", name.casefold())
    score = 0
    for token in tokens:
        if token in norm:
            score += 10
    if "code" in norm or norm.endswith("id"):
        score += 3
    if "label" in norm or "title" in norm or "name" in norm:
        score -= 5
    return score


def choose_correspondence_columns(frame: pd.DataFrame, *, source_tokens: Sequence[str], target_tokens: Sequence[str], path: Path) -> tuple[str, str]:
    """Choose source/target code columns from a RAMON correspondence table."""

    columns = [str(column) for column in frame.columns]
    source_ranked = sorted(((column_score(column, source_tokens), column) for column in columns), reverse=True)
    target_ranked = sorted(((column_score(column, target_tokens), column) for column in columns), reverse=True)
    if source_ranked and target_ranked and source_ranked[0][0] > 0 and target_ranked[0][0] > 0 and source_ranked[0][1] != target_ranked[0][1]:
        return source_ranked[0][1], target_ranked[0][1]
    code_like: list[str] = []
    for column in columns:
        values = frame[column].astype(str).head(100).map(canonical_code)
        if values.map(lambda value: bool(re.search(r"\d{2}", value))).mean() > 0.5:
            code_like.append(column)
    if len(code_like) >= 2:
        return code_like[0], code_like[1]
    raise ValueError(f"Could not infer source/target code columns in {path}; columns={columns}")


def read_correspondence(path: Path, *, source_tokens: Sequence[str], target_tokens: Sequence[str]) -> pd.DataFrame:
    """Read a RAMON/Eurostat correspondence preserving original code levels.

    The returned frame intentionally keeps detailed, division-level, section-level,
    and mixed-level rows.  Later economic composition must decide which rows are
    admissible; this parser must not discard official broad/detailed evidence such
    as ``C -> 09.90.19`` merely because it cannot be represented as a two-digit
    division edge.
    """

    raw = read_any_table(path)
    if raw.empty:
        raise ValueError(f"Empty correspondence table: {path}")
    source_col, target_col = choose_correspondence_columns(raw, source_tokens=source_tokens, target_tokens=target_tokens, path=path)
    assert_raw_edge_table_complete(step_name=str(path), frame=raw, source_col=source_col, target_col=target_col)

    records: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []
    bad_rows: list[dict[str, str]] = []
    for index, row in raw.reset_index(drop=True).iterrows():
        source = classify_correspondence_code(row.get(source_col, ""))
        target = classify_correspondence_code(row.get(target_col, ""))
        edge_kind = correspondence_edge_kind(source["level"], target["level"])
        if edge_kind == "non_mapping_row":
            if source["level"] == "header" or target["level"] == "header":
                skipped_rows.append(
                    {
                        "row_index": str(index),
                        "source_raw": source["raw"],
                        "target_raw": target["raw"],
                        "message": "header/source-target row skipped",
                    }
                )
                continue
            bad_rows.append(
                {
                    "row_index": str(index),
                    "source_raw": source["raw"],
                    "target_raw": target["raw"],
                    "source_level": source["level"],
                    "target_level": target["level"],
                    "message": "nonblank row is not a recognized correspondence code edge",
                }
            )
            continue
        records.append(
            {
                "source_code": source["code"],
                "target_code": target["code"],
                "source_raw": source["raw"],
                "target_raw": target["raw"],
                "source_display_code": source["display_code"],
                "target_display_code": target["display_code"],
                "source_level": source["level"],
                "target_level": target["level"],
                "source_division": source["division"],
                "target_division": target["division"],
                "edge_kind": edge_kind,
            }
        )
    _print_mapping_rows(f"Correspondence header/non-edge rows skipped [{path}]:", skipped_rows)
    _print_mapping_rows(f"Correspondence unrecognized code rows [{path}]:", bad_rows)
    if bad_rows:
        raise ValueError(f"Correspondence {path} contains nonblank rows that are not recognized as classification-code edges.")
    out = pd.DataFrame.from_records(
        records,
        columns=[
            "source_code",
            "target_code",
            "source_raw",
            "target_raw",
            "source_display_code",
            "target_display_code",
            "source_level",
            "target_level",
            "source_division",
            "target_division",
            "edge_kind",
        ],
    ).drop_duplicates().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No usable source/target code pairs parsed from {path}")
    print(
        f"Read correspondence {path}: source_codes={out['source_code'].nunique()} "
        f"target_codes={out['target_code'].nunique()} rows={len(out)} "
        f"edge_kinds={dict(sorted(out['edge_kind'].value_counts().to_dict().items()))}"
    )
    return out


def audit_correspondence_level_step(frame: pd.DataFrame, *, step_name: str) -> CorrespondenceLevelAudit:
    """Summarize original-level correspondence edges without reducing to divisions."""

    required = {"source_code", "target_code", "source_level", "target_level", "edge_kind"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Level-aware correspondence audit {step_name!r} missing columns: {missing}")
    edge_kind_counts = {str(key): int(value) for key, value in frame["edge_kind"].value_counts().sort_index().items()}
    source_level_counts = {str(key): int(value) for key, value in frame["source_level"].value_counts().sort_index().items()}
    target_level_counts = {str(key): int(value) for key, value in frame["target_level"].value_counts().sort_index().items()}
    mixed = frame.loc[frame["source_level"].astype(str) != frame["target_level"].astype(str)].copy()
    print(
        f"Level-aware correspondence audit [{step_name}]: rows={len(frame)} "
        f"source_levels={source_level_counts} target_levels={target_level_counts} "
        f"edge_kinds={edge_kind_counts} mixed_level_edges={len(mixed)}"
    )
    _print_mapping_rows(
        f"Level-aware mixed correspondence edge examples [{step_name}]:",
        [
            {
                "source_raw": clean_text(row.get("source_raw", "")),
                "source_level": clean_text(row.get("source_level", "")),
                "target_raw": clean_text(row.get("target_raw", "")),
                "target_level": clean_text(row.get("target_level", "")),
                "edge_kind": clean_text(row.get("edge_kind", "")),
            }
            for row in mixed.head(25).to_dict(orient="records")
        ],
    )
    return CorrespondenceLevelAudit(
        name=step_name,
        edges=frame.copy(),
        source_level_counts=source_level_counts,
        target_level_counts=target_level_counts,
        edge_kind_counts=edge_kind_counts,
        mixed_level_edges=mixed.reset_index(drop=True),
    )


NUMERIC_CORRESPONDENCE_LEVELS: Final[set[str]] = {"division", "group", "class", "category", "subcategory", "aggregate"}
DIRECT_ONLY_ANCHOR_SOURCE_MODE: Final[str] = "direct-only"
DIRECT_PLUS_CURATED_GAPS_ANCHOR_SOURCE_MODE: Final[str] = "direct-plus-curated-gaps"
ANCHOR_SOURCE_MODES: Final[set[str]] = {DIRECT_ONLY_ANCHOR_SOURCE_MODE, DIRECT_PLUS_CURATED_GAPS_ANCHOR_SOURCE_MODE}
DEFAULT_ANCHOR_SOURCE_MODE: Final[str] = DIRECT_ONLY_ANCHOR_SOURCE_MODE
CLEAN_OUTPUT_STATUS: Final[str] = "clean_no_diagnostics"
DIAGNOSTICS_BLOCKED_OUTPUT_STATUS: Final[str] = "diagnostics_blocked_no_outputs"
DIAGNOSTIC_EXPLORATORY_OUTPUT_STATUS: Final[str] = "diagnostic_exploratory_unresolved_anchors_present"
TARGET_DIVISION_SPLIT_WARNING: Final[str] = "CPA2.1 division maps to multiple CPA64/CPA65 targets; runtime output-share split required."
FINAL_AGGREGATION_SPLIT_WARNING: Final[str] = "EXIOBASE3 product maps to multiple CPA64/CPA65 targets; runtime output-share split required."


def numeric_code_ancestors(code: object) -> list[str]:
    """Return exact numeric code and numeric ancestors, stopping at two digits.

    This is deliberately not a section fallback.  Without an explicit hierarchy
    universe, a section-level source such as ``C`` cannot be expanded to all of its
    product descendants for product-composition purposes.
    """

    compact = canonical_code(code)
    if not compact.isdigit() or len(compact) < 2:
        return []
    out: list[str] = []
    current = compact
    while len(current) >= 2:
        out.append(current)
        if len(current) == 2:
            break
        current = current[:-1]
    return out


def _append_economic_warning(existing: object, warning: str) -> str:
    """Append a semicolon-separated economic warning without duplicates."""

    current = [part.strip() for part in clean_text(existing).split(" | ") if part.strip()]
    warning_text = clean_text(warning)
    if warning_text and warning_text not in current:
        current.append(warning_text)
    return " | ".join(current)


def _records_by_source_code(frame: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    """Index level-aware correspondence rows by normalized source code."""

    records = cast(list[dict[str, Any]], frame.to_dict(orient="records"))
    by_source: dict[str, list[dict[str, str]]] = {}
    for record in records:
        source_code = clean_text(record.get("source_code", ""))
        source_level = clean_text(record.get("source_level", ""))
        target_code = clean_text(record.get("target_code", ""))
        if not source_code or not target_code:
            continue
        if source_level == "section":
            # Preserve section rows in the parser/audit, but never use them as an
            # implicit product-composition fallback.  They are too broad without a
            # trusted child universe.
            continue
        by_source.setdefault(source_code, []).append({key: clean_text(value) for key, value in record.items()})
    return by_source


def _match_level_aware_edges(
    by_source: dict[str, list[dict[str, str]]],
    *,
    source_code: str,
    source_level: str,
    step_name: str,
) -> tuple[list[dict[str, str]], str, dict[str, str] | None]:
    """Match exact source code, then numeric ancestors; never infer sections."""

    code = canonical_code(source_code)
    level = clean_text(source_level)
    if level == "unresolved_broad_division":
        return [], "", {
            "diagnostic_kind": "missing_curated_cpa2002_product_anchor",
            "source_code": source_code,
            "source_label": source_level,
            "message": "ESA-only broad old-CPA division has no direct detailed CPA2002 row and no curated product-anchor row; refusing implicit broad fan-out.",
        }
    if level == "unresolved_direct_only_missing_detail":
        return [], "", {
            "diagnostic_kind": "missing_direct_cpa2002_anchor",
            "source_code": source_code,
            "source_label": source_level,
            "message": "Direct-only anchor mode found no direct detailed CPA2002 row for this EXIOBASE2 product; curated ESA59 gap anchors are disabled.",
        }
    if level == "section" or not code.isdigit():
        return [], "", {
            "diagnostic_kind": f"{step_name}_section_or_nonnumeric_anchor",
            "source_code": source_code,
            "source_label": source_level,
            "message": "Product composition refused section/non-numeric correspondence fallback without an explicit child universe.",
        }
    for candidate in numeric_code_ancestors(code):
        matches = by_source.get(candidate, [])
        if matches:
            rule = "exact" if candidate == code else f"ancestor:{candidate}"
            return matches, rule, None
    return [], "", {
        "diagnostic_kind": f"{step_name}_unresolved",
        "source_code": source_code,
        "source_label": source_level,
        "message": "No exact or numeric-ancestor correspondence row resolved this product-level code.",
    }


def _direct_cpa2002_detail_rows_by_product(direct_cpa2002: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    """Group detailed direct CPA2002 rows by EXIOBASE2 product code."""

    out: dict[str, list[dict[str, str]]] = {}
    if direct_cpa2002.empty:
        return out
    for row in direct_cpa2002.itertuples(index=False):
        exio_code = clean_text(getattr(row, "exio20_product_code", ""))
        cpa_code = clean_text(getattr(row, "cpa2002_code", ""))
        if not exio_code or not cpa_code:
            continue
        endpoint = classify_correspondence_code(cpa_code)
        out.setdefault(exio_code, []).append(
            {
                "cpa2002_code": endpoint["code"],
                "cpa2002_display_code": endpoint["display_code"],
                "cpa2002_level": endpoint["level"],
                "cpa2002_division": clean_text(getattr(row, "cpa2002_division", "")) or endpoint["division"],
                "cpa2002_label": clean_text(getattr(row, "cpa2002_label", "")),
            }
        )
    return out


def _curated_product_anchors_by_product(product_anchor_conciliation: pd.DataFrame) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Group curated product-anchor rows by EXIOBASE2 product and ESA division."""

    out: dict[tuple[str, str], list[dict[str, str]]] = {}
    if product_anchor_conciliation.empty:
        return out
    records = cast(list[dict[str, Any]], product_anchor_conciliation.to_dict(orient="records"))
    for record in records:
        exio_code = clean_text(record.get("exio20_product_code", ""))
        esa_division = clean_text(record.get("esa59_old_cpa2002_division", ""))
        normalized = clean_text(record.get("chosen_cpa2002_normalized_code", ""))
        if not exio_code or not esa_division or not normalized:
            continue
        out.setdefault((exio_code, esa_division), []).append(
            {
                "anchor_code": normalized,
                "anchor_display_code": clean_text(record.get("chosen_cpa2002_display_code", "")) or clean_text(record.get("chosen_cpa2002_code", "")),
                "anchor_level": clean_text(record.get("chosen_cpa2002_level", "")),
                "anchor_label": clean_text(record.get("chosen_cpa2002_label", "")),
                "anchor_source": "curated_CPA2002_product_anchor",
                "anchor_basis": clean_text(record.get("anchor_basis", "")),
            }
        )
    return out


def _product_anchors_for_closed_row(
    row: Any,
    *,
    direct_details_by_product: dict[str, list[dict[str, str]]],
    curated_anchors_by_product: dict[tuple[str, str], list[dict[str, str]]],
    anchor_source_mode: str,
) -> list[dict[str, str]]:
    """Return CPA2002 anchors for one closed first-step row under the selected source mode."""

    mode = clean_text(anchor_source_mode) or DEFAULT_ANCHOR_SOURCE_MODE
    if mode not in ANCHOR_SOURCE_MODES:
        raise ValueError(f"Unknown anchor_source_mode {anchor_source_mode!r}; expected one of {sorted(ANCHOR_SOURCE_MODES)}")
    exio_code = clean_text(getattr(row, "exio20_product_code", ""))
    old_code = clean_text(getattr(row, "esa59_product_code", ""))
    first_step_source = clean_text(getattr(row, "first_step_source", ""))
    direct_details = [
        detail
        for detail in direct_details_by_product.get(exio_code, [])
        if clean_text(detail.get("cpa2002_division", "")) == old_code
    ]
    anchors: list[dict[str, str]] = []
    for detail in direct_details:
        anchors.append(
            {
                "anchor_code": clean_text(detail.get("cpa2002_code", "")),
                "anchor_display_code": clean_text(detail.get("cpa2002_display_code", "")),
                "anchor_level": clean_text(detail.get("cpa2002_level", "")),
                "anchor_label": clean_text(detail.get("cpa2002_label", "")),
                "anchor_source": "CPA2002_6digit_EXIOBASE_FINAL" if first_step_source != "curated_first_step_conciliation" else "curated_first_step_conciliation_direct_CPA2002_detail",
                "anchor_basis": "direct_detailed_cpa2002_within_closed_first_step_division",
            }
        )
    if anchors:
        return anchors
    endpoint = classify_correspondence_code(old_code)
    if mode == DIRECT_ONLY_ANCHOR_SOURCE_MODE:
        return [
            {
                "anchor_code": endpoint["code"],
                "anchor_display_code": endpoint["display_code"] or old_code,
                "anchor_level": "unresolved_direct_only_missing_detail",
                "anchor_label": clean_text(getattr(row, "esa59_product_label", "")),
                "anchor_source": "Direct_CPA2002_detail_missing_in_direct_only_mode",
                "anchor_basis": "direct_only_missing_direct_detail",
            }
        ]
    curated = curated_anchors_by_product.get((exio_code, old_code), [])
    if curated:
        return curated
    return [
        {
            "anchor_code": endpoint["code"],
            "anchor_display_code": endpoint["display_code"] or old_code,
            "anchor_level": "unresolved_broad_division",
            "anchor_label": clean_text(getattr(row, "esa59_product_label", "")),
            "anchor_source": "ESA59_old_CPA_division_requires_curated_product_anchor",
            "anchor_basis": "missing_direct_detail_and_missing_product_anchor",
        }
    ]


def compose_products_to_cpa21_level_aware(
    closed_first_step: pd.DataFrame,
    direct_cpa2002: pd.DataFrame,
    cpa2002_to_2008: pd.DataFrame,
    cpa2008_to_21: pd.DataFrame,
    product_anchor_conciliation: pd.DataFrame,
    *,
    anchor_source_mode: str = DEFAULT_ANCHOR_SOURCE_MODE,
) -> ProductPathResult:
    """Compose EXIOBASE products through finest available CPA codes to CPA2.1.

    Direct CPA2002 product codes are used before ESA/old-division fallbacks.  The
    matcher uses exact correspondence rows first, then numeric ancestors.  It does
    not use section-level correspondence rows as implicit product fallbacks.
    """

    cpa2002_index = _records_by_source_code(cpa2002_to_2008)
    cpa2008_index = _records_by_source_code(cpa2008_to_21)
    direct_details_by_product = _direct_cpa2002_detail_rows_by_product(direct_cpa2002)
    curated_anchors_by_product = _curated_product_anchors_by_product(product_anchor_conciliation)
    path_rows: list[dict[str, str]] = []
    diagnostics: list[dict[str, str]] = []

    for row in closed_first_step.itertuples(index=False):
        exio20_code = clean_text(getattr(row, "exio20_product_code", ""))
        exio20_name = clean_text(getattr(row, "exio20_product_name", ""))
        exio3_code = clean_text(getattr(row, "exio3_product_code", ""))
        exio3_name = clean_text(getattr(row, "exio3_product_name", ""))
        esa59_code = clean_text(getattr(row, "esa59_product_code", ""))
        esa59_label = clean_text(getattr(row, "esa59_product_label", ""))
        first_step_source = clean_text(getattr(row, "first_step_source", ""))
        anchors = _product_anchors_for_closed_row(
            row,
            direct_details_by_product=direct_details_by_product,
            curated_anchors_by_product=curated_anchors_by_product,
            anchor_source_mode=anchor_source_mode,
        )
        for anchor in anchors:
            first_edges, first_rule, first_diagnostic = _match_level_aware_edges(
                cpa2002_index,
                source_code=anchor["anchor_code"],
                source_level=anchor["anchor_level"],
                step_name="cpa2002_to_cpa2008",
            )
            if first_diagnostic is not None:
                diagnostics.append(
                    {
                        **first_diagnostic,
                        "source_code": exio20_code,
                        "source_label": exio20_name,
                        "message": f"{first_diagnostic['message']} anchor={anchor['anchor_display_code']} ({anchor['anchor_level']})",
                    }
                )
                continue
            for first_edge in first_edges:
                cpa2008_code = clean_text(first_edge.get("target_code", ""))
                cpa2008_display = clean_text(first_edge.get("target_display_code", "")) or clean_text(first_edge.get("target_raw", "")) or cpa2008_code
                cpa2008_level = clean_text(first_edge.get("target_level", ""))
                second_edges, second_rule, second_diagnostic = _match_level_aware_edges(
                    cpa2008_index,
                    source_code=cpa2008_code,
                    source_level=cpa2008_level,
                    step_name="cpa2008_to_cpa21",
                )
                if second_diagnostic is not None:
                    diagnostics.append(
                        {
                            **second_diagnostic,
                            "source_code": exio20_code,
                            "source_label": exio20_name,
                            "message": f"{second_diagnostic['message']} cpa2008={cpa2008_display} ({cpa2008_level}) from anchor={anchor['anchor_display_code']}",
                        }
                    )
                    continue
                for second_edge in second_edges:
                    cpa21_division = clean_text(second_edge.get("target_division", ""))
                    if not cpa21_division:
                        diagnostics.append(
                            {
                                "diagnostic_kind": "cpa21_without_numeric_division",
                                "source_code": exio20_code,
                                "source_label": exio20_name,
                                "message": f"CPA2.1 target {second_edge.get('target_raw', '')} has no numeric division; not usable for CPA64/CPA65 aggregation.",
                            }
                        )
                        continue
                    path_rows.append(
                        {
                            "exio20_product_code": exio20_code,
                            "exio20_product_name": exio20_name,
                            "exio3_product_code": exio3_code,
                            "exio3_product_name": exio3_name,
                            "esa59_product_code": esa59_code,
                            "esa59_product_label": esa59_label,
                            "first_step_source": first_step_source,
                            "cpa2002_anchor_code": anchor["anchor_code"],
                            "cpa2002_anchor_display_code": anchor["anchor_display_code"],
                            "cpa2002_anchor_level": anchor["anchor_level"],
                            "cpa2002_anchor_label": anchor["anchor_label"],
                            "cpa2002_anchor_source": anchor["anchor_source"],
                            "cpa2002_anchor_basis": clean_text(anchor.get("anchor_basis", "")),
                            "cpa2002_match_rule": first_rule,
                            "cpa2008_code": cpa2008_code,
                            "cpa2008_display_code": cpa2008_display,
                            "cpa2008_level": cpa2008_level,
                            "cpa2008_match_rule": second_rule,
                            "cpa21_code": clean_text(second_edge.get("target_code", "")),
                            "cpa21_display_code": clean_text(second_edge.get("target_display_code", "")) or clean_text(second_edge.get("target_raw", "")),
                            "cpa21_level": clean_text(second_edge.get("target_level", "")),
                            "cpa21_division": cpa21_division,
                        }
                    )

    path_columns = [
        "exio20_product_code",
        "exio20_product_name",
        "exio3_product_code",
        "exio3_product_name",
        "esa59_product_code",
        "esa59_product_label",
        "first_step_source",
        "cpa2002_anchor_code",
        "cpa2002_anchor_display_code",
        "cpa2002_anchor_level",
        "cpa2002_anchor_label",
        "cpa2002_anchor_source",
        "cpa2002_anchor_basis",
        "cpa2002_match_rule",
        "cpa2008_code",
        "cpa2008_display_code",
        "cpa2008_level",
        "cpa2008_match_rule",
        "cpa21_code",
        "cpa21_display_code",
        "cpa21_level",
        "cpa21_division",
    ]
    diagnostic_columns = ["diagnostic_kind", "source_code", "source_label", "message"]
    paths = pd.DataFrame.from_records(path_rows, columns=path_columns).drop_duplicates().reset_index(drop=True)
    diagnostic_frame = pd.DataFrame.from_records(diagnostics, columns=diagnostic_columns).drop_duplicates().reset_index(drop=True)
    print(
        "Level-aware product composition: "
        f"closed_edges={len(closed_first_step)} product_paths={len(paths)} diagnostics={len(diagnostic_frame)}"
    )
    _print_mapping_rows(
        "Level-aware product-composition diagnostics:",
        cast(list[dict[str, str]], diagnostic_frame.head(50).to_dict(orient="records")),
    )
    return ProductPathResult(paths=paths, diagnostics=diagnostic_frame)


def expand_division_token(token: str) -> set[str]:
    """Expand a target-dictionary division token such as ``10-12`` or ``31_32``."""

    token = token.strip().replace("_", "-")
    if "-" in token:
        left, right = token.split("-", 1)
        if left.isdigit() and right.isdigit():
            return {f"{value:02d}" for value in range(int(left), int(right) + 1)}
    if token.isdigit():
        return {f"{int(token):02d}"}
    return set()


def canonical_target_version_label(target_version: str) -> str:
    """Return canonical mapping target-version metadata label."""

    normalized = target_version.strip().casefold().replace("_", "")
    if normalized == "cpa64":
        return "CPA_64"
    if normalized == "cpa65":
        return "CPA_65"
    return target_version.upper()


TARGET_DIVISION_MAP_COLUMNS: Final[set[str]] = {
    "target_version",
    "target_code",
    "target_label",
    "iot_a64_code",
    "cpa21_division",
}


def _read_dictionary_label_map(path: Path) -> dict[str, str]:
    """Read target dictionary labels for lock-step validation only."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    missing = sorted({"code", "label"} - set(frame.columns))
    if missing:
        raise ValueError(f"Target dictionary {path} missing columns: {missing}")
    labels: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        code = clean_text(getattr(row, "code"))
        label = clean_text(getattr(row, "label"))
        if not code or not label:
            continue
        if code in labels and labels[code] != label:
            raise ValueError(f"Target dictionary {path} has inconsistent labels for {code}")
        labels[code] = label
    return labels


def load_target_division_map(
    path: Path,
    *,
    target_dictionary: Path | None = None,
    target_version: str | None = None,
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Load the curated CPA target -> CPA2.1 division bridge.

    This file is the machine-readable target topology source.  The CPA64/CPA65
    dictionaries remain label/description dictionaries and are used only for
    optional lock-step validation, never for parsing division coverage.
    """

    frame = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    missing = sorted(TARGET_DIVISION_MAP_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Target division map {path} missing columns: {missing}")
    expected_target_version = canonical_target_version_label(target_version) if target_version is not None else ""

    labels: dict[str, str] = {}
    division_to_targets: dict[str, set[str]] = {}
    bad_rows: list[str] = []
    for row in frame.itertuples(index=False):
        row_target_version = clean_text(getattr(row, "target_version"))
        target_code = clean_text(getattr(row, "target_code"))
        target_label = clean_text(getattr(row, "target_label"))
        division = clean_text(getattr(row, "cpa21_division"))
        if expected_target_version and row_target_version != expected_target_version:
            bad_rows.append(f"target_version {row_target_version!r} for {target_code or '<blank>'} does not match expected {expected_target_version!r}")
            continue
        if not target_code or not target_label or not division:
            bad_rows.append(f"blank required field in row for target={target_code!r} division={division!r}")
            continue
        if not re.fullmatch(r"\d{2}", division):
            bad_rows.append(f"invalid CPA2.1 division {division!r} for {target_code}")
            continue
        old_label = labels.get(target_code)
        if old_label is not None and old_label != target_label:
            bad_rows.append(f"inconsistent labels for {target_code}: {old_label!r} vs {target_label!r}")
            continue
        labels[target_code] = target_label
        division_to_targets.setdefault(division, set()).add(target_code)

    if bad_rows:
        raise ValueError(f"Invalid target division map {path}: " + "; ".join(bad_rows[:20]))
    if not labels:
        raise ValueError(f"Target division map {path} contains no usable target rows")

    if target_dictionary is not None:
        dictionary_labels = _read_dictionary_label_map(target_dictionary)
        map_codes = set(labels)
        dictionary_codes = set(dictionary_labels)
        missing_from_dictionary = sorted(map_codes - dictionary_codes)
        missing_from_map = sorted(dictionary_codes - map_codes)
        if missing_from_dictionary or missing_from_map:
            raise ValueError(
                f"Target division map {path} and dictionary {target_dictionary} are not in lock-step: "
                f"missing_from_dictionary={missing_from_dictionary[:50]} "
                f"missing_from_map={missing_from_map[:50]}"
            )
        mismatched = sorted(
            code for code, label in labels.items()
            if clean_text(dictionary_labels.get(code, "")) != clean_text(label)
        )
        if mismatched:
            sample = [f"{code}: map={labels[code]!r} dictionary={dictionary_labels.get(code, '')!r}" for code in mismatched[:20]]
            raise ValueError(
                f"Target division map {path} labels do not match dictionary {target_dictionary}: "
                + "; ".join(sample)
            )

    return labels, division_to_targets


def build_official_mapping(
    *,
    esa59_path: Path,
    exio20_exio30_path: Path,
    cpa2002_cpa2008_path: Path,
    cpa2008_cpa21_path: Path,
    target_dictionary: Path,
    target_division_map: Path,
    target_version: str,
    cpa2002_exiobase_path: Path | None = None,
    first_step_conciliation_path: Path | None = None,
    product_anchor_conciliation_path: Path | None = None,
    anchor_source_mode: str = DEFAULT_ANCHOR_SOURCE_MODE,
) -> BuildResult:
    """Build EXIOBASE3 -> CPA64/65 using official/source concordance chains."""

    anchor_source_mode = clean_text(anchor_source_mode) or DEFAULT_ANCHOR_SOURCE_MODE
    if anchor_source_mode not in ANCHOR_SOURCE_MODES:
        raise ValueError(f"Unknown anchor_source_mode {anchor_source_mode!r}; expected one of {sorted(ANCHOR_SOURCE_MODES)}")
    esa = read_esa59_mapping(esa59_path)
    bridge = read_exio20_to_exio30_bridge(exio20_exio30_path)
    cpa2002_to_2008 = read_correspondence(cpa2002_cpa2008_path, source_tokens=("source", "from", "2002"), target_tokens=("target", "to", "2008"))
    cpa2008_to_21 = read_correspondence(cpa2008_cpa21_path, source_tokens=("source", "from", "2008"), target_tokens=("target", "to", "21", "2.1"))
    labels, division_to_targets = load_target_division_map(target_division_map, target_dictionary=target_dictionary, target_version=target_version)
    direct_cpa2002 = read_cpa2002_exiobase_mapping(cpa2002_exiobase_path) if cpa2002_exiobase_path is not None else pd.DataFrame()
    conciliation_path = first_step_conciliation_path if first_step_conciliation_path is not None else DEFAULT_FIRST_STEP_CONCILIATION
    first_step_conciliation = read_first_step_conciliation_map(conciliation_path)
    product_anchor_path = product_anchor_conciliation_path if product_anchor_conciliation_path is not None else DEFAULT_PRODUCT_ANCHOR_CONCILIATION
    product_anchor_conciliation_used = anchor_source_mode == DIRECT_PLUS_CURATED_GAPS_ANCHOR_SOURCE_MODE
    product_anchor_conciliation = read_product_anchor_conciliation_map(product_anchor_path) if product_anchor_conciliation_used else pd.DataFrame()

    joined = build_closed_first_step_edges(esa, bridge, direct_cpa2002, first_step_conciliation)

    diagnostics: list[dict[str, object]] = []
    composition = compose_products_to_cpa21_level_aware(
        joined,
        direct_cpa2002,
        cpa2002_to_2008,
        cpa2008_to_21,
        product_anchor_conciliation,
        anchor_source_mode=anchor_source_mode,
    )
    diagnostics.extend(cast(list[dict[str, object]], composition.diagnostics.to_dict(orient="records")))

    edge_rows: list[dict[str, object]] = []
    for row in composition.paths.itertuples(index=False):
        exio3_code = clean_text(getattr(row, "exio3_product_code", ""))
        if not exio3_code:
            diagnostics.append(
                {
                    "diagnostic_kind": "exio20_without_exio3_bridge",
                    "source_code": clean_text(getattr(row, "exio20_product_code", "")),
                    "source_label": clean_text(getattr(row, "exio20_product_name", "")),
                    "message": "EXIOBASE2 product did not match the EXIOBASE2->EXIOBASE3 bridge.",
                }
            )
            continue
        division = clean_text(getattr(row, "cpa21_division", ""))
        targets = division_to_targets.get(division, set())
        if not targets:
            diagnostics.append(
                {
                    "diagnostic_kind": "cpa21_division_without_target_aggregation",
                    "source_code": clean_text(getattr(row, "cpa21_display_code", "")),
                    "source_label": clean_text(getattr(row, "cpa21_level", "")),
                    "message": f"CPA2.1 division {division} has no target in {target_division_map}.",
                }
            )
            continue
        economic_warning = ""
        if len(targets) > 1:
            economic_warning = TARGET_DIVISION_SPLIT_WARNING
        for target_code in sorted(targets):
            edge_rows.append(
                {
                    "exio3_product_code": exio3_code,
                    "exio3_product_name": clean_text(getattr(row, "exio3_product_name", "")),
                    "exio20_product_code": clean_text(getattr(row, "exio20_product_code", "")),
                    "exio20_product_name": clean_text(getattr(row, "exio20_product_name", "")),
                    "esa59_product_code": clean_text(getattr(row, "esa59_product_code", "")),
                    "esa59_product_label": clean_text(getattr(row, "esa59_product_label", "")),
                    "first_step_source": clean_text(getattr(row, "first_step_source", "")),
                    "cpa2002_anchor_code": clean_text(getattr(row, "cpa2002_anchor_code", "")),
                    "cpa2002_anchor_display_code": clean_text(getattr(row, "cpa2002_anchor_display_code", "")),
                    "cpa2002_anchor_level": clean_text(getattr(row, "cpa2002_anchor_level", "")),
                    "cpa2002_anchor_label": clean_text(getattr(row, "cpa2002_anchor_label", "")),
                    "cpa2002_anchor_source": clean_text(getattr(row, "cpa2002_anchor_source", "")),
                    "cpa2002_anchor_basis": clean_text(getattr(row, "cpa2002_anchor_basis", "")),
                    "cpa2002_match_rule": clean_text(getattr(row, "cpa2002_match_rule", "")),
                    "cpa2008_code": clean_text(getattr(row, "cpa2008_code", "")),
                    "cpa2008_display_code": clean_text(getattr(row, "cpa2008_display_code", "")),
                    "cpa2008_level": clean_text(getattr(row, "cpa2008_level", "")),
                    "cpa2008_match_rule": clean_text(getattr(row, "cpa2008_match_rule", "")),
                    "cpa21_code": clean_text(getattr(row, "cpa21_code", "")),
                    "cpa21_display_code": clean_text(getattr(row, "cpa21_display_code", "")),
                    "cpa21_level": clean_text(getattr(row, "cpa21_level", "")),
                    "cpa21_division": division,
                    "target_code": target_code,
                    "target_label": labels.get(target_code, target_code),
                    "economic_warning": economic_warning,
                }
            )
    edges = pd.DataFrame.from_records(edge_rows)
    target_division_split_edge_rows = 0
    curated_product_anchor_edge_rows = 0
    exio20_products_with_multiple_cpa2002_anchors = 0
    final_output_share_split_mapping_rows = 0
    distinct_exio3_products_requiring_output_share_split = 0
    if not edges.empty:
        target_division_split_edge_rows = int((edges["economic_warning"].astype(str) == TARGET_DIVISION_SPLIT_WARNING).sum())
        curated_product_anchor_edge_rows = int((edges["cpa2002_anchor_source"].astype(str) == "curated_CPA2002_product_anchor").sum())
        anchor_counts = cast(pd.Series, edges.groupby("exio20_product_code")["cpa2002_anchor_display_code"].nunique())
        exio20_products_with_multiple_cpa2002_anchors = int((anchor_counts > 1).sum())
        _print_mapping_rows(
            "Final CPA64/CPA65 target-division split edge rows:",
            cast(
                list[dict[str, str]],
                edges.loc[edges["economic_warning"].astype(str) == TARGET_DIVISION_SPLIT_WARNING,
                          ["exio3_product_code", "exio3_product_name", "cpa21_division", "target_code", "target_label"]]
                .drop_duplicates()
                .head(50)
                .to_dict(orient="records"),
            ),
        )
        _print_mapping_rows(
            "Curated CPA2002 product-anchor edge rows:",
            cast(
                list[dict[str, str]],
                edges.loc[edges["cpa2002_anchor_source"].astype(str) == "curated_CPA2002_product_anchor",
                          ["exio20_product_code", "exio20_product_name", "cpa2002_anchor_display_code", "cpa2002_anchor_basis", "target_code"]]
                .drop_duplicates()
                .head(50)
                .to_dict(orient="records"),
            ),
        )
    if edges.empty:
        mapping = pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    else:
        grouped = cast(
            pd.DataFrame,
            edges.groupby(["exio3_product_code", "exio3_product_name", "target_code", "target_label"], as_index=False).agg(
                esa59_product_codes=("esa59_product_code", lambda values: ";".join(sorted(set(map(str, values))))),
                esa59_product_labels=("esa59_product_label", lambda values: " | ".join(sorted(set(map(str, values))))),
                cpa2002_anchor_codes=("cpa2002_anchor_display_code", lambda values: ";".join(sorted({str(value) for value in values if str(value)}))),
                cpa2002_anchor_levels=("cpa2002_anchor_level", lambda values: ";".join(sorted({str(value) for value in values if str(value)}))),
                cpa2002_anchor_sources=("cpa2002_anchor_source", lambda values: ";".join(sorted({str(value) for value in values if str(value)}))),
                cpa2002_anchor_bases=("cpa2002_anchor_basis", lambda values: ";".join(sorted({str(value) for value in values if str(value)}))),
                cpa2002_match_rules=("cpa2002_match_rule", lambda values: ";".join(sorted({str(value) for value in values if str(value)}))),
                cpa2008_codes=("cpa2008_display_code", lambda values: ";".join(sorted({str(value) for value in values if str(value)}))),
                cpa2008_match_rules=("cpa2008_match_rule", lambda values: ";".join(sorted({str(value) for value in values if str(value)}))),
                cpa21_codes=("cpa21_display_code", lambda values: ";".join(sorted({str(value) for value in values if str(value)}))),
                cpa21_levels=("cpa21_level", lambda values: ";".join(sorted({str(value) for value in values if str(value)}))),
                cpa21_divisions=("cpa21_division", lambda values: ";".join(sorted(set(map(str, values))))),
                exio20_product_codes=("exio20_product_code", lambda values: ";".join(sorted({str(value) for value in values if str(value)}))),
                economic_warnings=("economic_warning", lambda values: " | ".join(sorted({str(value) for value in values if str(value)}))),
            ),
        )
        target_counts = grouped.groupby("exio3_product_code")["target_code"].transform("nunique").astype(int)
        grouped.loc[:, "n_targets_for_exio3_product"] = target_counts
        split_mask = grouped["n_targets_for_exio3_product"].astype(int) > 1
        if bool(split_mask.any()):
            for idx in list(grouped.index[split_mask]):
                grouped.at[idx, "economic_warnings"] = _append_economic_warning(
                    grouped.at[idx, "economic_warnings"],
                    FINAL_AGGREGATION_SPLIT_WARNING,
                )
            final_output_share_split_mapping_rows = int(split_mask.sum())
            distinct_exio3_products_requiring_output_share_split = int(grouped.loc[split_mask, "exio3_product_code"].nunique())
        _print_mapping_rows(
            "Final CPA64/CPA65 output-share split mapping rows:",
            cast(
                list[dict[str, str]],
                grouped.loc[split_mask, ["exio3_product_code", "exio3_product_name", "target_code", "target_label", "cpa21_divisions", "economic_warnings"]]
                .drop_duplicates()
                .head(50)
                .to_dict(orient="records"),
            ),
        )
        rows: list[dict[str, object]] = []
        sorted_grouped = cast(pd.DataFrame, grouped.sort_values(by=["exio3_product_code", "target_code"]))
        grouped_records = cast(list[dict[str, Any]], sorted_grouped.to_dict(orient="records"))
        for order, row in enumerate(grouped_records, start=1):
            allocation_rule = "split-by-output-share" if int(row["n_targets_for_exio3_product"]) > 1 else "direct-single-target"
            esa59_product_codes = clean_text(row.get("esa59_product_codes", ""))
            esa59_product_labels = clean_text(row.get("esa59_product_labels", ""))
            cpa2002_anchor_codes = clean_text(row.get("cpa2002_anchor_codes", ""))
            cpa2002_anchor_levels = clean_text(row.get("cpa2002_anchor_levels", ""))
            cpa2002_anchor_sources = clean_text(row.get("cpa2002_anchor_sources", ""))
            cpa2002_anchor_bases = clean_text(row.get("cpa2002_anchor_bases", ""))
            cpa2002_match_rules = clean_text(row.get("cpa2002_match_rules", ""))
            cpa2008_codes = clean_text(row.get("cpa2008_codes", ""))
            cpa2008_match_rules = clean_text(row.get("cpa2008_match_rules", ""))
            cpa21_codes = clean_text(row.get("cpa21_codes", ""))
            cpa21_levels = clean_text(row.get("cpa21_levels", ""))
            cpa21_divisions = clean_text(row.get("cpa21_divisions", ""))
            exio20_product_codes = clean_text(row.get("exio20_product_codes", ""))
            economic_warnings = clean_text(row.get("economic_warnings", ""))
            rows.append(
                {
                    "order": order,
                    "kind": "weighted",
                    "source_dataset": "EXIOBASE",
                    "source_system": "pxp",
                    "source_version": "EXIOBASE2-to-EXIOBASE3-source-concordance",
                    "source_level": "product",
                    "target_dataset": "Eurostat_IO",
                    "target_system": "pxp",
                    "target_version": canonical_target_version_label(target_version),
                    "target_level": "CPA_product",
                    "source_code": clean_text(row.get("exio3_product_code", "")),
                    "source_label": clean_text(row.get("exio3_product_name", "")),
                    "target_code": clean_text(row.get("target_code", "")),
                    "target_label": clean_text(row.get("target_label", "")),
                    "enabled": "1",
                    "provenance": "EXIOBASE_ESA59_direct_CPA2002_and_level_aware_RAMON_correspondence_chain",
                    "certainty": "source-concordance-derived",
                    "allocation_rule": allocation_rule,
                    "source_name": "EXIOBASE20p_ESA59p + CPA2002_6digit_EXIOBASE_FINAL + EXIOBASE2->3 + level-aware RAMON CPA correspondences",
                    "official_source": "; ".join(
                        str(path)
                        for path in [
                            esa59_path,
                            exio20_exio30_path,
                            cpa2002_exiobase_path or "",
                            conciliation_path,
                            product_anchor_path if product_anchor_conciliation_used else "",
                            cpa2002_cpa2008_path,
                            cpa2008_cpa21_path,
                        ]
                        if str(path)
                    ),
                    "notes": (
                        f"ESA59 old product group(s) {esa59_product_codes}: {esa59_product_labels}. "
                        f"CPA2002 anchor(s) {cpa2002_anchor_codes} at level(s) {cpa2002_anchor_levels}; "
                        f"anchor source(s): {cpa2002_anchor_sources}; anchor basis: {cpa2002_anchor_bases}; "
                        f"CPA2002->CPA2008 match rule(s): {cpa2002_match_rules}; CPA2008 code(s): {cpa2008_codes}; "
                        f"CPA2008->CPA2.1 match rule(s): {cpa2008_match_rules}; CPA2.1 code(s): {cpa21_codes} "
                        f"at level(s) {cpa21_levels}; CPA2.1 division(s): {cpa21_divisions}. "
                        f"EXIOBASE2 source code(s): {exio20_product_codes}. "
                        f"{economic_warnings}"
                    ).strip(),
                    "source_balance_role": "",
                    "target_sector_role": "",
                    "include_in_direct_energy": "1",
                }
            )
        mapping = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))

    diagnostics_df = pd.DataFrame.from_records(diagnostics, columns=["diagnostic_kind", "source_code", "source_label", "message"]).fillna("")
    coverage = {
        "esa59_path": str(esa59_path),
        "esa59_sha256": sha256_file(esa59_path),
        "exio20_exio30_path": str(exio20_exio30_path),
        "exio20_exio30_sha256": sha256_file(exio20_exio30_path),
        "cpa2002_exiobase_path": str(cpa2002_exiobase_path) if cpa2002_exiobase_path is not None else "",
        "cpa2002_exiobase_sha256": sha256_file(cpa2002_exiobase_path) if cpa2002_exiobase_path is not None else "",
        "first_step_conciliation_path": str(conciliation_path),
        "first_step_conciliation_sha256": sha256_file(conciliation_path),
        "product_anchor_conciliation_path": str(product_anchor_path),
        "product_anchor_conciliation_sha256": sha256_file(product_anchor_path),
        "product_anchor_conciliation_used": product_anchor_conciliation_used,
        "anchor_source_mode": anchor_source_mode,
        "cpa2002_cpa2008_path": str(cpa2002_cpa2008_path),
        "cpa2002_cpa2008_sha256": sha256_file(cpa2002_cpa2008_path),
        "cpa2008_cpa21_path": str(cpa2008_cpa21_path),
        "cpa2008_cpa21_sha256": sha256_file(cpa2008_cpa21_path),
        "target_version": canonical_target_version_label(target_version),
        "target_division_map_path": str(target_division_map),
        "target_division_map_sha256": sha256_file(target_division_map),
        "esa59_rows": int(len(esa)),
        "level_aware_product_path_rows": int(len(composition.paths)),
        "edge_rows": int(len(edges)),
        "mapping_rows": int(len(mapping)),
        "distinct_exio3_products": int(mapping["source_code"].nunique()) if not mapping.empty else 0,
        "distinct_target_codes": int(mapping["target_code"].nunique()) if not mapping.empty else 0,
        "target_division_split_edge_rows": target_division_split_edge_rows,
        "curated_product_anchor_edge_rows": curated_product_anchor_edge_rows,
        "exio20_products_with_multiple_cpa2002_anchors": exio20_products_with_multiple_cpa2002_anchors,
        "final_output_share_split_mapping_rows": final_output_share_split_mapping_rows,
        "distinct_exio3_products_requiring_output_share_split": distinct_exio3_products_requiring_output_share_split,
        "diagnostic_rows": int(len(diagnostics_df)),
    }
    return BuildResult(mapping=mapping, edges=edges, diagnostics=diagnostics_df, coverage=coverage)


def write_outputs(
    result: BuildResult,
    *,
    out_dir: Path,
    target_version: str,
    edge_audit_csv: Path | None = None,
) -> dict[str, object]:
    """Write Layer-3 mapping outputs, diagnostics, manifest, and optional edge audit."""

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"exio_to_eurostat_{target_version.lower()}_products_from_official_concordances"
    anchor_source_mode = clean_text(result.coverage.get("anchor_source_mode", "")) or DEFAULT_ANCHOR_SOURCE_MODE
    if anchor_source_mode != DEFAULT_ANCHOR_SOURCE_MODE:
        stem = f"{stem}_{anchor_source_mode.replace('-', '_')}"
    mapping_path = out_dir / f"{stem}.csv"
    diagnostics_path = out_dir / f"{stem}_diagnostics.csv"
    manifest_path = out_dir / f"{stem}_manifest.json"
    result.mapping.to_csv(mapping_path, index=False)
    result.diagnostics.to_csv(diagnostics_path, index=False)

    output_files: dict[str, str] = {
        "mapping": mapping_path.name,
        "diagnostics": diagnostics_path.name,
    }
    edge_audit_path = edge_audit_csv
    if edge_audit_path is None:
        edge_audit_path = out_dir / "diagnostics" / f"{stem}_edges.csv"
    edge_audit_path.parent.mkdir(parents=True, exist_ok=True)
    result.edges.to_csv(edge_audit_path, index=False)
    diagnostics_allowed = bool(result.coverage.get("diagnostics_allowed", False))
    # Coverage is stored as dict[str, object] for JSON metadata, but this key is
    # produced by build_official_mapping as an int.  Keep the runtime behaviour
    # unchanged and document the static type boundary locally.
    diagnostic_rows_value = cast(int, result.coverage.get("diagnostic_rows", 0))
    diagnostic_rows = int(diagnostic_rows_value or 0)
    output_status = clean_text(result.coverage.get("output_status", "")) or CLEAN_OUTPUT_STATUS
    anchor_source_mode = clean_text(result.coverage.get("anchor_source_mode", "")) or DEFAULT_ANCHOR_SOURCE_MODE
    method_notes = [
        "No hard-coded NSI->CPA table is used.",
        "EXIOBASE2 products are mapped through a closed first-step comparison of EXIOBASE20p_ESA59p.txt and CPA2002_6digit_EXIOBASE_FINAL.xlsx, with product-level conflicts resolved only by scripts/mappings/exiobase2_old_cpa2002_first_step_conciliation.csv.",
        "After first-step division closure, direct detailed CPA2002 anchors are composed through level-aware RAMON CPA2002->CPA2008 and CPA2008->CPA2.1 correspondences.",
        "ESA-only broad CPA2002 division anchors are not fanned out implicitly.",
        "CPA2.1 two-digit divisions are aggregated to repo CPA64/CPA65 targets via scripts/mappings/eurostat_cpa64_to_cpa21_divisions.csv or scripts/mappings/eurostat_cpa65_to_cpa21_divisions.csv.",
        "Products reaching multiple final CPA64/CPA65 targets are retained as explicit weighted mapping rows and marked split-by-output-share for runtime monetary allocation.",
        "CPA64/CPA65 dictionary descriptions are not parsed for machine topology; dictionaries are label/description references only.",
        "The internal EXIOBASE3/ESA59/CPA2.1 edge audit is written under out-dir/diagnostics by default; --edge-audit-csv may override the path.",
    ]
    if anchor_source_mode == DIRECT_ONLY_ANCHOR_SOURCE_MODE:
        method_notes.append("Anchor source mode is direct-only: missing direct CPA2002 workbook detail is reported as diagnostics; curated ESA59 gap anchors are not used for mapping rows.")
    elif anchor_source_mode == DIRECT_PLUS_CURATED_GAPS_ANCHOR_SOURCE_MODE:
        method_notes.append("Anchor source mode is direct-plus-curated-gaps: curated product anchors are used only when direct detailed CPA2002 rows are absent; broad ESA59 divisions are never fanned out implicitly.")
    if diagnostics_allowed and diagnostic_rows:
        method_notes.append("This output was written with --allow-diagnostics. It contains unresolved source-chain diagnostics and is intended for audit/stress-testing, not as a clean production concordance.")

    manifest = {
        "mapping_name": stem,
        "created_utc": datetime.now(UTC).isoformat(),
        "created_by_script": "scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_bridge_nsi.py",
        "diagnostics_allowed": diagnostics_allowed,
        "output_status": output_status,
        "method_notes": method_notes,
        "coverage": result.coverage,
        "output_files": output_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concordances-zip", type=Path, default=DEFAULT_CONCORDANCES_ZIP)
    parser.add_argument("--esa59-member", default=ESA59_MEMBER)
    parser.add_argument("--exio20-exio30-member", default=EXIO20_EXIO30_MEMBER)
    parser.add_argument("--cpa2002-exiobase-member", default=CPA2002_EXIOBASE_MEMBER)
    parser.add_argument("--first-step-conciliation", type=Path, default=DEFAULT_FIRST_STEP_CONCILIATION)
    parser.add_argument("--product-anchor-conciliation", type=Path, default=DEFAULT_PRODUCT_ANCHOR_CONCILIATION)
    parser.add_argument(
        "--anchor-source-mode",
        choices=sorted(ANCHOR_SOURCE_MODES),
        default=DEFAULT_ANCHOR_SOURCE_MODE,
        help=(
            "CPA2002 anchor source policy. direct-only uses only the direct EXIOBASE2->CPA2002 workbook; "
            "direct-plus-curated-gaps additionally uses curated anchors for products absent from that workbook."
        ),
    )
    parser.add_argument("--cpa2002-cpa2008", type=Path, default=DEFAULT_CPA2002_CPA2008)
    parser.add_argument("--cpa2008-cpa21", type=Path, default=DEFAULT_CPA2008_CPA21)
    parser.add_argument("--target-version", choices=["cpa64", "cpa65"], default="cpa65")
    parser.add_argument("--target-dictionary", type=Path, default=None)
    parser.add_argument("--target-division-map", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--edge-audit-csv", type=Path, default=None, help="Optional override for the internal edge diagnostic CSV; defaults to out-dir/diagnostics/<stem>_edges.csv.")
    parser.add_argument("--allow-diagnostics", action="store_true", help="Write outputs even if source-chain diagnostics are present.")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the official-concordance mapping generator."""

    args = parse_args(argv)
    target_dictionary = args.target_dictionary
    if target_dictionary is None:
        target_dictionary = DEFAULT_CPA65_DICTIONARY if args.target_version == "cpa65" else DEFAULT_CPA64_DICTIONARY
    target_division_map = args.target_division_map
    if target_division_map is None:
        target_division_map = DEFAULT_CPA65_DIVISION_MAP if args.target_version == "cpa65" else DEFAULT_CPA64_DIVISION_MAP

    esa59_bytes, esa59_member = read_zip_member_bytes(args.concordances_zip, args.esa59_member)
    exio_bridge_bytes, exio_bridge_member = read_zip_member_bytes(args.concordances_zip, args.exio20_exio30_member)
    cpa2002_exiobase_bytes, cpa2002_exiobase_member = read_zip_member_bytes(args.concordances_zip, args.cpa2002_exiobase_member)

    with tempfile.TemporaryDirectory(prefix="exio_cpa_official_sources_") as tmpdir_text:
        tmpdir = Path(tmpdir_text)
        esa59_path = tmpdir / ESA59_SUFFIX
        exio_bridge_path = tmpdir / EXIO20_EXIO30_SUFFIX
        cpa2002_exiobase_path = tmpdir / CPA2002_EXIOBASE_SUFFIX
        esa59_path.write_text(esa59_bytes.decode("utf-8-sig", errors="replace"), encoding="utf-8")
        exio_bridge_path.write_bytes(exio_bridge_bytes)
        cpa2002_exiobase_path.write_bytes(cpa2002_exiobase_bytes)
        print_exio20_to_exio30_bridge_multiplicity(read_exio20_to_exio30_bridge(exio_bridge_path))
        result = build_official_mapping(
            esa59_path=esa59_path,
            exio20_exio30_path=exio_bridge_path,
            cpa2002_cpa2008_path=args.cpa2002_cpa2008,
            cpa2008_cpa21_path=args.cpa2008_cpa21,
            target_dictionary=target_dictionary,
            target_division_map=target_division_map,
            target_version=args.target_version,
            cpa2002_exiobase_path=cpa2002_exiobase_path,
            first_step_conciliation_path=args.first_step_conciliation,
            product_anchor_conciliation_path=args.product_anchor_conciliation,
            anchor_source_mode=args.anchor_source_mode,
        )

    official_source = "; ".join(
        [
            f"{args.concordances_zip}::{esa59_member}",
            f"{args.concordances_zip}::{exio_bridge_member}",
            f"{args.concordances_zip}::{cpa2002_exiobase_member}",
            str(args.cpa2002_cpa2008),
            str(args.cpa2008_cpa21),
            str(args.first_step_conciliation),
            str(args.product_anchor_conciliation),
            str(target_division_map),
        ]
    )
    mapping = result.mapping.copy()
    if "official_source" in mapping.columns:
        mapping.loc[:, "official_source"] = official_source
    coverage = dict(result.coverage)
    coverage.update(
        {
            "concordances_zip": str(args.concordances_zip),
            "concordances_zip_sha256": sha256_file(args.concordances_zip),
            "esa59_member": esa59_member,
            "esa59_path": f"{args.concordances_zip}::{esa59_member}",
            "esa59_sha256": sha256_bytes(esa59_bytes),
            "exio20_exio30_member": exio_bridge_member,
            "exio20_exio30_path": f"{args.concordances_zip}::{exio_bridge_member}",
            "exio20_exio30_sha256": sha256_bytes(exio_bridge_bytes),
            "cpa2002_exiobase_member": cpa2002_exiobase_member,
            "cpa2002_exiobase_path": f"{args.concordances_zip}::{cpa2002_exiobase_member}",
            "cpa2002_exiobase_sha256": sha256_bytes(cpa2002_exiobase_bytes),
            "first_step_conciliation_path": str(args.first_step_conciliation),
            "first_step_conciliation_sha256": sha256_file(args.first_step_conciliation),
            "product_anchor_conciliation_path": str(args.product_anchor_conciliation),
            "product_anchor_conciliation_sha256": sha256_file(args.product_anchor_conciliation),
            "anchor_source_mode": args.anchor_source_mode,
            "diagnostics_allowed": bool(args.allow_diagnostics),
            "output_status": (
                DIAGNOSTIC_EXPLORATORY_OUTPUT_STATUS
                if result.coverage["diagnostic_rows"] and args.allow_diagnostics
                else DIAGNOSTICS_BLOCKED_OUTPUT_STATUS
                if result.coverage["diagnostic_rows"]
                else CLEAN_OUTPUT_STATUS
            ),
            "target_division_map_path": str(target_division_map),
            "target_division_map_sha256": sha256_file(target_division_map),
        }
    )
    result = BuildResult(mapping=mapping, edges=result.edges, diagnostics=result.diagnostics, coverage=coverage)
    if result.coverage["diagnostic_rows"] and not args.allow_diagnostics:
        print(json.dumps({"coverage": result.coverage, "out_dir": str(args.out_dir), "wrote_outputs": False}, indent=2, sort_keys=True))
        raise SystemExit("Official-concordance mapping has diagnostics; inspect printed diagnostics or rerun with --allow-diagnostics for an explicitly marked exploratory output.")
    if not args.check_only:
        write_outputs(
            result,
            out_dir=args.out_dir,
            target_version=args.target_version,
            edge_audit_csv=args.edge_audit_csv,
        )
    print(json.dumps({"coverage": result.coverage, "out_dir": str(args.out_dir), "wrote_outputs": not args.check_only}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
