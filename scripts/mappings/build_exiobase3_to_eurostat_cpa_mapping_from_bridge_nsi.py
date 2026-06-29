#!/usr/bin/env python3
"""Build EXIOBASE3 -> Eurostat CPA mappings from official concordances.

This replaces the earlier Bridge_NSI_AT label-derived fallback.  The generator
uses only source concordance files under ``data/exiobase_concordance_sources``
and ``data/eurostat_concordance_sources``:

* EXIOBASE20p -> ESA59p aggregation, with old CPA/NACE-style division codes;
* EXIOBASE2 product -> EXIOBASE3 product bridge; and
* RAMON/Eurostat CPA 2002 -> CPA 2008 -> CPA 2.1 correspondence tables.

The final CPA64/CPA65 targets are generated from two-digit CPA 2.1 divisions and
the committed CPA64/CPA65 target dictionaries.  CPA65 splits division 68 into the
repo's two real-estate targets if those split targets are present.
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
DEFAULT_CPA2002_CPA2008: Final[Path] = DEFAULT_EUROSTAT_SOURCE_ROOT / "CPA_2002_to_CPA_2008.csv"
DEFAULT_CPA2008_CPA21: Final[Path] = DEFAULT_EUROSTAT_SOURCE_ROOT / "CPA_2008_to_CPA_21.csv"
DEFAULT_OUT_DIR: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "generated"
DEFAULT_CPA64_DICTIONARY: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "eurostat_cpa64_products_dictionary.csv"
DEFAULT_CPA65_DICTIONARY: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "eurostat_cpa65_products_dictionary.csv"

ESA59_SUFFIX: Final[str] = "EXIOBASE20p_ESA59p.txt"
EXIO20_EXIO30_SUFFIX: Final[str] = "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
ESA59_MEMBER: Final[str] = "Concordances/Different aggregations/EXIOBASE20p_ESA59p.txt"
EXIO20_EXIO30_MEMBER: Final[str] = "Concordances/EXIOBASE/EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"

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


def is_prefix_related(left: str, right: str) -> bool:
    """Return True if two compact classification keys match by hierarchy prefix."""

    return bool(left and right and (left == right or left.startswith(right) or right.startswith(left)))


def read_esa59_mapping(path: Path) -> pd.DataFrame:
    """Read EXIOBASE2 product -> ESA59 old-product-group mapping text.

    The EXIOBASE aggregation text files use group headers containing old product
    division codes in parentheses, followed by EXIOBASE2 product rows.  This
    reader also accepts two/three-column tabular variants for testability and
    for future upstream formatting changes.
    """

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
            group_label = line[: header_match.start()].strip()
            continue
        parts = [part.strip() for part in re.split(r"\t+|\s{2,}|,", line, maxsplit=2) if part.strip()]
        exio_code = ""
        exio_label = line
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
                    "esa59_product_label": group_label,
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
    """Read a RAMON/Eurostat correspondence into normalized source/target code pairs."""

    raw = read_any_table(path)
    if raw.empty:
        raise ValueError(f"Empty correspondence table: {path}")
    source_col, target_col = choose_correspondence_columns(raw, source_tokens=source_tokens, target_tokens=target_tokens, path=path)
    out = pd.DataFrame(
        {
            "source_code": raw[source_col].map(canonical_code),
            "target_code": raw[target_col].map(canonical_code),
            "source_raw": raw[source_col].map(clean_text),
            "target_raw": raw[target_col].map(clean_text),
        }
    )
    out = out.loc[(out["source_code"] != "") & (out["target_code"] != "")].drop_duplicates().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No usable source/target code pairs parsed from {path}")
    return out


def compose_old_division_to_cpa21_divisions(cpa2002_to_2008: pd.DataFrame, cpa2008_to_21: pd.DataFrame, *, old_division: str) -> tuple[set[str], list[str]]:
    """Compose old CPA2002 division code to current CPA2.1 division codes."""

    diagnostics: list[str] = []
    old = canonical_code(old_division)
    first = cpa2002_to_2008.loc[cpa2002_to_2008["source_code"].map(lambda code: is_prefix_related(code, old))].copy()
    if first.empty:
        diagnostics.append(f"No CPA2002->CPA2008 rows matched old division {old_division!r}")
        return set(), diagnostics
    cpa21_codes: set[str] = set()
    for cpa2008_code in sorted(set(first["target_code"].astype(str))):
        second = cpa2008_to_21.loc[cpa2008_to_21["source_code"].map(lambda code: is_prefix_related(code, cpa2008_code))].copy()
        if second.empty:
            diagnostics.append(f"No CPA2008->CPA2.1 rows matched CPA2008 code {cpa2008_code!r} from old division {old_division!r}")
            continue
        cpa21_codes.update(second["target_code"].astype(str).tolist())
    divisions = {division for division in (cpa_division_from_code(code) for code in cpa21_codes) if division}
    if not divisions:
        diagnostics.append(f"No CPA2.1 divisions resolved for old division {old_division!r}")
    return divisions, diagnostics


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


def load_target_division_map(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Load target labels and derive CPA division coverage from target codes/descriptions."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    missing = sorted({"code", "label", "description"} - set(frame.columns))
    if missing:
        raise ValueError(f"Target dictionary {path} missing columns: {missing}")
    labels = dict(zip(frame["code"].astype(str).str.strip(), frame["label"].astype(str).str.strip(), strict=False))
    division_to_targets: dict[str, set[str]] = {}
    for row in frame.itertuples(index=False):
        target_code = clean_text(getattr(row, "code"))
        description = clean_text(getattr(row, "description"))
        divisions: set[str] = set()
        for token in re.findall(r"\b\d{2}(?:[-_]\d{2})?\b", target_code):
            divisions.update(expand_division_token(token))
        for token in re.findall(r"\b\d{2}(?:[-_]\d{2})?\b", description):
            divisions.update(expand_division_token(token))
        # CPA65 L68A/L68B encode the deliberate 65-sector split of CPA division 68.
        if re.search(r"CPA[_-]?L68[A-Z]?", target_code, flags=re.IGNORECASE):
            divisions.add("68")
        for division in divisions:
            division_to_targets.setdefault(division, set()).add(target_code)
    return labels, division_to_targets


def build_official_mapping(
    *,
    esa59_path: Path,
    exio20_exio30_path: Path,
    cpa2002_cpa2008_path: Path,
    cpa2008_cpa21_path: Path,
    target_dictionary: Path,
    target_version: str,
) -> BuildResult:
    """Build EXIOBASE3 -> CPA64/65 using official/source concordance chains."""

    esa = read_esa59_mapping(esa59_path)
    bridge = read_exio20_to_exio30_bridge(exio20_exio30_path)
    cpa2002_to_2008 = read_correspondence(cpa2002_cpa2008_path, source_tokens=("source", "from", "2002"), target_tokens=("target", "to", "2008"))
    cpa2008_to_21 = read_correspondence(cpa2008_cpa21_path, source_tokens=("source", "from", "2008"), target_tokens=("target", "to", "21", "2.1"))
    labels, division_to_targets = load_target_division_map(target_dictionary)

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
    joined = pd.concat([with_code, without_code], ignore_index=True, sort=False).fillna("")

    diagnostics: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    old_to_current_cache: dict[str, set[str]] = {}
    for row in joined.itertuples(index=False):
        exio3_code = clean_text(getattr(row, "exio3_product_code", ""))
        if not exio3_code:
            diagnostics.append(
                {
                    "diagnostic_kind": "exio20_without_exio3_bridge",
                    "source_code": clean_text(getattr(row, "exio20_product_code", "")),
                    "source_label": clean_text(getattr(row, "exio20_product_name", "")),
                    "message": "ESA59 EXIOBASE2 product did not match the EXIOBASE2->EXIOBASE3 bridge.",
                }
            )
            continue
        old_code = clean_text(getattr(row, "esa59_product_code"))
        if old_code not in old_to_current_cache:
            divisions, messages = compose_old_division_to_cpa21_divisions(cpa2002_to_2008, cpa2008_to_21, old_division=old_code)
            old_to_current_cache[old_code] = divisions
            for message in messages:
                diagnostics.append(
                    {
                        "diagnostic_kind": "old_cpa_to_cpa21_unresolved",
                        "source_code": old_code,
                        "source_label": clean_text(getattr(row, "esa59_product_label")),
                        "message": message,
                    }
                )
        for division in sorted(old_to_current_cache[old_code]):
            targets = division_to_targets.get(division, set())
            if not targets:
                diagnostics.append(
                    {
                        "diagnostic_kind": "cpa21_division_without_target_aggregation",
                        "source_code": old_code,
                        "source_label": clean_text(getattr(row, "esa59_product_label")),
                        "message": f"CPA2.1 division {division} has no target in {target_dictionary}.",
                    }
                )
                continue
            for target_code in sorted(targets):
                edge_rows.append(
                    {
                        "exio3_product_code": exio3_code,
                        "exio3_product_name": clean_text(getattr(row, "exio3_product_name", "")),
                        "exio20_product_code": clean_text(getattr(row, "exio20_product_code", "")) or clean_text(getattr(row, "exio20_product_code_bridge", "")),
                        "exio20_product_name": clean_text(getattr(row, "exio20_product_name", "")) or clean_text(getattr(row, "exio20_product_name_bridge", "")),
                        "esa59_product_code": old_code,
                        "esa59_product_label": clean_text(getattr(row, "esa59_product_label", "")),
                        "cpa21_division": division,
                        "target_code": target_code,
                        "target_label": labels.get(target_code, target_code),
                    }
                )
    edges = pd.DataFrame.from_records(edge_rows)
    if edges.empty:
        mapping = pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    else:
        grouped = cast(
            pd.DataFrame,
            edges.groupby(["exio3_product_code", "exio3_product_name", "target_code", "target_label"], as_index=False).agg(
                esa59_product_codes=("esa59_product_code", lambda values: ";".join(sorted(set(map(str, values))))),
                esa59_product_labels=("esa59_product_label", lambda values: " | ".join(sorted(set(map(str, values))))),
                cpa21_divisions=("cpa21_division", lambda values: ";".join(sorted(set(map(str, values))))),
                exio20_product_codes=("exio20_product_code", lambda values: ";".join(sorted({str(value) for value in values if str(value)}))),
            ),
        )
        target_counts = grouped.groupby("exio3_product_code")["target_code"].transform("nunique").astype(int)
        grouped.loc[:, "n_targets_for_exio3_product"] = target_counts
        rows: list[dict[str, object]] = []
        sorted_grouped = cast(pd.DataFrame, grouped.sort_values(by=["exio3_product_code", "target_code"]))
        grouped_records = cast(list[dict[str, Any]], sorted_grouped.to_dict(orient="records"))
        for order, row in enumerate(grouped_records, start=1):
            allocation_rule = "split-by-output-share" if int(row["n_targets_for_exio3_product"]) > 1 else "direct-single-target"
            esa59_product_codes = clean_text(row.get("esa59_product_codes", ""))
            esa59_product_labels = clean_text(row.get("esa59_product_labels", ""))
            cpa21_divisions = clean_text(row.get("cpa21_divisions", ""))
            exio20_product_codes = clean_text(row.get("exio20_product_codes", ""))
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
                    "target_version": target_version.upper(),
                    "target_level": "CPA_product",
                    "source_code": clean_text(row.get("exio3_product_code", "")),
                    "source_label": clean_text(row.get("exio3_product_name", "")),
                    "target_code": clean_text(row.get("target_code", "")),
                    "target_label": clean_text(row.get("target_label", "")),
                    "enabled": "1",
                    "provenance": "EXIOBASE_ESA59_and_Eurostat_RAMON_official_correspondence_chain",
                    "certainty": "source-concordance-derived",
                    "allocation_rule": allocation_rule,
                    "source_name": "EXIOBASE20p_ESA59p + EXIOBASE2->3 + RAMON CPA correspondences",
                    "official_source": "; ".join(str(path) for path in [esa59_path, exio20_exio30_path, cpa2002_cpa2008_path, cpa2008_cpa21_path]),
                    "notes": (
                        f"ESA59 old product group(s) {esa59_product_codes}: {esa59_product_labels}. "
                        f"Resolved CPA2.1 division(s) {cpa21_divisions}. "
                        f"EXIOBASE2 source code(s): {exio20_product_codes}. "
                        "No manual NSI->CPA table is used; many-target rows are resolved from target-side values at runtime."
                    ),
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
        "cpa2002_cpa2008_path": str(cpa2002_cpa2008_path),
        "cpa2002_cpa2008_sha256": sha256_file(cpa2002_cpa2008_path),
        "cpa2008_cpa21_path": str(cpa2008_cpa21_path),
        "cpa2008_cpa21_sha256": sha256_file(cpa2008_cpa21_path),
        "target_version": target_version,
        "esa59_rows": int(len(esa)),
        "edge_rows": int(len(edges)),
        "mapping_rows": int(len(mapping)),
        "distinct_exio3_products": int(mapping["source_code"].nunique()) if not mapping.empty else 0,
        "distinct_target_codes": int(mapping["target_code"].nunique()) if not mapping.empty else 0,
        "diagnostic_rows": int(len(diagnostics_df)),
    }
    return BuildResult(mapping=mapping, edges=edges, diagnostics=diagnostics_df, coverage=coverage)


def write_outputs(result: BuildResult, *, out_dir: Path, target_version: str) -> dict[str, object]:
    """Write mapping, edge audit, diagnostics, and manifest outputs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"exio_to_eurostat_{target_version.lower()}_products_from_official_concordances"
    mapping_path = out_dir / f"{stem}.csv"
    edges_path = out_dir / f"{stem}_edge_audit.csv"
    diagnostics_path = out_dir / f"{stem}_diagnostics.csv"
    manifest_path = out_dir / f"{stem}_manifest.json"
    result.mapping.to_csv(mapping_path, index=False)
    result.edges.to_csv(edges_path, index=False)
    result.diagnostics.to_csv(diagnostics_path, index=False)
    manifest = {
        "mapping_name": stem,
        "created_utc": datetime.now(UTC).isoformat(),
        "created_by_script": "scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_bridge_nsi.py",
        "method_notes": [
            "No hard-coded NSI->CPA table is used.",
            "EXIOBASE2 products are mapped to ESA59 old product groups from EXIOBASE20p_ESA59p.txt.",
            "Old CPA2002-style divisions are chained through RAMON CPA2002->CPA2008 and CPA2008->CPA2.1 correspondences.",
            "CPA2.1 two-digit divisions are aggregated to the repo CPA64/CPA65 targets; CPA65 splits division 68 when target dictionary contains split real-estate targets.",
        ],
        "coverage": result.coverage,
        "output_files": {
            "mapping": mapping_path.name,
            "edge_audit": edges_path.name,
            "diagnostics": diagnostics_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concordances-zip", type=Path, default=DEFAULT_CONCORDANCES_ZIP)
    parser.add_argument("--esa59-member", default=ESA59_MEMBER)
    parser.add_argument("--exio20-exio30-member", default=EXIO20_EXIO30_MEMBER)
    parser.add_argument("--cpa2002-cpa2008", type=Path, default=DEFAULT_CPA2002_CPA2008)
    parser.add_argument("--cpa2008-cpa21", type=Path, default=DEFAULT_CPA2008_CPA21)
    parser.add_argument("--target-version", choices=["cpa64", "cpa65"], default="cpa65")
    parser.add_argument("--target-dictionary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--allow-diagnostics", action="store_true", help="Write outputs even if source-chain diagnostics are present.")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the official-concordance mapping generator."""

    args = parse_args(argv)
    target_dictionary = args.target_dictionary
    if target_dictionary is None:
        target_dictionary = DEFAULT_CPA65_DICTIONARY if args.target_version == "cpa65" else DEFAULT_CPA64_DICTIONARY

    esa59_bytes, esa59_member = read_zip_member_bytes(args.concordances_zip, args.esa59_member)
    exio_bridge_bytes, exio_bridge_member = read_zip_member_bytes(args.concordances_zip, args.exio20_exio30_member)

    with tempfile.TemporaryDirectory(prefix="exio_cpa_official_sources_") as tmpdir_text:
        tmpdir = Path(tmpdir_text)
        esa59_path = tmpdir / ESA59_SUFFIX
        exio_bridge_path = tmpdir / EXIO20_EXIO30_SUFFIX
        esa59_path.write_text(esa59_bytes.decode("utf-8-sig", errors="replace"), encoding="utf-8")
        exio_bridge_path.write_bytes(exio_bridge_bytes)
        result = build_official_mapping(
            esa59_path=esa59_path,
            exio20_exio30_path=exio_bridge_path,
            cpa2002_cpa2008_path=args.cpa2002_cpa2008,
            cpa2008_cpa21_path=args.cpa2008_cpa21,
            target_dictionary=target_dictionary,
            target_version=args.target_version,
        )

    official_source = "; ".join(
        [
            f"{args.concordances_zip}::{esa59_member}",
            f"{args.concordances_zip}::{exio_bridge_member}",
            str(args.cpa2002_cpa2008),
            str(args.cpa2008_cpa21),
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
        }
    )
    result = BuildResult(mapping=mapping, edges=result.edges, diagnostics=result.diagnostics, coverage=coverage)
    if not args.check_only:
        write_outputs(result, out_dir=args.out_dir, target_version=args.target_version)
    print(json.dumps({"coverage": result.coverage, "out_dir": str(args.out_dir), "wrote_outputs": not args.check_only}, indent=2, sort_keys=True))
    if result.coverage["diagnostic_rows"] and not args.allow_diagnostics:
        raise SystemExit("Official-concordance mapping has diagnostics; inspect diagnostics CSV or rerun with --allow-diagnostics after review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
