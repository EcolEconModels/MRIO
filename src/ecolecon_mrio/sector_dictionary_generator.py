"""Generate sector dictionaries from official metadata sources.

The generator is intentionally conservative.  Eurostat IOT product/activity
labels are read from SDMX metadata, EXIOBASE labels are read from the sector
metadata packaged with ``pymrio``, and committed dictionary rows are retained
when the upstream metadata does not cover an existing code.  This prevents a
metadata refresh from silently dropping locally used mapping keys.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Final

import pandas as pd

from ecolecon_mrio.eurostat_iot import (
    EurostatMetadata,
    load_eurostat_metadata,
)
from ecolecon_mrio.exio_concordance import load_exiobase_metadata


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_DIR: Final[Path] = REPO_ROOT / "scripts" / "mappings"
DEFAULT_OUTPUT_DIR: Final[Path] = DEFAULT_MAPPING_DIR / "generated" / "dictionaries"
DEFAULT_EUROSTAT_STORAGE: Final[Path] = REPO_ROOT / "data" / "eurostat"

EUROSTAT_PRODUCT_DATASET: Final[str] = "naio_10_cp1700"
EUROSTAT_INDUSTRY_DATASET: Final[str] = "naio_10_cp1750"
DEFAULT_EUROSTAT_FILTERS: Final[dict[str, str]] = {
    "geo": "AT",
    "time": "2021",
    "freq": "A",
    "unit": "MIO_EUR",
    "stk_flow": "TOTAL",
}

DICTIONARY_COLUMNS: Final[tuple[str, ...]] = ("code", "label", "description", "provenance_source")
TARGET_DICTIONARY_COLUMNS: Final[tuple[str, ...]] = ("code", "label", "description")

CPA64_OUTPUT_NAME: Final[str] = "eurostat_cpa64_products_dictionary.csv"
CPA65_OUTPUT_NAME: Final[str] = "eurostat_cpa65_products_dictionary.csv"
EUROSTAT_PRODUCTS_OUTPUT_NAME: Final[str] = "eurostat_products_dictionary.csv"
EUROSTAT_INDUSTRIES_OUTPUT_NAME: Final[str] = "eurostat_industries_dictionary.csv"
EXIO_PRODUCTS_OUTPUT_NAME: Final[str] = "exiobase_products_dictionary.csv"
EXIO_INDUSTRIES_OUTPUT_NAME: Final[str] = "exiobase_industries_dictionary.csv"
MANIFEST_OUTPUT_NAME: Final[str] = "sector_dictionary_generation_manifest.json"

VALID_IOT_ACTIVITY_CODES: Final[set[str]] = {
    "A01",
    "A02",
    "A03",
    "B",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "C10-12",
    "C10",
    "C11",
    "C12",
    "C13-15",
    "C13",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
    "C22",
    "C23",
    "C24",
    "C25",
    "C26",
    "C27",
    "C28",
    "C29",
    "C30",
    "C31_32",
    "C31",
    "C32",
    "C33",
    "D",
    "E36",
    "E37-39",
    "E37",
    "E38",
    "E39",
    "F",
    "F41",
    "F42",
    "F43",
    "G45",
    "G46",
    "G47",
    "H49",
    "H50",
    "H51",
    "H52",
    "H53",
    "I",
    "I55",
    "I56",
    "J58",
    "J59_60",
    "J59",
    "J60",
    "J61",
    "J62_63",
    "J62",
    "J63",
    "K64",
    "K65",
    "K66",
    "L68A",
    "L68B",
    "M69_70",
    "M69",
    "M70",
    "M71",
    "M72",
    "M73",
    "M74_75",
    "M74",
    "M75",
    "N77",
    "N78",
    "N79",
    "N80-82",
    "N80",
    "N81",
    "N82",
    "O",
    "P",
    "Q86",
    "Q87_88",
    "Q87",
    "Q88",
    "R90-92",
    "R90",
    "R91",
    "R92",
    "R93",
    "S94",
    "S95",
    "S96",
    "T",
    "T97",
    "T98",
    "U",
}


@dataclass(frozen=True)
class GeneratedDictionaries:
    """Generated dictionary frames and build metadata."""

    frames: dict[str, pd.DataFrame]
    manifest: dict[str, Any]


def _clean_text(value: object) -> str:
    """Collapse whitespace and normalize missing values to an empty string."""

    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return " ".join(str(value).strip().split())


def _read_existing_dictionary(path: Path, *, columns: Sequence[str]) -> pd.DataFrame:
    """Read an existing dictionary or return an empty frame with the requested columns."""

    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    frame = pd.read_csv(path, dtype=str).fillna("")
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame.loc[:, list(columns)].copy()


def _validate_dictionary(frame: pd.DataFrame, *, name: str, columns: Sequence[str]) -> None:
    """Validate core dictionary invariants before writing."""

    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    for column in ("code", "label", "description"):
        if column in columns and frame[column].astype(str).str.strip().eq("").any():
            bad = frame.loc[frame[column].astype(str).str.strip().eq(""), "code"].head(20).tolist()
            raise ValueError(f"{name} has blank {column!r} values for codes: {bad}")
    duplicates = frame["code"].astype(str).str.strip().duplicated(keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, "code"].head(20).tolist()
        raise ValueError(f"{name} has duplicate codes: {sample}")


def _assert_no_existing_codes_dropped(generated: pd.DataFrame, existing: pd.DataFrame, *, name: str) -> None:
    """Fail if a generated dictionary omits any code from the existing dictionary."""

    if existing.empty:
        return
    generated_codes = set(generated["code"].astype(str).str.strip())
    existing_codes = set(existing["code"].astype(str).str.strip())
    missing = sorted(existing_codes - generated_codes)
    if missing:
        raise ValueError(f"{name} would drop existing codes: {missing[:50]}")


def _merge_metadata_with_existing(
    metadata_rows: Sequence[Mapping[str, str]],
    existing: pd.DataFrame,
    *,
    output_columns: Sequence[str],
    preserve_existing_descriptions: bool = True,
) -> pd.DataFrame:
    """Merge official metadata rows with existing rows without dropping local codes."""

    rows_by_code: dict[str, dict[str, str]] = {}
    existing_by_code: dict[str, dict[str, str]] = {}
    for row in existing.to_dict("records"):
        code = _clean_text(row.get("code"))
        if code:
            existing_by_code[code] = {column: _clean_text(row.get(column)) for column in output_columns}

    for row in metadata_rows:
        code = _clean_text(row.get("code"))
        label = _clean_text(row.get("label"))
        if not code or not label:
            continue
        old = existing_by_code.get(code, {})
        description = (
            _clean_text(old.get("description"))
            if preserve_existing_descriptions and _clean_text(old.get("description"))
            else label
        )
        out = {column: "" for column in output_columns}
        out.update(
            {
                "code": code,
                "label": label,
                "description": description,
            }
        )
        if "provenance_source" in output_columns:
            out["provenance_source"] = _clean_text(row.get("provenance_source")) or _clean_text(
                old.get("provenance_source")
            )
        rows_by_code[code] = out

    for code, row in existing_by_code.items():
        if code in rows_by_code:
            continue
        retained = {column: _clean_text(row.get(column)) for column in output_columns}
        if "provenance_source" in output_columns:
            prior = retained.get("provenance_source", "")
            retained["provenance_source"] = (
                f"{prior}; retained_existing_dictionary_row_not_in_current_metadata"
                if prior
                else "retained_existing_dictionary_row_not_in_current_metadata"
            )
        rows_by_code[code] = retained

    existing_order = [_clean_text(code) for code in existing.get("code", pd.Series(dtype=str)).tolist()]
    metadata_order = [_clean_text(row.get("code")) for row in metadata_rows]
    ordered_codes = list(dict.fromkeys([code for code in existing_order + metadata_order if code]))
    out = pd.DataFrame([rows_by_code[code] for code in ordered_codes], columns=list(output_columns))
    return out.reset_index(drop=True)


def _ordered_labels(labels: Mapping[str, str], order: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """Return metadata labels in SDMX category order when an index is available."""

    if order:
        return [(str(code), str(labels.get(code, code))) for code in order if str(code) in labels]
    return [(str(code), str(label)) for code, label in labels.items()]


def _dimension_label_pairs(metadata: EurostatMetadata, dimension: str) -> list[tuple[str, str]]:
    """Extract ordered code-label pairs for one Eurostat dimension."""

    labels = metadata.dimensions.get(dimension, {})
    order = metadata.dimension_order.get(dimension, [])
    return _ordered_labels(labels, order)


def _load_metadata(
    dataset: str,
    *,
    filters: Mapping[str, Any],
    storage_folder: Path,
    force_download: bool,
) -> tuple[EurostatMetadata, dict[str, Any]]:
    """Download or read cached Eurostat metadata through the Eurostat API wrapper."""

    metadata = load_eurostat_metadata(
        dataset,
        filters=filters,
        storage_folder=storage_folder,
        force_download=force_download,
    )
    request_meta = {
        "dataset": dataset,
        "filters": {key: list(value) for key, value in metadata.filters.items()},
        "request_url": metadata.request_url,
        "raw_path": str(metadata.raw_path),
    }
    return metadata, request_meta


def _eurostat_rows_from_dimensions(
    metadata: EurostatMetadata,
    *,
    dimensions: Sequence[str],
    provenance_prefix: str,
    code_transform: str,
    code_kind: str,
) -> list[dict[str, str]]:
    """Build dictionary rows from one or more Eurostat dimensions."""

    seen: dict[str, dict[str, str]] = {}
    for dimension in dimensions:
        for code, label in _dimension_label_pairs(metadata, dimension):
            code_s = _clean_text(code)
            label_s = _clean_text(label)
            if not code_s or not label_s:
                continue
            if code_kind == "cpa" and not code_s.startswith("CPA_"):
                continue
            if code_kind == "nace" and code_s not in VALID_IOT_ACTIVITY_CODES:
                continue
            if code_transform == "strip_cpa_prefix" and code_s.startswith("CPA_"):
                code_out = code_s.removeprefix("CPA_").replace("_", "-")
            else:
                code_out = code_s
            if code_out in {"TOTAL", "CPA_TOTAL"}:
                continue
            seen.setdefault(
                code_out,
                {
                    "code": code_out,
                    "label": label_s,
                    "description": label_s,
                    "provenance_source": f"{provenance_prefix}:{dimension}",
                },
            )
    return list(seen.values())


def _description_from_cpa_code(code: str) -> str:
    """Return a parseable CPA target description from an IOT aggregate code."""

    if code == "CPA_L68A":
        return "CPA_65 real-estate split of A*64 L68; imputed rents of owner-occupied dwellings."
    if code == "CPA_L68B":
        return "CPA_65 real-estate split of A*64 L68; real estate services excluding imputed rents."
    compact = code.removeprefix("CPA_")
    if compact == "B":
        divisions = "05-09"
    elif compact == "D":
        divisions = "35"
    elif compact == "I":
        divisions = "55-56"
    elif compact == "O":
        divisions = "84"
    elif compact == "P":
        divisions = "85"
    elif compact == "T":
        divisions = "97-98"
    elif compact == "U":
        divisions = "99"
    elif compact == "L68":
        divisions = "68"
    else:
        match = re.search(r"(\d{2}(?:[-_]\d{2})?)", compact)
        divisions = match.group(1).replace("_", "-") if match else compact
    return f"Eurostat IOT target {compact}; NACE/CPA division {divisions}."


def _build_cpa_target_dictionary(
    metadata: EurostatMetadata,
    *,
    existing: pd.DataFrame,
    split_real_estate: bool,
) -> pd.DataFrame:
    """Build CPA64 or CPA65 target dictionary rows from Eurostat product metadata."""

    label_by_code: dict[str, str] = {}
    for dimension in ("prd_ava", "prd_use"):
        for code, label in _dimension_label_pairs(metadata, dimension):
            if code.startswith("CPA_"):
                label_by_code.setdefault(code, label)

    existing_codes = [_clean_text(code) for code in existing.get("code", pd.Series(dtype=str)).tolist()]
    rows: list[dict[str, str]] = []
    for code in existing_codes:
        if not code:
            continue
        if code == "CPA_L68" and split_real_estate:
            continue
        if code in {"CPA_L68A", "CPA_L68B"}:
            label = label_by_code.get(code) or _clean_text(
                existing.loc[existing["code"].astype(str).str.strip().eq(code), "label"].iloc[0]
            )
        elif code == "CPA_L68":
            label = _clean_text(existing.loc[existing["code"].astype(str).str.strip().eq(code), "label"].iloc[0])
        else:
            label = label_by_code.get(code) or _clean_text(
                existing.loc[existing["code"].astype(str).str.strip().eq(code), "label"].iloc[0]
            )
        rows.append({"code": code, "label": _clean_text(label), "description": _description_from_cpa_code(code)})

    if split_real_estate:
        codes = [row["code"] for row in rows]
        if "CPA_L68A" not in codes:
            rows.append(
                {
                    "code": "CPA_L68A",
                    "label": label_by_code.get("CPA_L68A", "Imputed rents of owner-occupied dwellings"),
                    "description": _description_from_cpa_code("CPA_L68A"),
                }
            )
        if "CPA_L68B" not in codes:
            rows.append(
                {
                    "code": "CPA_L68B",
                    "label": label_by_code.get("CPA_L68B", "Real estate services excluding imputed rents"),
                    "description": _description_from_cpa_code("CPA_L68B"),
                }
            )

    out = pd.DataFrame(rows, columns=list(TARGET_DICTIONARY_COLUMNS))
    return out.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)


def _build_exio_dictionary(*, system: str, existing: pd.DataFrame, exio_year: int | None = None) -> pd.DataFrame:
    """Build an EXIOBASE product or industry dictionary from EXIO metadata."""

    metadata = load_exiobase_metadata(system=system, year=exio_year, data_root=REPO_ROOT / "data")
    sectors = metadata.sectors
    rows: list[dict[str, str]] = []
    source_label = metadata.source
    for row in sectors.itertuples(index=False):
        code = _clean_text(getattr(row, "ExioCode", ""))
        label = _clean_text(getattr(row, "ExioName", ""))
        if not code or not label:
            continue
        rows.append(
            {
                "code": code,
                "label": label,
                "description": label,
                "provenance_source": source_label,
            }
        )
    return _merge_metadata_with_existing(rows, existing, output_columns=DICTIONARY_COLUMNS)


def build_sector_dictionaries(
    *,
    existing_dir: Path = DEFAULT_MAPPING_DIR,
    eurostat_storage: Path = DEFAULT_EUROSTAT_STORAGE,
    eurostat_filters: Mapping[str, Any] | None = None,
    force_download: bool = False,
    exio_year: int | None = None,
) -> GeneratedDictionaries:
    """Build all sector dictionaries while preserving existing committed codes."""

    filters = dict(DEFAULT_EUROSTAT_FILTERS)
    if eurostat_filters:
        filters.update({str(key): value for key, value in eurostat_filters.items()})

    product_metadata, product_request = _load_metadata(
        EUROSTAT_PRODUCT_DATASET,
        filters=filters,
        storage_folder=eurostat_storage,
        force_download=force_download,
    )
    industry_metadata, industry_request = _load_metadata(
        EUROSTAT_INDUSTRY_DATASET,
        filters=filters,
        storage_folder=eurostat_storage,
        force_download=force_download,
    )

    existing_products = _read_existing_dictionary(existing_dir / EUROSTAT_PRODUCTS_OUTPUT_NAME, columns=DICTIONARY_COLUMNS)
    existing_industries = _read_existing_dictionary(
        existing_dir / EUROSTAT_INDUSTRIES_OUTPUT_NAME, columns=DICTIONARY_COLUMNS
    )
    existing_cpa64 = _read_existing_dictionary(existing_dir / CPA64_OUTPUT_NAME, columns=TARGET_DICTIONARY_COLUMNS)
    existing_cpa65 = _read_existing_dictionary(existing_dir / CPA65_OUTPUT_NAME, columns=TARGET_DICTIONARY_COLUMNS)
    existing_exio_products = _read_existing_dictionary(existing_dir / EXIO_PRODUCTS_OUTPUT_NAME, columns=DICTIONARY_COLUMNS)
    existing_exio_industries = _read_existing_dictionary(
        existing_dir / EXIO_INDUSTRIES_OUTPUT_NAME, columns=DICTIONARY_COLUMNS
    )

    eurostat_product_rows = _eurostat_rows_from_dimensions(
        product_metadata,
        dimensions=("prd_ava", "prd_use"),
        provenance_prefix=f"Eurostat SDMX metadata:{EUROSTAT_PRODUCT_DATASET}",
        code_transform="strip_cpa_prefix",
        code_kind="cpa",
    )
    eurostat_industry_rows = _eurostat_rows_from_dimensions(
        industry_metadata,
        dimensions=("ind_ava", "ind_use"),
        provenance_prefix=f"Eurostat SDMX metadata:{EUROSTAT_INDUSTRY_DATASET}",
        code_transform="none",
        code_kind="nace",
    )

    frames = {
        EUROSTAT_PRODUCTS_OUTPUT_NAME: _merge_metadata_with_existing(
            eurostat_product_rows,
            existing_products,
            output_columns=DICTIONARY_COLUMNS,
        ),
        EUROSTAT_INDUSTRIES_OUTPUT_NAME: _merge_metadata_with_existing(
            eurostat_industry_rows,
            existing_industries,
            output_columns=DICTIONARY_COLUMNS,
        ),
        CPA64_OUTPUT_NAME: _build_cpa_target_dictionary(
            product_metadata,
            existing=existing_cpa64,
            split_real_estate=False,
        ),
        CPA65_OUTPUT_NAME: _build_cpa_target_dictionary(
            product_metadata,
            existing=existing_cpa65,
            split_real_estate=True,
        ),
        EXIO_PRODUCTS_OUTPUT_NAME: _build_exio_dictionary(
            system="pxp",
            existing=existing_exio_products,
            exio_year=exio_year,
        ),
        EXIO_INDUSTRIES_OUTPUT_NAME: _build_exio_dictionary(
            system="ixi",
            existing=existing_exio_industries,
            exio_year=exio_year,
        ),
    }

    existing_frames = {
        EUROSTAT_PRODUCTS_OUTPUT_NAME: existing_products,
        EUROSTAT_INDUSTRIES_OUTPUT_NAME: existing_industries,
        CPA64_OUTPUT_NAME: existing_cpa64,
        CPA65_OUTPUT_NAME: existing_cpa65,
        EXIO_PRODUCTS_OUTPUT_NAME: existing_exio_products,
        EXIO_INDUSTRIES_OUTPUT_NAME: existing_exio_industries,
    }
    for name, frame in frames.items():
        columns = TARGET_DICTIONARY_COLUMNS if name in {CPA64_OUTPUT_NAME, CPA65_OUTPUT_NAME} else DICTIONARY_COLUMNS
        _validate_dictionary(frame, name=name, columns=columns)
        _assert_no_existing_codes_dropped(frame, existing_frames[name], name=name)

    manifest: dict[str, Any] = {
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "generator": "ecolecon_mrio.sector_dictionary_generator",
        "eurostat_requests": {
            "products": product_request,
            "industries": industry_request,
        },
        "exiobase_sources": {
            "loader": "ecolecon_mrio.exio_concordance.load_exiobase_metadata",
            "year": exio_year,
            "products": str(resources.files("pymrio.mrio_models.exio3_pxp").joinpath("sectors.tsv")),
            "industries": str(resources.files("pymrio.mrio_models.exio3_ixi").joinpath("sectors.tsv")),
        },
        "outputs": {
            name: {
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "existing_rows": int(len(existing_frames[name])),
            }
            for name, frame in frames.items()
        },
        "code_preservation": "Existing committed dictionary codes are retained when absent from current upstream metadata.",
    }
    return GeneratedDictionaries(frames=frames, manifest=manifest)


def write_sector_dictionaries(
    *,
    out_dir: Path = DEFAULT_OUTPUT_DIR,
    existing_dir: Path = DEFAULT_MAPPING_DIR,
    eurostat_storage: Path = DEFAULT_EUROSTAT_STORAGE,
    eurostat_filters: Mapping[str, Any] | None = None,
    force_download: bool = False,
    exio_year: int | None = None,
) -> GeneratedDictionaries:
    """Build and write sector dictionaries plus a manifest."""

    result = build_sector_dictionaries(
        existing_dir=existing_dir,
        eurostat_storage=eurostat_storage,
        eurostat_filters=eurostat_filters,
        force_download=force_download,
        exio_year=exio_year,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in result.frames.items():
        frame.to_csv(out_dir / name, index=False)
    (out_dir / MANIFEST_OUTPUT_NAME).write_text(json.dumps(result.manifest, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_filter_pairs(values: Sequence[str]) -> dict[str, str]:
    """Parse ``key=value`` command-line filter overrides."""

    out: dict[str, str] = {}
    for value in values:
        key, sep, raw = value.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"Expected Eurostat filter override as key=value, got: {value!r}")
        out[key.strip()] = raw.strip()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description="Generate Eurostat and EXIOBASE sector dictionaries.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--existing-dir", type=Path, default=DEFAULT_MAPPING_DIR)
    parser.add_argument("--eurostat-storage", type=Path, default=DEFAULT_EUROSTAT_STORAGE)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--exio-year",
        type=int,
        default=None,
        help="Optional EXIOBASE year whose source zip is used to verify sector names when available.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Override a Eurostat request filter as key=value. May be passed multiple times.",
    )
    args = parser.parse_args(argv)
    result = write_sector_dictionaries(
        out_dir=args.out_dir,
        existing_dir=args.existing_dir,
        eurostat_storage=args.eurostat_storage,
        eurostat_filters=parse_filter_pairs(args.filter),
        force_download=args.force_download,
        exio_year=args.exio_year,
    )
    for name, frame in result.frames.items():
        print(f"Wrote {len(frame)} rows: {args.out_dir / name}")
    print(f"Wrote manifest: {args.out_dir / MANIFEST_OUTPUT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
