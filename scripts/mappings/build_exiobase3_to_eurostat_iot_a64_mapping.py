"""Build an EXIOBASE3-products to Eurostat IOT A*64 concordance.

The script reads a local EXIOBASE concordance archive supplied by the user. The
archive is not committed to git.

Mapping chain:
    NACE Rev. 2 class -> EXIOBASE2 product -> EXIOBASE3 product -> IOT A*64.

The output is an unweighted relational concordance. Equal split weights are
written only as explicit placeholders for downstream users who choose that
assumption; they are not empirical allocation weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

NACE_EXIO20_SUFFIX = "NACE2_EXIOBASE20p_list.csv"
EXIO20_EXIO30_SUFFIX = "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"

A64_ROWS: tuple[tuple[str, str, str], ...] = (
    ("A01", "Products of crop and animal production, hunting and related services", "01"),
    ("A02", "Products of forestry and logging", "02"),
    ("A03", "Fish and other fishing products; aquaculture products; support services to fishing", "03"),
    ("B", "Mining and quarrying products", "05-09"),
    ("C10-C12", "Food products, beverages and tobacco products", "10-12"),
    ("C13-C15", "Textiles, wearing apparel, leather and related products", "13-15"),
    ("C16", "Wood and products of wood and cork, except furniture; articles of straw and plaiting materials", "16"),
    ("C17", "Paper and paper products", "17"),
    ("C18", "Printing and reproduction services of recorded media", "18"),
    ("C19", "Coke and refined petroleum products", "19"),
    ("C20", "Chemicals and chemical products", "20"),
    ("C21", "Basic pharmaceutical products and pharmaceutical preparations", "21"),
    ("C22", "Rubber and plastic products", "22"),
    ("C23", "Other non-metallic mineral products", "23"),
    ("C24", "Basic metals", "24"),
    ("C25", "Fabricated metal products, except machinery and equipment", "25"),
    ("C26", "Computer, electronic and optical products", "26"),
    ("C27", "Electrical equipment", "27"),
    ("C28", "Machinery and equipment n.e.c.", "28"),
    ("C29", "Motor vehicles, trailers and semi-trailers", "29"),
    ("C30", "Other transport equipment", "30"),
    ("C31-C32", "Furniture; other manufactured goods", "31-32"),
    ("C33", "Repair and installation services of machinery and equipment", "33"),
    ("D35", "Electricity, gas, steam and air-conditioning", "35"),
    ("E36", "Natural water; water treatment and supply services", "36"),
    ("E37-E39", "Sewerage; waste collection, treatment and disposal; remediation and other waste management services", "37-39"),
    ("F", "Constructions and construction works", "41-43"),
    ("G45", "Wholesale and retail trade and repair services of motor vehicles and motorcycles", "45"),
    ("G46", "Wholesale trade services, except of motor vehicles and motorcycles", "46"),
    ("G47", "Retail trade services, except of motor vehicles and motorcycles", "47"),
    ("H49", "Land transport services and transport services via pipelines", "49"),
    ("H50", "Water transport services", "50"),
    ("H51", "Air transport services", "51"),
    ("H52", "Warehousing and support services for transportation", "52"),
    ("H53", "Postal and courier services", "53"),
    ("I", "Accommodation and food services", "55-56"),
    ("J58", "Publishing services", "58"),
    ("J59-J60", "Motion picture, video, television programme production, sound recording and music publishing; programming and broadcasting services", "59-60"),
    ("J61", "Telecommunications services", "61"),
    ("J62-J63", "Computer programming, consultancy and related services; information services", "62-63"),
    ("K64", "Financial services, except insurance and pension funding", "64"),
    ("K65", "Insurance, reinsurance and pension funding services, except compulsory social security", "65"),
    ("K66", "Services auxiliary to financial services and insurance services", "66"),
    ("L68", "Real estate services", "68"),
    ("M69-M70", "Legal and accounting services; services of head offices; management consulting services", "69-70"),
    ("M71", "Architectural and engineering services; technical testing and analysis services", "71"),
    ("M72", "Scientific research and development services", "72"),
    ("M73", "Advertising and market research services", "73"),
    ("M74-M75", "Other professional, scientific and technical services; veterinary services", "74-75"),
    ("N77", "Rental and leasing services", "77"),
    ("N78", "Employment services", "78"),
    ("N79", "Travel agency, tour operator and other reservation services and related services", "79"),
    ("N80-N82", "Security and investigation; services to buildings and landscape; office administrative and support services", "80-82"),
    ("O84", "Public administration and defence services; compulsory social security services", "84"),
    ("P85", "Education services", "85"),
    ("Q86", "Human health services", "86"),
    ("Q87-Q88", "Residential care services; social work services without accommodation", "87-88"),
    ("R90-R92", "Creative, arts and entertainment; libraries, archives, museums and other cultural; gambling and betting services", "90-92"),
    ("R93", "Sporting services and amusement and recreation services", "93"),
    ("S94", "Services furnished by membership organisations", "94"),
    ("S95", "Repair services of computers and personal and household goods", "95"),
    ("S96", "Other personal services", "96"),
    ("T", "Services of households as employers; undifferentiated goods and services produced by households for own use", "97-98"),
    ("U", "Services provided by extraterritorial organisations and bodies", "99"),
)


@dataclass(frozen=True)
class A64Category:
    code: str
    label: str
    divisions: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def a64_categories() -> list[A64Category]:
    return [A64Category(*row) for row in A64_ROWS]


def expand_divisions(spec: str) -> set[str]:
    values: set[str] = set()
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            values.update(f"{i:02d}" for i in range(start, end + 1))
        else:
            values.add(f"{int(part):02d}")
    return values


def division_to_a64_lookup() -> dict[str, A64Category]:
    lookup: dict[str, A64Category] = {}
    for category in a64_categories():
        for division in expand_divisions(category.divisions):
            if division in lookup:
                raise ValueError(f"Division {division} assigned more than once")
            lookup[division] = category
    return lookup


def nace2_to_division(nace2_code: object) -> str:
    text = str(nace2_code).strip()
    match = re.match(r"^(\d{1,2})(?:\.|$)", text)
    if not match:
        raise ValueError(f"Cannot infer NACE division from code {text!r}")
    return f"{int(match.group(1)):02d}"


def find_member_by_suffix(zf: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in zf.namelist() if name.replace("\\", "/").endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"No zip member ending with {suffix!r}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple zip members ending with {suffix!r}: {matches}")
    return matches[0]


def read_nace_to_exio20(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as handle:
        df = pd.read_csv(handle, dtype=str, encoding="utf-8-sig")
    required = {"NACE2", "EXIOBASE name", "EXIOBASE code"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{member} missing columns: {sorted(missing)}")
    out = df.rename(
        columns={
            "NACE2": "nace2_code",
            "EXIOBASE name": "source_exio20_product_name",
            "EXIOBASE code": "source_exio_code",
        }
    ).copy()
    out["source_exio_code"] = out["source_exio_code"].astype(str).str.strip()
    out["nace2_code"] = out["nace2_code"].astype(str).str.strip()
    out["nace2_division"] = out["nace2_code"].map(nace2_to_division)
    return out


def read_exio20_to_exio30_bridge(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    data = zf.read(member)
    df = pd.read_excel(io.BytesIO(data), dtype=str)
    required = {
        "EXIOBASE 2 Name",
        "EXIOBASE 2 code",
        "EXIOBASE 3 name",
        "EXIOBASE 3 code",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{member} missing columns: {sorted(missing)}")
    base = df.rename(
        columns={
            "EXIOBASE 2 Name": "exio20_product_name",
            "EXIOBASE 2 code": "exio20_product_code",
            "EXIOBASE 3 name": "exio3_product_name",
            "EXIOBASE 3 code": "exio3_product_code",
        }
    )[
        ["exio20_product_name", "exio20_product_code", "exio3_product_name", "exio3_product_code"]
    ].copy()
    for col in base.columns:
        base[col] = base[col].astype(str).str.strip()

    records: list[dict[str, str]] = []
    for rec_obj in base.to_dict(orient="records"):
        rec = {str(k): str(v) for k, v in rec_obj.items()}
        records.append({"source_exio_code": rec["exio20_product_code"], "bridge_key_kind": "exio20_code", **rec})
        if rec["exio3_product_code"] != rec["exio20_product_code"]:
            records.append({"source_exio_code": rec["exio3_product_code"], "bridge_key_kind": "exio3_code_alias", **rec})
    return pd.DataFrame.from_records(records).drop_duplicates()


def build_mapping(concordances_zip: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not concordances_zip.exists():
        raise FileNotFoundError(f"Missing concordance archive: {concordances_zip}")

    with zipfile.ZipFile(concordances_zip) as zf:
        nace_member = find_member_by_suffix(zf, NACE_EXIO20_SUFFIX)
        bridge_member = find_member_by_suffix(zf, EXIO20_EXIO30_SUFFIX)
        nace_bytes = zf.read(nace_member)
        bridge_bytes = zf.read(bridge_member)
        nace = read_nace_to_exio20(zf, nace_member)
        bridge = read_exio20_to_exio30_bridge(zf, bridge_member)

    a64_lookup = division_to_a64_lookup()
    merged = nace.merge(bridge, on="source_exio_code", how="left", validate="many_to_many")
    unmatched = merged[merged["exio3_product_code"].isna()]["source_exio_code"].dropna().unique().tolist()
    if unmatched:
        raise RuntimeError(f"Unmatched EXIOBASE product codes after bridge: {sorted(unmatched)}")

    merged["iot_a64_code"] = merged["nace2_division"].map(lambda d: a64_lookup[str(d)].code)
    merged["iot_a64_label"] = merged["nace2_division"].map(lambda d: a64_lookup[str(d)].label)
    merged["iot_a64_divisions"] = merged["nace2_division"].map(lambda d: a64_lookup[str(d)].divisions)
    merged["source_chain"] = "NACE2_EXIOBASE20p_list.csv -> EXIOBASE2/3 product comparison -> A64 division rule"
    long_cols = [
        "nace2_code",
        "nace2_division",
        "source_exio_code",
        "source_exio20_product_name",
        "bridge_key_kind",
        "exio20_product_code",
        "exio20_product_name",
        "exio3_product_code",
        "exio3_product_name",
        "iot_a64_code",
        "iot_a64_label",
        "iot_a64_divisions",
        "source_chain",
    ]
    long_df = merged[long_cols].drop_duplicates().sort_values(
        ["exio3_product_code", "iot_a64_code", "nace2_code", "source_exio_code"]
    )

    grouped = long_df.groupby(
        ["exio3_product_code", "exio3_product_name", "iot_a64_code", "iot_a64_label", "iot_a64_divisions"],
        dropna=False,
        sort=True,
    )
    collapsed = grouped.agg(
        n_source_nace2_codes=("nace2_code", "nunique"),
        source_nace2_codes=("nace2_code", lambda x: ";".join(sorted(set(map(str, x))))),
        source_nace2_divisions=("nace2_division", lambda x: ";".join(sorted(set(map(str, x))))),
        source_exio_codes=("source_exio_code", lambda x: ";".join(sorted(set(map(str, x))))),
        bridge_key_kinds=("bridge_key_kind", lambda x: ";".join(sorted(set(map(str, x))))),
    ).reset_index()

    target_counts = collapsed.groupby("exio3_product_code")["iot_a64_code"].transform("nunique")
    collapsed["n_iot_a64_targets_for_exio3_product"] = target_counts.astype(int)
    collapsed["is_multi_target"] = collapsed["n_iot_a64_targets_for_exio3_product"] > 1
    collapsed["equal_split_weight_if_required"] = 1.0 / collapsed["n_iot_a64_targets_for_exio3_product"]
    collapsed["weight_status"] = "placeholder_equal_split_not_empirical"
    collapsed = collapsed.sort_values(["exio3_product_code", "iot_a64_code"]).reset_index(drop=True)

    coverage = {
        "source_archive_sha256": sha256_file(concordances_zip),
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
        "source_nace_exio_rows": int(len(nace)),
        "long_link_rows": int(len(long_df)),
        "collapsed_rows": int(len(collapsed)),
        "distinct_source_nace2_codes": int(long_df["nace2_code"].nunique()),
        "distinct_exio3_products": int(collapsed["exio3_product_code"].nunique()),
        "distinct_iot_a64_categories": int(collapsed["iot_a64_code"].nunique()),
        "multi_target_exio3_products": int(collapsed.loc[collapsed["is_multi_target"], "exio3_product_code"].nunique()),
    }
    return long_df, collapsed, coverage


def write_a64_codelist(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iot_a64_code", "iot_a64_label", "nace_cpa_divisions"])
        writer.writerows(A64_ROWS)


def write_outputs(
    *,
    concordances_zip: Path,
    out_dir: Path,
    long_df: pd.DataFrame,
    collapsed: pd.DataFrame,
    coverage: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "exiobase3_products_to_eurostat_iot_a64_long_links.csv"
    collapsed_path = out_dir / "exiobase3_products_to_eurostat_iot_a64_collapsed.csv"
    codelist_path = out_dir / "eurostat_iot_a64_codelist_used.csv"
    manifest_path = out_dir / "manifest.json"
    readme_path = out_dir / "README.md"

    long_df.to_csv(long_path, index=False)
    collapsed.to_csv(collapsed_path, index=False)
    write_a64_codelist(codelist_path)

    manifest = {
        "mapping_name": "exiobase3_products_to_eurostat_iot_a64",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_by_script": "scripts/mappings/build_exiobase3_to_eurostat_iot_a64_mapping.py",
        "source_archive": {
            "path": str(concordances_zip),
            "sha256": coverage["source_archive_sha256"],
            "local_filename_expected": "data/exiobase_concordance_sources/exiobase_concordances.zip",
        },
        "source_files": coverage["source_files"],
        "target_classification": {
            "name": "Eurostat ESA/IOT A*64 product/activity aggregate",
            "basis": "NACE Rev.2 division ranges",
            "num_categories": len(A64_ROWS),
            "real_estate_imputed_rent_split": "not inferred; L68 remains one category",
        },
        "method_notes": [
            "NACE Rev.2 division is inferred from NACE2_EXIOBASE20p_list.csv.",
            "A64 category is assigned from NACE Rev.2 division ranges.",
            "Links are unweighted relational concordance links.",
            "Equal split weights are placeholders only and are not empirical allocation shares.",
            "The 65-sector real-estate/imputed-rent split is not generated.",
        ],
        "coverage": {k: v for k, v in coverage.items() if k != "source_files"},
        "output_files": {
            "long_links": long_path.name,
            "collapsed": collapsed_path.name,
            "a64_codelist": codelist_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme = """# EXIOBASE3 products to Eurostat IOT A*64 mapping

This directory contains generated concordance outputs. The raw EXIOBASE
concordance archive is not redistributed in this repository.

## Mapping chain

```text
NACE Rev. 2 class -> EXIOBASE2 product -> EXIOBASE3 product -> Eurostat IOT A*64
```

## Weighting

The mapping is an unweighted relational concordance. `equal_split_weight_if_required`
is a placeholder only and should not be treated as an empirical allocation share.

## 64 vs 65 sector target

The output targets the 64-sector A*64 aggregate. It does not split L68 real
estate services into actual-rent and imputed-rent categories.
"""
    readme_path.write_text(readme, encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concordances-zip",
        type=Path,
        default=Path("data/exiobase_concordance_sources/exiobase_concordances.zip"),
        help="Path to the local EXIOBASE concordance archive renamed to exiobase_concordances.zip.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/derived/concordances/exiobase_eurostat_iot_a64"),
        help="Output directory for generated mapping files.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build in memory and print coverage without writing outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    long_df, collapsed, coverage = build_mapping(args.concordances_zip)
    if not args.check_only:
        write_outputs(
            concordances_zip=args.concordances_zip,
            out_dir=args.out_dir,
            long_df=long_df,
            collapsed=collapsed,
            coverage=coverage,
        )
    print(json.dumps({"coverage": coverage, "out_dir": str(args.out_dir), "wrote_outputs": not args.check_only}, indent=2))
    if coverage["distinct_exio3_products"] != 200:
        raise RuntimeError(f"Expected 200 EXIOBASE3 products, got {coverage['distinct_exio3_products']}")
    if coverage["distinct_iot_a64_categories"] != 64:
        raise RuntimeError(f"Expected 64 A64 target categories, got {coverage['distinct_iot_a64_categories']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
