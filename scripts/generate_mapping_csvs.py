#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ecolecon_mrio.eurostat_iot import build_eurostat_product_use_reference, load_eurostat_iot
from ecolecon_mrio.exio_concordance import load_pymrio_exio3_sectors
from ecolecon_mrio.exio_to_eurostat_mapping import (
    _candidate_nace2_divisions_for_exio,
    build_cpa_division_index,
    build_exio_to_eurostat_mapping,
    candidate_cpa_codes_for_exio,
    filter_cpa_candidates_for_exio,
)
from ecolecon_mrio.mapping_csv import MAPPING_CSV_COLUMNS, read_mapping_csv


EXIO_CLASSIFICATIONS_XLSX = Path("data/exiobase_classifications/Classifications_v_3_3_18.xlsx")
WEIGHTED_OFFICIAL_SOURCE = (
    "https://acpr.banque-france.fr/system/files/2025-03/"
    "table_correspondance_nace-gics-exiobase_20230816.xlsx"
)
FALLBACK_OFFICIAL_SOURCE = (
    "https://raw.githubusercontent.com/eurostat/correspondenceTables/main/inst/extdata/NACE.csv | "
    "https://raw.githubusercontent.com/eurostat/correspondenceTables/main/inst/extdata/NACE_CPA21.csv | "
    "https://circabc.europa.eu/sd/a/acbc9240-03b9-43df-97b5-146d102fc18e/"
    "NACE%20Rev.%201.1%20-%20NACE%20Rev.%202%20-%20short%20version.pdf"
)
ENERGY_METADATA_SOURCE = "https://ec.europa.eu/eurostat/cache/metadata/en/nrg_bal_esms.htm"


def _find_any_eurostat_meta() -> dict[str, str]:
    roots = sorted(Path("data/eurostat/naio_10_cp1700").glob("*/parsed/meta.json"))
    if not roots:
        raise FileNotFoundError("Could not find a cached Eurostat meta.json under data/eurostat/naio_10_cp1700")
    with roots[0].open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if "dimensions" in meta and "prd_ava" in meta["dimensions"]:
        labels = meta["dimensions"]["prd_ava"]
    else:
        labels = meta["dimension"]["prd_ava"]["category"]["label"]
    return {str(code): str(label) for code, label in labels.items()}


def _load_exio_lookup() -> dict[str, dict[str, str]]:
    df = pd.read_excel(EXIO_CLASSIFICATIONS_XLSX, sheet_name="Products_HIOT")
    df = df.rename(columns={"Product name": "name", "Product code 1": "code", "Product code 2": "label"})
    df = df[["name", "code", "label"]].dropna()
    df["name"] = df["name"].astype(str).str.strip()
    df["code"] = df["code"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["name"] != ""].drop_duplicates(subset=["name"], keep="first")
    return {
        str(row.name): {"source_code": str(row.code), "source_label": str(row.name), "source_classification": str(row.label)}
        for row in df.itertuples(index=False)
    }


def _normalized_lookup(labels: dict[str, str]) -> dict[str, str]:
    return {" ".join(value.strip().casefold().split()): code for code, value in labels.items()}


def _match_target_code(target_label: str, cpa_labels: dict[str, str]) -> str:
    normalized = _normalized_lookup(cpa_labels)
    return normalized.get(" ".join(str(target_label).strip().casefold().split()), "")


def _write_mapping_csv(path: Path, frame: pd.DataFrame) -> None:
    extras = [column for column in frame.columns if column not in MAPPING_CSV_COLUMNS]
    ordered = list(MAPPING_CSV_COLUMNS) + extras
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, ordered].to_csv(path, index=False)


def _weights_for_candidates(candidates: list[str], x_series: pd.Series | None) -> dict[str, float]:
    if not candidates:
        return {}
    if x_series is None:
        w = 1.0 / len(candidates)
        return {candidate: w for candidate in candidates}
    values = [float(x_series.get(candidate, 0.0)) for candidate in candidates]
    values = [value if value > 0 else 0.0 for value in values]
    total = float(sum(values))
    if total <= 0:
        w = 1.0 / len(candidates)
        return {candidate: w for candidate in candidates}
    return {candidate: value / total for candidate, value in zip(candidates, values)}


def _generate_weighted_rows(*, geo: str, year: int, eurostat_storage_folder: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    parsed = load_eurostat_iot(
        dataset="naio_10_cp1700",
        filters={"geo": str(geo), "time": str(year), "freq": "A", "unit": "MIO_EUR", "stk_flow": "TOTAL"},
        row_dim="prd_ava",
        col_dim="prd_use",
        storage_folder=eurostat_storage_folder,
        force_download=False,
        force_reparse=False,
    )
    z = parsed.Z
    x = build_eurostat_product_use_reference(parsed)
    cpa_codes = list(map(str, z.index.astype(str)))
    cpa_labels = {str(code): str(label) for code, label in ((parsed.meta.get("dimensions", {}) or {}).get("prd_ava", {}) or {}).items()}

    exio_sectors = load_pymrio_exio3_sectors("pxp").sectors
    valid_exio = exio_sectors["ExioName"].astype(str).str.strip().tolist()
    exio_code_map = dict(
        zip(
            exio_sectors["ExioName"].astype(str).str.strip(),
            exio_sectors["ExioCode"].astype(str).str.strip(),
        )
    )

    base_df = build_exio_to_eurostat_mapping(
        valid_exio,
        cpa_codes,
        {code: cpa_labels.get(code, code) for code in cpa_codes},
        exio_code_map=exio_code_map,
    ).copy()
    if "weight" not in base_df.columns:
        base_df["weight"] = 1.0

    x_series = pd.to_numeric(x.astype("float64"), errors="coerce").fillna(0.0)
    division_index = build_cpa_division_index(cpa_codes)
    split_rows: list[dict[str, object]] = []
    split_exio: set[str] = set()
    for exio_name in valid_exio:
        exio_code = exio_code_map.get(exio_name, "")
        candidates = candidate_cpa_codes_for_exio(exio_code, division_index=division_index)
        candidates = filter_cpa_candidates_for_exio(exio_name, candidates)
        candidates = [candidate for candidate in candidates if candidate in cpa_codes]
        if len(candidates) <= 1:
            continue
        split_exio.add(exio_name)
        weights = _weights_for_candidates(candidates, x_series)
        for cpa_code in candidates:
            split_rows.append(
                {
                    "aggregate_sector": cpa_code,
                    "exio_sector": exio_name,
                    "weight": float(weights.get(cpa_code, 0.0)),
                    "enabled": "1",
                    "notes": "split-by-eurostat-output-share",
                }
            )
    if split_exio:
        base_df = base_df[~base_df["exio_sector"].isin(split_exio)].copy()
    if split_rows:
        base_df = pd.concat([base_df, pd.DataFrame(split_rows)], ignore_index=True)
    return base_df, cpa_labels


def _detailed_weighted_note(group: pd.DataFrame, row: pd.Series) -> str:
    exio_name = str(row["source_key"])
    exio_code = str(row["source_code"])
    target_code = str(row["target_code"])
    target_label = str(row["target_label"])
    note_marker = str(row["notes"])
    weight = float(row["weight"]) if str(row["weight"]).strip() != "" else 0.0
    targets = [
        f"{str(item.target_code)} ({str(item.target_label)}) = {float(item.weight):.6f}"
        for item in group[["target_code", "target_label", "weight"]].itertuples(index=False)
    ]
    if note_marker.startswith("official-exio-nace:"):
        workbook_ref = note_marker.split(":", 1)[1].strip()
        return (
            f"Semi-official override retained from the ACPR / Banque de France EXIOBASE-to-NACE workbook. "
            f"EXIO product {exio_code} ({exio_name}) is aligned to {workbook_ref}, which is represented in the Eurostat "
            f"CPA_65 target set by {target_code} ({target_label}). Weight = {weight:.6f} because the retained official signal "
            f"resolves to a single target sector in the current target set."
        )

    nace2_divisions, mapping_rule = _candidate_nace2_divisions_for_exio(exio_code)
    first_div = exio_code[1:3] if len(exio_code) >= 3 else exio_code
    division_text = ", ".join(str(div) for div in nace2_divisions) if nace2_divisions else "none"
    if len(group) == 1:
        return (
            f"No direct official EXIOBASE-to-Eurostat concordance row is used for {exio_code} ({exio_name}). "
            f"This row is derived deterministically from EXIO code prefix {first_div} through the repo's {mapping_rule} bridge, "
            f"yielding NACE Rev. 2 division(s) {division_text} and therefore Eurostat target {target_code} ({target_label}). "
            f"Weight = {weight:.6f} because only one target sector is retained after applying the bridge to the current CPA_65 target set."
        )

    return (
        f"No direct official EXIOBASE-to-Eurostat concordance row is used for {exio_code} ({exio_name}). "
        f"This row is derived deterministically from EXIO code prefix {first_div} through the repo's {mapping_rule} bridge, "
        f"yielding NACE Rev. 2 division(s) {division_text}. Because those divisions span multiple sectors in the current CPA_65 target set, "
        f"the EXIO product is split across targets using Austrian Eurostat product-use reference shares x = Z + atomic final demand. "
        f"Current split: {'; '.join(targets)}."
    )


def _build_weighted_mapping_csv(*, geo: str, year: int, eurostat_storage_folder: Path, exio_lookup: dict[str, dict[str, str]]) -> pd.DataFrame:
    weighted_rows, cpa_labels = _generate_weighted_rows(geo=geo, year=year, eurostat_storage_folder=eurostat_storage_folder)
    out = pd.DataFrame(index=weighted_rows.index)
    out["kind"] = "weighted"
    out["source_dataset"] = "EXIOBASE3"
    out["source_system"] = "pxp"
    out["source_version"] = "3"
    out["source_level"] = "product"
    out["target_dataset"] = "Eurostat_IO"
    out["target_system"] = "pxp"
    out["target_version"] = "CPA_65"
    out["target_level"] = "product"
    out["source_key"] = weighted_rows["exio_sector"].astype(str)
    out["source_code"] = out["source_key"].map(lambda key: exio_lookup.get(str(key), {}).get("source_code", ""))
    out["source_label"] = out["source_key"].map(lambda key: exio_lookup.get(str(key), {}).get("source_label", str(key)))
    out["target_key"] = weighted_rows["aggregate_sector"].astype(str)
    out["target_code"] = out["target_key"].astype(str)
    out["target_label"] = out["target_key"].map(lambda key: cpa_labels.get(str(key), str(key)))
    out["subtarget_key"] = ""
    out["weight"] = weighted_rows["weight"].astype(str)
    out["share"] = ""
    out["enabled"] = weighted_rows.get("enabled", pd.Series("1", index=weighted_rows.index)).astype(str)
    out["provenance"] = "semi-official"
    out["certainty"] = out.groupby("source_key")["target_key"].transform("count").map(
        lambda count: "derived-output-share-split" if int(count) > 1 else "derived-single-target"
    )
    out["transformation_rule"] = "weighted-exio-to-eurostat"
    out["source_name"] = "exio_to_eurostat_cpa.csv"
    out["official_source"] = weighted_rows["notes"].astype(str).map(
        lambda value: WEIGHTED_OFFICIAL_SOURCE if value.startswith("official-exio-nace:") else FALLBACK_OFFICIAL_SOURCE
    )
    out["notes"] = weighted_rows["notes"].astype(str)
    out.loc[out["notes"].str.startswith("official-exio-nace:"), "certainty"] = "semi-official-retained"

    enriched_notes: list[str] = []
    for _, group in out.groupby("source_key", sort=False):
        for _, row in group.iterrows():
            enriched_notes.append(_detailed_weighted_note(group, row))
    out["notes"] = enriched_notes
    return out.loc[:, list(MAPPING_CSV_COLUMNS)]


def _build_split_mapping_csv(*, electricity_by_source: bool, cpa_labels: dict[str, str]) -> pd.DataFrame:
    concordance = load_pymrio_exio3_sectors(system="pxp").sectors.copy()
    concordance["ExioNumber"] = pd.to_numeric(concordance["ExioNumber"], errors="coerce")

    def _pick(*, codes: list[str] | None = None, code_prefixes: list[str] | None = None) -> pd.DataFrame:
        match = pd.Series(False, index=concordance.index)
        if codes:
            match = match | concordance["ExioCode"].isin(list(map(str, codes)))
        if code_prefixes:
            for prefix in code_prefixes:
                match = match | concordance["ExioCode"].astype(str).str.startswith(str(prefix))
        return concordance[match].copy()

    rows: list[dict[str, str]] = []

    c19 = _pick(code_prefixes=["p23.1.", "p23.20."])
    c19 = pd.concat([c19, _pick(codes=["p23.3", "p24.g", "p24.h", "p24.i", "p10.e", "p40.2.a"])], ignore_index=True)
    c19 = c19.sort_values(["ExioNumber", "ExioCode", "ExioName"], na_position="last").drop_duplicates(subset=["ExioCode"], keep="first")
    for _, row in c19.iterrows():
        exio_code = str(row["ExioCode"])
        enabled = "1"
        constituent = "1"
        notes = ""
        if exio_code == "p23.3":
            enabled = "0"
            constituent = "0"
            notes = "Optional row retained for traceability: nuclear fuel is often excluded from CPA_C19 in Eurostat IOTs."
        rows.append(
            {
                "aggregate_sector": "CPA_C19",
                "subsector": str(row["ExioName"]),
                "exio_sector": str(row["ExioName"]),
                "source_code": exio_code,
                "source_label": str(row["ExioName"]),
                "constituent": constituent,
                "enabled": enabled,
                "notes": notes,
            }
        )

    d = pd.concat(
        [
            _pick(code_prefixes=["p40.11."]),
            _pick(codes=["p40.12", "p40.13", "p40.2.1", "p40.2.d", "p40.2.e"]),
        ],
        ignore_index=True,
    )
    d = d.sort_values(["ExioNumber", "ExioCode", "ExioName"], na_position="last").drop_duplicates(subset=["ExioCode"], keep="first")
    for _, row in d.iterrows():
        exio_code = str(row["ExioCode"])
        subsector = str(row["ExioName"])
        notes = ""
        if exio_code.startswith("p40.11.") and not electricity_by_source:
            subsector = "Electricity generation"
            notes = "Electricity source rows are grouped into one subsector to avoid an electricity-by-source split."
        elif exio_code == "p40.2.1":
            subsector = "Gas distribution"
        elif exio_code == "p40.2.d":
            subsector = "Manufactured gas"
        rows.append(
            {
                "aggregate_sector": "CPA_D",
                "subsector": subsector,
                "exio_sector": str(row["ExioName"]),
                "source_code": exio_code,
                "source_label": str(row["ExioName"]),
                "constituent": "1",
                "enabled": "1",
                "notes": notes,
            }
        )

    frame = pd.DataFrame(rows)
    out = pd.DataFrame(index=frame.index)
    out["kind"] = "split"
    out["source_dataset"] = "EXIOBASE3"
    out["source_system"] = "pxp"
    out["source_version"] = "3"
    out["source_level"] = "product"
    out["target_dataset"] = "Eurostat_IO"
    out["target_system"] = "pxp"
    out["target_version"] = "CPA_65"
    out["target_level"] = "product"
    out["source_key"] = frame["exio_sector"].astype(str)
    out["source_code"] = frame["source_code"].astype(str)
    out["source_label"] = frame["source_label"].astype(str)
    out["target_key"] = frame["aggregate_sector"].astype(str)
    out["target_code"] = frame["aggregate_sector"].astype(str)
    out["target_label"] = frame["aggregate_sector"].map(lambda key: cpa_labels.get(str(key), str(key)))
    out["subtarget_key"] = frame["subsector"].astype(str)
    out["weight"] = ""
    out["share"] = ""
    out["enabled"] = frame["enabled"].astype(str)
    out["provenance"] = "methodological"
    out["certainty"] = "methodological-template"
    out["transformation_rule"] = "exio-energy-split-template"
    out["source_name"] = (
        "exio_energy_split_cpa_with_electricity_source.csv"
        if electricity_by_source
        else "exio_energy_split_cpa_no_electricity_source.csv"
    )
    out["official_source"] = "Repo-authored split template built on EXIOBASE classification codes"
    out["notes"] = frame.apply(
        lambda row: (
            f"Repo-authored split template row. EXIO product {row['source_code']} ({row['source_label']}) is included under "
            f"aggregate sector {row['aggregate_sector']} and subsector {row['subsector']}. "
            f"This is not an official pairwise concordance; it is a deliberate disaggregation rule for the energy split pipeline."
            + (f" Existing note: {row['notes']}" if str(row['notes']).strip() else "")
        ),
        axis=1,
    )
    out["constituent"] = frame["constituent"].astype(str)
    return out.loc[:, list(MAPPING_CSV_COLUMNS) + ["constituent"]]


def _canonicalize_category_mapping(path: Path, *, cpa_labels: dict[str, str]) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str).fillna("")
    if set(MAPPING_CSV_COLUMNS).issubset(raw.columns):
        out = read_mapping_csv(path, expected_kind="category-target").copy()
        out["target_code"] = out["target_code"].astype(str)
        blank = out["target_code"] == ""
        out.loc[blank, "target_code"] = out.loc[blank, "target_key"].map(lambda key: _match_target_code(str(key), cpa_labels))
        out.loc[out["target_label"].astype(str) == "", "target_label"] = out["target_key"].astype(str)
        out.loc[out["source_label"].astype(str) == "", "source_label"] = out["source_key"].astype(str)
        out.loc[out["official_source"].astype(str) == "", "official_source"] = ENERGY_METADATA_SOURCE
        return out

    out = pd.DataFrame(index=raw.index)
    out["kind"] = "category-target"
    out["source_dataset"] = "Eurostat_Energy_Balances"
    out["source_system"] = "nrg_bal"
    out["source_version"] = "current"
    out["source_level"] = "category"
    out["target_dataset"] = "Eurostat_IO"
    out["target_system"] = "pxp"
    out["target_version"] = "CPA_65"
    out["target_level"] = "target-label"
    out["source_key"] = raw["Category"].astype(str)
    out["source_code"] = ""
    out["source_label"] = raw["Category"].astype(str)
    out["target_key"] = raw["TargetSector"].astype(str)
    out["target_code"] = raw["TargetSector"].astype(str).map(lambda label: _match_target_code(str(label), cpa_labels))
    out["target_label"] = raw["TargetSector"].astype(str)
    out["subtarget_key"] = ""
    out["weight"] = ""
    out["share"] = raw["Share_total"].astype(str)
    out["enabled"] = "1"
    out["provenance"] = "methodological"
    out["certainty"] = "methodological-seed-share"
    out["transformation_rule"] = "energy-balance-category-share-seed"
    out["source_name"] = path.name
    out["official_source"] = ENERGY_METADATA_SOURCE
    out["notes"] = out.apply(
        lambda row: (
            f"Repo-authored bridge from Eurostat energy-balance category '{row['source_key']}' to Eurostat IO target '{row['target_key']}'. "
            f"Share_total = {row['share']} is the checked-in seed share in the source-of-truth CSV; runtime code may recompute Share_used "
            f"from observed Eurostat outputs for the chosen year."
        ),
        axis=1,
    )
    return out.loc[:, list(MAPPING_CSV_COLUMNS)]


def _canonicalize_aggregation_mapping(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str).fillna("")
    if set(MAPPING_CSV_COLUMNS).issubset(raw.columns):
        out = read_mapping_csv(path, expected_kind="aggregate").copy()
        out.loc[out["source_label"].astype(str) == "", "source_label"] = out["source_key"].astype(str)
        out.loc[out["target_label"].astype(str) == "", "target_label"] = out["target_key"].astype(str)
        out.loc[out["official_source"].astype(str) == "", "official_source"] = "Repo analytical aggregation rule"
        return out

    out = pd.DataFrame(index=raw.index)
    out["kind"] = "aggregate"
    out["source_dataset"] = "Eurostat_IO"
    out["source_system"] = "pxp"
    out["source_version"] = "CPA_65"
    out["source_level"] = "product"
    out["target_dataset"] = "Repo_Analytical"
    out["target_system"] = "energy_2sector"
    out["target_version"] = "current"
    out["target_level"] = "aggregate"
    out["source_key"] = raw["from_sector"].astype(str)
    out["source_code"] = ""
    out["source_label"] = raw["from_sector"].astype(str)
    out["target_key"] = raw["to_sector"].astype(str)
    out["target_code"] = raw["to_sector"].astype(str)
    out["target_label"] = raw["to_sector"].astype(str)
    out["subtarget_key"] = ""
    out["weight"] = ""
    out["share"] = ""
    out["enabled"] = raw.get("enabled", pd.Series("1", index=raw.index)).astype(str)
    out["provenance"] = "methodological"
    out["certainty"] = "analytical-aggregation"
    out["transformation_rule"] = "energy-green-brown-aggregation"
    out["source_name"] = path.name
    out["official_source"] = "Repo analytical aggregation rule"
    out["notes"] = raw.apply(
        lambda row: (
            f"Repo analytical aggregation rule. Source sector '{row['from_sector']}' is assigned to aggregate '{row['to_sector']}'."
            + (f" Existing note: {row['notes']}" if str(row.get('notes', '')).strip() else "")
        ),
        axis=1,
    )
    return out.loc[:, list(MAPPING_CSV_COLUMNS)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the canonical annotated mapping CSVs used by the MRIO repo.")
    parser.add_argument("--geo", default="AT")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--eurostat-storage-folder", default="data/eurostat")
    parser.add_argument("--outdir", default="scripts/mappings")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    exio_lookup = _load_exio_lookup()
    cached_cpa_labels = _find_any_eurostat_meta()

    weighted = _build_weighted_mapping_csv(
        geo=str(args.geo),
        year=int(args.year),
        eurostat_storage_folder=Path(args.eurostat_storage_folder),
        exio_lookup=exio_lookup,
    )
    _write_mapping_csv(outdir / "exio_to_eurostat_cpa.csv", weighted)

    split_with = _build_split_mapping_csv(electricity_by_source=True, cpa_labels=cached_cpa_labels)
    _write_mapping_csv(outdir / "exio_energy_split_cpa_with_electricity_source.csv", split_with)

    split_without = _build_split_mapping_csv(electricity_by_source=False, cpa_labels=cached_cpa_labels)
    _write_mapping_csv(outdir / "exio_energy_split_cpa_no_electricity_source.csv", split_without)

    category_path = outdir / "energy_balance_to_nace2_mapping.csv"
    category = _canonicalize_category_mapping(category_path, cpa_labels=cached_cpa_labels)
    _write_mapping_csv(category_path, category)

    aggregate_path = outdir / "energy_green_brown_2sector_from_split_subsectors.csv"
    aggregate = _canonicalize_aggregation_mapping(aggregate_path)
    _write_mapping_csv(aggregate_path, aggregate)

    print(f"Wrote canonical annotated mapping CSVs to {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())