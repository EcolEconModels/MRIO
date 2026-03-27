from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


def repo_root_from_file(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parents[2]


def default_postgrowth_root(repo_root: Path) -> Path:
    return repo_root.parent / "PostGrowth-SFC-Austria"


def default_output_dir(repo_root: Path) -> Path:
    return default_postgrowth_root(repo_root) / "mrio_energy_balance_diagnostics"


def bootstrap_ecolecon_mrio(repo_root: Path) -> None:
    package_root = repo_root / "src" / "ecolecon_mrio"
    if not package_root.exists():
        raise FileNotFoundError(f"Could not find package root: {package_root}")
    existing = sys.modules.get("ecolecon_mrio")
    if existing is not None:
        if not hasattr(existing, "__path__"):
            existing.__path__ = [str(package_root)]
        return
    package = types.ModuleType("ecolecon_mrio")
    package.__path__ = [str(package_root)]
    sys.modules["ecolecon_mrio"] = package


def import_repo_module(repo_root: Path, module_name: str):
    bootstrap_ecolecon_mrio(repo_root)
    return importlib.import_module(module_name)


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_weighted_mapping(repo_root: Path, mapping_csv: Path | None = None) -> pd.DataFrame:
    mapping_mod = import_repo_module(repo_root, "ecolecon_mrio.mapping_csv")
    path = mapping_csv or (repo_root / "scripts" / "mappings" / "exio_to_eurostat_cpa.csv")
    return mapping_mod.load_weighted_mapping_frame(path)


def load_eurostat_iot_bundle(repo_root: Path, *, geo: str, year: int, flow_code: str):
    eurostat_iot = import_repo_module(repo_root, "ecolecon_mrio.eurostat_iot")
    data_mod = import_repo_module(repo_root, "ecolecon_mrio.data")
    parsed = eurostat_iot.load_eurostat_iot(
        dataset="naio_10_cp1700",
        filters={
            "geo": str(geo),
            "time": str(year),
            "freq": "A",
            "unit": "MIO_EUR",
            "stk_flow": str(flow_code),
        },
        row_dim="prd_ava",
        col_dim="prd_use",
        storage_folder=Path(data_mod.default_data_root()) / "eurostat",
        force_download=False,
        force_reparse=False,
    )
    output = (parsed.Z.sum(axis=1) + parsed.Y.sum(axis=1)).astype("float64")
    output.index = output.index.astype(str)
    labels = {
        str(code): str(label)
        for code, label in (((parsed.meta.get("dimensions", {}) or {}).get("prd_ava", {}) or {}).items())
    }
    return parsed, output, labels


def build_eurostat_product_use_reference(
    parsed,
    *,
    exclude_final_demand_codes: tuple[str, ...] = (),
) -> pd.Series:
    y_block = parsed.Y.copy()
    if exclude_final_demand_codes:
        drop_cols = [col for col in y_block.columns if str(col) in set(map(str, exclude_final_demand_codes))]
        if drop_cols:
            y_block = y_block.drop(columns=drop_cols)
    reference = (parsed.Z.sum(axis=1) + y_block.sum(axis=1)).astype("float64")
    reference.index = reference.index.astype(str)
    return reference


def load_exio_slice(repo_root: Path, *, region: str, year: int, flow_name: str):
    exio_mod = import_repo_module(repo_root, "ecolecon_mrio.exiobase3")
    data_mod = import_repo_module(repo_root, "ecolecon_mrio.data")
    return exio_mod.load_exiobase3_region_slice(
        year=year,
        region=region,
        system="pxp",
        flow=flow_name,
        cache_root=Path(data_mod.default_data_root()),
    )


def eurostat_order_map(reference_index: pd.Index) -> dict[str, int]:
    return {str(code): idx + 1 for idx, code in enumerate(reference_index.astype(str).tolist())}


def mapping_label_maps(mapping_df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    target_labels = (
        mapping_df.loc[:, ["aggregate_sector", "target_label"]]
        .drop_duplicates(subset=["aggregate_sector"])
        .set_index("aggregate_sector")["target_label"]
        .astype(str)
        .to_dict()
    )
    source_labels = (
        mapping_df.loc[:, ["exio_sector", "source_label"]]
        .drop_duplicates(subset=["exio_sector"])
        .set_index("exio_sector")["source_label"]
        .astype(str)
        .to_dict()
    )
    return target_labels, source_labels


def build_signed_value_summary(
    repo_root: Path,
    *,
    source_values: pd.Series,
    reference_values: pd.Series,
    mapping_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    concordance = import_repo_module(repo_root, "ecolecon_mrio.diagnostic_concordance")
    alloc = concordance.allocate_weighted_values(source_values, mapping_df)
    summary = alloc.aggregate_totals.rename("mapped_total").to_frame()
    ref = reference_values.astype("float64").copy()
    ref.index = ref.index.astype(str)
    summary["reference_total"] = ref.reindex(summary.index)
    summary["gap"] = summary["mapped_total"] - summary["reference_total"]
    summary["gap_pct"] = np.where(
        summary["reference_total"].abs() > 0,
        100.0 * summary["gap"] / summary["reference_total"],
        np.nan,
    )
    summary = summary.reset_index().rename(columns={"aggregate_sector": "target_sector"})
    return alloc.detail.copy(), summary


def read_energy_balance_table(postgrowth_root: Path, *, year: int) -> pd.DataFrame:
    path = postgrowth_root / f"Energy_Balances_{year}_Eurostat.xlsx"
    df = pd.read_excel(path)
    df = df.rename(columns={"SIEC (Labels)": "energy_balance_sector_name", "Total": "energy_balance_total_energy_gwh_2021"})
    return df


def load_ghg_protocol_factors(postgrowth_root: Path) -> dict[str, float]:
    workbook = postgrowth_root / "Emission_Factors_for_Cross_Sector_Tools_V2.0_0.xlsx"
    raw = pd.read_excel(workbook, sheet_name="Stationary Combustion", header=None)
    header_rows = raw.index[
        (raw[1].astype(str) == "Fuel")
        & (raw[4].astype(str).str.contains("CO2 emission factors", regex=False, na=False))
    ]
    if len(header_rows) == 0:
        raise ValueError("Could not locate GHG Protocol stationary-combustion factor table header.")
    header_row = int(header_rows[0])
    table2 = raw.index[raw[1].astype(str).str.contains(r"^Table 2", regex=True, na=False)]
    data_end = int(table2[0] - 1) if len(table2) else len(raw) - 1
    ef = raw.iloc[header_row + 3 : data_end + 1, [1, 2, 4]].copy()
    ef.columns = ["category", "fuel", "kg_co2_per_tj"]
    ef["category"] = ef["category"].ffill()
    ef["fuel"] = ef["fuel"].astype(str)
    ef["kg_co2_per_tj"] = pd.to_numeric(ef["kg_co2_per_tj"], errors="coerce")
    ef = ef[ef["kg_co2_per_tj"].notna()].copy()

    def convert(value: float) -> float:
        return float(value) * 3.6 / 1000.0

    def pick_one(patterns: list[str], category: str | None = None) -> float:
        candidates = ef
        if category is not None:
            candidates = candidates[candidates["category"].str.contains(category, case=False, na=False)]
        for pattern in patterns:
            hit = candidates[candidates["fuel"].str.contains(pattern, case=False, regex=True, na=False)]
            if not hit.empty:
                return float(hit.iloc[0]["kg_co2_per_tj"])
        raise ValueError(f"Could not find GHG factor for patterns={patterns} category={category}")

    oil_components = [
        pick_one([r"^Gas/Diesel oil"], "Oil products"),
        pick_one([r"^Motor gasoline"], "Oil products"),
        pick_one([r"^Residual fuel oil"], "Oil products"),
    ]
    waste_fossil_share = 0.5
    return {
        "Solid fossil fuels": convert(
            pick_one([r"^Other bituminous coal$", r"^Coking coal$", r"^Sub bituminous coal$", r"^Lignite$"], "Coal products")
        ),
        "Fossil gas": convert(pick_one([r"^Natural gas"], "Natural gas")),
        "Oil and petroleum products (excluding biofuel portion)": convert(float(np.mean(oil_components))),
        "Hydro": 0.0,
        "Wind": 0.0,
        "Solar": 0.0,
        "Biofuels": convert(pick_one([r"^Biodiesels$", r"^Biogasoline$", r"^Other liquid biofuels$"], "Biomass")),
        "Waste": convert(pick_one([r"^Municipal waste \(Non biomass fraction\)$"], "Other wastes")) * waste_fossil_share
        + convert(pick_one([r"^Municipal wastes \(Biomass fraction\)$"], "Biomass")) * (1.0 - waste_fossil_share),
        "Other Renewables": 0.0,
    }


def compute_weighted_energy_balance_factor(energy_df: pd.DataFrame, source_factors: dict[str, float]) -> pd.Series:
    source_cols = list(source_factors)
    weighted_emissions = sum(energy_df[col].astype("float64") * float(source_factors[col]) for col in source_cols)
    total = energy_df["energy_balance_total_energy_gwh_2021"].astype("float64")
    return np.where(total > 0, weighted_emissions / total, 0.0)


def sector_total(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())