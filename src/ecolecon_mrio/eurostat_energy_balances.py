"""
Eurostat energy balances -> primary energy-by-sector utilities.

This module builds a "Tabelle1"-style table for a given country/year:
  - Direct final energy use by primary source (fossil fuels, biofuels, waste, etc.)
  - Electricity and derived heat final use, "unrolled" into primary sources using
    Eurostat production-of-electricity-and-heat-by-fuel statistics.

It is designed to be cache-first and to reuse the SDMX-JSON downloader/parser in
`ecolecon_mrio.eurostat_iot`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.eurostat_iot import (
    EurostatParseError,
    eurostat_dimensions,
    load_eurostat_table_2d,
)
from ecolecon_mrio.mapping_csv import read_product_source_mapping, read_source_bucket_mapping


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_PRIMARY_SOURCE_BUCKETS_PATH = _repo_root() / "scripts" / "mappings" / "energy_primary_source_buckets.csv"
DEFAULT_PRODUCT_SOURCE_MAPPING_PATH = _repo_root() / "scripts" / "mappings" / "siec_to_primary_energy_sources.csv"


PRIMARY_SOURCES: tuple[str, ...] = tuple(read_source_bucket_mapping(DEFAULT_PRIMARY_SOURCE_BUCKETS_PATH))
PRODUCT_SOURCE_MAPPING: dict[str, tuple[str, ...]] = read_product_source_mapping(DEFAULT_PRODUCT_SOURCE_MAPPING_PATH)

NRG_BAL_LABEL_ALIASES: Mapping[str, str] = {
    # Some Eurostat labels appear in older spreadsheets / documentation in shortened forms.
    "Final consumption - households - energy use": "Final consumption - other sectors - households - energy use",
}


DEFAULT_ENERGY_BALANCE_DATASET = "nrg_bal_c"
DEFAULT_ELECTRICITY_HEAT_MIX_DATASET = "nrg_bal_peh"

# SIEC code scaffolding used to map Eurostat energy products to our 9 primary-source buckets.
# These are designed to be:
# - non-overlapping (avoid mixing aggregates with their subcomponents)
# - stable across `nrg_bal_c` and `nrg_bal_peh`
#
# Notes:
# - `RA000` ("Renewables and biofuels") is an aggregate; we do NOT use it directly.
# - For fossil gas, `C0350-0370` ("Manufactured gases") is treated as gas and we do NOT
#   include its subcodes (C0350, C0360, C0371, ...) to avoid double counting.
SIEC_TOTAL = "TOTAL"
SIEC_ELECTRICITY = "E7000"
SIEC_HEAT = "H8000"

SIEC_RENEWABLES_AND_BIOFUELS_AGG = "RA000"


def _codes_for_source_bucket(source_bucket: str, *, code_to_label: Mapping[str, str]) -> tuple[str, ...]:
    declared_codes = PRODUCT_SOURCE_MAPPING.get(str(source_bucket), tuple())
    return _present_codes(code_to_label, declared_codes)


def _as_path(p: str | Path | None) -> Path:
    if p is None:
        return default_data_root() / "eurostat"
    return Path(p)


def _norm_label(s: str) -> str:
    s = str(s).strip().casefold()
    s = re.sub(r"[\s\u00a0]+", " ", s)
    # Keep only alphanumerics, spaces, and hyphens.
    # NOTE: the hyphen must be escaped (or placed first/last) inside a character class.
    s = re.sub(r"[^a-z0-9 -]+", "", s)
    return s


def _match_codes_by_label(
    code_to_label: Mapping[str, str],
    *,
    patterns: Sequence[str],
    exclude_patterns: Sequence[str] = (),
) -> list[str]:
    inc = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    exc = [re.compile(p, flags=re.IGNORECASE) for p in exclude_patterns]
    out: list[str] = []
    for code, label in code_to_label.items():
        lab = str(label)
        if not any(r.search(lab) for r in inc):
            continue
        if any(r.search(lab) for r in exc):
            continue
        out.append(str(code))
    return out


def _sum_codes(values: pd.Series, codes: Sequence[str]) -> float:
    if not codes:
        return 0.0
    v = pd.to_numeric(values.reindex(list(codes)), errors="coerce").fillna(0.0)
    return float(v.sum())


def _present_codes(code_to_label: Mapping[str, str], codes: Sequence[str]) -> tuple[str, ...]:
    have = set(map(str, code_to_label.keys()))
    return tuple([str(c) for c in codes if str(c) in have])


def _resolve_nrg_bal_code(label: str, *, label_to_code: Mapping[str, str]) -> str | None:
    """
    Resolve a human-readable `nrg_bal` label to its code using normalized matching plus aliases.
    """
    key = _norm_label(label)
    if key in label_to_code:
        return label_to_code[key]
    alias = NRG_BAL_LABEL_ALIASES.get(str(label))
    if alias:
        key2 = _norm_label(alias)
        if key2 in label_to_code:
            return label_to_code[key2]
    return None


@dataclass(frozen=True)
class EnergyBalanceResult:
    """
    Tabellen-style energy balance decomposition.

    `tabelle1` layout mirrors the original standalone script:
      Category | Total | 9 direct sources | Electricity: | 9 electricity sources | Heat | 9 heat sources
    """

    tabelle1: pd.DataFrame
    unrolled_sources: pd.DataFrame
    diagnostics: pd.DataFrame
    meta: dict[str, Any]


def read_category_target_mapping_csv(path: str | Path) -> pd.DataFrame:
    """
    Read a Category -> TargetSector mapping CSV.

    Supports both the legacy three-column format and the canonical
    `kind=category-target` schema defined in `mapping_csv.py`.

    Returned columns:
      - Category
      - TargetSector
    """
    df = pd.read_csv(path)
    if {"Category", "TargetSector"}.issubset(set(df.columns)):
        out = df.loc[:, ["Category", "TargetSector"]].copy()
    elif {"kind", "source_key", "target_key", "share"}.issubset(set(df.columns)):
        from ecolecon_mrio.mapping_csv import load_category_mapping_frame

        canonical = load_category_mapping_frame(path)
        out = canonical.loc[:, ["Category", "TargetSector"]].copy()
    else:
        raise ValueError(
            "Mapping CSV must have legacy columns Category, TargetSector or the canonical "
            f"category-target schema. Got: {df.columns.tolist()}"
        )
    out["Category"] = out["Category"].astype(str).str.strip()
    out["TargetSector"] = out["TargetSector"].astype(str).str.strip()
    return out


def match_labels_to_codes(
    labels: Sequence[str],
    *,
    code_to_label: Mapping[str, str],
) -> dict[str, str]:
    """
    Map a list of human-readable labels to codes using normalized exact matching.
    """
    inv: dict[str, list[str]] = {}
    for code, lab in code_to_label.items():
        inv.setdefault(_norm_label(lab), []).append(str(code))

    out: dict[str, str] = {}
    for lab in labels:
        key = _norm_label(lab)
        codes = inv.get(key, [])
        if not codes:
            raise KeyError(f"Could not match label to code: {lab}")
        if len(codes) > 1:
            raise KeyError(f"Ambiguous label match for {lab}: codes={codes}")
        out[str(lab)] = codes[0]
    return out


def compute_mapping_shares_from_outputs(
    mapping_df: pd.DataFrame,
    *,
    target_output_by_target: Mapping[str, float],
    category_col: str = "Category",
    target_col: str = "TargetSector",
) -> pd.DataFrame:
    """
    Compute Category -> TargetSector shares using an external weighting reference.

    Returns a copy of mapping_df with a new column `Share_used`.
    """
    df = mapping_df.copy()
    if category_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"mapping_df must include columns {category_col}, {target_col}")

    outputs = {str(k): float(v) for k, v in target_output_by_target.items()}
    share_used: list[float] = []

    for cat, grp in df.groupby(category_col, sort=False):
        if len(grp) == 1:
            share_used.append(1.0)
            continue
        w = np.array([max(0.0, outputs.get(str(t), 0.0)) for t in grp[target_col].astype(str)], dtype="float64")
        s = float(w.sum())
        if s > 0:
            shares = (w / s).tolist()
        else:
            raise ValueError(f"No positive weighting reference available for category {cat!r}.")
        share_used.extend(shares)

    df["Share_used"] = np.array(share_used, dtype="float64")
    return df


def build_postgrowth_energy_balance(
    mapped_df: pd.DataFrame,
    *,
    source_group_by_source: Mapping[str, str],
    sector_col: str = "TargetSector",
    split_energy_green_brown: bool = False,
    brown_label: str = "Energy Brown",
    green_label: str = "Energy Green",
    coke_label: str = "Coke and refined petroleum products",
    electricity_label: str = "Electricity, gas, steam and air conditioning",
) -> pd.DataFrame:
    source_cols = [src for src in PRIMARY_SOURCES if src in mapped_df.columns]
    if not source_cols:
        raise ValueError("Mapped energy balance is missing primary source columns.")
    if sector_col not in mapped_df.columns:
        raise ValueError(f"Mapped energy balance is missing sector column {sector_col!r}.")

    out = mapped_df.loc[:, [sector_col, *source_cols]].copy()
    out = out.rename(columns={sector_col: "Sector"})
    out["Sector"] = out["Sector"].astype(str).str.strip()

    if split_energy_green_brown:
        coke_rows = out.index[out["Sector"] == coke_label].tolist()
        electricity_rows = out.index[out["Sector"] == electricity_label].tolist()
        if len(coke_rows) != 1 or len(electricity_rows) != 1:
            raise ValueError(
                "Expected exactly one coke row and one electricity row when splitting to Energy Brown/Green."
            )
        coke_idx = coke_rows[0]
        electricity_idx = electricity_rows[0]

        combined = pd.to_numeric(out.loc[coke_idx, source_cols], errors="coerce").fillna(0.0) + pd.to_numeric(
            out.loc[electricity_idx, source_cols], errors="coerce"
        ).fillna(0.0)
        brown_sources = [src for src in source_cols if str(source_group_by_source.get(src, "")).casefold() == "fossil"]
        green_sources = [src for src in source_cols if src not in set(brown_sources)]

        brown_row = combined.copy()
        green_row = combined.copy()
        for source in green_sources:
            brown_row[source] = 0.0
        for source in brown_sources:
            green_row[source] = 0.0

        out.loc[coke_idx, source_cols] = brown_row[source_cols].to_numpy(dtype="float64")
        out.loc[electricity_idx, source_cols] = green_row[source_cols].to_numpy(dtype="float64")
        out.at[coke_idx, "Sector"] = brown_label
        out.at[electricity_idx, "Sector"] = green_label

    out["Total_energy_GWh"] = pd.to_numeric(out[source_cols].stack(), errors="coerce").unstack(fill_value=0.0).sum(axis=1)
    return out.loc[:, ["Sector", *source_cols, "Total_energy_GWh"]]


def disaggregate_unrolled_sources(
    unrolled_sources: pd.DataFrame,
    mapping_with_shares: pd.DataFrame,
    *,
    category_col: str = "Category",
    target_col: str = "TargetSector",
    share_col: str = "Share_used",
) -> pd.DataFrame:
    """
    Disaggregate `unrolled_sources` (Category x PRIMARY_SOURCES) to TargetSector x PRIMARY_SOURCES.
    """
    required = {category_col, target_col, share_col}
    if not required.issubset(set(mapping_with_shares.columns)):
        raise ValueError(f"mapping_with_shares must include {sorted(required)}")

    base = unrolled_sources.set_index("Category")
    missing = [c for c in mapping_with_shares[category_col].unique().tolist() if c not in base.index]
    if missing:
        raise ValueError(f"Unrolled sources missing categories present in mapping: {missing[:10]}")

    targets = sorted(mapping_with_shares[target_col].astype(str).unique().tolist())
    out = pd.DataFrame(0.0, index=targets, columns=list(PRIMARY_SOURCES))

    for cat, grp in mapping_with_shares.groupby(category_col, sort=False):
        vec = base.loc[str(cat), list(PRIMARY_SOURCES)].astype("float64").to_numpy()
        for _, row in grp.iterrows():
            tgt = str(row[target_col])
            share = float(row[share_col])
            out.loc[tgt, :] += share * vec
    out.index.name = target_col
    return out


def load_eurostat_energy_balance_slice(
    *,
    geo: str,
    time: int,
    nrg_bal: Sequence[str] | None = None,
    siec: Sequence[str] | None = None,
    unit: str = "GWH",
    freq: str = "A",
    dataset: str = DEFAULT_ENERGY_BALANCE_DATASET,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    force_reparse: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load Eurostat energy balance data as a 2D matrix: nrg_bal x siec.
    """
    filters: dict[str, Any] = {"geo": str(geo), "time": str(time), "unit": str(unit), "freq": str(freq)}
    if nrg_bal is not None:
        filters["nrg_bal"] = list(map(str, nrg_bal))
    if siec is not None:
        filters["siec"] = list(map(str, siec))
    return load_eurostat_table_2d(
        dataset=dataset,
        filters=filters,
        row_dim="nrg_bal",
        col_dim="siec",
        storage_folder=_as_path(storage_folder) if storage_folder is not None else None,
        force_download=force_download,
        force_reparse=force_reparse,
    )


def load_eurostat_electricity_heat_mix(
    *,
    geo: str,
    time: int,
    unit: str = "GWH",
    freq: str = "A",
    dataset: str = DEFAULT_ELECTRICITY_HEAT_MIX_DATASET,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    force_reparse: bool = False,
    electricity_nrg_bal_prefixes: Sequence[str] = ("GEP",),
    heat_nrg_bal_prefixes: Sequence[str] = ("GHP",),
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """
    Compute electricity and derived heat production shares by primary source.

    Uses Eurostat dataset `nrg_bal_peh` by default (production of electricity and derived heat by type of fuel).
    """
    df, meta = load_eurostat_energy_balance_slice(
        geo=geo,
        time=time,
        nrg_bal=None,
        siec=None,
        unit=unit,
        freq=freq,
        dataset=dataset,
        storage_folder=storage_folder,
        force_download=force_download,
        force_reparse=force_reparse,
    )

    nrg_bal_codes = list(map(str, df.index.astype(str)))
    elec_items = [c for c in nrg_bal_codes if any(str(c).startswith(p) for p in electricity_nrg_bal_prefixes)]
    heat_items = [c for c in nrg_bal_codes if any(str(c).startswith(p) for p in heat_nrg_bal_prefixes)]
    if not elec_items:
        raise EurostatParseError(
            f"Could not identify electricity production rows in {dataset}: "
            f"expected prefixes {list(electricity_nrg_bal_prefixes)} among nrg_bal codes."
        )
    if not heat_items:
        raise EurostatParseError(
            f"Could not identify heat production rows in {dataset}: "
            f"expected prefixes {list(heat_nrg_bal_prefixes)} among nrg_bal codes."
        )

    dim_siec = (meta.get("dimensions", {}) or {}).get("siec", {})
    if not isinstance(dim_siec, dict) or not dim_siec:
        raise EurostatParseError(f"Missing siec dimension metadata in dataset={dataset}.")
    siec_labels: dict[str, str] = {str(k): str(v) for k, v in dim_siec.items()}

    # Use a non-overlapping SIEC-to-primary-source mapping (codes, not regex) and allocate
    # any small remainder to Other Renewables to ensure closure.
    c_solid = _codes_for_source_bucket("Solid fossil fuels", code_to_label=siec_labels)
    c_gas = _codes_for_source_bucket("Fossil gas", code_to_label=siec_labels)
    c_oil = _codes_for_source_bucket("Oil and petroleum products (excluding biofuel portion)", code_to_label=siec_labels)
    c_hydro = _codes_for_source_bucket("Hydro", code_to_label=siec_labels)
    c_wind = _codes_for_source_bucket("Wind", code_to_label=siec_labels)
    c_solar = _codes_for_source_bucket("Solar", code_to_label=siec_labels)
    c_other = _codes_for_source_bucket("Other Renewables", code_to_label=siec_labels)
    c_waste = _codes_for_source_bucket("Waste", code_to_label=siec_labels)
    c_bio = _codes_for_source_bucket("Biofuels", code_to_label=siec_labels)
    have_ra000 = str(SIEC_RENEWABLES_AND_BIOFUELS_AGG) in set(map(str, siec_labels.keys()))

    def _shares_for(items: Sequence[str]) -> pd.Series:
        totals = df.loc[list(items)].sum(axis=0)
        total_all = float(pd.to_numeric(totals.get(SIEC_TOTAL, 0.0), errors="coerce") or 0.0)
        if not np.isfinite(total_all) or total_all <= 0:
            # Fallback: if TOTAL isn't present/usable, use sum of mapped buckets.
            total_all = float(
                _sum_codes(totals, c_solid)
                + _sum_codes(totals, c_gas)
                + _sum_codes(totals, c_oil)
                + _sum_codes(totals, c_hydro)
                + _sum_codes(totals, c_wind)
                + _sum_codes(totals, c_solar)
                + _sum_codes(totals, c_bio)
                + _sum_codes(totals, c_waste)
                + _sum_codes(totals, c_other)
            )
        if not np.isfinite(total_all) or total_all <= 0:
            raise EurostatParseError(
                f"Zero/invalid total production for items={list(items)} in dataset={dataset} "
                f"(geo={geo} time={time} unit={unit})."
            )

        mapped = {
            "Solid fossil fuels": _sum_codes(totals, c_solid),
            "Fossil gas": _sum_codes(totals, c_gas),
            "Oil and petroleum products (excluding biofuel portion)": _sum_codes(totals, c_oil),
            "Hydro": _sum_codes(totals, c_hydro),
            "Wind": _sum_codes(totals, c_wind),
            "Solar": _sum_codes(totals, c_solar),
            "Biofuels": _sum_codes(totals, c_bio),
            "Waste": _sum_codes(totals, c_waste),
            "Other Renewables": _sum_codes(totals, c_other),
        }
        if mapped["Biofuels"] == 0.0 and have_ra000:
            ra000 = float(pd.to_numeric(totals.get(SIEC_RENEWABLES_AND_BIOFUELS_AGG, 0.0), errors="coerce") or 0.0)
            mapped["Biofuels"] = max(
                0.0,
                ra000
                - mapped["Hydro"]
                - mapped["Wind"]
                - mapped["Solar"]
                - mapped["Other Renewables"],
            )

        residual = float(total_all - sum(mapped.values()))
        if residual > 0:
            mapped["Other Renewables"] += residual

        s = pd.Series({k: float(mapped.get(k, 0.0)) for k in PRIMARY_SOURCES}, dtype="float64")
        ssum = float(s.sum())
        if not np.isfinite(ssum) or ssum <= 0:
            raise EurostatParseError(
                f"Zero/invalid mapped production for items={list(items)} in dataset={dataset} "
                f"(geo={geo} time={time} unit={unit})."
            )
        return s / ssum

    elec_shares = _shares_for(elec_items)
    heat_shares = _shares_for(heat_items)
    return elec_shares, heat_shares, {
        "mix_dataset": dataset,
        "mix_meta": meta,
        "siec_codes": {
            "solid_fossil": c_solid,
            "fossil_gas": c_gas,
            "oil_ex_bio": c_oil,
            "hydro": c_hydro,
            "wind": c_wind,
            "solar": c_solar,
            "biofuels_detailed": c_bio,
            "waste": c_waste,
            "other_renew": c_other,
        },
    }


def build_tabelle1_from_eurostat(
    *,
    geo: str,
    time: int,
    categories: Sequence[str],
    composite_categories: Mapping[str, Sequence[str]] | None = None,
    unit: str = "GWH",
    freq: str = "A",
    energy_dataset: str = DEFAULT_ENERGY_BALANCE_DATASET,
    mix_dataset: str = DEFAULT_ELECTRICITY_HEAT_MIX_DATASET,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    force_reparse: bool = False,
    electricity_label_patterns: Sequence[str] = ("^Electricity$", "Electricity"),
    heat_label_patterns: Sequence[str] = ("^Derived heat$", "^Heat$", "Derived heat"),
    siec_direct_patterns: Mapping[str, Sequence[str]] | None = None,
    siec_direct_exclude: Mapping[str, Sequence[str]] | None = None,
    total_siec_code: str = "TOTAL",
) -> EnergyBalanceResult:
    """
    Build a Tabelle1-style table for selected energy-balance categories.

    Parameters
    ----------
    categories:
        Output category labels. Each label must exist as an `nrg_bal` label in the Eurostat energy balance dataset,
        unless overridden via `composite_categories`.
    composite_categories:
        Optional mapping from output category label -> list of component `nrg_bal` labels to sum.
        This is useful for custom aggregates (e.g. "metal and other minerals").
    """
    composites = {str(k): [str(x) for x in v] for k, v in (composite_categories or {}).items()}

    # 1) Load an nrg_bal label map (cheap: siec=TOTAL slice) to resolve category labels -> codes.
    df_total, meta_total = load_eurostat_energy_balance_slice(
        geo=geo,
        time=time,
        nrg_bal=None,
        siec=[str(total_siec_code)],
        unit=unit,
        freq=freq,
        dataset=energy_dataset,
        storage_folder=storage_folder,
        force_download=force_download,
        force_reparse=force_reparse,
    )
    dim_nrg_bal = (meta_total.get("dimensions", {}) or {}).get("nrg_bal", {})
    if not isinstance(dim_nrg_bal, dict) or not dim_nrg_bal:
        raise EurostatParseError(f"Missing nrg_bal dimension metadata in dataset={energy_dataset}.")

    label_to_code: dict[str, str] = {_norm_label(v): str(k) for k, v in dim_nrg_bal.items()}
    code_to_label: dict[str, str] = {str(k): str(v) for k, v in dim_nrg_bal.items()}

    # Build a mapping from output category -> list of nrg_bal codes (to sum).
    category_to_codes: dict[str, list[str]] = {}
    all_component_codes: set[str] = set()
    for cat in categories:
        cat_s = str(cat)
        components = composites.get(cat_s, [cat_s])
        codes: list[str] = []
        for c in components:
            code = _resolve_nrg_bal_code(str(c), label_to_code=label_to_code)
            if code is None:
                keys = sorted(label_to_code.keys())
                sample = ", ".join(keys[:10])
                raise EurostatParseError(
                    f"Could not resolve nrg_bal label '{c}' in dataset={energy_dataset} (geo={geo} time={time}). "
                    f"Example available labels (normalized): {sample}"
                )
            codes.append(code)
        category_to_codes[cat_s] = codes
        all_component_codes.update(codes)

    # 2) Download final consumption slice for all required nrg_bal codes (all siec columns present in slice).
    df_fc, meta_fc = load_eurostat_energy_balance_slice(
        geo=geo,
        time=time,
        nrg_bal=sorted(all_component_codes),
        unit=unit,
        freq=freq,
        dataset=energy_dataset,
        storage_folder=storage_folder,
        force_download=force_download,
        force_reparse=force_reparse,
    )
    dim_siec = (meta_fc.get("dimensions", {}) or {}).get("siec", {})
    if not isinstance(dim_siec, dict) or not dim_siec:
        raise EurostatParseError(f"Missing siec dimension metadata in dataset={energy_dataset}.")
    siec_labels: dict[str, str] = {str(k): str(v) for k, v in dim_siec.items()}

    # 3) Identify electricity and heat siec codes by label.
    elec_codes = _match_codes_by_label(siec_labels, patterns=[str(p) for p in electricity_label_patterns])
    heat_codes = _match_codes_by_label(siec_labels, patterns=[str(p) for p in heat_label_patterns])
    elec_codes = sorted(elec_codes, key=lambda x: (len(x), x))
    heat_codes = sorted(heat_codes, key=lambda x: (len(x), x))
    if not elec_codes:
        raise EurostatParseError(
            f"Could not identify electricity siec code(s) in dataset={energy_dataset}. "
            f"Patterns={list(electricity_label_patterns)}"
        )
    if not heat_codes:
        raise EurostatParseError(
            f"Could not identify heat siec code(s) in dataset={energy_dataset}. "
            f"Patterns={list(heat_label_patterns)}"
        )
    elec_code = elec_codes[0]
    heat_code = heat_codes[0]

    # 4) Identify direct-fuel codes using the same non-overlapping code sets used for the mix.
    # We intentionally avoid aggregates like RA000/BIOE to prevent double counting.
    c_solid = _codes_for_source_bucket("Solid fossil fuels", code_to_label=siec_labels)
    c_gas = _codes_for_source_bucket("Fossil gas", code_to_label=siec_labels)
    c_oil = _codes_for_source_bucket("Oil and petroleum products (excluding biofuel portion)", code_to_label=siec_labels)
    c_hydro = _codes_for_source_bucket("Hydro", code_to_label=siec_labels)
    c_wind = _codes_for_source_bucket("Wind", code_to_label=siec_labels)
    c_solar = _codes_for_source_bucket("Solar", code_to_label=siec_labels)
    c_other = _codes_for_source_bucket("Other Renewables", code_to_label=siec_labels)
    c_waste = _codes_for_source_bucket("Waste", code_to_label=siec_labels)
    c_bio = _codes_for_source_bucket("Biofuels", code_to_label=siec_labels)
    have_ra000 = str(SIEC_RENEWABLES_AND_BIOFUELS_AGG) in set(map(str, siec_labels.keys()))

    # 5) Load electricity/heat production shares by source.
    elec_mix, heat_mix, mix_meta = load_eurostat_electricity_heat_mix(
        geo=geo,
        time=time,
        unit=unit,
        freq=freq,
        dataset=mix_dataset,
        storage_folder=storage_folder,
        force_download=force_download,
        force_reparse=force_reparse,
    )

    # 6) Build Tabelle1.
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

    for cat, codes in category_to_codes.items():
        block = df_fc.loc[codes].sum(axis=0)
        total = float(pd.to_numeric(block.get(str(total_siec_code), 0.0), errors="coerce") or 0.0)
        elec_total = float(block.get(elec_code, 0.0))
        heat_total = float(block.get(heat_code, 0.0))
        direct_total = float(total - elec_total - heat_total)
        if not np.isfinite(direct_total) or direct_total < 0:
            direct_total = 0.0

        direct_vals: dict[str, float] = {
            "Solid fossil fuels": _sum_codes(block, c_solid),
            "Fossil gas": _sum_codes(block, c_gas),
            "Oil and petroleum products (excluding biofuel portion)": _sum_codes(block, c_oil),
            "Hydro": _sum_codes(block, c_hydro),
            "Wind": _sum_codes(block, c_wind),
            "Solar": _sum_codes(block, c_solar),
            "Biofuels": _sum_codes(block, c_bio),
            "Waste": _sum_codes(block, c_waste),
            "Other Renewables": _sum_codes(block, c_other),
        }
        if direct_vals["Biofuels"] == 0.0 and have_ra000:
            ra000 = float(pd.to_numeric(block.get(SIEC_RENEWABLES_AND_BIOFUELS_AGG, 0.0), errors="coerce") or 0.0)
            direct_vals["Biofuels"] = max(
                0.0,
                ra000
                - direct_vals["Hydro"]
                - direct_vals["Wind"]
                - direct_vals["Solar"]
                - direct_vals["Other Renewables"],
            )

        residual = float(direct_total - sum(direct_vals.values()))
        if residual > 0:
            direct_vals["Other Renewables"] += residual

        elec_by_src = (elec_mix * elec_total).to_dict()
        heat_by_src = (heat_mix * heat_total).to_dict()

        total_direct = float(sum(direct_vals.values()))

        row: dict[str, Any] = {"Category": cat, "Total": total}
        for src in PRIMARY_SOURCES:
            row[src] = direct_vals[src]
        row["Electricity:"] = elec_total
        for src in PRIMARY_SOURCES:
            row[f"Electricity::{src}"] = float(elec_by_src.get(src, 0.0))
        row["Heat"] = heat_total
        for src in PRIMARY_SOURCES:
            row[f"Heat::{src}"] = float(heat_by_src.get(src, 0.0))
        rows.append(row)

        diag_rows.append(
            {
                "Category": cat,
                "Elec_total": elec_total,
                "Elec_sources_sum": float(sum(elec_by_src.values())),
                "Elec_residual_total_minus_sources": elec_total - float(sum(elec_by_src.values())),
                "Heat_total": heat_total,
                "Heat_sources_sum": float(sum(heat_by_src.values())),
                "Heat_residual_total_minus_sources": heat_total - float(sum(heat_by_src.values())),
                "Direct_sources_sum": total_direct,
                "Unrolled_sum": float(sum(direct_vals.values())) + float(sum(elec_by_src.values())) + float(sum(heat_by_src.values())),
            }
        )

    tabelle1 = pd.DataFrame(rows)

    # Unrolled: direct + electricity + heat contributions per primary source.
    unrolled = pd.DataFrame({"Category": tabelle1["Category"].astype(str)})
    for src in PRIMARY_SOURCES:
        unrolled[src] = (
            pd.to_numeric(tabelle1[src], errors="coerce").fillna(0.0)
            + pd.to_numeric(tabelle1[f"Electricity::{src}"], errors="coerce").fillna(0.0)
            + pd.to_numeric(tabelle1[f"Heat::{src}"], errors="coerce").fillna(0.0)
        )
    unrolled.insert(1, "Total_unrolled", unrolled[list(PRIMARY_SOURCES)].sum(axis=1))

    diagnostics = pd.DataFrame(diag_rows)
    totals = {
        "Category": "== SUM ==",
        "Elec_total": float(diagnostics["Elec_total"].sum()),
        "Elec_sources_sum": float(diagnostics["Elec_sources_sum"].sum()),
        "Elec_residual_total_minus_sources": float(diagnostics["Elec_residual_total_minus_sources"].sum()),
        "Heat_total": float(diagnostics["Heat_total"].sum()),
        "Heat_sources_sum": float(diagnostics["Heat_sources_sum"].sum()),
        "Heat_residual_total_minus_sources": float(diagnostics["Heat_residual_total_minus_sources"].sum()),
        "Direct_sources_sum": float(diagnostics["Direct_sources_sum"].sum()),
        "Unrolled_sum": float(diagnostics["Unrolled_sum"].sum()),
    }
    diagnostics = pd.concat([diagnostics, pd.DataFrame([totals])], ignore_index=True)

    meta: dict[str, Any] = {
        "energy_dataset": energy_dataset,
        "mix_dataset": mix_dataset,
        "geo": str(geo),
        "time": int(time),
        "unit": str(unit),
        "freq": str(freq),
        "resolved_nrg_bal_codes": category_to_codes,
        "nrg_bal_code_to_label": code_to_label,
        "siec_code_to_label": siec_labels,
        "electricity_siec_code": elec_code,
        "heat_siec_code": heat_code,
        "direct_siec_codes": {
            "solid_fossil": c_solid,
            "fossil_gas": c_gas,
            "oil_ex_bio": c_oil,
            "hydro": c_hydro,
            "wind": c_wind,
            "solar": c_solar,
            "biofuels_detailed": c_bio,
            "waste": c_waste,
            "other_renew": c_other,
        },
        "mix": mix_meta,
    }

    return EnergyBalanceResult(tabelle1=tabelle1, unrolled_sources=unrolled, diagnostics=diagnostics, meta=meta)


def validate_energy_balance_closure(
    diagnostics: pd.DataFrame,
    *,
    tol_abs: float = 1e-6,
) -> None:
    """
    Raise if electricity/heat closure residuals exceed tolerance.
    """
    diag = diagnostics.copy()
    diag = diag[diag["Category"].astype(str) != "== SUM =="]
    elec_bad = diag.loc[diag["Elec_residual_total_minus_sources"].abs() > tol_abs]
    heat_bad = diag.loc[diag["Heat_residual_total_minus_sources"].abs() > tol_abs]
    if not elec_bad.empty or not heat_bad.empty:
        msg = []
        if not elec_bad.empty:
            msg.append(
                "Electricity closure failures: "
                + ", ".join(elec_bad["Category"].astype(str).head(5).tolist())
            )
        if not heat_bad.empty:
            msg.append(
                "Heat closure failures: "
                + ", ".join(heat_bad["Category"].astype(str).head(5).tolist())
            )
        raise ValueError("; ".join(msg))
