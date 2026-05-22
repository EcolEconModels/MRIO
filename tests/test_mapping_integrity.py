from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPINGS = REPO_ROOT / "scripts" / "mappings"
PIPELINES = REPO_ROOT / "scripts" / "pipelines"


def _read_csv(name: str, root: Path = MAPPINGS) -> pd.DataFrame:
    return pd.read_csv(root / name, dtype=str).fillna("")


def _active_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "enabled" not in df.columns:
        return df.copy()
    enabled = df["enabled"].astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
    return df.loc[enabled].copy()


def _split_children(split_csv: str, concordance_csv: str) -> set[str]:
    split_df = _active_rows(_read_csv(split_csv, PIPELINES))
    concordance_df = _active_rows(_read_csv(concordance_csv))
    sources = set(split_df["source_code"].astype(str).str.strip())
    children = set()
    for source in sources:
        members = concordance_df.loc[concordance_df["target_code"].astype(str).str.strip() == source, "source_code"]
        children.update(members.astype(str).str.strip())
    return children


def _mapping_by_source_target(df: pd.DataFrame) -> dict[str, set[str]]:
    active = _active_rows(df)
    return {
        source: set(group["target_code"].astype(str).str.strip())
        for source, group in active.groupby(active["source_code"].astype(str).str.strip(), sort=False)
    }


def _active_targets_for(code: str, concordance: pd.DataFrame | None = None) -> set[str]:
    if concordance is None:
        concordance = _active_rows(_read_csv("exio_to_eurostat_cpa_products.csv"))
    mask = concordance["source_code"].astype(str).str.strip() == code
    return set(concordance.loc[mask, "target_code"].astype(str).str.strip())


def test_exio_concordance_covers_all_dictionary_codes_and_biofuels_map_to_c20() -> None:
    exio_dict = _read_csv("exiobase_products_dictionary.csv")
    concordance = _active_rows(_read_csv("exio_to_eurostat_cpa_products.csv"))

    exio_codes = set(exio_dict["code"].astype(str).str.strip())
    mapped_codes = set(concordance["source_code"].astype(str).str.strip())
    missing = sorted(exio_codes - mapped_codes)
    assert not missing, f"Missing EXIO source codes in concordance: {missing[:20]}"

    for code in ["p24.a", "p24.a.w", "p24.b", "p24.c", "p24.d", "p24.e", "p24.f", "p24.g", "p24.h", "p24.i"]:
        assert _active_targets_for(code, concordance) == {"CPA_C20"}, f"{code} must map only to CPA_C20"

    for code in ["p24.e", "p24.f", "p24.g", "p24.h", "p24.i"]:
        targets = _active_targets_for(code, concordance)
        assert "CPA_C21" not in targets, f"{code} must not map to CPA_C21; got {sorted(targets)!r}"


def test_eurostat_dictionary_snapshots_include_provenance_source() -> None:
    for name in ["eurostat_products_dictionary.csv", "eurostat_industries_dictionary.csv"]:
        df = _read_csv(name)
        assert "provenance_source" in df.columns, f"{name} must expose provenance_source"
        assert df["description"].astype(str).str.strip().ne("").all(), f"{name} must keep nonblank descriptions"
        assert df["provenance_source"].astype(str).str.strip().ne("").all(), f"{name} must keep nonblank provenance_source values"


def test_concordance_allocation_rules_are_explicit() -> None:
    concordance = _active_rows(_read_csv("exio_to_eurostat_cpa_products.csv"))
    by_source = concordance.groupby(concordance["source_code"].astype(str).str.strip(), sort=False)

    if "weight" in concordance.columns:
        weights = pd.to_numeric(concordance["weight"], errors="coerce").fillna(0.0)
        sums = weights.groupby(concordance["source_code"].astype(str).str.strip()).sum()
        bad = sums[(sums - 1.0).abs() > 1e-9]
        assert bad.empty, f"Concordance weights must sum to 1 per source: {bad.head(20).to_dict()}"
        return

    bad_sources: list[str] = []
    for source, group in by_source:
        if len(group) == 1:
            continue
        allocation_rules = group.get("allocation_rule", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
        if not allocation_rules.str.contains("split-by-output-share|derived-output-share-split", regex=True).any():
            bad_sources.append(source)
    assert not bad_sources, f"Sources with multiple active rows need explicit split allocation rules: {bad_sources[:20]}"


def test_p24_energy_and_fuel_relevant_block_stays_single_target_c20() -> None:
    concordance = _active_rows(_read_csv("exio_to_eurostat_cpa_products.csv"))
    for code in ["p24.a", "p24.a.w", "p24.b", "p24.c", "p24.d", "p24.e", "p24.f", "p24.g", "p24.h", "p24.i"]:
        assert _active_targets_for(code, concordance) == {"CPA_C20"}, f"{code} must map only to CPA_C20"


def test_green_brown_split_includes_c20_and_reaggregation_covers_children() -> None:
    split = _active_rows(_read_csv("energy_green_brown_split.csv", PIPELINES))
    assert set(split["source_code"].astype(str).str.strip()) == {"CPA_C19", "CPA_C20", "CPA_D"}

    children = _split_children("energy_green_brown_split.csv", "exio_to_eurostat_cpa_products.csv")
    reagg = _active_rows(_read_csv("energy_green_brown_reaggregation.csv", PIPELINES))
    reagg_sources = set(reagg["source_code"].astype(str).str.strip())
    missing = sorted(children - reagg_sources)
    extra = sorted(reagg_sources - children)
    assert not missing, (
        "Green/brown reaggregation is missing split children: "
        f"n_children={len(children)} n_reagg_sources={len(reagg_sources)} missing={missing[:20]} extra={extra[:20]}"
    )

    def _targets_for(code: str) -> set[str]:
        return set(reagg.loc[reagg["source_code"].astype(str).str.strip() == code, "target_code"].astype(str).str.strip())

    for code in ["p24.g", "p24.h", "p24.i"]:
        assert _targets_for(code) == {"ENERGY_GREEN"}, f"{code} must map to ENERGY_GREEN"

    for code in ["p24.a", "p24.a.w", "p24.b", "p24.c", "p24.d", "p24.e", "p24.f"]:
        assert _targets_for(code) == {"CPA_C20_others"}, f"{code} must map to CPA_C20_others"


def _assert_split_liquid_biofuel_semantics(reagg: pd.DataFrame, csv_name: str) -> None:
    reagg_sources = set(reagg["source_code"].astype(str).str.strip())

    for bad in ["p27.a", "p27.a.w", "p27.41", "p27.41.w", "p27.42", "p27.42.w", "p27.43", "p27.43.w", "p27.44", "p27.44.w", "p27.45", "p27.45.w", "p27.5"]:
        assert bad not in reagg_sources, f"{bad} must not appear in {csv_name}"

    expectations = {
        "p24.e": ("CPA_C20_others", "CPA_C20: Chemicals and chemical products_others", "CPA_C20_others"),
        "p24.g": ("CPA_C20_BIOGASOLINE", "CPA_C20: Biogasoline", "assigned to aggregate 'ENERGY_GREEN'"),
        "p24.h": ("CPA_C20_BIODIESEL", "CPA_C20: Biodiesel", "assigned to aggregate 'ENERGY_GREEN'"),
        "p24.i": ("CPA_C20_OTHER_LIQUID_BIOFUELS", "CPA_C20: Other liquid biofuels", "assigned to aggregate 'ENERGY_GREEN'"),
    }
    for code, (target_code, target_label, _) in expectations.items():
        row = reagg.loc[reagg["source_code"].astype(str).str.strip() == code]
        assert not row.empty, f"{code} must be explicitly reaggregated in {csv_name}"
        assert set(row["target_code"].astype(str).str.strip()) == {target_code}
        assert set(row["target_label"].astype(str).str.strip()) == {target_label}
        assert row["notes"].astype(str).str.strip().ne("").all(), f"{code} notes must not be blank in {csv_name}"


def _assert_electricity_target_codes(reagg: pd.DataFrame, csv_name: str) -> None:
    expectations = {
        "p40.11.k": ("CPA_D_Electricity_by_Geothermal", "CPA_D: Electricity by Geothermal"),
        "p40.11.g": ("CPA_D_Electricity_by_biomass_and_waste", "CPA_D: Electricity by biomass and waste"),
        "p40.11.a": ("CPA_D_Electricity_by_coal", "CPA_D: Electricity by coal"),
        "p40.11.b": ("CPA_D_Electricity_by_gas", "CPA_D: Electricity by gas"),
        "p40.11.d": ("CPA_D_Electricity_by_hydro", "CPA_D: Electricity by hydro"),
        "p40.11.c": ("CPA_D_Electricity_by_nuclear", "CPA_D: Electricity by nuclear"),
        "p40.11.f": ("CPA_D_Electricity_by_petroleum_and_other_oil_derivatives", "CPA_D: Electricity by petroleum and other oil derivatives"),
        "p40.11.h": ("CPA_D_Electricity_by_solar_photovoltaic", "CPA_D: Electricity by solar photovoltaic"),
        "p40.11.i": ("CPA_D_Electricity_by_solar_thermal", "CPA_D: Electricity by solar thermal"),
        "p40.11.j": ("CPA_D_Electricity_by_tide_wave_ocean", "CPA_D: Electricity by tide, wave, ocean"),
        "p40.11.e": ("CPA_D_Electricity_by_wind", "CPA_D: Electricity by wind"),
        "p40.11.l": ("CPA_D_Electricity_nec", "CPA_D: Electricity nec"),
    }
    for code, (target_code, target_label) in expectations.items():
        row = reagg.loc[reagg["source_code"].astype(str).str.strip() == code]
        assert not row.empty, f"{code} must be explicitly reaggregated in {csv_name}"
        assert set(row["target_code"].astype(str).str.strip()) == {target_code}
        assert set(row["target_label"].astype(str).str.strip()) == {target_label}


def test_fuels_reaggregation_covers_all_c20_children_and_split_biofuels() -> None:
    children = _split_children("fuels_split.csv", "exio_to_eurostat_cpa_products.csv")
    reagg = _active_rows(_read_csv("fuels_reaggregation.csv", PIPELINES))
    reagg_sources = set(reagg["source_code"].astype(str).str.strip())

    missing = sorted(children - reagg_sources)
    extra = sorted(reagg_sources - children)
    assert not missing, (
        "Fuels reaggregation is missing split children: "
        f"n_children={len(children)} n_reagg_sources={len(reagg_sources)} missing={missing[:20]} extra={extra[:20]}"
    )

    _assert_split_liquid_biofuel_semantics(reagg, "fuels_reaggregation.csv")
    _assert_electricity_target_codes(reagg, "fuels_reaggregation.csv")

    for bad in ["p23.1", "p23.2", "p24.4", "p40.2.2"]:
        assert bad not in reagg_sources, f"{bad} must not appear in fuels_reaggregation.csv"


def test_static_energy_split_templates_keep_liquid_biofuels_out_of_c19() -> None:
    for csv_name in ["exio_energy_split_cpa_with_electricity_source.csv", "exio_energy_split_cpa_no_electricity_source.csv"]:
        split = _active_rows(_read_csv(csv_name))
        biofuel_rows = split.loc[split["source_code"].astype(str).str.strip().isin({"p24.g", "p24.h", "p24.i"})]
        assert not biofuel_rows.empty, f"{csv_name} must keep the liquid biofuel rows"
        assert set(biofuel_rows["target_code"].astype(str).str.strip()) == {"CPA_C20"}, f"{csv_name} must map liquid biofuels to CPA_C20"
        assert set(biofuel_rows["target_label"].astype(str).str.strip()) == {"Chemicals and chemical products"}, f"{csv_name} must label CPA_C20 consistently"
        assert biofuel_rows["notes"].astype(str).str.contains("Liquid biofuels", regex=False).all(), f"{csv_name} must document liquid biofuel semantics"


def test_fuels_more_reaggregation_matches_fuels_semantics() -> None:
    reagg = _active_rows(_read_csv("fuels_more_reaggregation.csv", PIPELINES))
    _assert_split_liquid_biofuel_semantics(reagg, "fuels_more_reaggregation.csv")
    _assert_electricity_target_codes(reagg, "fuels_more_reaggregation.csv")


def test_reaggregation_source_codes_do_not_map_to_multiple_targets() -> None:
    for csv_name in ["energy_green_brown_reaggregation.csv", "fuels_reaggregation.csv", "fuels_more_reaggregation.csv"]:
        df = _active_rows(_read_csv(csv_name, PIPELINES))
        counts = df.groupby(df["source_code"].astype(str).str.strip())["target_code"].nunique()
        bad = counts[counts > 1]
        assert bad.empty, f"{csv_name} maps source_code values to multiple targets: {bad.head(20).to_dict()}"
