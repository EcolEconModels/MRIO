from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPINGS = REPO_ROOT / "scripts" / "mappings"


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(MAPPINGS / name, dtype=str).fillna("")


def _active_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "enabled" not in df.columns:
        return df.copy()
    enabled = df["enabled"].astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
    return df.loc[enabled].copy()


def _split_children(split_csv: str, concordance_csv: str) -> set[str]:
    split_df = _active_rows(_read_csv(split_csv))
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
    split = _active_rows(_read_csv("energy_green_brown_split.csv"))
    assert set(split["source_code"].astype(str).str.strip()) == {"CPA_C19", "CPA_C20", "CPA_D"}

    children = _split_children("energy_green_brown_split.csv", "exio_to_eurostat_cpa_products.csv")
    reagg = _active_rows(_read_csv("energy_green_brown_reaggregation.csv"))
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


def test_fuels_reaggregation_covers_all_c20_children_and_biofuels() -> None:
    children = _split_children("fuels_split.csv", "exio_to_eurostat_cpa_products.csv")
    reagg = _active_rows(_read_csv("fuels_reaggregation.csv"))
    reagg_sources = set(reagg["source_code"].astype(str).str.strip())

    missing = sorted(children - reagg_sources)
    extra = sorted(reagg_sources - children)
    assert not missing, (
        "Fuels reaggregation is missing split children: "
        f"n_children={len(children)} n_reagg_sources={len(reagg_sources)} missing={missing[:20]} extra={extra[:20]}"
    )

    for code in ["p24.g", "p24.h", "p24.i"]:
        row = reagg.loc[reagg["source_code"].astype(str).str.strip() == code]
        assert not row.empty, f"{code} must be explicitly reaggregated in fuels_reaggregation.csv"
        assert set(row["target_code"].astype(str).str.strip()) == {"CPA_C20_BIOFUELS"}

    for bad in ["p23.1", "p23.2", "p24.4", "p40.2.2"]:
        assert bad not in reagg_sources, f"{bad} must not appear in fuels_reaggregation.csv"


def test_reaggregation_source_codes_do_not_map_to_multiple_targets() -> None:
    for csv_name in ["energy_green_brown_reaggregation.csv", "fuels_reaggregation.csv"]:
        df = _active_rows(_read_csv(csv_name))
        counts = df.groupby(df["source_code"].astype(str).str.strip())["target_code"].nunique()
        bad = counts[counts > 1]
        assert bad.empty, f"{csv_name} maps source_code values to multiple targets: {bad.head(20).to_dict()}"
