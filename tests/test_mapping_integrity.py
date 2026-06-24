# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportGeneralTypeIssues=false, reportCallIssue=false
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


def _codes(df: pd.DataFrame) -> set[str]:
    return set(df["code"].astype(str).str.strip())


def test_legacy_exio_concordance_covers_all_dictionary_codes() -> None:
    exio_dict = _read_csv("exiobase_products_dictionary.csv")
    concordance = _active_rows(_read_csv("exio_to_eurostat_cpa_products.csv"))

    exio_codes = _codes(exio_dict)
    mapped_codes = set(concordance["source_code"].astype(str).str.strip())
    missing = sorted(exio_codes - mapped_codes)
    assert not missing, f"Missing EXIO source codes in concordance: {missing[:20]}"


def test_committed_cpa64_and_cpa65_target_dictionaries_have_explicit_real_estate_granularity() -> None:
    cpa64 = _read_csv("eurostat_cpa64_products_dictionary.csv")
    cpa65 = _read_csv("eurostat_cpa65_products_dictionary.csv")

    cpa64_codes = _codes(cpa64)
    cpa65_codes = _codes(cpa65)

    assert len(cpa64_codes) == 64
    assert len(cpa65_codes) == 65
    assert "CPA_L68" in cpa64_codes
    assert {"CPA_L68A", "CPA_L68B"}.isdisjoint(cpa64_codes)
    assert "CPA_L68" not in cpa65_codes
    assert {"CPA_L68A", "CPA_L68B"}.issubset(cpa65_codes)
    assert cpa64_codes - {"CPA_L68"} == cpa65_codes - {"CPA_L68A", "CPA_L68B"}

    for name, df in [("eurostat_cpa64_products_dictionary.csv", cpa64), ("eurostat_cpa65_products_dictionary.csv", cpa65)]:
        assert df["code"].astype(str).str.strip().ne("").all(), f"{name} has blank code(s)"
        assert df["label"].astype(str).str.strip().ne("").all(), f"{name} has blank label(s)"
        assert df["description"].astype(str).str.strip().ne("").all(), f"{name} has blank description(s)"


def test_eurostat_dictionary_snapshots_include_provenance_source() -> None:
    for name in ["eurostat_products_dictionary.csv", "eurostat_industries_dictionary.csv"]:
        df = _read_csv(name)
        assert "provenance_source" in df.columns, f"{name} must expose provenance_source"
        assert df["description"].astype(str).str.strip().ne("").all(), f"{name} must keep nonblank descriptions"
        assert df["provenance_source"].astype(str).str.strip().ne("").all(), f"{name} must keep nonblank provenance_source values"


def test_concordance_allocation_rules_are_explicit() -> None:
    concordance = _active_rows(_read_csv("exio_to_eurostat_cpa_products.csv"))
    source_codes = concordance["source_code"].astype(str).str.strip()

    if "weight" in concordance.columns:
        weights = pd.Series(
            pd.to_numeric(concordance["weight"], errors="coerce"),
            index=concordance.index,
            dtype="float64",
        ).fillna(0.0)
        sums = weights.groupby(source_codes).sum()
        bad_by_source: dict[str, float] = {}
        for source, total in sums.items():
            numeric_total = float(total)
            if abs(numeric_total - 1.0) > 1e-9:
                bad_by_source[str(source)] = numeric_total
        assert not bad_by_source, f"Concordance weights must sum to 1 per source: {dict(list(bad_by_source.items())[:20])}"
        return

    bad_sources: list[str] = []
    for source, group in concordance.groupby(source_codes, sort=False):
        if len(group) == 1:
            continue
        if "allocation_rule" in group.columns:
            allocation_rules = group["allocation_rule"].astype(str).str.strip().str.lower()
        else:
            allocation_rules = pd.Series([], dtype=str)
        if not allocation_rules.str.contains("split-by-output-share|derived-output-share-split", regex=True).any():
            bad_sources.append(str(source))
    assert not bad_sources, f"Sources with multiple active rows need explicit split allocation rules: {bad_sources[:20]}"


def test_reaggregation_source_codes_do_not_map_to_multiple_targets() -> None:
    for csv_name in ["energy_green_brown_reaggregation.csv", "fuels_reaggregation.csv", "fuels_more_reaggregation.csv"]:
        df = _active_rows(_read_csv(csv_name, PIPELINES))
        counts = df.groupby(df["source_code"].astype(str).str.strip())["target_code"].nunique()
        bad = counts[counts > 1]
        assert bad.empty, f"{csv_name} maps source_code values to multiple targets: {bad.head(20).to_dict()}"
