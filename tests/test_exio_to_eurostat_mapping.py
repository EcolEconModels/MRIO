from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def test_build_exio_to_eurostat_mapping_uses_validated_official_override():
    from ecolecon_mrio.exio_to_eurostat_mapping import build_exio_to_eurostat_mapping

    exio_names = [
        "Nuclear fuel",
    ]
    exio_code_map = {
        "Nuclear fuel": "p23.3",
    }
    cpa_codes = ["CPA_C19", "CPA_C24"]
    cpa_label_map = {code: code for code in cpa_codes}

    df = build_exio_to_eurostat_mapping(
        exio_names,
        cpa_codes,
        cpa_label_map,
        exio_code_map=exio_code_map,
    )

    got = dict(zip(df["exio_sector"], df["aggregate_sector"]))
    assert got["Nuclear fuel"] == "CPA_C24"


def test_candidate_cpa_codes_for_exio_keeps_required_fallback_splits():
    from ecolecon_mrio.exio_to_eurostat_mapping import build_cpa_division_index, candidate_cpa_codes_for_exio

    division_index = build_cpa_division_index(
        ["CPA_C18", "CPA_C19", "CPA_C20", "CPA_C21", "CPA_C24", "CPA_C28", "CPA_C33", "CPA_H53", "CPA_J58", "CPA_J61", "CPA_J59_60", "CPA_R90-92", "CPA_R93", "CPA_S95", "CPA_S96"]
    )

    assert candidate_cpa_codes_for_exio("p22", division_index=division_index) == ["CPA_C18", "CPA_J58"]
    assert candidate_cpa_codes_for_exio("p24.d", division_index=division_index) == ["CPA_C20", "CPA_C21"]
    assert candidate_cpa_codes_for_exio("p29", division_index=division_index) == ["CPA_C28", "CPA_C33"]
    assert candidate_cpa_codes_for_exio("p64", division_index=division_index) == ["CPA_H53", "CPA_J61"]
    assert candidate_cpa_codes_for_exio("p92", division_index=division_index) == ["CPA_J59_60", "CPA_R90-92", "CPA_R93"]
    assert candidate_cpa_codes_for_exio("p93", division_index=division_index) == ["CPA_S95", "CPA_S96"]


def test_exiobase_dictionary_covers_official_sector_list():
    from ecolecon_mrio.exio_concordance import load_pymrio_exio3_sectors

    repo_root = Path(__file__).resolve().parents[1]
    dictionary = pd.read_csv(repo_root / "scripts" / "mappings" / "exiobase_products_dictionary.csv", dtype=str).fillna("")
    official = load_pymrio_exio3_sectors("pxp").sectors

    dict_codes = set(dictionary["code"].astype(str).str.strip())
    missing = official[~official["ExioCode"].astype(str).str.strip().isin(dict_codes)]

    assert missing.empty


def test_exiobase_industries_dictionary_covers_official_sector_list():
    from ecolecon_mrio.exio_concordance import load_pymrio_exio3_sectors

    repo_root = Path(__file__).resolve().parents[1]
    dictionary = pd.read_csv(repo_root / "scripts" / "mappings" / "exiobase_industries_dictionary.csv", dtype=str).fillna("")
    official = load_pymrio_exio3_sectors("ixi").sectors

    dict_codes = set(dictionary["code"].astype(str).str.strip())
    missing = official[~official["ExioCode"].astype(str).str.strip().isin(dict_codes)]

    assert missing.empty


def test_exio_to_eurostat_mapping_has_no_blank_source_codes():
    repo_root = Path(__file__).resolve().parents[1]
    source_codes: set[str] = set()
    with (repo_root / "scripts" / "mappings" / "exio_to_eurostat_cpa_products.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_code = str(row.get("source_code", "")).strip()
            assert source_code != ""
            source_codes.add(source_code)

    assert "p10.a" in source_codes
    assert "p23.1.a" in source_codes
    assert "p40.2.a" in source_codes


def test_exio_to_eurostat_mapping_has_nonblank_source_codes():
    repo_root = Path(__file__).resolve().parents[1]
    source_codes: list[str] = []
    allocation_rules: list[str] = []
    with (repo_root / "scripts" / "mappings" / "exio_to_eurostat_cpa_products.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_codes.append(str(row.get("source_code", "")).strip())
            allocation_rules.append(str(row.get("allocation_rule", "")).strip())

    assert all(source_codes)
    assert all(allocation_rules)
    assert "split-by-output-share" in allocation_rules

def test_reviewed_official_direct_production_concordance_changes_only_accepted_sources() -> None:
    """The production map narrows only p23.3, p24.*, and p33 target topology."""
    root = Path(__file__).resolve().parents[1]
    raw = pd.read_csv(
        root
        / "scripts/mappings/generated/official_direct/exio_to_eurostat_cpa65_products_from_official_direct.csv",
        dtype=str,
        keep_default_na=False,
    )
    reviewed = pd.read_csv(
        root / "scripts/mappings/exio_to_eurostat_cpa65_products_official_direct_reviewed.csv",
        dtype=str,
        keep_default_na=False,
    )

    def target_sets(frame: pd.DataFrame) -> dict[str, set[str]]:
        return {
            str(source): set(group["target_code"].astype(str))
            for source, group in frame.groupby("source_code", sort=False)
        }

    raw_targets = target_sets(raw)
    reviewed_targets = target_sets(reviewed)
    assert set(reviewed_targets) == set(raw_targets)

    p24_sources = {source for source in raw_targets if source.startswith("p24.")}
    reviewed_sources = p24_sources | {"p23.3", "p33"}
    for source in set(raw_targets) - reviewed_sources:
        assert reviewed_targets[source] == raw_targets[source]
        raw_rows = raw.loc[raw["source_code"] == source].to_dict(orient="records")
        reviewed_rows = reviewed.loc[reviewed["source_code"] == source].to_dict(orient="records")
        assert reviewed_rows == raw_rows

    assert reviewed_targets["p23.3"] == {"CPA_C24"}
    assert reviewed_targets["p33"] == {"CPA_C26"}
    assert reviewed_targets["p24.d"] == {"CPA_C20", "CPA_C21"}
    for source in p24_sources - {"p24.d"}:
        assert reviewed_targets[source] == {"CPA_C20"}

    p24_d = reviewed.loc[reviewed["source_code"] == "p24.d"]
    assert set(p24_d["allocation_rule"]) == {"split-by-output-share"}
    singles = reviewed.loc[
        reviewed["source_code"].isin(
            tuple((p24_sources - {"p24.d"}) | {"p23.3", "p33"})
        )
    ]
    assert set(singles["allocation_rule"]) == {"direct-single-target"}


def test_reviewed_official_direct_production_concordance_preserves_domains() -> None:
    """Target narrowing must retain all EXIO products and all CPA65 target sectors."""
    root = Path(__file__).resolve().parents[1]
    raw = pd.read_csv(
        root
        / "scripts/mappings/generated/official_direct/exio_to_eurostat_cpa65_products_from_official_direct.csv",
        dtype=str,
        keep_default_na=False,
    )
    reviewed = pd.read_csv(
        root / "scripts/mappings/exio_to_eurostat_cpa65_products_official_direct_reviewed.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert set(reviewed["source_code"]) == set(raw["source_code"])
    assert set(reviewed["target_code"]) == set(raw["target_code"])
