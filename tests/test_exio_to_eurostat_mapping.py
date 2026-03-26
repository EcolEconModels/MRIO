from __future__ import annotations


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
