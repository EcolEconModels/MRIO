import pandas as pd


def test_compute_mapping_shares_from_outputs_and_disaggregate():
    from ecolecon_mrio.eurostat_energy_balances import (
        PRIMARY_SOURCES,
        compute_mapping_shares_from_outputs,
        disaggregate_unrolled_sources,
    )

    unrolled = pd.DataFrame(
        [
            {"Category": "A", **{s: 1.0 for s in PRIMARY_SOURCES}},
            {"Category": "B", **{s: 2.0 for s in PRIMARY_SOURCES}},
        ]
    )

    mapping = pd.DataFrame(
        [
            {"Category": "A", "TargetSector": "T1"},
            {"Category": "A", "TargetSector": "T2"},
            {"Category": "B", "TargetSector": "T2"},
        ]
    )

    outputs = {"T1": 1.0, "T2": 3.0}
    mapping_w = compute_mapping_shares_from_outputs(mapping, target_output_by_target=outputs)

    # Category A: shares should be 1/4 and 3/4
    a = mapping_w[mapping_w["Category"] == "A"].set_index("TargetSector")["Share_used"]
    assert abs(a.loc["T1"] - 0.25) < 1e-12
    assert abs(a.loc["T2"] - 0.75) < 1e-12

    out = disaggregate_unrolled_sources(unrolled, mapping_w)
    assert set(out.index) == {"T1", "T2"}
    # All sources identical; check one
    assert abs(out.loc["T1", PRIMARY_SOURCES[0]] - 0.25 * 1.0) < 1e-12
    assert abs(out.loc["T2", PRIMARY_SOURCES[0]] - (0.75 * 1.0 + 1.0 * 2.0)) < 1e-12


def test_norm_label_allows_hyphen():
    from ecolecon_mrio import eurostat_energy_balances as eb

    assert eb._norm_label("Final-consumption") == "final-consumption"
