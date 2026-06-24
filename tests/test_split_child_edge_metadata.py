from __future__ import annotations

import pytest

from scripts.split_eurostat_energy_by_exiobase import _build_split_child_edge_metadata


def test_split_child_edge_metadata_marks_many_target_child() -> None:
    selected_children = {
        "CPA_C20": ["p24.d", "p24.g"],
    }
    sector_mapping = {
        "CPA_C20": {"p24.d": 0.75, "p24.g": 1.0},
        "CPA_C21": {"p24.d": 0.25},
    }

    frame = _build_split_child_edge_metadata(
        selected_children=selected_children,
        sector_mapping=sector_mapping,
        variant_dim="stk_flow",
        variant_value="DOM",
        exio_flow="domestic",
    )

    by_child = frame.set_index("current_child_code")
    assert by_child.loc["p24.d", "proposed_edge_child_code"] == "p24.d@CPA_C20"
    assert bool(by_child.loc["p24.d", "is_edge_child_candidate"])
    assert int(by_child.loc["p24.d", "source_n_targets"]) == 2
    assert float(by_child.loc["p24.d", "edge_weight_current_reference"]) == pytest.approx(0.75)
    assert by_child.loc["p24.g", "proposed_edge_child_code"] == "p24.g"
    assert not bool(by_child.loc["p24.g", "is_edge_child_candidate"])
    assert int(by_child.loc["p24.g", "source_n_targets"]) == 1


def test_split_child_edge_metadata_flattens_grouped_selected_children() -> None:
    selected_children = {
        "CPA_C20": {
            "biofuels": ["p24.g", "p24.h"],
            "residual": ["p24.d", "p24.d"],
        }
    }
    sector_mapping = {
        "CPA_C20": {"p24.d": 0.6, "p24.g": 1.0, "p24.h": 1.0},
        "CPA_C21": {"p24.d": 0.4},
    }

    frame = _build_split_child_edge_metadata(
        selected_children=selected_children,
        sector_mapping=sector_mapping,
    )

    assert frame["current_child_code"].tolist() == ["p24.g", "p24.h", "p24.d"]
    p24d = frame.loc[frame["current_child_code"].eq("p24.d")].iloc[0]
    assert p24d["proposed_edge_child_code"] == "p24.d@CPA_C20"
    assert bool(p24d["selected_parent_in_concordance"])
