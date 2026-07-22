from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio.disaggregation import SplitProfiles, disaggregate_and_balance
from ecolecon_mrio.energy_balance_pxp import build_column_allocator


def _profile() -> SplitProfiles:
    share = pd.Series([0.5, 0.5], index=["EB_A", "EB_B"], dtype="float64")
    return SplitProfiles(
        aggregate_sector="CPA_C20",
        subsectors=("EB_A", "EB_B"),
        output_share=share.copy(),
        intermediate_sales_share=share.copy(),
        intermediate_input_share=share.copy(),
        final_demand_share=share.copy(),
    )


def test_column_allocator_rejects_bare_non_original_supplier_token() -> None:
    Z = pd.DataFrame([[1.0]], index=["CPA_C10-12"], columns=["CPA_C20"])
    tech = pd.DataFrame(
        [
            {
                "order": 1,
                "enabled": True,
                "parent_code": "CPA_C20",
                "parent_label": "",
                "buyer_split_code": "EB_A",
                "buyer_split_label": "",
                "supplier_selector_system": "target_or_sector_code",
                "supplier_selector": "EB_NOT_AN_ORIGINAL_ROW",
                "allocation_weight": 1.0,
                "apply_to": "Z_COLUMNS",
                "rule": "supplier_specific_column_allocation",
                "fallback_method": "output_share_for_unmatched_suppliers",
                "notes": "",
            }
        ]
    )

    with pytest.raises(ValueError, match="not original Eurostat sector rows"):
        build_column_allocator(Z, tech, split_profiles={"CPA_C20": _profile()})


def test_column_allocator_has_no_implicit_prefix_matching() -> None:
    Z = pd.DataFrame([[1.0]], index=["CPA_C10-12"], columns=["CPA_C20"])
    tech = pd.DataFrame(
        [
            {
                "order": 1,
                "enabled": True,
                "parent_code": "CPA_C20",
                "parent_label": "",
                "buyer_split_code": "EB_A",
                "buyer_split_label": "",
                "supplier_selector_system": "target_or_sector_code",
                "supplier_selector": "CPA_C1",
                "allocation_weight": 1.0,
                "apply_to": "Z_COLUMNS",
                "rule": "supplier_specific_column_allocation",
                "fallback_method": "output_share_for_unmatched_suppliers",
                "notes": "",
            }
        ]
    )

    with pytest.raises(ValueError, match="not original Eurostat sector rows"):
        build_column_allocator(Z, tech, split_profiles={"CPA_C20": _profile()})


def test_column_allocator_exact_match_and_diagnostics() -> None:
    Z = pd.DataFrame([[10.0]], index=["CPA_A01"], columns=["CPA_C20"])
    tech = pd.DataFrame(
        [
            {
                "order": 1,
                "enabled": True,
                "parent_code": "CPA_C20",
                "parent_label": "",
                "buyer_split_code": "EB_A",
                "buyer_split_label": "",
                "supplier_selector_system": "target_or_sector_code",
                "supplier_selector": "CPA_A01",
                "allocation_weight": 1.0,
                "apply_to": "Z_COLUMNS",
                "rule": "supplier_specific_column_allocation",
                "fallback_method": "output_share_for_unmatched_suppliers",
                "notes": "",
            }
        ]
    )

    allocator, diagnostics = build_column_allocator(Z, tech, split_profiles={"CPA_C20": _profile()})
    assert np.isclose(float(allocator["CPA_C20"]["CPA_A01"].loc["EB_A"]), 1.0)
    assert np.isclose(float(allocator["CPA_C20"]["CPA_A01"].loc["EB_B"]), 0.0)
    assert not diagnostics["column_allocator_matches"].empty
    assert diagnostics["column_allocator_summary"].loc[0, "n_matched_supplier_rows"] == 1


def test_disaggregation_preserves_allocator_column_target_in_local_mode() -> None:
    Z = pd.DataFrame(
        [[0.0, 100.0], [0.0, 0.0]],
        index=["SUP", "CPA_C20"],
        columns=["SUP", "CPA_C20"],
    )
    Y = pd.DataFrame([[0.0], [0.0]], index=["SUP", "CPA_C20"], columns=["FD"])
    prof = _profile()
    column_allocator = {
        "CPA_C20": {
            "SUP": pd.Series([1.0, 0.0], index=["EB_A", "EB_B"], dtype="float64"),
            "CPA_C20": pd.Series([0.5, 0.5], index=["EB_A", "EB_B"], dtype="float64"),
        }
    }

    result = disaggregate_and_balance(
        Z,
        Y,
        final_demand_columns=list(Y.columns),
        split_profiles={"CPA_C20": prof},
        column_allocator=column_allocator,
        balancing_scope="local",
        create_io_system=False,
    )

    assert np.isclose(float(result.Z.loc["SUP", "EB_A"]), 100.0)
    assert np.isclose(float(result.Z.loc["SUP", "EB_B"]), 0.0)
