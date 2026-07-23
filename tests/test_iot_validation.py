from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio.iot_validation import (
    InvariantReport,
    RowIdentityPolicy,
    validate_structural_and_row_invariants,
)


def _valid_blocks() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.Index(["a", "b"])
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx, columns=idx)
    Y = pd.DataFrame(
        {
            "P3": [3.0, 5.0],
            "P3_S13": [1.0, 2.0],
            "P3_S14": [2.0, 3.0],
            "TFU": [3.0, 5.0],
            "TU": [6.0, 12.0],
        },
        index=idx,
    )
    V = pd.DataFrame(
        {
            "a": [4.0, 2.0, 6.0],
            "b": [9.0, 3.0, 12.0],
        },
        index=pd.Index(["P1", "P7", "TS_BP"]),
    )
    return Z, Y, V


def test_report_collects_checks_and_require_ok() -> None:
    Z, Y, V = _valid_blocks()
    report = validate_structural_and_row_invariants(
        Z=Z,
        Y=Y,
        V=V,
        flow_role="TOTAL",
        final_demand_columns=["P3_S13", "P3_S14"],
        policy=RowIdentityPolicy(
            total_final_use_column="TFU",
            total_use_column="TU",
            output_row="P1",
            imports_row="P7",
            total_supply_basic_row="TS_BP",
        ),
    )
    assert report.ok
    report.require_ok()
    frame = report.to_frame()
    assert set(frame["status"]) == {"pass"}
    assert set(frame["check_id"]) == {
        "row.total_final_use",
        "row.total_use",
        "row.total_supply_equals_use",
        "row.total_supply_components",
    }


def test_generated_row_failure_is_reported_without_early_raise() -> None:
    Z, Y, V = _valid_blocks()
    Y.loc["a", "TU"] = 7.0
    V.loc["TS_BP", "b"] = 14.0
    report = validate_structural_and_row_invariants(
        Z=Z,
        Y=Y,
        V=V,
        flow_role="TOTAL",
        final_demand_columns=["P3_S13", "P3_S14"],
        policy=RowIdentityPolicy(
            total_final_use_column="TFU",
            total_use_column="TU",
            output_row="P1",
            imports_row="P7",
            total_supply_basic_row="TS_BP",
        ),
    )
    assert not report.ok
    failed = {check.check_id: check for check in report.checks if check.status == "fail"}
    assert failed["row.total_use"].affected_labels == ("a",)
    assert failed["row.total_supply_equals_use"].affected_labels == ("b",)
    with pytest.raises(ValueError, match="row.total_use"):
        report.require_ok()


def test_dom_and_imp_row_totals_have_distinct_economic_roles() -> None:
    idx = pd.Index(["a"])
    Z = pd.DataFrame([[2.0]], index=idx, columns=idx)
    Y = pd.DataFrame({"FD": [3.0]}, index=idx)
    domestic = pd.DataFrame([[5.0]], index=pd.Index(["P1"]), columns=idx)
    imported = pd.DataFrame([[5.0]], index=pd.Index(["P7"]), columns=idx)

    dom_report = validate_structural_and_row_invariants(
        Z=Z,
        Y=Y,
        V=domestic,
        flow_role="DOM",
        final_demand_columns=["FD"],
        policy=RowIdentityPolicy(output_row="P1"),
    )
    imp_report = validate_structural_and_row_invariants(
        Z=Z,
        Y=Y,
        V=imported,
        flow_role="IMP",
        final_demand_columns=["FD"],
        policy=RowIdentityPolicy(imports_row="P7"),
    )
    assert dom_report.ok
    assert imp_report.ok
    assert any(check.check_id == "row.domestic_output" for check in dom_report.checks)
    assert any(check.check_id == "row.imported_supply" for check in imp_report.checks)


def test_structural_failures_reject_missing_or_nonfinite_atomic_cells() -> None:
    idx = pd.Index(["a"])
    Z = pd.DataFrame([[1.0]], index=idx, columns=idx)
    Y = pd.DataFrame({"FD": [np.nan]}, index=idx)
    with pytest.raises(ValueError, match="non-finite"):
        validate_structural_and_row_invariants(
            Z=Z,
            Y=Y,
            flow_role="DOM",
            final_demand_columns=["FD"],
        )
    with pytest.raises(ValueError, match="absent from Y"):
        validate_structural_and_row_invariants(
            Z=Z,
            Y=pd.DataFrame({"FD": [1.0]}, index=idx),
            flow_role="DOM",
            final_demand_columns=["missing"],
        )


def test_undeclared_optional_controls_are_not_applicable() -> None:
    idx = pd.Index(["a"])
    report = validate_structural_and_row_invariants(
        Z=pd.DataFrame([[1.0]], index=idx, columns=idx),
        Y=pd.DataFrame({"FD": [2.0]}, index=idx),
        flow_role="DOM",
        final_demand_columns=["FD"],
    )
    assert report.ok
    assert all(check.status == "not_applicable" for check in report.checks)
    assert isinstance(report, InvariantReport)
