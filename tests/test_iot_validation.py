from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ecolecon_mrio.iot_validation as iot_validation_module

from ecolecon_mrio.iot_validation import (
    ColumnIdentityPolicy,
    CrossFlowPolicy,
    InvariantReport,
    RowIdentityPolicy,
    validate_column_invariants,
    validate_cross_flow_invariants,
    validate_structural_and_row_invariants,
    AdditivePartition,
    FinalDemandPartitionPolicy,
    ValueAddedPartitionPolicy,
    validate_final_demand_partitions,
    validate_value_added_partitions,
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
            supply_equals_use_row="TS_BP",
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
            supply_equals_use_row="TS_BP",
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


def test_column_accounts_use_explicit_valuation_adjustments() -> None:
    idx = pd.Index(["a", "b"])
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx, columns=idx)
    V = pd.DataFrame(
        {
            "a": [0.5, 4.5, 1.5, 6.0],
            "b": [1.0, 7.0, 2.0, 9.0],
        },
        index=pd.Index(["TAX", "P2_ADJ", "B1G", "P1"]),
    )
    report = validate_column_invariants(
        Z=Z,
        V=V,
        flow_role="TOTAL",
        policy=ColumnIdentityPolicy(
            intermediate_consumption_row="P2_ADJ",
            gross_value_added_row="B1G",
            output_row="P1",
            valuation_adjustment_rows=("TAX",),
        ),
    )
    assert report.ok
    assert {check.check_id for check in report.checks} == {
        "column.intermediate_consumption",
        "column.output_components",
    }


def test_column_accounts_do_not_infer_valuation_adjustments() -> None:
    idx = pd.Index(["a"])
    Z = pd.DataFrame([[4.0]], index=idx, columns=idx)
    V = pd.DataFrame(
        [[2.0], [4.0], [3.0], [7.0]],
        index=pd.Index(["TAX", "P2", "B1G", "P1"]),
        columns=idx,
    )
    report = validate_column_invariants(
        Z=Z,
        V=V,
        flow_role="TOTAL",
        policy=ColumnIdentityPolicy(
            intermediate_consumption_row="P2",
            gross_value_added_row="B1G",
            output_row="P1",
        ),
    )
    assert report.ok


def test_column_accounts_collect_specific_failed_columns() -> None:
    idx = pd.Index(["a", "b"])
    Z = pd.DataFrame([[1.0, 0.0], [2.0, 3.0]], index=idx, columns=idx)
    V = pd.DataFrame(
        {"a": [3.0, 2.0, 5.0], "b": [4.0, 1.0, 7.0]},
        index=pd.Index(["P2", "B1G", "P1"]),
    )
    report = validate_column_invariants(
        Z=Z,
        V=V,
        flow_role="TOTAL",
        policy=ColumnIdentityPolicy(
            intermediate_consumption_row="P2",
            gross_value_added_row="B1G",
            output_row="P1",
        ),
    )
    failed = {check.check_id: check for check in report.checks if check.status == "fail"}
    assert failed["column.intermediate_consumption"].affected_labels == ("b",)
    assert failed["column.output_components"].affected_labels == ("b",)


def test_column_policy_rejects_duplicate_or_missing_adjustment_rows() -> None:
    idx = pd.Index(["a"])
    Z = pd.DataFrame([[1.0]], index=idx, columns=idx)
    V = pd.DataFrame([[1.0]], index=pd.Index(["P2"]), columns=idx)
    with pytest.raises(ValueError, match="contains duplicates"):
        validate_column_invariants(
            Z=Z,
            V=V,
            flow_role="TOTAL",
            policy=ColumnIdentityPolicy(valuation_adjustment_rows=("TAX", "TAX")),
        )
    with pytest.raises(ValueError, match="valuation adjustment row is absent"):
        validate_column_invariants(
            Z=Z,
            V=V,
            flow_role="TOTAL",
            policy=ColumnIdentityPolicy(valuation_adjustment_rows=("TAX",)),
        )


def test_undeclared_column_controls_are_not_applicable() -> None:
    idx = pd.Index(["a"])
    report = validate_column_invariants(
        Z=pd.DataFrame([[1.0]], index=idx, columns=idx),
        V=None,
        flow_role="DOM",
    )
    assert report.ok
    assert all(check.status == "not_applicable" for check in report.checks)


def _valid_cross_flow_blocks() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.Index(["a", "b"])
    fd = pd.Index(["FD1", "FD2"])
    domestic_Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx, columns=idx)
    imports_Z = pd.DataFrame([[0.5, 1.0], [1.5, 2.0]], index=idx, columns=idx)
    domestic_Y = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=idx, columns=fd)
    imports_Y = pd.DataFrame([[0.5, 1.0], [1.0, 0.5]], index=idx, columns=fd)
    total_Z = domestic_Z + imports_Z
    total_Y = domestic_Y + imports_Y
    total_V = pd.DataFrame([domestic_Z.sum(axis=1) + domestic_Y.sum(axis=1), imports_Z.sum(axis=1) + imports_Y.sum(axis=1), imports_Z.sum(axis=0)], index=pd.Index(["P1", "P7", "UI"]), columns=idx)
    total_M = pd.DataFrame([imports_Y.sum(axis=0)], index=pd.Index(["UI"]), columns=fd)
    return total_Z, total_Y, domestic_Z, domestic_Y, imports_Z, imports_Y, total_V, total_M


def test_cross_flow_validates_complete_triplet_and_import_bridges() -> None:
    total_Z, total_Y, domestic_Z, domestic_Y, imports_Z, imports_Y, total_V, total_M = _valid_cross_flow_blocks()
    report = validate_cross_flow_invariants(total_Z=total_Z, total_Y=total_Y, domestic_Z=domestic_Z, domestic_Y=domestic_Y, imports_Z=imports_Z, imports_Y=imports_Y, final_demand_columns=["FD1", "FD2"], total_V=total_V, total_M=total_M, policy=CrossFlowPolicy(total_output_row="P1", total_imports_row="P7", imported_use_intermediate_row="UI", imported_use_final_row="UI"))
    assert report.ok
    assert {check.check_id for check in report.checks} == {"cross_flow.intermediate_transactions", "cross_flow.final_uses", "cross_flow.domestic_output_bridge", "cross_flow.imported_supply_bridge", "cross_flow.imported_intermediate_use", "cross_flow.imported_final_use"}


def test_cross_flow_reports_specific_transaction_cells() -> None:
    total_Z, total_Y, domestic_Z, domestic_Y, imports_Z, imports_Y, _, _ = _valid_cross_flow_blocks()
    total_Z.loc["a", "b"] += 1.0
    total_Y.loc["b", "FD2"] += 2.0
    report = validate_cross_flow_invariants(total_Z=total_Z, total_Y=total_Y, domestic_Z=domestic_Z, domestic_Y=domestic_Y, imports_Z=imports_Z, imports_Y=imports_Y, final_demand_columns=["FD1", "FD2"])
    failed = {check.check_id: check for check in report.checks if check.status == "fail"}
    assert failed["cross_flow.intermediate_transactions"].affected_labels == ("a|b",)
    assert failed["cross_flow.final_uses"].affected_labels == ("b|FD2",)


def test_comparison_rejects_silent_series_domain_loss() -> None:
    observed = pd.Series([1.0], index=pd.Index(["a"]))
    derived = pd.Series([1.0, 2.0], index=pd.Index(["a", "extra"]))
    with pytest.raises(ValueError, match="identical ordered labels"):
        iot_validation_module._comparison_check(
            check_id="test.domain",
            category="test",
            flow_role="TOTAL",
            block="test",
            observed=observed,
            derived=derived,
            rtol=1e-9,
            atol=1e-6,
            explanation="test",
        )


def test_cross_flow_reports_import_control_failures_separately() -> None:
    total_Z, total_Y, domestic_Z, domestic_Y, imports_Z, imports_Y, total_V, total_M = _valid_cross_flow_blocks()
    total_V.loc["P7", "a"] += 1.0
    total_V.loc["UI", "b"] += 1.0
    total_M.loc["UI", "FD1"] += 1.0
    report = validate_cross_flow_invariants(total_Z=total_Z, total_Y=total_Y, domestic_Z=domestic_Z, domestic_Y=domestic_Y, imports_Z=imports_Z, imports_Y=imports_Y, final_demand_columns=["FD1", "FD2"], total_V=total_V, total_M=total_M, policy=CrossFlowPolicy(total_imports_row="P7", imported_use_intermediate_row="UI", imported_use_final_row="UI"))
    failed = {check.check_id: check for check in report.checks if check.status == "fail"}
    assert failed["cross_flow.imported_supply_bridge"].affected_labels == ("a",)
    assert failed["cross_flow.imported_intermediate_use"].affected_labels == ("b",)
    assert failed["cross_flow.imported_final_use"].affected_labels == ("FD1",)


def test_cross_flow_optional_controls_are_not_applicable() -> None:
    total_Z, total_Y, domestic_Z, domestic_Y, imports_Z, imports_Y, _, _ = _valid_cross_flow_blocks()
    report = validate_cross_flow_invariants(total_Z=total_Z, total_Y=total_Y, domestic_Z=domestic_Z, domestic_Y=domestic_Y, imports_Z=imports_Z, imports_Y=imports_Y, final_demand_columns=["FD1", "FD2"])
    assert report.ok
    assert sum(check.status == "not_applicable" for check in report.checks) == 4


def test_generated_final_demand_partitions_validate_multiple_aggregates() -> None:
    idx = pd.Index(["a", "b"])
    Y = pd.DataFrame(
        {
            "P3_S13": [1.0, 2.0],
            "P3_S14": [2.0, 1.0],
            "P3": [3.0, 3.0],
            "P51G": [4.0, 1.0],
            "P52": [1.0, -0.5],
            "P5": [5.0, 0.5],
        },
        index=idx,
    )
    report = validate_final_demand_partitions(
        Y=Y,
        flow_role="TOTAL",
        policy=FinalDemandPartitionPolicy(
            partitions=(
                AdditivePartition(
                    check_id="partition.final_consumption",
                    aggregate="P3",
                    components=("P3_S13", "P3_S14"),
                ),
                AdditivePartition(
                    check_id="partition.capital_formation",
                    aggregate="P5",
                    components=("P51G", "P52"),
                ),
            )
        ),
    )
    assert report.ok
    assert {check.check_id for check in report.checks} == {
        "partition.final_consumption",
        "partition.capital_formation",
    }


def test_generated_final_demand_partition_reports_affected_products() -> None:
    idx = pd.Index(["a", "b"])
    Y = pd.DataFrame(
        {"P3_A": [1.0, 2.0], "P3_B": [2.0, 3.0], "P3": [3.0, 6.0]},
        index=idx,
    )
    report = validate_final_demand_partitions(
        Y=Y,
        flow_role="DOM",
        policy=FinalDemandPartitionPolicy(
            partitions=(
                AdditivePartition(
                    check_id="partition.final_consumption",
                    aggregate="P3",
                    components=("P3_A", "P3_B"),
                ),
            )
        ),
    )
    assert not report.ok
    assert report.checks[0].affected_labels == ("b",)


def test_partition_policy_rejects_ambiguous_definitions() -> None:
    idx = pd.Index(["a"])
    Y = pd.DataFrame({"P3": [1.0], "P3_A": [1.0]}, index=idx)
    with pytest.raises(ValueError, match="duplicate components"):
        validate_final_demand_partitions(
            Y=Y,
            flow_role="TOTAL",
            policy=FinalDemandPartitionPolicy(
                partitions=(
                    AdditivePartition(
                        check_id="partition.p3",
                        aggregate="P3",
                        components=("P3_A", "P3_A"),
                    ),
                )
            ),
        )
    with pytest.raises(ValueError, match="includes its aggregate"):
        validate_final_demand_partitions(
            Y=Y,
            flow_role="TOTAL",
            policy=FinalDemandPartitionPolicy(
                partitions=(
                    AdditivePartition(
                        check_id="partition.p3",
                        aggregate="P3",
                        components=("P3",),
                    ),
                )
            ),
        )


def test_generated_value_added_partition_supports_signed_components() -> None:
    columns = pd.Index(["a", "b"])
    V = pd.DataFrame(
        {
            "a": [3.0, -0.5, 1.0, 2.5, 6.0],
            "b": [4.0, -1.0, 2.0, 3.0, 8.0],
        },
        index=pd.Index(["D1", "D29_D39", "P51C", "B2A3N", "B1G"]),
    )
    report = validate_value_added_partitions(
        V=V,
        flow_role="TOTAL",
        policy=ValueAddedPartitionPolicy(
            partitions=(
                AdditivePartition(
                    check_id="partition.gross_value_added",
                    aggregate="B1G",
                    components=("D1", "D29_D39", "P51C", "B2A3N"),
                ),
            )
        ),
    )
    assert report.ok


def test_value_added_partition_reports_columns_and_optional_policy() -> None:
    columns = pd.Index(["a", "b"])
    V = pd.DataFrame(
        {"a": [2.0, 3.0, 5.0], "b": [1.0, 4.0, 6.0]},
        index=pd.Index(["D1", "B2A3G", "B1G"]),
    )
    report = validate_value_added_partitions(
        V=V,
        flow_role="DOM",
        policy=ValueAddedPartitionPolicy(
            partitions=(
                AdditivePartition(
                    check_id="partition.gross_value_added",
                    aggregate="B1G",
                    components=("D1", "B2A3G"),
                ),
            )
        ),
    )
    assert report.checks[0].affected_labels == ("b",)
    empty = validate_value_added_partitions(V=None, flow_role="DOM")
    assert empty.ok
    assert empty.checks[0].status == "not_applicable"
