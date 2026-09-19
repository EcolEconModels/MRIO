from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio.iot_flow_semantics import flow_use_total, technical_coefficients

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "derive_total_iot_flow.py"
SPEC = importlib.util.spec_from_file_location("derive_total_iot_flow", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_flow(root: Path, flow: str, scale: float) -> None:
    d = root / f"stk_flow={flow}"
    d.mkdir(parents=True)
    idx = pd.Index(["a", "b"])
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx, columns=idx) * scale
    Y = pd.DataFrame([[5.0], [6.0]], index=idx, columns=pd.Index(["P3"])) * scale
    V = pd.DataFrame([[7.0, 8.0]], index=pd.Index(["D1"]), columns=idx) * scale
    M = pd.DataFrame([[9.0]], index=pd.Index(["TU"]), columns=pd.Index(["P3"])) * scale
    for name, frame in (("Z_split.csv", Z), ("Y_split.csv", Y), ("V_split.csv", V), ("M_split.csv", M)):
        frame.to_csv(d / name)
    (d / "exio_selected_sectors.json").write_text(json.dumps({"a": ["a"], "b": ["b"]}), encoding="utf-8")


def _write_canonical_sidecars(root: Path, *, suffix: str) -> None:
    z_name = f"Z_{suffix}.csv"
    y_name = f"Y_{suffix}.csv"
    x_name = f"x_{suffix}.csv"
    a_name = f"A_{suffix}.csv"
    use_name = f"use_total_{suffix}.csv"
    dom = root / "stk_flow=DOM"
    output = flow_use_total(pd.read_csv(dom / z_name, index_col=0), pd.read_csv(dom / y_name, index_col=0)).rename("x")
    for flow in ("DOM", "IMP", "TOTAL"):
        directory = root / f"stk_flow={flow}"
        Z = pd.read_csv(directory / z_name, index_col=0)
        Y = pd.read_csv(directory / y_name, index_col=0)
        flow_use_total(Z, Y).to_csv(directory / use_name, header=True)
        output.to_csv(directory / x_name, header=True)
        technical_coefficients(Z, output).to_csv(directory / a_name)


def test_validate_additive_outputs_detects_and_accepts_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This focused fixture exercises additive file identities only; the complete
    # naio_10_cp1700 gate has its own real-schema fixtures.
    monkeypatch.setattr(MODULE, "_validate_generated_full_table", lambda **_: None)
    _write_flow(tmp_path, "DOM", 1.0)
    _write_flow(tmp_path, "IMP", 2.0)
    _write_flow(tmp_path, "TOTAL", 3.0)
    _write_canonical_sidecars(tmp_path, suffix="split")
    MODULE.validate_additive_outputs(tmp_path, suffix="split", rtol=1e-9, atol=1e-6)
    # A shared but wrong x/A bundle must not pass merely because all three flows
    # use the same denominator; x must equal the DOM transaction row-use margin.
    for flow in ("DOM", "IMP", "TOTAL"):
        directory = tmp_path / f"stk_flow={flow}"
        wrong_x = pd.read_csv(directory / "x_split.csv", index_col=0).iloc[:, 0] + 1.0
        wrong_x.to_csv(directory / "x_split.csv", header=True)
        z = pd.read_csv(directory / "Z_split.csv", index_col=0)
        technical_coefficients(z, wrong_x).to_csv(directory / "A_split.csv")
    with pytest.raises(ValueError, match="split DOM x=row-use"):
        MODULE.validate_additive_outputs(tmp_path, suffix="split", rtol=1e-9, atol=1e-6)
    _write_canonical_sidecars(tmp_path, suffix="split")
    total = tmp_path / "stk_flow=TOTAL"
    Z = pd.read_csv(total / "Z_split.csv", index_col=0)
    Z.iloc[0, 0] += 1.0
    Z.to_csv(total / "Z_split.csv")
    with pytest.raises(ValueError, match="split TOTAL use=DOM\\+IMP"):
        MODULE.validate_additive_outputs(tmp_path, suffix="split", rtol=1e-9, atol=1e-6)


def test_topology_mismatch_is_rejected(tmp_path: Path) -> None:
    _write_flow(tmp_path, "DOM", 1.0)
    _write_flow(tmp_path, "IMP", 2.0)
    (tmp_path / "stk_flow=IMP" / "exio_selected_sectors.json").write_text(json.dumps({"a": ["a"], "b": ["c"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="topology differs"):
        MODULE._load_equal_manifests(tmp_path / "stk_flow=DOM", tmp_path / "stk_flow=IMP")


def test_parent_aggregation_reconstructs_original_cells() -> None:
    split = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        index=pd.Index(["a1", "a2", "b"]),
        columns=pd.Index(["a1", "a2", "b"]),
    )
    mapping = {"a1": "a", "a2": "a", "b": "b"}
    got = MODULE._aggregate_square(split, mapping, pd.Index(["a", "b"]))
    expected = pd.DataFrame([[12.0, 9.0], [15.0, 9.0]], index=pd.Index(["a", "b"]), columns=pd.Index(["a", "b"]))
    pd.testing.assert_frame_equal(got, expected)


def test_assert_close_rejects_nan_mask_mismatch() -> None:
    left = pd.Series([1.0, float("nan")], index=pd.Index(["a", "b"]))
    right = pd.Series([1.0, 2.0], index=pd.Index(["a", "b"]))
    with pytest.raises(ValueError):
        MODULE._assert_close("nan mask", left, right, rtol=1e-9, atol=1e-6)


def test_assert_close_uses_elementwise_scale_and_accepts_roundoff() -> None:
    huge = pd.Series([1.0e12, 0.0], index=pd.Index(["large", "small"]))
    masked_small_error = pd.Series([1.0e12, 1.0e-4], index=huge.index)
    with pytest.raises(ValueError, match="worst_tolerance"):
        MODULE._assert_close(
            "elementwise", huge, masked_small_error, rtol=1e-9, atol=1e-6
        )

    roundoff = pd.Series([1.0 + 7.27595761418e-12], index=pd.Index(["x"]))
    exact = pd.Series([1.0], index=roundoff.index)
    assert MODULE._assert_close(
        "roundoff",
        roundoff,
        exact,
        rtol=MODULE._GENERATED_IDENTITY_RTOL,
        atol=MODULE._GENERATED_IDENTITY_ATOL,
    ) < MODULE._GENERATED_IDENTITY_ATOL


def test_aggregate_rows_rejects_orphan_parent() -> None:
    frame = pd.DataFrame({"P3": [1.0, 2.0]}, index=pd.Index(["child", "orphan"]))
    with pytest.raises(ValueError, match="orphan"):
        MODULE._aggregate_rows(frame, {"child": "parent"}, pd.Index(["parent"]))


def _write_primitive(root: Path, flow: str, z_scale: float, y_scale: float) -> None:
    directory = root / f"stk_flow={flow}"
    directory.mkdir(parents=True)
    idx = pd.Index(["a", "b"])
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx, columns=idx) * z_scale
    Y = pd.DataFrame(
        {
            "P3_S13": [1.0, 2.0], "P3_S14": [2.0, 1.0], "P3_S15": [3.0, 1.0],
            "P51G": [1.0, 2.0], "P52": [2.0, 3.0], "P53": [3.0, 4.0],
            "P3": [6.0, 4.0], "P5M": [5.0, 7.0], "P5": [6.0, 9.0],
            "TFU": [12.0, 13.0], "TU": [15.0, 20.0],
        }, index=idx,
    ) * y_scale
    Y["TFU"] = Y[["P3_S13", "P3_S14", "P3_S15", "P51G", "P52", "P53"]].sum(axis=1)
    Y["TU"] = Z.sum(axis=1) + Y["TFU"]
    Z.to_csv(directory / "Z_split.csv")
    Y.to_csv(directory / "Y_split.csv")
    (directory / "exio_selected_sectors.json").write_text(json.dumps({"a": ["a"], "b": ["b"]}), encoding="utf-8")
    pd.DataFrame(columns=pd.Index(["parent", "child", "output_share"])).to_csv(directory / "split_output_shares.csv", index=False)



def _joint_reconciliation_fixture() -> tuple[
    pd.Index,
    dict[str, str],
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, SimpleNamespace],
]:
    children = pd.Index(["a1", "a2"])
    child_parent = {"a1": "A", "a2": "A"}
    output_share = pd.Series([0.5, 0.5], index=children, dtype="float64")

    # All intermediate purchases are structurally supported only in producer a1.
    # With preliminary Y fixed, a1 output is too small; Step 2 must reallocate
    # atomic final demand and child output while preserving the parent Y cell.
    Z_dom = pd.DataFrame(
        [[3.0, 0.0], [3.0, 0.0]], index=children, columns=children
    )
    Z_imp = pd.DataFrame(0.0, index=children, columns=children)
    Y_dom = pd.DataFrame({"P3": [1.0, 9.0]}, index=children)
    Y_imp = pd.DataFrame({"P3": [1.0, 1.0]}, index=children)

    parent = pd.Index(["A"])
    refs = {
        "DOM": SimpleNamespace(
            Z=pd.DataFrame([[6.0]], index=parent, columns=parent),
            Y=pd.DataFrame({"P3": [10.0]}, index=parent),
            V=None,
        ),
        "IMP": SimpleNamespace(
            Z=pd.DataFrame([[0.0]], index=parent, columns=parent),
            Y=pd.DataFrame({"P3": [2.0]}, index=parent),
            V=None,
        ),
        "TOTAL": SimpleNamespace(
            Z=pd.DataFrame([[6.0]], index=parent, columns=parent),
            Y=pd.DataFrame({"P3": [12.0]}, index=parent),
            V=pd.DataFrame(
                [[10.0], [0.0]],
                index=pd.Index(["B1G", "D21X31"]),
                columns=parent,
            ),
        ),
    }
    return (
        children,
        child_parent,
        output_share,
        Z_dom,
        Z_imp,
        Y_dom,
        Y_imp,
        refs,
    )


def test_joint_iot_reconciliation_moves_atomic_y_and_output_while_preserving_parent_cells() -> None:
    (
        _children,
        child_parent,
        output_share,
        Z_dom,
        Z_imp,
        Y_dom,
        Y_imp,
        refs,
    ) = _joint_reconciliation_fixture()
    preliminary_x = MODULE.flow_use_total(Z_dom, Y_dom)

    (
        dom,
        imp,
        y_dom,
        y_imp,
        output,
        production_v,
        diagnostic,
        checks,
    ) = MODULE._reconcile_iot_intermediate_transactions(
        Z_dom,
        Z_imp,
        Y_dom,
        Y_imp,
        child_parent=child_parent,
        dom_output_share=output_share,
        references=refs,
    )

    assert production_v is not None
    assert float(y_dom["P3"].sum()) == pytest.approx(10.0, abs=1e-10)
    assert float(y_imp["P3"].sum()) == pytest.approx(2.0, abs=1e-10)
    pd.testing.assert_series_equal(
        output,
        MODULE.flow_use_total(dom, y_dom).rename("x"),
        check_names=False,
        rtol=1e-10,
        atol=1e-10,
    )
    total_columns = dom.sum(axis=0).add(imp.sum(axis=0))
    pd.testing.assert_series_equal(
        total_columns + production_v.loc["D21X31"] + production_v.loc["B1G"],
        output,
        check_names=False,
        rtol=1e-10,
        atol=1e-10,
    )
    assert float(output.loc["a1"]) > float(preliminary_x.loc["a1"]) + 1e-6
    assert float(y_dom.loc["a1", "P3"]) > 1.0
    assert float(production_v.loc["B1G"].min()) >= -1e-10
    assert {row["status"] for row in checks} == {"pass"}
    assert {
        ("DOM", "tc.parent_atomic_y_cell"),
        ("IMP", "tc.parent_atomic_y_cell"),
        ("DOM", "tc.dom_output_identity"),
        ("TOTAL", "tc.production_account_identity"),
    }.issubset({(row["flow"], row["check_id"]) for row in checks})
    assert not np.allclose(
        diagnostic["preliminary_domestic_output"],
        diagnostic["reconciled_domestic_output"],
    )


def test_joint_iot_reconciliation_preserves_signed_investment_parent_cell() -> None:
    children = pd.Index(["a1", "a2"])
    child_parent = {"a1": "A", "a2": "A"}
    output_share = pd.Series([0.5, 0.5], index=children, dtype="float64")
    Z_dom = pd.DataFrame(
        [[1.0, 0.0], [1.0, 0.0]], index=children, columns=children
    )
    Z_imp = pd.DataFrame(0.0, index=children, columns=children)
    Y_dom = pd.DataFrame(
        {"P51G": [2.0, 2.0], "P52": [2.0, -1.0], "P53": [0.0, 0.0]},
        index=children,
    )
    Y_dom["P5M"] = Y_dom["P52"] + Y_dom["P53"]
    Y_dom["P5"] = Y_dom["P51G"] + Y_dom["P5M"]
    Y_imp = Y_dom * 0.0
    parent = pd.Index(["A"])
    refs = {
        "DOM": SimpleNamespace(
            Z=pd.DataFrame([[2.0]], index=parent, columns=parent),
            Y=pd.DataFrame(
                {
                    "P51G": [4.0],
                    "P52": [1.0],
                    "P53": [0.0],
                    "P5M": [1.0],
                    "P5": [5.0],
                },
                index=parent,
            ),
            V=None,
        ),
        "IMP": SimpleNamespace(
            Z=pd.DataFrame([[0.0]], index=parent, columns=parent),
            Y=pd.DataFrame(
                {
                    "P51G": [0.0],
                    "P52": [0.0],
                    "P53": [0.0],
                    "P5M": [0.0],
                    "P5": [0.0],
                },
                index=parent,
            ),
            V=None,
        ),
        "TOTAL": SimpleNamespace(
            Z=pd.DataFrame([[2.0]], index=parent, columns=parent),
            Y=pd.DataFrame(
                {
                    "P51G": [4.0],
                    "P52": [1.0],
                    "P53": [0.0],
                    "P5M": [1.0],
                    "P5": [5.0],
                },
                index=parent,
            ),
            V=pd.DataFrame(
                [[3.0], [0.0]],
                index=pd.Index(["B1G", "D21X31"]),
                columns=parent,
            ),
        ),
    }

    (
        _dom,
        _imp,
        final_y_dom,
        _final_y_imp,
        output,
        production_v,
        _diag,
        _checks,
    ) = MODULE._reconcile_iot_intermediate_transactions(
        Z_dom,
        Z_imp,
        Y_dom,
        Y_imp,
        child_parent=child_parent,
        dom_output_share=output_share,
        references=refs,
    )
    assert production_v is not None
    assert float(final_y_dom["P52"].sum()) == pytest.approx(1.0, abs=1e-10)
    assert float(output.min()) >= -1e-10
    pd.testing.assert_series_equal(
        final_y_dom["P5M"],
        final_y_dom["P52"] + final_y_dom["P53"],
        check_names=False,
    )


def test_joint_iot_reconciliation_preserves_negative_d21_source_sign() -> None:
    (
        _children,
        child_parent,
        output_share,
        Z_dom,
        Z_imp,
        Y_dom,
        Y_imp,
        refs,
    ) = _joint_reconciliation_fixture()
    refs["TOTAL"].V = pd.DataFrame(
        [[12.0], [-2.0]],
        index=pd.Index(["B1G", "D21X31"]),
        columns=pd.Index(["A"]),
    )
    _dom, _imp, _yd, _yi, _x, production_v, _diag, _checks = (
        MODULE._reconcile_iot_intermediate_transactions(
            Z_dom,
            Z_imp,
            Y_dom,
            Y_imp,
            child_parent=child_parent,
            dom_output_share=output_share,
            references=refs,
        )
    )
    assert production_v is not None
    assert float(production_v.loc["D21X31"].sum()) == pytest.approx(
        -2.0, abs=1e-10
    )
    assert bool((production_v.loc["D21X31"] <= 1e-12).all())


def test_joint_iot_reconciliation_rejects_positive_output_without_exio_support() -> None:
    (
        _children,
        child_parent,
        _output_share,
        Z_dom,
        Z_imp,
        Y_dom,
        Y_imp,
        refs,
    ) = _joint_reconciliation_fixture()
    output_share = pd.Series([0.0, 1.0], index=Z_dom.index, dtype="float64")
    with pytest.raises(ValueError, match="zero DOM EXIO output support"):
        MODULE._reconcile_iot_intermediate_transactions(
            Z_dom,
            Z_imp,
            Y_dom,
            Y_imp,
            child_parent=child_parent,
            dom_output_share=output_share,
            references=refs,
        )


def test_derive_constructs_total_shared_x_a_and_source_m(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_primitive(tmp_path, "DOM", 1.0, 1.0)
    _write_primitive(tmp_path, "IMP", 2.0, 3.0)
    sources = {
        "DOM": pd.DataFrame([[1.0, float("nan")]], index=pd.Index(["m"]), columns=pd.Index(["P3", "P5"])),
        "IMP": pd.DataFrame([[0.0, 2.0]], index=pd.Index(["m"]), columns=pd.Index(["P3", "P5"])),
        "TOTAL": pd.DataFrame([[3.0, 4.0]], index=pd.Index(["m"]), columns=pd.Index(["P3", "P5"])),
    }

    def fake_source(**kwargs):
        flow = kwargs["filters"]["stk_flow"]
        dom = tmp_path / "stk_flow=DOM"
        imp = tmp_path / "stk_flow=IMP"
        Z_dom = pd.read_csv(dom / "Z_split.csv", index_col=0)
        Z_imp = pd.read_csv(imp / "Z_split.csv", index_col=0)
        Y_dom = pd.read_csv(dom / "Y_split.csv", index_col=0)
        Y_imp = pd.read_csv(imp / "Y_split.csv", index_col=0)
        if flow == "TOTAL":
            Z = Z_dom + Z_imp
            # Published TOTAL controls are rounded independently from DOM+IMP.
            Z.iloc[0, 0] += 0.02
            Y = MODULE.enforce_final_demand_display_identities(Y_dom + Y_imp, Z=Z)
        else:
            Z = Z_dom if flow == "DOM" else Z_imp
            Y = Y_dom if flow == "DOM" else Y_imp
        V = None
        if flow in {"DOM", "TOTAL"}:
            p1 = MODULE.flow_use_total(Z_dom, Y_dom)
            d21 = pd.Series(0.0, index=p1.index, dtype="float64")
            total_input = Z_dom.add(Z_imp).sum(axis=0).reindex(p1.index)
            b1g = p1 - total_input - d21
            V = pd.DataFrame(
                [p1, d21, b1g],
                index=pd.Index(["P1", "D21X31", "B1G"]),
            )
        return SimpleNamespace(Z=Z, Y=Y, M=sources[flow], V=V, meta={"source_rounding_quantum": 1.0})

    monkeypatch.setattr(MODULE, "load_eurostat_iot", fake_source)
    MODULE.derive_total_flow(tmp_path, eurostat_dataset="x", eurostat_filters={}, eurostat_row_dim="r", eurostat_col_dim="c")
    coherence = pd.read_csv(tmp_path / MODULE._TRANSFORMATION_COHERENCE_REPORT)
    assert set(coherence["validation_class"]) == {"transformation_coherence"}
    assert set(coherence["status"]) == {"pass"}
    assert {
        ("DOM", "tc.parent_z_cell"),
        ("IMP", "tc.parent_z_cell"),
        ("DOM", "tc.parent_atomic_y_cell"),
        ("IMP", "tc.parent_atomic_y_cell"),
        ("DOM", "tc.dom_output_identity"),
        ("DOM", "tc.parent_dom_output"),
        ("IMP", "tc.parent_import_use"),
        ("TOTAL", "tc.production_account_identity"),
        ("TOTAL", "tc.parent_d21x31"),
        ("TOTAL", "tc.nonnegative_child_b1g"),
        ("TOTAL", "tc.separable_reconciliation_convergence"),
        ("TOTAL", "tc.total_use_additivity"),
    }.issubset(set(zip(coherence["flow"], coherence["check_id"])))
    dom = tmp_path / "stk_flow=DOM"
    imp = tmp_path / "stk_flow=IMP"
    total = tmp_path / "stk_flow=TOTAL"
    Z_dom, Z_imp, Z_total = (pd.read_csv(path / "Z_split.csv", index_col=0) for path in (dom, imp, total))
    Y_dom, Y_imp, Y_total = (pd.read_csv(path / "Y_split.csv", index_col=0) for path in (dom, imp, total))
    pd.testing.assert_frame_equal(Z_total, Z_dom + Z_imp)
    assert Y_total.columns.equals(Y_dom.columns)
    for col in ("P3_S13", "P51G", "P52", "P53"):
        pd.testing.assert_series_equal(Y_total[col], Y_dom[col] + Y_imp[col], check_names=False)
    pd.testing.assert_series_equal(Y_total["P5M"], Y_total["P52"] + Y_total["P53"], check_names=False)
    pd.testing.assert_series_equal(Y_total["P5"], Y_total["P51G"] + Y_total["P5M"], check_names=False)
    pd.testing.assert_series_equal(
        Y_total["TFU"], MODULE.source_atomic_final_demand(Y_total).sum(axis=1), check_names=False
    )
    pd.testing.assert_series_equal(
        Y_total["TU"], Z_total.sum(axis=1) + Y_total["TFU"], check_names=False
    )
    x = pd.read_csv(dom / "x_split.csv", index_col=0).iloc[:, 0]
    for directory, Z in ((dom, Z_dom), (imp, Z_imp), (total, Z_total)):
        pd.testing.assert_series_equal(pd.read_csv(directory / "x_split.csv", index_col=0).iloc[:, 0], x, check_names=False)
        expected_a = technical_coefficients(Z, x)
        pd.testing.assert_frame_equal(pd.read_csv(directory / "A_split.csv", index_col=0), expected_a)
    for flow, expected in sources.items():
        got = pd.read_csv(tmp_path / f"stk_flow={flow}" / "M_split.csv", index_col=0)
        pd.testing.assert_frame_equal(got, expected)
    assert (total / "V_split.csv").exists()

    identity = pd.read_csv(tmp_path / "derived_total_flow_identity.csv")
    analytical = identity.loc[
        identity["stage"].astype(str).eq("analytical_contract")
        & identity["flow"].astype(str).eq("DOM")
        & identity["object"].astype(str).eq("P1")
    ]
    assert len(analytical) == 1
    row = analytical.iloc[0]
    assert row["contract"] == "joint_transaction_derived_DOM_row_use"
    assert bool(row["ok"])
    assert float(row["max_abs_gap"]) <= MODULE._GENERATED_IDENTITY_ATOL


def test_derive_rejects_different_child_order(tmp_path: Path) -> None:
    _write_primitive(tmp_path, "DOM", 1.0, 1.0)
    _write_primitive(tmp_path, "IMP", 1.0, 1.0)
    path = tmp_path / "stk_flow=IMP" / "Z_split.csv"
    Z = pd.read_csv(path, index_col=0).iloc[::-1, ::-1]
    Z.to_csv(path)
    with pytest.raises(ValueError, match="child topology/order differs"):
        MODULE.derive_total_flow(tmp_path, eurostat_dataset="x", eurostat_filters={}, eurostat_row_dim="r", eurostat_col_dim="c")


def test_published_m_mask_validation_fails() -> None:
    with pytest.raises(ValueError, match="published M"):
        MODULE._assert_close("published M", pd.DataFrame([[float("nan")]]), pd.DataFrame([[0.0]]), rtol=0.0, atol=0.0)


def test_transaction_v_rejects_signed_unknown_row() -> None:
    parents = pd.Index(["A", "B"])
    reference = SimpleNamespace(
        V=pd.DataFrame([[1.0, -1.0]], index=pd.Index(["UNKNOWN"]), columns=parents),
        Z=pd.DataFrame(0.0, index=parents, columns=parents),
    )
    children = pd.Index(["A", "B"])
    zeros = pd.DataFrame(0.0, index=children, columns=children)
    with pytest.raises(ValueError, match="Unknown nonzero V row"):
        MODULE._construct_transaction_v(
            flow="DOM", reference=reference, child_parent={"A": "A", "B": "B"},
            Z=zeros, Z_total=zeros, Z_imp=zeros,
            x_domestic=pd.Series([1.0, 1.0], index=children),
            import_supply_proxy=pd.Series([0.0, 0.0], index=children),
        )


def test_import_partition_rejects_negative_or_excess_source_component() -> None:
    kwargs = {
        "aggregate": pd.Series([5.0], index=["a"]),
        "published_total": pd.Series([4.0], index=["A"]),
        "child_parent": {"a": "A"},
        "children": pd.Index(["a"]),
        "row_code": "P7_B0",
    }
    with pytest.raises(ValueError, match="non-negative partition"):
        MODULE._allocate_import_partition(
            published_part=pd.Series([5.0], index=["A"]), **kwargs
        )


def test_transaction_v_closes_rows_columns_and_value_added() -> None:
    """Transaction-derived output and residual surplus close every child account."""
    parent = pd.Index(["A"])
    children = pd.Index(["a1", "a2"])
    rows = [
        "B1G", "B2A3G", "B2A3N", "CPA_TOTAL", "D1", "D11", "D21X31",
        "D29X39", "P1", "P2_ADJ", "P51C", "P7", "P7_B0", "P7_D0",
        "P7_U2", "P7_U3", "TS_BP", "B3G",
    ]
    source_values = {
        "B1G": 32.0, "B2A3G": 28.0, "B2A3N": 26.0, "CPA_TOTAL": 20.0,
        "D1": 10.0, "D11": 8.0, "D21X31": -2.0, "D29X39": -1.0,
        "P1": 50.0, "P2_ADJ": 18.0, "P51C": 2.0, "P7": 10.0,
        "P7_B0": 6.0, "P7_D0": 4.0, "P7_U2": 4.0, "P7_U3": 2.0,
        "TS_BP": 60.0, "B3G": float("nan"),
    }
    reference = SimpleNamespace(
        V=pd.DataFrame([[source_values[row]] for row in rows], index=pd.Index(rows), columns=parent),
    )
    Z_total = pd.DataFrame([[10.0, 5.0], [2.0, 3.0]], index=children, columns=children)
    Z_imp = pd.DataFrame([[1.0, 2.0], [1.0, 1.0]], index=children, columns=children)
    x = pd.Series([30.0, 20.0], index=children)
    imports = pd.Series([4.0, 6.0], index=children)
    V, _ = MODULE._construct_transaction_v(
        flow="TOTAL", reference=reference, child_parent={"a1": "A", "a2": "A"},
        Z=Z_total, Z_total=Z_total, Z_imp=Z_imp,
        x_domestic=x, import_supply_proxy=imports,
    )
    assert V is not None
    pd.testing.assert_series_equal(V.loc["P1"], x, check_names=False)
    pd.testing.assert_series_equal(V.loc["P7"], imports, check_names=False)
    pd.testing.assert_series_equal(V.loc["TS_BP"], x + imports, check_names=False)
    pd.testing.assert_series_equal(V.loc["P2_ADJ"], Z_total.sum(axis=0) + V.loc["D21X31"], check_names=False)
    pd.testing.assert_series_equal(V.loc["B1G"], V.loc["P1"] - V.loc["P2_ADJ"], check_names=False)
    pd.testing.assert_series_equal(
        V.loc["B1G"], V.loc["D1"] + V.loc["D29X39"] + V.loc["P51C"] + V.loc["B2A3N"],
        check_names=False,
    )
    pd.testing.assert_series_equal(V.loc["B2A3G"], V.loc["B2A3N"] + V.loc["P51C"], check_names=False)
    assert float(V.loc["D1"].sum()) == 10.0
    assert float(V.loc["D21X31"].sum()) == -2.0
    assert V.loc["B3G"].isna().all()

    # Coarser consolidated-EU sources can publish gross B2A3G without P51C/B2A3N.
    coarse = SimpleNamespace(V=reference.V.drop(index=["D11", "P51C", "B2A3N"]))
    coarse_v, _ = MODULE._construct_transaction_v(
        flow="TOTAL", reference=coarse, child_parent={"a1": "A", "a2": "A"},
        Z=Z_total, Z_total=Z_total, Z_imp=Z_imp,
        x_domestic=x, import_supply_proxy=imports,
    )
    assert coarse_v is not None
    pd.testing.assert_series_equal(
        coarse_v.loc["B2A3G"],
        coarse_v.loc["B1G"] - coarse_v.loc["D1"] - coarse_v.loc["D29X39"],
        check_names=False,
    )


def test_transaction_m_rebuilds_controls_partitions_and_missingness() -> None:
    columns = pd.Index(
        [
            "P3", "P3_S13", "P3_S14", "P3_S15", "P5", "P51G",
            "P52", "P53", "P5M", "P6", "TFU", "TOTAL", "TU",
        ]
    )
    products = pd.Index(["a", "b"])
    y_imp = pd.DataFrame(1.0, index=products, columns=columns)
    y_imp["P3"] = y_imp[["P3_S13", "P3_S14", "P3_S15"]].sum(axis=1)
    y_imp["P5M"] = y_imp[["P52", "P53"]].sum(axis=1)
    y_imp["P5"] = y_imp[["P51G", "P52", "P53"]].sum(axis=1)
    y_imp["TFU"] = y_imp[["P3_S13", "P3_S14", "P3_S15", "P51G", "P52", "P53", "P6"]].sum(axis=1)
    z_imp = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=products, columns=products)
    y_imp["TU"] = z_imp.sum(axis=1) + y_imp["TFU"]
    y_total = y_imp * 2.0
    z_total = z_imp * 2.0

    rows = pd.Index(
        ["CPA_TOTAL", "D21X31", "P2_ADJ", "P7", "P7_B0", "P7_D0", "P7_U2", "P7_U3", "B3G"]
    )
    source_m = pd.DataFrame(np.nan, index=rows, columns=columns)
    source_m.loc["D21X31"] = 1.0
    source_m.loc["P2_ADJ"] = 1.0
    # P7/P7_* are supply-by-product rows and are non-applicable on final-use M.
    reference = SimpleNamespace(M=source_m)
    V = pd.DataFrame(0.0, index=rows, columns=products)
    V.loc["B3G"] = np.nan
    V.loc["CPA_TOTAL"] = z_total.sum(axis=0)
    V.loc["P7"] = [4.0, 6.0]
    V.loc["P7_B0"] = [2.4, 3.6]
    V.loc["P7_D0"] = [1.6, 2.4]
    V.loc["P7_U2"] = [1.6, 2.4]
    V.loc["P7_U3"] = [0.8, 1.2]

    M = MODULE._construct_transaction_m(
        flow="TOTAL", reference=reference, Y=y_total, Z=z_total,
        Y_imp=y_imp, Z_imp=z_imp, V=V,
    )
    assert M is not None
    ordinary_columns = columns.drop("TOTAL")
    pd.testing.assert_series_equal(
        M.loc["CPA_TOTAL", ordinary_columns], y_total[ordinary_columns].sum(axis=0), check_names=False
    )
    assert M.loc[["P7", "P7_B0", "P7_D0", "P7_U2", "P7_U3"]].isna().all().all()
    assert M.loc["P2_ADJ", "P3"] == pytest.approx(y_total["P3"].sum() + M.loc["D21X31", "P3"])
    assert M.loc["B3G"].isna().all()


def test_transaction_m_rebuilds_p2_display_after_rounded_atomic_source_cells() -> None:
    columns = pd.Index(
        [
            "P3", "P3_S13", "P3_S14", "P3_S15", "P5", "P51G",
            "P52", "P53", "P5M", "P6", "TFU", "TOTAL", "TU",
        ]
    )
    products = pd.Index(["a", "b"])
    y = pd.DataFrame(1.0, index=products, columns=columns)
    y["P3"] = y[["P3_S13", "P3_S14", "P3_S15"]].sum(axis=1)
    y["P5M"] = y[["P52", "P53"]].sum(axis=1)
    y["P5"] = y[["P51G", "P52", "P53"]].sum(axis=1)
    y["TFU"] = y[["P3_S13", "P3_S14", "P3_S15", "P51G", "P52", "P53", "P6"]].sum(axis=1)
    z = pd.DataFrame(1.0, index=products, columns=products)
    y["TU"] = z.sum(axis=1) + y["TFU"]
    rows = pd.Index(["P2_ADJ", "D21X31"])
    source_m = pd.DataFrame(1.0, index=rows, columns=columns)
    source_m.loc["D21X31", "P6"] = np.nan
    source_m.loc["P2_ADJ", "P6"] = 999.0
    source_m.loc["P2_ADJ", "TFU"] = 999.0
    result = MODULE._construct_transaction_m(
        flow="TOTAL",
        reference=SimpleNamespace(M=source_m),
        Y=y,
        Z=z,
        Y_imp=y,
        Z_imp=z,
        V=pd.DataFrame(0.0, index=rows, columns=products),
    )
    assert result is not None
    expected = float(y["TFU"].sum()) + float(result.loc["D21X31", "TFU"])
    assert result.loc["P2_ADJ", "TFU"] == pytest.approx(expected)
    assert result.loc["P2_ADJ", "TFU"] != pytest.approx(
        float(result.loc["P2_ADJ", MODULE.source_atomic_final_demand(y).columns].sum())
    )


def test_signed_primitive_requires_a_transaction_proxy() -> None:
    with pytest.raises(ValueError, match="No usable V allocation proxy"):
        MODULE._allocate_source_parent_vector(
            published=pd.Series([-2.0], index=pd.Index(["A"])),
            child_parent={"a1": "A", "a2": "A"},
            children=pd.Index(["a1", "a2"]),
            proxy=pd.Series([0.0, 0.0], index=pd.Index(["a1", "a2"])),
            flow="TOTAL",
            row_code="D29X39",
        )


def test_assert_close_rejects_infinite_values() -> None:
    left = pd.Series([float("inf")], index=pd.Index(["a"]))
    right = pd.Series([float("inf")], index=pd.Index(["a"]))
    with pytest.raises(ValueError, match="non-finite values are present"):
        MODULE._assert_close("infinite values", left, right, rtol=1e-9, atol=1e-6)

def test_nested_p5m_source_control_is_atomic_without_p52_p53() -> None:
    products = pd.Index(["a", "b"])
    z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=products, columns=products)
    y = pd.DataFrame(
        {
            "P3": [5.0, 6.0],
            "P51G": [2.0, 3.0],
            "P5M": [-4.0, 1.0],
            "P5": [-2.0, 4.0],
            "P6": [1.0, 1.0],
        },
        index=products,
    )
    reference = MODULE.ParsedEurostatIOT(Z=z, Y=y, V=None, M=None, meta={})
    p5m, p5m_counts = MODULE._derived_y_control(reference, "P5M")
    p5, p5_counts = MODULE._derived_y_control(reference, "P5")
    pd.testing.assert_series_equal(p5m, y["P5M"], check_names=False)
    pd.testing.assert_series_equal(p5, y["P51G"] + y["P5M"], check_names=False)
    assert (p5m_counts == 1).all()
    assert (p5_counts == 2).all()

def test_transaction_v_preserves_published_p7_supply_not_imp_row_use() -> None:
    products = pd.Index(["a1", "a2"])
    parent = pd.Index(["A"])
    source_v = pd.DataFrame(
        [[10.0], [6.0], [4.0], [60.0]],
        index=pd.Index(["P7", "P7_B0", "P7_D0", "TS_BP"]),
        columns=parent,
    )
    reference = SimpleNamespace(V=source_v)
    z = pd.DataFrame(0.0, index=products, columns=products)
    x = pd.Series([20.0, 30.0], index=products)
    # The supply proxy deliberately differs from any imported transaction row-use vector.
    proxy = pd.Series([1.0, 3.0], index=products)
    v, _ = MODULE._construct_transaction_v(
        flow="TOTAL",
        reference=reference,
        child_parent={"a1": "A", "a2": "A"},
        Z=z,
        Z_total=z,
        Z_imp=z,
        x_domestic=x,
        import_supply_proxy=proxy,
    )
    assert v is not None
    pd.testing.assert_series_equal(v.loc["P7"], pd.Series([2.5, 7.5], index=products), check_names=False)
    pd.testing.assert_series_equal(v.loc["TS_BP"], x + v.loc["P7"], check_names=False)
    assert pytest.approx(float(v.loc["P7"].sum())) == 10.0

    standalone = SimpleNamespace(V=source_v.drop(index=["P7_B0"]))
    standalone_v, _ = MODULE._construct_transaction_v(
        flow="TOTAL", reference=standalone,
        child_parent={"a1": "A", "a2": "A"},
        Z=z, Z_total=z, Z_imp=z, x_domestic=x, import_supply_proxy=proxy,
    )
    assert standalone_v is not None
    assert float(standalone_v.loc["P7_D0"].sum()) == pytest.approx(4.0)
    assert "P7_B0" not in standalone_v.index


def test_transaction_v_uses_domestic_output_for_dom_ts_bp() -> None:
    products = pd.Index(["a1", "a2"])
    parent = pd.Index(["A"])
    source_v = pd.DataFrame(
        [[50.0], [10.0], [50.0]],
        index=pd.Index(["P1", "P7", "TS_BP"]),
        columns=parent,
    )
    z = pd.DataFrame(0.0, index=products, columns=products)
    output = pd.Series([20.0, 30.0], index=products)
    imports = pd.Series([1.0, 3.0], index=products)
    v, _ = MODULE._construct_transaction_v(
        flow="DOM",
        reference=SimpleNamespace(V=source_v),
        child_parent={"a1": "A", "a2": "A"},
        Z=z,
        Z_total=z,
        Z_imp=z,
        x_domestic=output,
        import_supply_proxy=imports,
    )
    assert v is not None
    pd.testing.assert_series_equal(v.loc["P1"], output, check_names=False)
    pd.testing.assert_series_equal(v.loc["P7"], pd.Series([2.5, 7.5], index=products), check_names=False)
    pd.testing.assert_series_equal(v.loc["TS_BP"], output, check_names=False)
    assert not v.loc["TS_BP"].equals(v.loc["P1"] + v.loc["P7"])


def test_source_ts_bp_controls_reject_flow_specific_identity_drift() -> None:
    columns = pd.Index(["A", "B"])
    dom = SimpleNamespace(V=pd.DataFrame(
        [[10.0, 20.0], [10.0, 20.0]],
        index=pd.Index(["TS_BP", "P1"]), columns=columns,
    ))
    total = SimpleNamespace(V=pd.DataFrame(
        [[14.0, 27.0], [10.0, 20.0], [4.0, 7.0]],
        index=pd.Index(["TS_BP", "P1", "P7"]), columns=columns,
    ))
    diagnostics = MODULE._validate_source_ts_bp_controls(
        references={"DOM": dom, "TOTAL": total},
        source_rounding_quantum=0.01,
        rtol=1e-9,
        atol=1e-6,
    )
    assert {item["source_identity"] for item in diagnostics} == {
        "DOM_TS_BP_equals_P1", "TOTAL_TS_BP_equals_P1_plus_P7",
    }

    dom.V.loc["TS_BP", "A"] += 1.0
    with pytest.raises(ValueError, match="DOM_TS_BP_equals_P1"):
        MODULE._validate_source_ts_bp_controls(
            references={"DOM": dom, "TOTAL": total},
            source_rounding_quantum=0.01,
            rtol=1e-9,
            atol=1e-6,
        )

    dom.V.loc["TS_BP", "A"] -= 1.0
    total.V.loc["TS_BP", "B"] += 1.0
    with pytest.raises(ValueError, match="TOTAL_TS_BP_equals_P1_plus_P7"):
        MODULE._validate_source_ts_bp_controls(
            references={"DOM": dom, "TOTAL": total},
            source_rounding_quantum=0.01,
            rtol=1e-9,
            atol=1e-6,
        )

# STEP2_MONETARY_SCALE_INTEGRATION_TEST_V3
def test_joint_iot_reconciliation_is_scale_equivariant_on_homogeneous_accounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step-2 zero-target identities must remain stable under monetary rescaling."""
    (
        _children,
        child_parent,
        output_share,
        Z_dom,
        Z_imp,
        Y_dom,
        Y_imp,
        refs,
    ) = _joint_reconciliation_fixture()

    base = MODULE._reconcile_iot_intermediate_transactions(
        Z_dom,
        Z_imp,
        Y_dom,
        Y_imp,
        child_parent=child_parent,
        dom_output_share=output_share,
        references=refs,
    )

    factor = 1.0e6
    monkeypatch.setattr(
        MODULE,
        "_GENERATED_IDENTITY_ATOL",
        MODULE._GENERATED_IDENTITY_ATOL * factor,
    )
    scaled_refs: dict[str, SimpleNamespace] = {}
    for flow, reference in refs.items():
        scaled_refs[flow] = SimpleNamespace(
            Z=reference.Z * factor,
            Y=reference.Y * factor,
            V=None if reference.V is None else reference.V * factor,
        )
    scaled = MODULE._reconcile_iot_intermediate_transactions(
        Z_dom * factor,
        Z_imp * factor,
        Y_dom * factor,
        Y_imp * factor,
        child_parent=child_parent,
        dom_output_share=output_share,
        references=scaled_refs,
    )

    for base_frame, scaled_frame in zip(base[:4], scaled[:4], strict=True):
        pd.testing.assert_frame_equal(
            scaled_frame / factor,
            base_frame,
            check_exact=False,
            rtol=1e-8,
            atol=1e-8,
        )
    pd.testing.assert_series_equal(
        scaled[4] / factor,
        base[4],
        check_exact=False,
        check_names=False,
        rtol=1e-8,
        atol=1e-8,
    )
    assert base[5] is not None and scaled[5] is not None
    pd.testing.assert_frame_equal(
        scaled[5] / factor,
        base[5],
        check_exact=False,
        rtol=1e-8,
        atol=1e-8,
    )
    assert {row["status"] for row in scaled[7]} == {"pass"}
