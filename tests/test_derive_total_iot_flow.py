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



def _conditioning_fixture(tmp_path: Path) -> tuple[pd.Index, dict[str, str], pd.Series]:
    children = pd.Index(["fuel", "coal", "wind"])
    child_parent = {"fuel": "fuel", "coal": "electricity", "wind": "electricity"}
    dom = tmp_path / "stk_flow=DOM"
    dom.mkdir()
    pd.DataFrame(
        [
            {"parent": "electricity", "child": "coal", "output_share": 0.9},
            {"parent": "electricity", "child": "wind", "output_share": 0.1},
        ]
    ).to_csv(dom / "split_output_shares.csv", index=False)
    shares = MODULE._load_dom_split_output_shares(
        dom, child_parent=child_parent, children=children
    )
    return children, child_parent, shares


def test_producer_column_conditioning_removes_exio_output_scale(tmp_path: Path) -> None:
    children, child_parent, shares = _conditioning_fixture(tmp_path)
    Z = pd.DataFrame(0.0, index=children, columns=children)
    Z.loc["fuel", ["coal", "wind"]] = [81.0, 9.0]
    Z.loc["coal", "fuel"], Z.loc["wind", "fuel"] = 1.0, 9.0
    x = pd.Series([90.0, 10.0, 90.0], index=children)
    before_parent = MODULE._aggregate_square(
        Z, child_parent, pd.Index(["fuel", "electricity"])
    )
    got, diagnostic = MODULE._condition_producer_columns_on_domestic_output(
        Z, child_parent=child_parent, output_share=shares, x_domestic=x, flow="DOM"
    )
    pd.testing.assert_series_equal(got.sum(axis=1), Z.sum(axis=1))
    pd.testing.assert_frame_equal(
        MODULE._aggregate_square(got, child_parent, pd.Index(["fuel", "electricity"])),
        before_parent,
    )
    assert got.loc["fuel", "coal"] == pytest.approx(9.0)
    assert got.loc["fuel", "wind"] == pytest.approx(81.0)
    assert not diagnostic.empty


def test_producer_column_conditioning_is_noop_when_scales_match(tmp_path: Path) -> None:
    children, child_parent, shares = _conditioning_fixture(tmp_path)
    Z = pd.DataFrame(0.0, index=children, columns=children)
    Z.loc["fuel", ["coal", "wind"]] = [81.0, 9.0]
    x = pd.Series([90.0, 90.0, 10.0], index=children)
    got, _ = MODULE._condition_producer_columns_on_domestic_output(
        Z, child_parent=child_parent, output_share=shares, x_domestic=x, flow="DOM"
    )
    pd.testing.assert_frame_equal(got, Z)


def test_producer_column_conditioning_is_invariant_to_donor_production_scale(
    tmp_path: Path,
) -> None:
    """Column rescaling that leaves donor A unchanged must not change the target split."""
    children, child_parent, base_shares = _conditioning_fixture(tmp_path)
    target_x = pd.Series([90.0, 10.0, 90.0], index=children)

    base = pd.DataFrame(0.0, index=children, columns=children)
    base.loc["fuel", ["coal", "wind"]] = [81.0, 9.0]
    conditioned_base, _ = MODULE._condition_producer_columns_on_domestic_output(
        base,
        child_parent=child_parent,
        output_share=base_shares,
        x_domestic=target_x,
        flow="DOM",
    )

    # Scale the underlying donor producer columns by 2 and 0.5.  Re-normalising
    # the same Eurostat parent cell changes the preliminary donor transaction
    # shares and the donor output shares, while A = Z/x is unchanged.
    donor_mass = pd.Series({"coal": 18.0, "wind": 0.5})
    scaled_shares = base_shares.copy()
    scaled_shares.loc[["coal", "wind"]] = donor_mass / float(donor_mass.sum())
    scaled = base.copy()
    scaled.loc["fuel", ["coal", "wind"]] = (
        90.0 * donor_mass / float(donor_mass.sum())
    ).to_numpy()
    conditioned_scaled, _ = MODULE._condition_producer_columns_on_domestic_output(
        scaled,
        child_parent=child_parent,
        output_share=scaled_shares,
        x_domestic=target_x,
        flow="DOM",
    )
    pd.testing.assert_frame_equal(conditioned_scaled, conditioned_base)


def test_producer_column_conditioning_zeros_inputs_to_zero_target_output(
    tmp_path: Path,
) -> None:
    children, child_parent, shares = _conditioning_fixture(tmp_path)
    Z = pd.DataFrame(0.0, index=children, columns=children)
    Z.loc["fuel", ["coal", "wind"]] = [45.0, 45.0]
    x = pd.Series([90.0, 0.0, 100.0], index=children)
    got, _ = MODULE._condition_producer_columns_on_domestic_output(
        Z,
        child_parent=child_parent,
        output_share=shares,
        x_domestic=x,
        flow="DOM",
    )
    assert float(got["coal"].sum()) == pytest.approx(0.0, abs=1e-12)
    assert float(got["wind"].sum()) == pytest.approx(90.0)
    checks = MODULE._validate_producer_conditioning_transformation(
        Z, got, child_parent=child_parent, x_domestic=x, flow="DOM"
    )
    assert {row["status"] for row in checks} == {"pass"}


def test_transformation_coherence_rejects_row_parent_shuffle_hidden_by_aggregates(
    tmp_path: Path,
) -> None:
    """Whole-row and parent-cell equality cannot hide a changed row/parent submargin."""
    children, child_parent, _ = _conditioning_fixture(tmp_path)
    before = pd.DataFrame(0.0, index=children, columns=children)
    before.loc["coal", ["fuel", "coal"]] = [2.0, 3.0]
    before.loc["wind", ["fuel", "wind"]] = [4.0, 5.0]
    after = before.copy()
    after.loc["coal", "fuel"] -= 1.0
    after.loc["coal", "coal"] += 1.0
    after.loc["wind", "fuel"] += 1.0
    after.loc["wind", "wind"] -= 1.0
    pd.testing.assert_series_equal(before.sum(axis=1), after.sum(axis=1))
    pd.testing.assert_frame_equal(
        MODULE._aggregate_square(
            before, child_parent, pd.Index(["fuel", "electricity"])
        ),
        MODULE._aggregate_square(
            after, child_parent, pd.Index(["fuel", "electricity"])
        ),
    )
    x = pd.Series([10.0, 5.0, 9.0], index=children)
    with pytest.raises(ValueError, match="row-child/buyer-parent mass"):
        MODULE._validate_producer_conditioning_transformation(
            before, after, child_parent=child_parent, x_domestic=x, flow="DOM"
        )


def test_producer_column_conditioning_rejects_zero_output_support(tmp_path: Path) -> None:
    children, child_parent, shares = _conditioning_fixture(tmp_path)
    shares.loc[["coal", "wind"]] = [0.0, 1.0]
    Z = pd.DataFrame(0.0, index=children, columns=children)
    Z.loc["fuel", "wind"] = 1.0
    x = pd.Series([1.0, 1.0, 1.0], index=children)
    with pytest.raises(ValueError, match="zero EXIO producer-output support"):
        MODULE._condition_producer_columns_on_domestic_output(
            Z, child_parent=child_parent, output_share=shares,
            x_domestic=x, flow="DOM",
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
            V = pd.DataFrame([p1], index=pd.Index(["P1"]))
        return SimpleNamespace(Z=Z, Y=Y, M=sources[flow], V=V, meta={"source_rounding_quantum": 1.0})

    monkeypatch.setattr(MODULE, "load_eurostat_iot", fake_source)
    MODULE.derive_total_flow(tmp_path, eurostat_dataset="x", eurostat_filters={}, eurostat_row_dim="r", eurostat_col_dim="c")
    coherence = pd.read_csv(tmp_path / MODULE._TRANSFORMATION_COHERENCE_REPORT)
    assert set(coherence["validation_class"]) == {"transformation_coherence"}
    assert set(coherence["status"]) == {"pass"}
    assert {
        ("DOM", "tc.row_child_mass"),
        ("IMP", "tc.parent_z_cell"),
        ("DOM", "tc.dom_output_margin"),
        ("IMP", "tc.imp_row_use_margin"),
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
