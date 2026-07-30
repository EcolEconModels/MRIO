from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_validate_additive_outputs_detects_and_accepts_identity(tmp_path: Path) -> None:
    _write_flow(tmp_path, "DOM", 1.0)
    _write_flow(tmp_path, "IMP", 2.0)
    _write_flow(tmp_path, "TOTAL", 3.0)
    _write_canonical_sidecars(tmp_path, suffix="split")
    MODULE.validate_additive_outputs(tmp_path, suffix="split", rtol=1e-9, atol=1e-6)
    total = tmp_path / "stk_flow=TOTAL"
    Z = pd.read_csv(total / "Z_split.csv", index_col=0)
    Z.iloc[0, 0] += 1.0
    Z.to_csv(total / "Z_split.csv")
    with pytest.raises(ValueError, match="DOM\\+IMP=TOTAL"):
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
        dom = tmp_path / "stk_flow=DOM"; imp = tmp_path / "stk_flow=IMP"
        Z_dom = pd.read_csv(dom / "Z_split.csv", index_col=0); Z_imp = pd.read_csv(imp / "Z_split.csv", index_col=0)
        Y_dom = pd.read_csv(dom / "Y_split.csv", index_col=0); Y_imp = pd.read_csv(imp / "Y_split.csv", index_col=0)
        if flow == "TOTAL":
            controls = set(MODULE.derived_final_demand_control_columns(Y_dom.columns))
            Z = Z_dom + Z_imp
            Y = MODULE.enforce_final_demand_display_identities(Y_dom.drop(columns=list(controls)) + Y_imp.drop(columns=list(controls)), Z=Z)
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
    dom = tmp_path / "stk_flow=DOM"; imp = tmp_path / "stk_flow=IMP"; total = tmp_path / "stk_flow=TOTAL"
    Z_dom, Z_imp, Z_total = (pd.read_csv(path / "Z_split.csv", index_col=0) for path in (dom, imp, total))
    Y_dom, Y_imp, Y_total = (pd.read_csv(path / "Y_split.csv", index_col=0) for path in (dom, imp, total))
    pd.testing.assert_frame_equal(Z_total, Z_dom + Z_imp)
    for col in ("P3_S13", "P51G", "P52", "P53"):
        pd.testing.assert_series_equal(Y_total[col], Y_dom[col] + Y_imp[col], check_names=False)
    pd.testing.assert_series_equal(Y_total["P5M"], Y_total["P52"] + Y_total["P53"], check_names=False)
    pd.testing.assert_series_equal(Y_total["P5"], Y_total["P51G"] + Y_total["P5M"], check_names=False)
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


def test_construct_canonical_v_rejects_signed_unknown_row() -> None:
    parents = pd.Index(["A", "B"])
    reference = SimpleNamespace(
        V=pd.DataFrame([[1.0, -1.0]], index=pd.Index(["UNKNOWN"]), columns=parents),
        Z=pd.DataFrame(0.0, index=parents, columns=parents),
    )
    children = pd.Index(["A", "B"])
    zeros = pd.DataFrame(0.0, index=children, columns=children)
    with pytest.raises(ValueError, match="Unknown nonzero V row"):
        MODULE._construct_canonical_v(
            flow="DOM", reference=reference, child_parent={"A": "A", "B": "B"},
            Z=zeros, x_domestic=pd.Series([1.0, 1.0], index=children),
            Z_total=zeros, Z_imp=zeros, source_total_Z=zeros,
            imp_product_proxy=pd.Series([1.0, 1.0], index=children),
            quantum=1.0, rtol=0.0, atol=1e-9,
        )


def test_v_policy_preflight_marks_unknown_one_child_as_unknown() -> None:
    reference = SimpleNamespace(
        V=pd.DataFrame([[2.0]], index=pd.Index(["UNKNOWN"]), columns=pd.Index(["A"])),
    )
    preflight = MODULE._v_policy_preflight(
        references={"DOM": reference}, child_parent={"child": "A"},
        children=pd.Index(["child"]), x=pd.Series([1.0], index=["child"]),
        total_inputs=pd.Series([1.0], index=["child"]),
        imp_inputs=pd.Series([1.0], index=["child"]),
        imp_products=pd.Series([1.0], index=["child"]),
    )
    assert preflight.loc[0, "status"] == "unknown_row"


def test_construct_canonical_v_defers_ts_bp_until_p7_is_allocated() -> None:
    parent = pd.Index(["A"])
    children = pd.Index(["a1", "a2"])
    reference = SimpleNamespace(
        V=pd.DataFrame(
            [[150.0], [50.0], [100.0]], index=pd.Index(["TS_BP", "P7", "P1"]), columns=parent,
        ),
        Z=pd.DataFrame(0.0, index=parent, columns=parent),
    )
    zeros = pd.DataFrame(0.0, index=children, columns=children)
    V, _, _ = MODULE._construct_canonical_v(
        flow="TOTAL", reference=reference, child_parent={"a1": "A", "a2": "A"},
        Z=zeros, x_domestic=pd.Series([40.0, 60.0], index=children),
        Z_total=zeros, Z_imp=zeros, source_total_Z=pd.DataFrame(0.0, index=parent, columns=parent),
        imp_product_proxy=pd.Series([1.0, 3.0], index=children),
        quantum=1.0, rtol=0.0, atol=1e-9,
    )
    assert V is not None
    pd.testing.assert_series_equal(V.loc["P7"], pd.Series([12.5, 37.5], index=children), check_names=False)
    pd.testing.assert_series_equal(V.loc["TS_BP"], pd.Series([52.5, 97.5], index=children), check_names=False)
    assert float(V.loc["TS_BP"].sum()) == 150.0

def test_v_preflight_materiality_uses_atol_and_ignores_nan() -> None:
    from scripts.derive_total_iot_flow import _material_v_policy_problems

    preflight = pd.DataFrame(
        {
            "status": [
                "unknown_row",
                "unknown_row",
                "zero_proxy_nonzero_source",
                "allocatable",
            ],
            "source_value": [1e-12, float("nan"), 2e-6, 10.0],
        }
    )
    problems = _material_v_policy_problems(preflight, atol=1e-6)
    assert problems.index.tolist() == [2]


def test_assert_close_rejects_infinite_values() -> None:
    left = pd.Series([float("inf")], index=pd.Index(["a"]))
    right = pd.Series([float("inf")], index=pd.Index(["a"]))
    with pytest.raises(ValueError, match="max_abs_gap=inf"):
        MODULE._assert_close("infinite values", left, right, rtol=1e-9, atol=1e-6)
