from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
