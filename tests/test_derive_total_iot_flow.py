from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "derive_total_iot_flow.py"
SPEC = importlib.util.spec_from_file_location("derive_total_iot_flow", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_flow(root: Path, flow: str, *, scale: float) -> None:
    out = root / f"stk_flow={flow}"
    out.mkdir(parents=True)
    Z = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]], index=pd.Index(["a", "b"]), columns=pd.Index(["a", "b"])
    ) * scale
    Y = pd.DataFrame([[5.0], [6.0]], index=pd.Index(["a", "b"]), columns=pd.Index(["P3_S14"])) * scale
    V = pd.DataFrame([[7.0, 8.0]], index=pd.Index(["D1"]), columns=pd.Index(["a", "b"])) * scale
    M = pd.DataFrame([[9.0]], index=pd.Index(["TU"]), columns=pd.Index(["P3_S14"])) * scale
    for name, frame in (("Z_split.csv", Z), ("Y_split.csv", Y), ("V_split.csv", V), ("M_split.csv", M)):
        frame.to_csv(out / name)
    (out / "split_info.json").write_text(json.dumps({"variant_value": flow}), encoding="utf-8")


def test_derive_total_flow_is_additive_and_recomputes_x_a(tmp_path: Path) -> None:
    _write_flow(tmp_path, "DOM", scale=1.0)
    _write_flow(tmp_path, "IMP", scale=2.0)
    _write_flow(tmp_path, "TOTAL", scale=99.0)

    MODULE.derive_total_flow(tmp_path)

    total = tmp_path / "stk_flow=TOTAL"
    Z = pd.read_csv(total / "Z_split.csv", index_col=0)
    Y = pd.read_csv(total / "Y_split.csv", index_col=0)
    x = pd.read_csv(total / "x_split.csv", index_col=0).iloc[:, 0]
    A = pd.read_csv(total / "A_split.csv", index_col=0)
    V = pd.read_csv(total / "V_split.csv", index_col=0)
    M = pd.read_csv(total / "M_split.csv", index_col=0)

    expected_Z = pd.DataFrame(
        [[3.0, 6.0], [9.0, 12.0]], index=pd.Index(["a", "b"]), columns=pd.Index(["a", "b"])
    )
    expected_Y = pd.DataFrame(
        [[15.0], [18.0]], index=pd.Index(["a", "b"]), columns=pd.Index(["P3_S14"])
    )
    pd.testing.assert_frame_equal(Z, expected_Z)
    pd.testing.assert_frame_equal(Y, expected_Y)
    pd.testing.assert_series_equal(x, expected_Z.sum(axis=1) + expected_Y.sum(axis=1), check_names=False)
    np.testing.assert_allclose(A.to_numpy(), expected_Z.divide(x, axis="columns").to_numpy())
    np.testing.assert_allclose(V.to_numpy(), [[21.0, 24.0]])
    np.testing.assert_allclose(M.to_numpy(), [[27.0]])

    info = json.loads((total / "split_info.json").read_text(encoding="utf-8"))
    assert info["flow_construction"] == "derived_dom_plus_imp"
    assert (tmp_path / "derived_total_flow_identity.csv").is_file()
