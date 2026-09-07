from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_environmental_support.py"
spec = importlib.util.spec_from_file_location("audit_environmental_support", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _write_vector(path: Path, values: dict[str, float]) -> None:
    pd.Series(values, name="x", dtype="float64").to_csv(path)


def _artifact_dir(tmp_path: Path) -> Path:
    flow = tmp_path / "stk_flow=DOM"
    flow.mkdir()
    _write_vector(flow / "x_split.csv", {"a": 0.0, "d@T3": 4.0, "T2": 5.0, "T4": 6.0})
    pd.DataFrame(
        [
            {"source_code": "b", "target_code": "T2", "weight": 1.0},
            {"source_code": "c", "target_code": "T2", "weight": 1.0},
            {"source_code": "d", "target_code": "T3", "weight": 0.4},
            {"source_code": "d", "target_code": "T4", "weight": 0.6},
        ]
    ).to_csv(flow / "flow_specific_exio_to_eurostat_weights.csv", index=False)
    (flow / "exio_selected_sectors.json").write_text(
        json.dumps({"PARENT_A": ["a"], "T3": ["d@T3"]}), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "parent_code": "T3",
                "current_child_code": "d@T3",
                "source_code": "d",
                "concordance_target_code": "T3",
                "edge_weight_current_reference": 0.4,
            }
        ]
    ).to_csv(flow / "split_child_edges.csv", index=False)
    return flow


def test_projection_reconstruction_preserves_raw_partition(tmp_path: Path) -> None:
    flow = _artifact_dir(tmp_path)
    P, parent = module.build_projection_from_split_artifacts(
        flow, raw_exio_sectors=("a", "b", "c", "d")
    )
    assert np.allclose(P.sum(axis=0).to_numpy(), 1.0, rtol=0.0, atol=1e-12)
    assert P.loc["a", "a"] == pytest.approx(1.0)
    assert P.loc["d@T3", "d"] == pytest.approx(0.4)
    assert P.loc["T4", "d"] == pytest.approx(0.6)
    assert P.loc["T2", "b"] == pytest.approx(1.0)
    assert P.loc["T2", "c"] == pytest.approx(1.0)
    assert parent["a"] == "PARENT_A"
    assert parent["d@T3"] == "T3"


def test_projection_reconstruction_fails_on_unproven_group_child(tmp_path: Path) -> None:
    flow = _artifact_dir(tmp_path)
    data = json.loads((flow / "exio_selected_sectors.json").read_text(encoding="utf-8"))
    data["GROUP"] = ["custom_group_child"]
    (flow / "exio_selected_sectors.json").write_text(json.dumps(data), encoding="utf-8")
    vector = pd.read_csv(flow / "x_split.csv", index_col=0)
    vector.loc["custom_group_child"] = 0.0
    vector.to_csv(flow / "x_split.csv")
    with pytest.raises(ValueError, match="cannot be reconstructed exactly"):
        module.build_projection_from_split_artifacts(
            flow, raw_exio_sectors=("a", "b", "c", "d")
        )


def test_production_support_state_machine_distinguishes_unavailable_from_blocking() -> None:
    labels = pd.Index(["gas_coke", "zero_zero", "supported", "no_projection"])
    result = module.classify_production_support(
        mapped_value=pd.Series([3.0, 0.0, 4.0, 0.0], index=labels),
        mapped_output=pd.Series([0.0, 0.0, 2.0, 0.0], index=labels),
        target_output=pd.Series([0.0, 5.0, 7.0, 0.0], index=labels),
        projection_support=pd.Series([True, True, True, False], index=labels),
    ).set_index("sector")
    assert result.loc["gas_coke", "support_state"] == "NONZERO_F_ZERO_X"
    assert not bool(result.loc["gas_coke", "method_blocking_positive_target"])
    assert result.loc["zero_zero", "support_state"] == "ZERO_ZERO"
    assert bool(result.loc["zero_zero", "method_blocking_positive_target"])
    assert result.loc["supported", "support_state"] == "SUPPORTED"
    assert not bool(result.loc["supported", "method_blocking_positive_target"])
    assert result.loc["no_projection", "support_state"] == "NO_PROJECTION_SUPPORT"


def test_final_demand_support_keeps_nonzero_zero_denominator_fail_closed() -> None:
    labels = pd.Index(["P3", "P52", "P6"])
    result = module.classify_final_demand_support(
        physical=pd.Series([2.0, 5.0, 0.0], index=labels),
        monetary=pd.Series([10.0, 0.0, 0.0], index=labels),
    ).set_index("final_demand_category")
    assert result.loc["P3", "support_state"] == "SUPPORTED"
    assert result.loc["P52", "support_state"] == "NONZERO_FY_ZERO_DENOMINATOR"
    assert bool(result.loc["P52", "method_blocking"])
    assert result.loc["P6", "support_state"] == "ZERO_ZERO"
