from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/validate_energy_balance_pxp_outputs.py"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_energy_balance_pxp_outputs", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _write_system(folder: Path, *, suffix: str, labels: list[str]) -> None:
    Z = pd.DataFrame([[1.0 if i == j else 0.2 for j, _ in enumerate(labels)] for i, _ in enumerate(labels)], index=labels, columns=labels)
    Y = pd.DataFrame([[2.0] for _ in labels], index=labels, columns=["FD"])
    x = Z.sum(axis=1) + Y.sum(axis=1)
    A = Z.div(x.replace(0.0, pd.NA), axis=1).fillna(0.0)
    Z.to_csv(folder / f"Z_{suffix}.csv")
    Y.to_csv(folder / f"Y_{suffix}.csv")
    x.to_csv(folder / f"x_{suffix}.csv", header=True)
    A.to_csv(folder / f"A_{suffix}.csv")


def _write_valid_tree(root: Path) -> None:
    plan = root / "plan_diagnostics"
    plan.mkdir(parents=True)
    pd.DataFrame(
        [
            {"parent_code": "CPA_C19", "child_code": "EB_C19_RESIDUAL", "child_share": 1.0},
            {"parent_code": "CPA_C20", "child_code": "EB_C20_OTHERS", "child_share": 1.0},
            {"parent_code": "CPA_D", "child_code": "EB_D_OTHERS", "child_share": 1.0},
        ]
    ).to_csv(plan / "split_shares.csv", index=False)
    pd.DataFrame(
        [
            {"child_code": "EB_C19_RESIDUAL", "value_MEUR": 1.0},
            {"child_code": "EB_C20_OTHERS", "value_MEUR": 1.0},
            {"child_code": "EB_D_OTHERS", "value_MEUR": 1.0},
        ]
    ).to_csv(plan / "component_values.csv", index=False)
    for name in ["selector_residuals", "parent_residuals", "price_resolution"]:
        pd.DataFrame([{ "ok": 1 }]).to_csv(plan / f"{name}.csv", index=False)
    (plan / "diagnostics_metadata.json").write_text('{"diagnostic_keys": []}', encoding="utf-8")

    for flow in ["DOM", "IMP", "TOTAL"]:
        flow_dir = root / f"stk_flow={flow}"
        flow_dir.mkdir(parents=True)
        _write_system(flow_dir, suffix="split", labels=["EB_C20_OTHERS", "OTHER"])
        _write_system(flow_dir, suffix="reaggregated", labels=["CPA_C20_others", "OTHER"])

    pd.DataFrame([{ "variant_value": "TOTAL", "n_split_rows": 2 }]).to_csv(root / "run_summary.csv", index=False)


def test_validator_accepts_valid_synthetic_tree(tmp_path: Path) -> None:
    module = _load_validator_module()
    _write_valid_tree(tmp_path)
    summary = module.validate_output_tree(tmp_path, expected_flows=["DOM", "IMP", "TOTAL"], require_reaggregated=True)
    assert set(summary["flow"]) == {"DOM", "IMP", "TOTAL"}
    assert set(summary["kind"]) == {"split", "reaggregated"}


def test_validator_rejects_duplicated_plan_diagnostics_in_flow_folder(tmp_path: Path) -> None:
    module = _load_validator_module()
    _write_valid_tree(tmp_path)
    (tmp_path / "stk_flow=TOTAL" / "component_values.csv").write_text("bad\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Plan diagnostics appear duplicated"):
        module.validate_output_tree(tmp_path, expected_flows=["DOM", "IMP", "TOTAL"], require_reaggregated=True)


def test_validator_rejects_reaggregated_eb_child_labels(tmp_path: Path) -> None:
    module = _load_validator_module()
    _write_valid_tree(tmp_path)
    flow_dir = tmp_path / "stk_flow=TOTAL"
    _write_system(flow_dir, suffix="reaggregated", labels=["EB_C20_OTHERS", "OTHER"])
    with pytest.raises(SystemExit, match="still contains EB_"):
        module.validate_output_tree(tmp_path, expected_flows=["DOM", "IMP", "TOTAL"], require_reaggregated=True)
