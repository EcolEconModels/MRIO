from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib.util

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cli_module():
    script = REPO_ROOT / "scripts" / "split_eurostat_iot_by_energy_balances.py"
    spec = importlib.util.spec_from_file_location("split_eurostat_iot_by_energy_balances", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_cli_compiles_once_writes_plan_diagnostics_and_outputs(tmp_path: Path, monkeypatch) -> None:
    cli = _load_cli_module()

    Z_total = pd.DataFrame([[10.0]], index=["CPA_C20"], columns=["CPA_C20"])
    Y_total = pd.DataFrame([[5.0]], index=["CPA_C20"], columns=["P3_S14"])
    Z_dom = pd.DataFrame([[6.0]], index=["CPA_C20"], columns=["CPA_C20"])
    Y_dom = pd.DataFrame([[2.0]], index=["CPA_C20"], columns=["P3_S14"])

    def fake_load_variants(**kwargs):
        assert kwargs["variant_values"] == ["DOM", "TOTAL"]
        return {
            "DOM": SimpleNamespace(Z=Z_dom, Y=Y_dom, meta={}),
            "TOTAL": SimpleNamespace(Z=Z_total, Y=Y_total, meta={}),
        }

    monkeypatch.setattr(cli, "load_eurostat_iot_variants", fake_load_variants)
    monkeypatch.setattr(cli, "load_energy_balance_pxp_split_frame", lambda path: pd.DataFrame())
    monkeypatch.setattr(cli, "load_energy_balance_pxp_reaggregation_frame", lambda path: pd.DataFrame())
    monkeypatch.setattr(cli, "load_energy_balance_pxp_input_technology_frame", lambda path: pd.DataFrame())
    monkeypatch.setattr(cli, "load_energy_balance_pxp_prices_frame", lambda path: pd.DataFrame())
    monkeypatch.setattr(cli, "load_energy_balance_dictionary_frame", lambda path: pd.DataFrame())

    compile_calls = []
    run_calls = []

    plan = SimpleNamespace(
        split_profiles={"CPA_C20": object()},
        component_values=pd.DataFrame([{"child_code": "EB_C20_OTHERS"}]),
        split_shares=pd.DataFrame([{"parent_code": "CPA_C20", "child_code": "EB_C20_OTHERS", "child_share": 1.0}]),
        selector_residuals=pd.DataFrame(),
        parent_residuals=pd.DataFrame([{"child_code": "EB_C20_OTHERS"}]),
        price_resolution=pd.DataFrame([{"price_code": "p"}]),
        diagnostics={
            "missing_or_zero_selectors": pd.DataFrame(),
            "quantity_lookup": pd.DataFrame([{"child_code": "EB_C20_OTHERS"}]),
            "validation_summary": pd.DataFrame([{"parents": "CPA_C20"}]),
        },
        reaggregation_mapping={"EB_C20_OTHERS": "CPA_C20_others"},
    )

    def fake_compile(**kwargs):
        compile_calls.append(kwargs)
        assert kwargs["Z"].equals(Z_total)
        assert kwargs["energy_balance_dictionary_frame"] is not None
        return plan

    def fake_run(Z, Y, plan_arg, **kwargs):
        run_calls.append((Z.copy(), Y.copy(), kwargs))
        return SimpleNamespace(
            Z=pd.DataFrame([[1.0]], index=["EB_C20_OTHERS"], columns=["EB_C20_OTHERS"]),
            Y=pd.DataFrame([[2.0]], index=["EB_C20_OTHERS"], columns=["P3_S14"]),
            x=pd.Series([3.0], index=["EB_C20_OTHERS"]),
            A=pd.DataFrame([[1.0 / 3.0]], index=["EB_C20_OTHERS"], columns=["EB_C20_OTHERS"]),
        )

    def fake_aggregate_iot(Z, Y, mapping, **kwargs):
        assert mapping == {"EB_C20_OTHERS": "CPA_C20_others"}
        return SimpleNamespace(
            Z=pd.DataFrame([[1.0]], index=["CPA_C20_others"], columns=["CPA_C20_others"]),
            Y=pd.DataFrame([[2.0]], index=["CPA_C20_others"], columns=["P3_S14"]),
            x=pd.Series([3.0], index=["CPA_C20_others"]),
            A=pd.DataFrame([[1.0 / 3.0]], index=["CPA_C20_others"], columns=["CPA_C20_others"]),
        )

    monkeypatch.setattr(cli, "compile_energy_balance_pxp_split_plan", fake_compile)
    monkeypatch.setattr(cli, "run_energy_balance_pxp_split", fake_run)
    monkeypatch.setattr(cli, "aggregate_iot", fake_aggregate_iot)

    outdir = tmp_path / "out"
    rc = cli.main(
        [
            "--eurostat-stk-flows",
            '["DOM","TOTAL"]',
            "--outdir",
            str(outdir),
            "--storage-folder",
            str(tmp_path / "cache"),
            "--energy-balance-dictionary-csv",
            str(tmp_path / "dict.csv"),
            "--write-reaggregated",
        ]
    )

    assert rc == 0
    assert len(compile_calls) == 1
    assert len(run_calls) == 2
    assert (outdir / "plan_diagnostics" / "component_values.csv").exists()
    assert (outdir / "plan_diagnostics" / "validation_summary.csv").exists()
    for flow in ["DOM", "TOTAL"]:
        flow_dir = outdir / f"stk_flow={flow}"
        assert (flow_dir / "Z_split.csv").exists()
        assert (flow_dir / "Y_split.csv").exists()
        assert (flow_dir / "x_split.csv").exists()
        assert (flow_dir / "A_split.csv").exists()
        assert (flow_dir / "Z_reaggregated.csv").exists()
        assert not (flow_dir / "component_values.csv").exists()
    assert (outdir / "run_summary.csv").exists()


def test_shell_wrapper_is_pure_cli_record() -> None:
    shell = (REPO_ROOT / "scripts" / "pipelines" / "at_2021_energy_balance_pxp.sh").read_text(encoding="utf-8")
    assert "--energy-balance-dictionary-csv" in shell
    assert "energy_balance_selector_quantities" not in shell
    assert "python -" not in shell
