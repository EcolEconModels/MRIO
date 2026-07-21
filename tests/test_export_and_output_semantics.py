from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio.exiobase3 import read_exiobase3_region_export_vector
from ecolecon_mrio.exio_eurostat_concordance_audit import ConcordanceAuditResult


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "derive_total_iot_flow.py"
SPEC = importlib.util.spec_from_file_location("derive_total_iot_flow_semantics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_table(zf: zipfile.ZipFile, name: str, rows: list[str]) -> None:
    header = "region\tsector\tAT\tDE\tFR\nsector\tsector\ta\tb\tc\nunit\tunit\tEUR\tEUR\tEUR\n"
    zf.writestr(name, header + "".join(rows))


def test_region_export_vector_sums_foreign_intermediate_and_final_use(tmp_path: Path) -> None:
    source = tmp_path / "exio.zip"
    with zipfile.ZipFile(source, "w") as zf:
        _write_table(zf, "Z.txt", ["AT\tp1\t1\t2\t3\n", "DE\tp1\t9\t9\t9\n"])
        _write_table(zf, "Y.txt", ["AT\tp1\t4\t5\t6\n", "AT\tp2\t7\t8\t9\n"])
    result = read_exiobase3_region_export_vector(source_zip=source, region="AT")
    assert result.loc["p1"] == 16.0
    assert result.loc["p2"] == 17.0


def _write_flow(root: Path, flow: str, Z: pd.DataFrame, Y: pd.DataFrame) -> None:
    directory = root / f"stk_flow={flow}"
    directory.mkdir(parents=True)
    Z.to_csv(directory / "Z_agg.csv")
    Y.to_csv(directory / "Y_agg.csv")


def test_output_contract_uses_domestic_output_and_keeps_additive_use(tmp_path: Path) -> None:
    idx = pd.Index(["a", "b"])
    Z_dom = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx, columns=idx)
    Y_dom = pd.DataFrame([[5.0], [6.0]], index=idx, columns=pd.Index(["P6"]))
    Z_imp = Z_dom * 2.0
    Y_imp = Y_dom * 2.0
    _write_flow(tmp_path, "DOM", Z_dom, Y_dom)
    _write_flow(tmp_path, "IMP", Z_imp, Y_imp)
    _write_flow(tmp_path, "TOTAL", Z_dom + Z_imp, Y_dom + Y_imp)
    MODULE.validate_additive_outputs(tmp_path, suffix="agg", atol=1e-9, rtol=1e-12)
    x_dom = pd.read_csv(tmp_path / "stk_flow=DOM" / "x_agg.csv", index_col=0).iloc[:, 0]
    x_imp = pd.read_csv(tmp_path / "stk_flow=IMP" / "x_agg.csv", index_col=0).iloc[:, 0]
    x_total = pd.read_csv(tmp_path / "stk_flow=TOTAL" / "x_agg.csv", index_col=0).iloc[:, 0]
    pd.testing.assert_series_equal(x_imp, x_dom, check_names=False)
    pd.testing.assert_series_equal(x_total, x_dom, check_names=False)
    use_dom = pd.read_csv(tmp_path / "stk_flow=DOM" / "use_total_agg.csv", index_col=0).iloc[:, 0]
    use_imp = pd.read_csv(tmp_path / "stk_flow=IMP" / "use_total_agg.csv", index_col=0).iloc[:, 0]
    use_total = pd.read_csv(tmp_path / "stk_flow=TOTAL" / "use_total_agg.csv", index_col=0).iloc[:, 0]
    np.testing.assert_allclose(use_total.to_numpy(), (use_dom + use_imp).to_numpy())
    A_dom = pd.read_csv(tmp_path / "stk_flow=DOM" / "A_agg.csv", index_col=0)
    A_imp = pd.read_csv(tmp_path / "stk_flow=IMP" / "A_agg.csv", index_col=0)
    A_total = pd.read_csv(tmp_path / "stk_flow=TOTAL" / "A_agg.csv", index_col=0)
    np.testing.assert_allclose(A_total.to_numpy(), (A_dom + A_imp).to_numpy())


def test_additive_audit_requires_mapped_result_fields() -> None:
    audit_script = Path(__file__).resolve().parents[1] / "scripts" / "run_exio_eurostat_concordance_audit_all_flows.py"
    spec = importlib.util.spec_from_file_location("additive_audit", audit_script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    incomplete = cast(
        ConcordanceAuditResult,
        SimpleNamespace(
            mapped_z=None,
            mapped_y=None,
            mapped_x=None,
            eurostat_z=None,
            eurostat_y_total=None,
            eurostat_x=None,
        ),
    )
    with pytest.raises(ValueError, match="missing audit result fields"):
        module._require_additive_audit_fields(incomplete)
