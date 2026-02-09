import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def test_xlsx_sanitize_removes_nan_literals(tmp_path: Path):
    from scripts.build_energy_balances_65sectors_primary_sources import _xlsx_sanitize_df

    df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", np.nan]})
    out = tmp_path / "t.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        _xlsx_sanitize_df(df).to_excel(w, sheet_name="S", index=False)

    with zipfile.ZipFile(out, "r") as z:
        xml = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert ">nan<" not in xml

