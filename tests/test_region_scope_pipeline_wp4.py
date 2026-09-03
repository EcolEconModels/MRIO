from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio.exiobase3 import read_exiobase3_scope_export_vector
from ecolecon_mrio.iot_aggregation import aggregate_environmental_extensions
from ecolecon_mrio.region_aggregation import RegionScope
from scripts.split_eurostat_energy_by_exiobase import _resolve_pipeline_scope


def _write_table(zf: zipfile.ZipFile, name: str, regions: list[str], labels: list[str], rows: list[tuple[str,str,list[float]]]) -> None:
    text = "\t\t" + "\t".join(regions) + "\n"
    text += "region\tsector\t" + "\t".join(labels) + "\n"
    text += "unit\tunit\t" + "\t".join(["M.EUR"] * len(labels)) + "\n"
    for region, sector, values in rows:
        text += region + "\t" + sector + "\t" + "\t".join(map(str, values)) + "\n"
    zf.writestr(name, text)


def test_scope_export_excludes_intra_scope_trade(tmp_path: Path) -> None:
    zp = tmp_path / "iot.zip"
    # one sector; destination columns AT, DE, US. AT+DE is the scope.
    rows = [
        ("AT", "S", [1.0, 2.0, 3.0]),
        ("DE", "S", [4.0, 5.0, 6.0]),
        ("US", "S", [7.0, 8.0, 9.0]),
    ]
    with zipfile.ZipFile(zp, "w") as zf:
        _write_table(zf, "Z.txt", ["AT", "DE", "US"], ["S", "S", "S"], rows)
        _write_table(zf, "Y.txt", ["AT", "DE", "US"], ["HH", "HH", "HH"], rows)
    scope = RegionScope(code="EU_TEST", members=("AT", "DE"))
    exports = read_exiobase3_scope_export_vector(source_zip=zp, scope=scope)
    # AT->US (3) + DE->US (6), in both Z and Y = 18. Intra AT<->DE is excluded.
    assert exports.loc["S"] == pytest.approx(18.0)


def test_singleton_scope_needs_no_mapping() -> None:
    scope = _resolve_pipeline_scope(scope_code="AT", mapping_csv=None, source_regions=None)
    assert scope == RegionScope(code="AT", members=("AT",))


def test_mapping_backed_scope_uses_file_members(tmp_path: Path) -> None:
    mapping = tmp_path / "scope.csv"
    pd.DataFrame(
        {
            "scope_code": ["EU_TEST", "EU_TEST"],
            "scope_label": ["EU test", "EU test"],
            "member_region": ["AT", "DE"],
            "member_order": [1, 2],
            "enabled": [1, 1],
            "notes": ["", ""],
        }
    ).to_csv(mapping, index=False)
    scope = _resolve_pipeline_scope(
        scope_code="EU_TEST", mapping_csv=mapping, source_regions=("AT", "DE", "US")
    )
    assert scope.members == ("AT", "DE")


def test_boundary_aware_environmental_reaggregation_uses_flow_denominator() -> None:
    frame = pd.DataFrame(
        {
            "sector": ["a", "b"],
            "boundary": ["external_imports_direct", "external_imports_direct"],
            "denominator_role": ["target_import_supply", "target_import_supply"],
            "extension": ["air", "air"],
            "stressor": ["CO2", "CO2"],
            "unit": ["kg", "kg"],
            "value": [2.0, 6.0],
            "monetary_output": [10.0, 30.0],
            "intensity": [0.2, 0.2],
        }
    )
    out = aggregate_environmental_extensions(
        frame,
        {"a": "A", "b": "A"},
        output=pd.Series({"a": 10.0, "b": 30.0}),
    )
    assert out is not None
    row = out.iloc[0]
    assert row["boundary"] == "external_imports_direct"
    assert row["value"] == pytest.approx(8.0)
    assert row["monetary_output"] == pytest.approx(40.0)
    assert row["intensity"] == pytest.approx(0.2)


def test_boundary_reaggregation_rejects_domestic_x_as_import_denominator() -> None:
    frame = pd.DataFrame(
        {
            "sector": ["a"],
            "boundary": ["external_imports_direct"],
            "denominator_role": ["target_import_supply"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [2.0],
            "monetary_output": [10.0],
            "intensity": [0.2],
        }
    )
    with pytest.raises(ValueError, match="denominator disagrees"):
        aggregate_environmental_extensions(
            frame, {"a": "A"}, output=pd.Series({"a": 99.0})
        )
