from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import pytest

from ecolecon_mrio import exiobase3_monetary_unit
from ecolecon_mrio.exiobase3_extensions import load_exiobase3_extension_row_with_unit


def _write_extension_zip(path: Path, *, include_unit: bool = True) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "emissions/F.txt",
            "\tAT\tAT\n"
            "\tsector_a\tsector_b\n"
            "\t\t\n"
            "CO2 - combustion - air\t1.5\t2.5\n"
            "Other stressor\t3.0\t4.0\n",
        )
        zf.writestr(
            "emissions/F_Y.txt",
            "\tAT\tAT\n"
            "\tHouseholds\tGovernment\n"
            "\t\t\n"
            "CO2 - combustion - air\t0.5\t0.7\n"
            "Other stressor\t1.0\t1.2\n",
        )
        if include_unit:
            zf.writestr(
                "emissions/unit.txt",
                "stressor\tunit\nCO2 - combustion - air\tkg\nOther stressor\tt\n",
            )


def test_exiobase3_monetary_unit_default_and_override() -> None:
    default = exiobase3_monetary_unit(system="pxp")
    assert default.unit == "MEUR"
    assert default.unit_source == "exiobase3_monetary_table_convention"
    override = exiobase3_monetary_unit(system="pxp", override="EUR")
    assert override.unit == "EUR"
    assert override.unit_source == "explicit_override"


def test_extension_row_with_unit_reads_unit_txt(tmp_path: Path) -> None:
    archive = tmp_path / "extension.zip"
    _write_extension_zip(archive)
    result = load_exiobase3_extension_row_with_unit(
        archive,
        extension="emissions",
        kind="F",
        exact_priority=("CO2 - combustion - air",),
    )
    assert result.stressor == "CO2 - combustion - air"
    assert result.unit == "kg"
    assert result.unit_source == "emissions/unit.txt"
    assert isinstance(result.values.index, pd.MultiIndex)
    assert list(result.values.index.names) == ["region", "sector"]
    assert result.values.loc[("AT", "sector_a")] == pytest.approx(1.5)



def test_extension_row_with_unit_aligns_official_codes_to_canonical_basis(tmp_path: Path) -> None:
    """Raw F sector codes must map exactly onto the x/Z/Y name-based basis."""

    archive = tmp_path / "extension_codes.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "emissions/F.txt",
            "\tAT\tAT\r\n"
            "\tp01.a\tp01.b\r\n"
            "\t\t\r\n"
            "CO2 - combustion - air\t1.5\t2.5\r\n",
        )
        zf.writestr(
            "emissions/unit.txt",
            "stressor\tunit\nCO2 - combustion - air\tkg\n",
        )
    canonical = pd.MultiIndex.from_tuples(
        [("AT", "Wheat"), ("AT", "Paddy rice")],
        names=["region", "sector"],
    )
    result = load_exiobase3_extension_row_with_unit(
        archive,
        extension="emissions",
        kind="F",
        exact_priority=("CO2 - combustion - air",),
        system="pxp",
        basis_index=canonical,
    )
    assert result.values.index.equals(canonical)
    assert result.values.loc[("AT", "Wheat")] == pytest.approx(2.5)
    assert result.values.loc[("AT", "Paddy rice")] == pytest.approx(1.5)


def test_extension_row_with_unit_f_y_uses_final_demand_category_index(tmp_path: Path) -> None:
    archive = tmp_path / "extension_f_y.zip"
    _write_extension_zip(archive)
    result = load_exiobase3_extension_row_with_unit(
        archive,
        extension="emissions",
        kind="F_Y",
        exact_priority=("CO2 - combustion - air",),
    )
    assert result.stressor == "CO2 - combustion - air"
    assert list(result.values.index.names) == ["region", "final_demand_category"]
    assert result.values.loc[("AT", "Households")] == pytest.approx(0.5)
    assert result.metadata["row_index_names"] == ["region", "final_demand_category"]

def test_extension_row_with_unit_override_when_unit_file_missing(tmp_path: Path) -> None:
    archive = tmp_path / "extension_missing_unit.zip"
    _write_extension_zip(archive, include_unit=False)
    result = load_exiobase3_extension_row_with_unit(
        archive,
        extension="emissions",
        kind="F",
        exact_priority=("CO2 - combustion - air",),
        unit_override="kg",
    )
    assert result.unit == "kg"
    assert result.unit_source == "explicit_override"


def test_extension_row_with_unit_missing_unit_fails(tmp_path: Path) -> None:
    archive = tmp_path / "extension_missing_unit.zip"
    _write_extension_zip(archive, include_unit=False)
    with pytest.raises(ValueError, match="unit metadata"):
        load_exiobase3_extension_row_with_unit(
            archive,
            extension="emissions",
            kind="F",
            exact_priority=("CO2 - combustion - air",),
        )
