from __future__ import annotations

import zipfile

import pandas as pd
import pytest


def _write_member(zf: zipfile.ZipFile, name: str, text: str) -> None:
    zf.writestr(name, text.encode("utf-8"))


def test_read_extension_units_accepts_explicit_and_unnamed_key_headers(tmp_path):
    """Source row labels remain authoritative under both observed header forms."""
    from ecolecon_mrio.exiobase3_extensions import read_extension_units

    source = tmp_path / "unit_headers.zip"
    with zipfile.ZipFile(source, "w") as zf:
        _write_member(zf, "air_emissions/unit.txt", "stressor\tunit\nCH4\tkg\n")
        _write_member(
            zf,
            "factor_inputs/unit.txt",
            "\tunit\n"
            "Compensation of employees; wages, salaries, & employers' social "
            "contributions: Low-skilled\tM.EUR\n",
        )

    explicit = read_extension_units(source, extension="air_emissions")
    unnamed = read_extension_units(source, extension="factor_inputs")

    assert explicit.to_dict() == {"CH4": "kg"}
    assert unnamed.to_dict() == {
        "Compensation of employees; wages, salaries, & employers' social "
        "contributions: Low-skilled": "M.EUR"
    }
    assert explicit.index.name == "stressor"
    assert unnamed.index.name == "stressor"


def test_read_extension_units_rejects_duplicate_row_labels(tmp_path):
    """Duplicate physical-row keys cannot select an ambiguous source unit."""
    from ecolecon_mrio.exiobase3_extensions import read_extension_units

    source = tmp_path / "duplicate_units.zip"
    with zipfile.ZipFile(source, "w") as zf:
        _write_member(
            zf,
            "synthetic/unit.txt",
            "stressor\tunit\nSynthetic indicator\tkg\nSynthetic indicator\tg\n",
        )

    with pytest.raises(ValueError, match="Duplicate stressor/indicator labels"):
        read_extension_units(source, extension="synthetic")


@pytest.mark.parametrize(
    "metadata, message",
    [
        ("stressor\tquantity_unit\nSynthetic indicator\tkg\n", "Unexpected unit file format"),
        ("stressor\tunit\tunit\nSynthetic indicator\tkg\tg\n", "Unexpected unit file format"),
        ("stressor\tunit\nSynthetic indicator\t\n", "Blank unit"),
        ("stressor\tunit\n\tkg\n", "Blank stressor/indicator label"),
    ],
)
def test_read_extension_units_rejects_missing_or_ambiguous_metadata(
    tmp_path, metadata, message
):
    """Malformed metadata must fail instead of inventing labels or dimensions."""
    from ecolecon_mrio.exiobase3_extensions import read_extension_units

    source = tmp_path / "malformed_units.zip"
    with zipfile.ZipFile(source, "w") as zf:
        _write_member(zf, "synthetic/unit.txt", metadata)

    with pytest.raises(ValueError, match=message):
        read_extension_units(source, extension="synthetic")


def test_extract_quantity_and_price_from_minimal_zip(tmp_path):
    from ecolecon_mrio.exiobase3_extensions import extract_quantity_and_price

    zp = tmp_path / "IOT_2020_pxp.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        _write_member(
            zf,
            "energy/unit.txt",
            "stressor\tunit\nEnergy use - Gross\tTJ\n",
        )
        _write_member(
            zf,
            "energy/F.txt",
            "\n".join(
                [
                    "region\tAT\tAT",
                    "sector\tCoal\tElectricity",
                    "stressor\t\t",
                    "Energy use - Gross\t1\t2",
                    "",
                ]
            ),
        )
        _write_member(
            zf,
            "energy/F_Y.txt",
            "\n".join(
                [
                    "region\tAT",
                    "category\tExports: Total (fob)",
                    "stressor\t",
                    "Energy use - Gross\t0",
                    "",
                ]
            ),
        )

    mon_out = pd.Series({"Coal": 10.0, "Electricity": 40.0})
    res = extract_quantity_and_price(
        container=zp,
        region="AT",
        sectors=["Coal", "Electricity"],
        monetary_output=mon_out,
        extensions=("energy",),
        stressor_keywords=("Energy use - Gross",),
        cache_dir=tmp_path / "cache",
    )

    assert res is not None
    assert set(res["exio_sector"]) == {"Coal", "Electricity"}
    assert (res["unit"] == "TJ").all()
    coal = res[res["exio_sector"] == "Coal"].iloc[0]
    elec = res[res["exio_sector"] == "Electricity"].iloc[0]
    assert coal["quantity"] == 1.0
    assert elec["quantity"] == 2.0
    assert abs(float(coal["quantity_share"]) - (1.0 / 3.0)) < 1e-12
    assert abs(float(elec["quantity_share"]) - (2.0 / 3.0)) < 1e-12
    assert abs(float(coal["value_per_physical_input"]) - 10.0) < 1e-12
    assert abs(float(elec["value_per_physical_input"]) - 20.0) < 1e-12
