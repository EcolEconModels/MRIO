from __future__ import annotations

import zipfile

import pandas as pd


def _write_member(zf: zipfile.ZipFile, name: str, text: str) -> None:
    zf.writestr(name, text.encode("utf-8"))


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
    assert abs(float(coal["implied_price"]) - 10.0) < 1e-12
    assert abs(float(elec["implied_price"]) - 20.0) < 1e-12

