from __future__ import annotations

import zipfile

import pandas as pd


def _write_member(zf: zipfile.ZipFile, name: str, text: str) -> None:
    zf.writestr(name, text.encode("utf-8"))


def test_split_eurostat_energy_sectors_extract_quantity_and_price(tmp_path):
    from mrio_pyramids.exiobase3 import ExioRegionSlice
    from split_eurostat_energy_by_exiobase import split_eurostat_energy_sectors_using_exiobase

    exio_sectors = ["coke oven products", "refined petroleum products", "electricity from coal", "gas supply"]

    # Minimal EXIO slice (domestic block + one final-demand column).
    Z_domestic = pd.DataFrame(
        [
            [1.0, 2.0, 0.5, 0.1],
            [0.2, 1.0, 0.4, 0.2],
            [0.1, 0.3, 1.5, 0.4],
            [0.0, 0.2, 0.6, 1.0],
        ],
        index=exio_sectors,
        columns=exio_sectors,
    )
    Y_region = pd.DataFrame(
        {"FD": [1.0, 2.0, 3.0, 4.0]},
        index=exio_sectors,
    )
    x_region = (Z_domestic.sum(axis=1) + Y_region.sum(axis=1)).astype("float64")

    # Minimal extension zip for quantity/price extraction.
    zp = tmp_path / "IOT_2020_pxp.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        _write_member(zf, "energy/unit.txt", "stressor\tunit\nEnergy use - Gross\tTJ\n")
        _write_member(
            zf,
            "energy/F.txt",
            "\n".join(
                [
                    "region\tAT\tAT\tAT\tAT",
                    "sector\tcoke oven products\trefined petroleum products\telectricity from coal\tgas supply",
                    "stressor\t\t\t\t",
                    "Energy use - Gross\t1\t2\t4\t3",
                    "",
                ]
            ),
        )

    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=x_region,
        source_zip=zp,
    )

    euro_sectors = ["A", "C19", "D35"]
    euro_Z = pd.DataFrame(
        [
            [10.0, 1.0, 2.0],
            [1.0, 5.0, 1.0],
            [2.0, 1.0, 6.0],
        ],
        index=euro_sectors,
        columns=euro_sectors,
    )
    euro_Y = pd.DataFrame({"FD": [3.0, 2.0, 1.0]}, index=euro_sectors)

    res = split_eurostat_energy_sectors_using_exiobase(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_region="AT",
        exio_slice=exio_slice,
        eurostat_sector_19="C19",
        eurostat_sector_35="D35",
        exio_patterns_19=("coke", "refined"),
        exio_patterns_35=("electricity", "gas"),
        extract_quantity_and_price=True,
        ras_max_iter=5_000,
        ras_tol=1e-12,
    )

    assert "C19" not in res.Z.index
    assert "D35" not in res.Z.index
    assert any(s.startswith("C19::") for s in res.Z.index)
    assert any(s.startswith("D35::") for s in res.Z.index)

    assert res.exio_quantity_and_price is not None
    assert set(res.exio_quantity_and_price["exio_sector"]) == set(exio_sectors)
    assert (res.exio_quantity_and_price["unit"] == "TJ").all()


def test_write_energy_split_report_xlsx(tmp_path):
    from mrio_pyramids.exiobase3 import ExioRegionSlice
    from split_eurostat_energy_by_exiobase import (
        split_eurostat_energy_sectors_using_exiobase,
        write_energy_split_report_xlsx,
    )

    exio_sectors = ["coke oven products", "refined petroleum products", "electricity", "gas"]
    Z_domestic = pd.DataFrame(
        1.0,
        index=exio_sectors,
        columns=exio_sectors,
    )
    Y_region = pd.DataFrame({"FD": [1.0, 1.0, 1.0, 1.0]}, index=exio_sectors)
    x_region = (Z_domestic.sum(axis=1) + Y_region.sum(axis=1)).astype("float64")

    zp = tmp_path / "IOT_2020_pxp.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        _write_member(zf, "energy/unit.txt", "stressor\tunit\nEnergy use - Gross\tTJ\n")
        _write_member(
            zf,
            "energy/F.txt",
            "\n".join(
                [
                    "region\tAT\tAT\tAT\tAT",
                    "sector\tcoke oven products\trefined petroleum products\telectricity\tgas",
                    "stressor\t\t\t\t",
                    "Energy use - Gross\t1\t1\t1\t1",
                    "",
                ]
            ),
        )

    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=x_region,
        source_zip=zp,
    )

    euro_sectors = ["A", "C19", "D35"]
    euro_Z = pd.DataFrame(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 3.0],
        ],
        index=euro_sectors,
        columns=euro_sectors,
    )
    euro_Y = pd.DataFrame({"FD": [1.0, 1.0, 1.0]}, index=euro_sectors)

    res = split_eurostat_energy_sectors_using_exiobase(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_region="AT",
        exio_slice=exio_slice,
        eurostat_sector_19="C19",
        eurostat_sector_35="D35",
        exio_patterns_19=("coke", "refined"),
        exio_patterns_35=("electricity", "gas"),
        extract_quantity_and_price=True,
        ras_max_iter=5_000,
        ras_tol=1e-12,
    )

    out_xlsx = write_energy_split_report_xlsx(
        tmp_path / "energy_split_report.xlsx",
        result=res,
        aggregates=("C19", "D35"),
    )
    assert out_xlsx.exists()

