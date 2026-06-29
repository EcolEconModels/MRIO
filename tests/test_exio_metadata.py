from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


def test_load_exiobase_metadata_uses_packaged_codes_without_zip() -> None:
    from ecolecon_mrio.exio_concordance import load_exiobase_metadata

    meta = load_exiobase_metadata(system="pxp")

    assert meta.system == "pxp"
    assert meta.source_zip is None
    assert {"ExioName", "ExioCode", "ExioLabel"}.issubset(meta.sectors.columns)
    assert "p01.a" in set(meta.sectors["ExioCode"])


def test_load_exiobase_metadata_checks_source_zip_names(tmp_path: Path) -> None:
    from ecolecon_mrio.exio_concordance import load_exiobase_metadata

    packaged = load_exiobase_metadata(system="pxp").sectors
    names = packaged["ExioName"].astype(str).tolist()
    rows = [{"region": "AT", "sector": name, "indout": "0"} for name in names]
    zip_path = tmp_path / "IOT_2020_pxp.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("x.txt", pd.DataFrame(rows).to_csv(sep="\t", index=False))
        zf.writestr("metadata.json", json.dumps({"version": "test"}))

    meta = load_exiobase_metadata(system="pxp", source_zip=zip_path)

    assert meta.source_zip == zip_path
    assert meta.version == "test"
    assert meta.zip_sector_names[:3] == tuple(names[:3])
