from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import inspect
import pytest

from ecolecon_mrio.data import (
    derive_A_from_Z_and_x,
    derive_nonexport_final_demand_from_Y,
    load_mrio_iot_matrices,
)


def _write_tiny_exio_zip(root: Path, year: int) -> Path:
    data_dir = root / f"exiobase3_{year}"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"IOT_{year}_pxp.zip"
    prefix = "nested_source_dir/"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file_parameters.json", "{}")
        zf.writestr(
            prefix + "x.txt",
            "region\tsector\tindout\n"
            "R1\ts1\t10\n"
            "R1\ts2\t20\n",
        )
        zf.writestr(
            prefix + "Z.txt",
            "\t\tR1\tR1\n"
            "\t\ts1\ts2\n"
            "unit\tunit\tEUR\tEUR\n"
            "R1\ts1\t1\t4\n"
            "R1\ts2\t2\t8\n",
        )
        zf.writestr(
            prefix + "Y.txt",
            "\t\tR1\tR1\n"
            "\t\tFinal consumption\tExport\n"
            "unit\tunit\tEUR\tEUR\n"
            "R1\ts1\t5\t100\n"
            "R1\ts2\t6\t200\n",
        )
    (data_dir / "download_log.json").write_text(
        '{"doi": "10.5281/zenodo.20051562"}', encoding="utf-8"
    )
    return zip_path


def test_load_mrio_iot_matrices_derives_sparse_A_from_Z_without_A_txt(tmp_path):
    year = 2099
    _write_tiny_exio_zip(tmp_path, year)

    matrices, basis_index, metadata = load_mrio_iot_matrices(
        "exiobase",
        year,
        data_root=tmp_path,
        matrices=("Z", "x", "Y"),
        allow_download_missing=False,
    )
    Z = matrices["Z"]
    x = matrices["x"]
    Y = matrices["Y"]
    A = derive_A_from_Z_and_x(Z, x, basis_index=basis_index)
    y = derive_nonexport_final_demand_from_Y(Y)

    assert list(basis_index.tolist()) == [("R1", "s1"), ("R1", "s2")]
    np.testing.assert_allclose(x.to_numpy(dtype=float), [10.0, 20.0])
    np.testing.assert_allclose(y.to_numpy(dtype=float), [5.0, 6.0])
    np.testing.assert_allclose(
        A.toarray(),
        np.array(
            [
                [1.0 / 10.0, 4.0 / 20.0],
                [2.0 / 10.0, 8.0 / 20.0],
            ]
        ),
    )
    assert metadata["requested_matrices"] == ["Z", "x", "Y"]
    assert "A" not in matrices


def test_iot_loader_does_not_accept_delete_source_zip_compatibility_kwarg(tmp_path):
    assert "delete_source_zip_after_parse" not in inspect.signature(
        load_mrio_iot_matrices
    ).parameters
    with pytest.raises(TypeError, match="delete_source_zip_after_parse"):
        load_mrio_iot_matrices(
            "exiobase", 2098, matrices=("x",),
            data_root=tmp_path,
            allow_download_missing=False,
            delete_source_zip_after_parse=True,
        )
