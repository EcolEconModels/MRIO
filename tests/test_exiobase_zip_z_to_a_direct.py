from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from ecolecon_mrio.data import load_mrio_matrices


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
    return zip_path


def test_load_mrio_matrices_derives_sparse_A_from_Z_without_A_txt(tmp_path):
    year = 2099
    _write_tiny_exio_zip(tmp_path, year)

    A, x, y, basis_index, metadata = load_mrio_matrices(
        year,
        data_root=tmp_path,
        allow_download_missing=False,
        cache_matrices=False,
        read_existing_cache=False,
    )

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
    assert metadata["matrix_source"] == "Z.txt/x.txt"
    assert "A.txt" not in " ".join(map(str, metadata.values()))


def test_delete_source_zip_after_parse_is_rejected_for_zip_stream(tmp_path):
    year = 2098
    _write_tiny_exio_zip(tmp_path, year)
    try:
        load_mrio_matrices(
            year,
            data_root=tmp_path,
            allow_download_missing=False,
            delete_source_zip_after_parse=True,
        )
    except ValueError as exc:
        assert "source zip must be retained" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("delete_source_zip_after_parse=True should be rejected")
