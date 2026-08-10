from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from ecolecon_mrio import data
from ecolecon_mrio.data import DEFAULT_EXIOBASE3_DOI, MrioMatrixResult


def _write_tiny_exio_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("file_parameters.json", "{}")
        zf.writestr(
            "x.txt",
            "region\tsector\tindout\n"
            "AT\tA\t100\n"
            "AT\tB\t200\n",
        )
        zf.writestr(
            "Z.txt",
            "\t\tAT\tAT\n"
            "\t\tA\tB\n"
            "\t\tunit\tunit\n"
            "AT\tA\t1\t2\n"
            "AT\tB\t3\t4\n",
        )
        zf.writestr(
            "Y.txt",
            "\t\tAT\tAT\n"
            "\t\tHFCE\tGFCF\n"
            "\t\tunit\tunit\n"
            "AT\tA\t10\t1\n"
            "AT\tB\t20\t2\n",
        )




def _sparse_scalar(matrix: csr_matrix, row: int, col: int) -> float:
    """Return one sparse entry as a Python float for typed test assertions."""
    dense = np.asarray(matrix.toarray(), dtype=float)
    return float(dense[row, col])

def _prepare_cache_root(tmp_path: Path) -> Path:
    data_dir = tmp_path / "exiobase3_2020"
    _write_tiny_exio_zip(data_dir / "IOT_2020_pxp.zip")
    (data_dir / "download_log.json").write_text(
        json.dumps({"version": DEFAULT_EXIOBASE3_DOI}) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_matrix_result_can_return_z_and_full_final_demand_matrix(tmp_path: Path) -> None:
    cache_root = _prepare_cache_root(tmp_path)

    result = data.load_mrio_matrix_result(
        2020,
        matrix="Z",
        final_demand="matrix",
        data_root=cache_root,
        allow_download_missing=False,
        doi=DEFAULT_EXIOBASE3_DOI,
    )

    assert isinstance(result, MrioMatrixResult)
    assert result.A is None
    Z_result = result.Z
    assert isinstance(Z_result, csr_matrix)
    assert Z_result.shape == (2, 2)
    assert _sparse_scalar(Z_result, 0, 1) == 2.0
    assert isinstance(result.y, pd.DataFrame)
    assert list(result.y.columns) == [("AT", "HFCE"), ("AT", "GFCF")]
    assert float(result.y.loc[("AT", "A"), ("AT", "HFCE")]) == 10.0
    assert result.metadata["matrix_request"] == "Z"
    assert result.metadata["final_demand"] == "matrix"
    assert result.metadata["has_A"] is False
    assert result.metadata["has_Z"] is True


def test_matrix_result_can_return_a_and_z_without_mutating_z(tmp_path: Path) -> None:
    cache_root = _prepare_cache_root(tmp_path)

    result = data.load_mrio_matrix_result(
        2020,
        matrix="both",
        final_demand="none",
        data_root=cache_root,
        allow_download_missing=False,
        doi=DEFAULT_EXIOBASE3_DOI,
    )

    A_result = result.A
    Z_result = result.Z
    assert isinstance(A_result, csr_matrix)
    assert isinstance(Z_result, csr_matrix)
    assert result.y is None
    assert _sparse_scalar(Z_result, 0, 1) == 2.0
    assert _sparse_scalar(A_result, 0, 1) == pytest.approx(2.0 / 200.0)
    assert result.metadata["matrix_request"] == "both"
    assert result.metadata["final_demand"] == "none"
    assert result.metadata["has_A"] is True
    assert result.metadata["has_Z"] is True


def test_legacy_load_mrio_matrices_wraps_matrix_result(tmp_path: Path) -> None:
    cache_root = _prepare_cache_root(tmp_path)

    A, x, y, basis_index, metadata = data.load_mrio_matrices(
        2020,
        data_root=cache_root,
        allow_download_missing=False,
        doi=DEFAULT_EXIOBASE3_DOI,
    )

    assert isinstance(A, csr_matrix)
    assert list(x.index) == list(basis_index)
    assert isinstance(y, pd.Series)
    assert float(y.loc[("AT", "A")]) == 11.0
    assert metadata["matrix_request"] == "A"
    assert metadata["final_demand"] == "summed"
    assert metadata["has_A"] is True
    assert metadata["has_Z"] is False


def test_matrix_result_can_read_final_demand_without_z(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Y-only matrix requests must not parse the EXIO transaction matrix."""

    cache_root = _prepare_cache_root(tmp_path)

    def _unexpected_z_read(*_args: object, **_kwargs: object) -> csr_matrix:
        raise AssertionError("matrix='none' must not read Z.txt")

    monkeypatch.setattr(data, "_read_exio_Z_sparse_from_zip", _unexpected_z_read)
    result = data.load_mrio_matrix_result(
        2020,
        matrix="none",
        final_demand="matrix",
        data_root=cache_root,
        allow_download_missing=False,
        doi=DEFAULT_EXIOBASE3_DOI,
    )
    assert result.A is None
    assert result.Z is None
    assert isinstance(result.y, pd.DataFrame)
    assert result.metadata["matrix_request"] == "none"
    assert result.metadata["z_nnz"] is None
    assert result.metadata["z_density"] is None


def test_matrix_result_rejects_unsupported_requests(tmp_path: Path) -> None:
    cache_root = _prepare_cache_root(tmp_path)

    with pytest.raises(ValueError, match="matrix"):
        data.load_mrio_matrix_result(
            2020,
            matrix="L",
            data_root=cache_root,
            allow_download_missing=False,
            doi=DEFAULT_EXIOBASE3_DOI,
        )

    with pytest.raises(ValueError, match="final_demand"):
        data.load_mrio_matrix_result(
            2020,
            final_demand="exports_only",
            data_root=cache_root,
            allow_download_missing=False,
            doi=DEFAULT_EXIOBASE3_DOI,
        )
