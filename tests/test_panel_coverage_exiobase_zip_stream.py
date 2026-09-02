from __future__ import annotations

import pandas as pd
from scipy.sparse import csr_matrix


def test_attempt_mrio_year_uses_zip_stream_matrix_loader_for_exiobase(monkeypatch, tmp_path):
    import ecolecon_mrio.panel_coverage as coverage

    idx = pd.MultiIndex.from_tuples([("AT", "a"), ("AT", "b")], names=["region", "sector"])
    Z = csr_matrix([[1.0, 0.0], [2.0, 2.0]], dtype=float)
    x = pd.Series([10.0, 20.0], index=idx, name="x")
    Y_columns = pd.MultiIndex.from_tuples(
        [("AT", "Final consumption expenditure by households")],
        names=["region", "category"],
    )
    Y = pd.DataFrame([[7.0], [8.0]], index=idx, columns=Y_columns)
    calls = []

    def fake_matrices(dataset, year, *, matrices, **kwargs):
        calls.append((str(dataset), int(year), tuple(matrices), dict(kwargs)))
        return {"Z": Z, "x": x, "Y": Y}, idx, {
            "cache_kind": "exiobase_zip_stream_iot",
            "source_zip": "fake.zip",
        }

    monkeypatch.setattr(coverage, "load_mrio_iot_matrices", fake_matrices)
    monkeypatch.setattr(
        coverage,
        "load_mrio",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_mrio must not be used for EXIOBASE coverage")),
    )

    rec = coverage.attempt_mrio_year(
        "exiobase",
        2012,
        data_root=str(tmp_path),
        calc_all=True,
        include_ghosh=True,
        allow_download_missing=False,
        verbose=False,
    )

    assert rec.load_status == "ok"
    assert rec.metadata["mrio_type"] == "SimpleNamespace"
    assert rec.metadata["has_A"] is True
    assert rec.metadata["has_x"] is True
    assert calls and calls[0][0:3] == ("exiobase", 2012, ("Z", "x", "Y"))
    assert calls[0][3]["allow_download_missing"] is False
