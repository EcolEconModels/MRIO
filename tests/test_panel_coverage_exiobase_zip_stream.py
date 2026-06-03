from __future__ import annotations

import pandas as pd
from scipy.sparse import csr_matrix


def test_attempt_mrio_year_uses_zip_stream_matrix_loader_for_exiobase(monkeypatch, tmp_path):
    import ecolecon_mrio.panel_coverage as coverage

    idx = pd.MultiIndex.from_tuples([("AT", "a"), ("AT", "b")], names=["region", "sector"])
    A = csr_matrix([[0.1, 0.0], [0.2, 0.1]], dtype=float)
    x = pd.Series([10.0, 20.0], index=idx, name="x")
    y = pd.Series([7.0, 8.0], index=idx, name="y")
    calls = []

    def fake_matrices(year: int, **kwargs):
        calls.append((int(year), dict(kwargs)))
        return A, x, y, idx, {"cache_kind": "exiobase_zip_stream", "source_zip": "fake.zip"}

    monkeypatch.setattr(coverage, "load_mrio_matrices", fake_matrices)
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
    assert calls and calls[0][0] == 2012
    assert calls[0][1]["allow_download_missing"] is False
