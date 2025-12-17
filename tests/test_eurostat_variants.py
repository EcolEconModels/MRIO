import pandas as pd


def test_load_eurostat_iot_variants_passes_variant_dim(monkeypatch):
    import mrio_pyramids.eurostat_iot as ei

    calls = []

    def fake_load_eurostat_iot(*, dataset, filters, row_dim, col_dim, storage_folder="data/eurostat", force_download=False, force_reparse=False):
        calls.append((dataset, dict(filters), row_dim, col_dim))
        Z = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
        Y = pd.DataFrame([[2.0]], index=["A"], columns=["FD"])
        return ei.ParsedEurostatIOT(Z=Z, Y=Y, meta={})

    monkeypatch.setattr(ei, "load_eurostat_iot", fake_load_eurostat_iot)

    out = ei.load_eurostat_iot_variants(
        dataset="dummy",
        base_filters={"geo": "AT", "time": "2021"},
        variant_dim="stk_flow",
        variant_values=["DOM", "IMP"],
        row_dim="row",
        col_dim="col",
    )

    assert set(out.keys()) == {"DOM", "IMP"}
    assert calls[0][1]["stk_flow"] == "DOM"
    assert calls[1][1]["stk_flow"] == "IMP"

