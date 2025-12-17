import pytest


def _mini_payload():
    # Minimal Eurostat-like SDMX-JSON payload for dims: row x col
    return {
        "id": ["row", "col"],
        "size": [2, 3],
        "dimension": {
            "row": {"category": {"index": {"A": 0, "B": 1}, "label": {"A": "A", "B": "B"}}},
            "col": {
                "category": {
                    "index": {"A": 0, "B": 1, "FD": 2},
                    "label": {"A": "A", "B": "B", "FD": "Final demand"},
                }
            },
        },
        # Flattened order: row stride=3, col stride=1
        "value": {"0": 1.0, "1": 2.0, "2": 10.0, "3": 3.0, "4": 4.0, "5": 20.0},
    }


def test_parse_eurostat_iot_splits_Z_and_Y():
    from mrio_pyramids.eurostat_iot import parse_eurostat_iot

    parsed = parse_eurostat_iot(_mini_payload(), row_dim="row", col_dim="col")
    assert list(parsed.Z.index) == ["A", "B"]
    assert list(parsed.Z.columns) == ["A", "B"]
    assert parsed.Z.loc["A", "A"] == 1.0
    assert parsed.Z.loc["A", "B"] == 2.0
    assert parsed.Z.loc["B", "A"] == 3.0
    assert parsed.Z.loc["B", "B"] == 4.0

    assert list(parsed.Y.columns) == ["FD"]
    assert parsed.Y.loc["A", "FD"] == 10.0
    assert parsed.Y.loc["B", "FD"] == 20.0


def test_parse_eurostat_iot_bad_dim_raises_helpful_error():
    from mrio_pyramids.eurostat_iot import EurostatFilterError, parse_eurostat_iot

    with pytest.raises(EurostatFilterError) as e:
        parse_eurostat_iot(_mini_payload(), row_dim="bad", col_dim="col")
    assert "available_dims" in str(e.value)

