from __future__ import annotations

import pandas as pd
import pytest


def test_validate_mapping_weights_passes():
    from ecolecon_mrio.weighted_mapping import validate_mapping_weights

    mapping = {
        "CPA_A": {"S1": 0.4, "S2": 0.6},
        "CPA_B": {"S1": 0.6, "S2": 0.4},
        "CPA_C": {"S3": 1.0},
    }
    validate_mapping_weights(mapping)


def test_validate_mapping_weights_fails_on_sum():
    from ecolecon_mrio.weighted_mapping import validate_mapping_weights

    mapping = {
        "CPA_A": {"S1": 0.7},
        "CPA_B": {"S1": 0.2},
    }
    with pytest.raises(ValueError):
        validate_mapping_weights(mapping)


def test_read_weighted_mapping_csv(tmp_path):
    from ecolecon_mrio.weighted_mapping import read_weighted_mapping_csv

    csv_path = tmp_path / "mapping.csv"
    df = pd.DataFrame(
        {
            "aggregate_sector": ["CPA_A", "CPA_B", "CPA_A", "CPA_B"],
            "exio_sector": ["S1", "S1", "S2", "S2"],
            "weight": [0.4, 0.6, 0.6, 0.4],
            "enabled": ["1", "1", "1", "1"],
        }
    )
    df.to_csv(csv_path, index=False)

    mapping = read_weighted_mapping_csv(csv_path)
    assert mapping["CPA_A"]["S1"] == 0.4
    assert mapping["CPA_B"]["S1"] == 0.6


def test_read_weighted_mapping_csv_requires_weight(tmp_path):
    from ecolecon_mrio.weighted_mapping import read_weighted_mapping_csv

    csv_path = tmp_path / "mapping.csv"
    df = pd.DataFrame(
        {
            "aggregate_sector": ["CPA_A"],
            "exio_sector": ["S1"],
        }
    )
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        read_weighted_mapping_csv(csv_path)
