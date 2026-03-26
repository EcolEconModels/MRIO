from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _canonical_base(kind: str) -> dict[str, str]:
    return {
        "kind": kind,
        "source_dataset": "SRC",
        "source_system": "pxp",
        "source_version": "1",
        "source_level": "product",
        "target_dataset": "TGT",
        "target_system": "pxp",
        "target_version": "1",
        "target_level": "product",
        "source_key": "",
        "source_code": "",
        "source_label": "",
        "target_key": "",
        "target_code": "",
        "target_label": "",
        "subtarget_key": "",
        "weight": "",
        "share": "",
        "enabled": "1",
        "provenance": "methodological",
        "certainty": "derived",
        "transformation_rule": "test",
        "source_name": "sample.csv",
        "official_source": "",
        "notes": "",
    }


def test_validate_mapping_weights_passes() -> None:
    from ecolecon_mrio.mapping_csv import validate_mapping_weights

    mapping = {
        "CPA_A": {"S1": 0.4, "S2": 0.6},
        "CPA_B": {"S1": 0.6, "S2": 0.4},
        "CPA_C": {"S3": 1.0},
    }
    validate_mapping_weights(mapping)


def test_validate_mapping_weights_fails_on_sum() -> None:
    from ecolecon_mrio.mapping_csv import validate_mapping_weights

    mapping = {"CPA_A": {"S1": 0.7}, "CPA_B": {"S1": 0.2}}
    with pytest.raises(ValueError):
        validate_mapping_weights(mapping)


def test_read_weighted_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_weighted_mapping_frame, read_weighted_mapping

    rows = []
    for target, source, weight in [("CPA_A", "S1", "0.4"), ("CPA_B", "S1", "0.6"), ("CPA_A", "S2", "0.6"), ("CPA_B", "S2", "0.4")]:
        row = _canonical_base("weighted")
        row["target_key"] = target
        row["source_key"] = source
        row["weight"] = weight
        rows.append(row)
    csv_path = tmp_path / "weighted.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    frame = load_weighted_mapping_frame(csv_path)
    mapping = read_weighted_mapping(csv_path)
    assert set(frame["aggregate_sector"]) == {"CPA_A", "CPA_B"}
    assert mapping["CPA_A"]["S1"] == 0.4
    assert mapping["CPA_B"]["S2"] == 0.4


def test_read_split_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_split_mapping_frame, read_split_mapping

    rows = []
    for source, subtarget in [("Electricity by coal", "Electricity generation"), ("Electricity by gas", "Electricity generation")]:
        row = _canonical_base("split")
        row["target_key"] = "CPA_D"
        row["source_key"] = source
        row["subtarget_key"] = subtarget
        rows.append(row)
    csv_path = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    frame = load_split_mapping_frame(csv_path)
    mapping = read_split_mapping(csv_path)
    assert set(frame["aggregate_sector"]) == {"CPA_D"}
    assert mapping["CPA_D"]["Electricity generation"] == ["Electricity by coal", "Electricity by gas"]


def test_read_aggregation_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_aggregation_mapping_frame, read_aggregation_mapping

    rows = []
    for source, target in [("A", "X"), ("B", "X")]:
        row = _canonical_base("aggregate")
        row["source_key"] = source
        row["target_key"] = target
        rows.append(row)
    csv_path = tmp_path / "aggregate.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    frame = load_aggregation_mapping_frame(csv_path)
    mapping, returned_frame = read_aggregation_mapping(csv_path)
    assert set(frame["from_sector"]) == {"A", "B"}
    assert mapping == {"A": "X", "B": "X"}
    assert len(returned_frame) == 2


def test_read_category_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_category_mapping_frame

    row = _canonical_base("category-target")
    row["source_key"] = "Industry"
    row["target_key"] = "Chemicals"
    row["share"] = "0.75"
    csv_path = tmp_path / "category.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    frame = load_category_mapping_frame(csv_path)
    assert frame.loc[0, "Category"] == "Industry"
    assert frame.loc[0, "TargetSector"] == "Chemicals"
    assert float(frame.loc[0, "Share_total"]) == 0.75