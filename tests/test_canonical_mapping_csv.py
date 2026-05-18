from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _dictionary_rows() -> list[dict[str, object]]:
    return [
        {
            "code": "SRC_A",
            "label": "Source A",
            "parent_code": "",
            "can_be_source": True,
            "can_be_target": False,
            "can_be_split": False,
            "active": True,
        },
        {
            "code": "SPL_1",
            "label": "Split 1",
            "parent_code": "",
            "can_be_source": True,
            "can_be_target": True,
            "can_be_split": True,
            "active": True,
        },
        {
            "code": "TGT_X",
            "label": "Target X",
            "parent_code": "",
            "can_be_source": False,
            "can_be_target": True,
            "can_be_split": False,
            "active": True,
        },
    ]


def _mapping_rows() -> list[dict[str, object]]:
    return [
        {
            "mapping_set": "demo",
            "source_code": "SRC_A",
            "source_label": "Source A",
            "target_code": "SPL_1",
            "target_label": "Split 1",
            "subtarget_code": "",
            "subtarget_label": "",
            "weight": 1.0,
            "enabled": True,
            "notes": "ok",
        }
    ]


def test_load_sector_dictionary_and_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.canonical_mapping_csv import load_canonical_mapping_frame, load_sector_dictionary

    dictionary_path = tmp_path / "dictionary.csv"
    pd.DataFrame(_dictionary_rows()).to_csv(dictionary_path, index=False)
    mapping_path = tmp_path / "mapping.csv"
    pd.DataFrame(_mapping_rows()).to_csv(mapping_path, index=False)

    dictionary = load_sector_dictionary(dictionary_path)
    frame = load_canonical_mapping_frame(mapping_path, dictionary_path)

    assert set(dictionary["code"]) == {"SRC_A", "SPL_1", "TGT_X"}
    assert list(frame["source_code"]) == ["SRC_A"]
    assert list(frame["target_code"]) == ["SPL_1"]


def test_canonical_mapping_rejects_label_mismatch(tmp_path: Path) -> None:
    from ecolecon_mrio.canonical_mapping_csv import load_canonical_mapping_frame

    dictionary_path = tmp_path / "dictionary.csv"
    pd.DataFrame(_dictionary_rows()).to_csv(dictionary_path, index=False)
    mapping_path = tmp_path / "mapping.csv"
    rows = _mapping_rows()
    rows[0]["target_label"] = "Wrong"
    pd.DataFrame(rows).to_csv(mapping_path, index=False)

    with pytest.raises(ValueError, match="label mismatch"):
        load_canonical_mapping_frame(mapping_path, dictionary_path)


def test_canonical_mapping_rejects_repeated_code_prefix(tmp_path: Path) -> None:
    from ecolecon_mrio.canonical_mapping_csv import load_canonical_mapping_frame

    dictionary_path = tmp_path / "dictionary.csv"
    pd.DataFrame(_dictionary_rows()).to_csv(dictionary_path, index=False)
    mapping_path = tmp_path / "mapping.csv"
    rows = _mapping_rows()
    rows[0]["source_label"] = "SRC_A :: Source A :: Source A"
    rows[0]["target_label"] = "SPL_1 :: Split 1 :: Split 1"
    pd.DataFrame(rows).to_csv(mapping_path, index=False)

    with pytest.raises(ValueError, match="label mismatch"):
        load_canonical_mapping_frame(mapping_path, dictionary_path)


def test_committed_fuels_labels_do_not_stack_prefixes() -> None:
    from pathlib import Path

    from ecolecon_mrio.canonical_mapping_csv import load_canonical_mapping_frame

    dictionary_path = Path("scripts/mappings/sector_dictionary.csv")
    mapping_paths = [
        Path("scripts/mappings/fuels_split_canonical.csv"),
        Path("scripts/mappings/fuels_identity_canonical.csv"),
        Path("scripts/mappings/fuels_reaggregation_canonical.csv"),
        Path("scripts/mappings/fuels_more_split_canonical.csv"),
        Path("scripts/mappings/fuels_more_identity_canonical.csv"),
        Path("scripts/mappings/fuels_more_reaggregation_canonical.csv"),
    ]

    for mapping_path in mapping_paths:
        frame = load_canonical_mapping_frame(mapping_path, dictionary_path)
        assert frame["source_label"].map(lambda value: str(value).count("::") <= 1).all()
        assert frame["target_label"].map(lambda value: str(value).count("::") <= 1).all()


def test_validate_canonical_chain_rejects_missing_coverage(tmp_path: Path) -> None:
    from ecolecon_mrio.canonical_mapping_csv import validate_canonical_chain

    dictionary_path = tmp_path / "dictionary.csv"
    dictionary_rows = _dictionary_rows() + [
        {
            "code": "OTHER",
            "label": "Other",
            "parent_code": "",
            "can_be_source": True,
            "can_be_target": True,
            "can_be_split": False,
            "active": True,
        }
    ]
    pd.DataFrame(dictionary_rows).to_csv(dictionary_path, index=False)

    split_rows = _mapping_rows()
    split_rows[0]["target_code"] = "SPL_1"
    split_rows[0]["target_label"] = "Split 1"
    split_path = tmp_path / "split.csv"
    pd.DataFrame(split_rows).to_csv(split_path, index=False)

    reaggregation_rows = [
        {
            "mapping_set": "demo",
            "source_code": "OTHER",
            "source_label": "Other",
            "target_code": "TGT_X",
            "target_label": "Target X",
            "subtarget_code": "",
            "subtarget_label": "",
            "weight": 1.0,
            "enabled": True,
            "notes": "ok",
        }
    ]
    reagg_path = tmp_path / "reagg.csv"
    pd.DataFrame(reaggregation_rows).to_csv(reagg_path, index=False)

    with pytest.raises(ValueError, match="missing from reaggregation"):
        validate_canonical_chain(split_path, reagg_path, dictionary_path)