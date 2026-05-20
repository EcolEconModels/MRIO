from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _dictionary_rows() -> list[dict[str, object]]:
    return [
        {
            "code": "SRC_A",
            "label": "Source A",
            "description": "Source dictionary",
        },
        {
            "code": "SPL_1",
            "label": "Split 1",
            "description": "Source dictionary",
        },
        {
            "code": "TGT_X",
            "label": "Target X",
            "description": "Target dictionary",
        },
    ]


def _mapping_rows() -> list[dict[str, object]]:
    return [
        {
            "source_code": "SRC_A",
            "source_label": "Source A",
            "target_code": "SPL_1",
            "target_label": "Split 1",
            "order": 1,
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


def test_split_mapping_rejects_unknown_target_code(tmp_path: Path) -> None:
    from ecolecon_mrio.canonical_mapping_csv import load_canonical_mapping_frame

    dictionary_path = tmp_path / "dictionary.csv"
    pd.DataFrame(_dictionary_rows()).to_csv(dictionary_path, index=False)
    target_dictionary_path = tmp_path / "target_dictionary.csv"
    pd.DataFrame([{"code": "TGT_X", "label": "Target X", "description": "Target dictionary"}]).to_csv(
        target_dictionary_path,
        index=False,
    )
    mapping_path = tmp_path / "mapping.csv"
    rows = _mapping_rows()
    rows[0]["target_code"] = "UNKNOWN"
    rows[0]["target_label"] = "Unknown"
    pd.DataFrame(rows).to_csv(mapping_path, index=False)

    with pytest.raises(ValueError, match="unknown target code"):
        load_canonical_mapping_frame(
            mapping_path,
            dictionary_path,
            target_dictionary_path=target_dictionary_path,
        )


def test_committed_fuels_labels_do_not_stack_prefixes() -> None:
    from pathlib import Path

    from ecolecon_mrio.canonical_mapping_csv import load_canonical_mapping_frame
    from ecolecon_mrio.mapping_csv import load_split_manifest_frame

    eurostat_dictionary_path = Path("scripts/mappings/eurostat_cpa65_products_dictionary.csv")
    split_paths = [
        Path("scripts/mappings/fuels_split.csv"),
        Path("scripts/mappings/fuels_more_split.csv"),
        Path("scripts/mappings/energy_green_brown_split.csv"),
    ]
    reaggregation_paths = [
        Path("scripts/mappings/fuels_reaggregation.csv"),
        Path("scripts/mappings/fuels_more_reaggregation.csv"),
        Path("scripts/mappings/energy_green_brown_reaggregation.csv"),
    ]

    for mapping_path in split_paths:
        frame = load_split_manifest_frame(mapping_path, eurostat_dictionary_path)
        assert frame["source_label"].map(lambda value: str(value).count("::") == 0).all()

    for mapping_path in reaggregation_paths:
        frame = pd.read_csv(mapping_path, dtype=str).fillna("")
        assert frame["source_label"].map(lambda value: str(value).count("::") <= 1).all()
        assert frame["target_label"].map(lambda value: str(value).count("::") <= 1).all()


def test_validate_canonical_chain_rejects_missing_coverage(tmp_path: Path) -> None:
    from ecolecon_mrio.canonical_mapping_csv import validate_canonical_chain

    dictionary_path = tmp_path / "dictionary.csv"
    dictionary_rows = _dictionary_rows() + [
        {
            "code": "OTHER",
            "label": "Other",
            "description": "Target dictionary",
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
            "source_code": "OTHER",
            "source_label": "Other",
            "target_code": "TGT_X",
            "target_label": "Target X",
            "order": 1,
            "enabled": True,
            "notes": "ok",
        }
    ]
    reagg_path = tmp_path / "reagg.csv"
    pd.DataFrame(reaggregation_rows).to_csv(reagg_path, index=False)

    with pytest.raises(ValueError, match="missing from reaggregation"):
        validate_canonical_chain(split_path, reagg_path, dictionary_path)


def test_validate_canonical_chain_checks_reaggregation_source_dictionary(tmp_path: Path) -> None:
    from ecolecon_mrio.canonical_mapping_csv import validate_canonical_chain

    split_dictionary_path = tmp_path / "split_dictionary.csv"
    pd.DataFrame(_dictionary_rows()).to_csv(split_dictionary_path, index=False)

    reaggregation_dictionary_path = tmp_path / "reaggregation_dictionary.csv"
    pd.DataFrame(
        [
            {"code": "SRC_A", "label": "Source A", "description": "Source dictionary"},
            {"code": "SPL_1", "label": "Split 1", "description": "Source dictionary"},
        ]
    ).to_csv(reaggregation_dictionary_path, index=False)

    split_path = tmp_path / "split.csv"
    pd.DataFrame(_mapping_rows()).to_csv(split_path, index=False)

    reagg_path = tmp_path / "reagg.csv"
    pd.DataFrame(
        [
            {
                "source_code": "p23.2",
                "source_label": "Refined Petroleum",
                "target_code": "TGT_X",
                "target_label": "Target X",
                "order": 1,
                "enabled": True,
                "notes": "stale source code",
            }
        ]
    ).to_csv(reagg_path, index=False)

    with pytest.raises(ValueError, match="unknown source code"):
        validate_canonical_chain(
            split_path,
            reagg_path,
            split_dictionary_path,
            reaggregation_dictionary_path,
        )