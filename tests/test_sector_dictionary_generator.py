from __future__ import annotations

from pathlib import Path

import pandas as pd


def _metadata(row_dim: str, col_dim: str, labels: dict[str, str]):
    from pathlib import Path

    from ecolecon_mrio.eurostat_iot import EurostatMetadata

    order = list(labels)
    return EurostatMetadata(
        dataset="sample",
        filters={},
        request_url="https://example.invalid",
        raw_path=Path("raw.json"),
        dimensions={row_dim: labels, col_dim: labels},
        dimension_order={row_dim: order, col_dim: order},
    )


def test_eurostat_metadata_merge_preserves_existing_codes() -> None:
    from ecolecon_mrio.sector_dictionary_generator import (
        DICTIONARY_COLUMNS,
        _eurostat_rows_from_dimensions,
        _merge_metadata_with_existing,
    )

    metadata = _metadata("prd_ava", "prd_use", {"CPA_A01": "Official agriculture", "CPA_U": "Official U"})
    existing = pd.DataFrame(
        [
            {
                "code": "A01",
                "label": "Old A01",
                "description": "Existing detailed description",
                "provenance_source": "existing",
            },
            {
                "code": "01.11",
                "label": "Detailed current code",
                "description": "Detailed current description",
                "provenance_source": "existing",
            },
        ],
        columns=list(DICTIONARY_COLUMNS),
    )

    rows = _eurostat_rows_from_dimensions(
        metadata,
        dimensions=("prd_ava", "prd_use"),
        provenance_prefix="Eurostat SDMX metadata:sample",
        code_transform="strip_cpa_prefix",
        code_kind="cpa",
    )
    out = _merge_metadata_with_existing(rows, existing, output_columns=DICTIONARY_COLUMNS)

    assert set(out["code"]) == {"A01", "U", "01.11"}
    assert out.loc[out["code"].eq("A01"), "label"].item() == "Official agriculture"
    assert out.loc[out["code"].eq("A01"), "description"].item() == "Existing detailed description"
    retained = out.loc[out["code"].eq("01.11"), "provenance_source"].item()
    assert "retained_existing_dictionary_row_not_in_current_metadata" in retained


def test_eurostat_metadata_extraction_excludes_accounting_rows() -> None:
    from ecolecon_mrio.sector_dictionary_generator import _eurostat_rows_from_dimensions

    product_metadata = _metadata("prd_ava", "prd_use", {"CPA_A01": "Official agriculture", "B1G": "Value added"})
    industry_metadata = _metadata("ind_ava", "ind_use", {"A01": "Official agriculture", "B1G": "Value added"})

    product_rows = _eurostat_rows_from_dimensions(
        product_metadata,
        dimensions=("prd_ava", "prd_use"),
        provenance_prefix="Eurostat SDMX metadata:sample",
        code_transform="strip_cpa_prefix",
        code_kind="cpa",
    )
    industry_rows = _eurostat_rows_from_dimensions(
        industry_metadata,
        dimensions=("ind_ava", "ind_use"),
        provenance_prefix="Eurostat SDMX metadata:sample",
        code_transform="none",
        code_kind="nace",
    )

    assert [row["code"] for row in product_rows] == ["A01"]
    assert [row["code"] for row in industry_rows] == ["A01"]


def test_industry_metadata_extraction_keeps_iot_aggregates_but_excludes_d11() -> None:
    from ecolecon_mrio.sector_dictionary_generator import _eurostat_rows_from_dimensions

    metadata = _metadata(
        "ind_ava",
        "ind_use",
        {
            "B": "Mining and quarrying",
            "D": "Electricity, gas, steam and air conditioning supply",
            "D11": "Wages and salaries",
        },
    )

    rows = _eurostat_rows_from_dimensions(
        metadata,
        dimensions=("ind_ava", "ind_use"),
        provenance_prefix="Eurostat SDMX metadata:sample",
        code_transform="none",
        code_kind="nace",
    )

    assert [row["code"] for row in rows] == ["B", "D"]


def test_cpa65_dictionary_uses_metadata_labels_and_keeps_split() -> None:
    from ecolecon_mrio.sector_dictionary_generator import _build_cpa_target_dictionary

    metadata = _metadata(
        "prd_ava",
        "prd_use",
        {
            "CPA_A01": "Official A01",
            "CPA_L68A": "Official imputed rents",
            "CPA_L68B": "Official real estate excluding imputed rents",
        },
    )
    existing = pd.DataFrame(
        [
            {"code": "CPA_A01", "label": "Old A01", "description": "A*64 A01; NACE/CPA division 01."},
            {"code": "CPA_L68A", "label": "Old L68A", "description": "old"},
            {"code": "CPA_L68B", "label": "Old L68B", "description": "old"},
        ]
    )

    out = _build_cpa_target_dictionary(metadata, existing=existing, split_real_estate=True)

    assert set(out["code"]) == {"CPA_A01", "CPA_L68A", "CPA_L68B"}
    assert out.loc[out["code"].eq("CPA_A01"), "label"].item() == "Official A01"
    assert out.loc[out["code"].eq("CPA_L68A"), "label"].item() == "Official imputed rents"
    assert "NACE/CPA division 01" in out.loc[out["code"].eq("CPA_A01"), "description"].item()


def test_cpa_b_description_uses_mining_division_range() -> None:
    from ecolecon_mrio.sector_dictionary_generator import _description_from_cpa_code

    assert _description_from_cpa_code("CPA_B") == "Eurostat IOT target B; NACE/CPA division 05-09."


def test_write_sector_dictionaries_with_synthetic_metadata(tmp_path: Path, monkeypatch) -> None:
    import ecolecon_mrio.sector_dictionary_generator as generator

    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    pd.DataFrame(
        [{"code": "A01", "label": "Old product", "description": "Old product", "provenance_source": "existing"}]
    ).to_csv(existing_dir / generator.EUROSTAT_PRODUCTS_OUTPUT_NAME, index=False)
    pd.DataFrame(
        [{"code": "A01", "label": "Old industry", "description": "Old industry", "provenance_source": "existing"}]
    ).to_csv(existing_dir / generator.EUROSTAT_INDUSTRIES_OUTPUT_NAME, index=False)
    pd.DataFrame(
        [{"code": "CPA_A01", "label": "Old CPA", "description": "A*64 A01; NACE/CPA division 01."}]
    ).to_csv(existing_dir / generator.CPA64_OUTPUT_NAME, index=False)
    pd.DataFrame(
        [
            {"code": "CPA_A01", "label": "Old CPA", "description": "A*64 A01; NACE/CPA division 01."},
            {"code": "CPA_L68A", "label": "Old L68A", "description": "old"},
            {"code": "CPA_L68B", "label": "Old L68B", "description": "old"},
        ]
    ).to_csv(existing_dir / generator.CPA65_OUTPUT_NAME, index=False)
    pd.DataFrame(
        [{"code": "p01.a", "label": "Old product", "description": "Old product", "provenance_source": "existing"}]
    ).to_csv(existing_dir / generator.EXIO_PRODUCTS_OUTPUT_NAME, index=False)
    pd.DataFrame(
        [
            {"code": "i01.a", "label": "Old industry", "description": "Old industry", "provenance_source": "existing"},
            {
                "code": "i40.2.a",
                "label": "Legacy split",
                "description": "Legacy split",
                "provenance_source": "existing",
            },
        ]
    ).to_csv(existing_dir / generator.EXIO_INDUSTRIES_OUTPUT_NAME, index=False)

    def fake_load_metadata(dataset: str, **_kwargs):
        if dataset == generator.EUROSTAT_PRODUCT_DATASET:
            return (
                _metadata(
                    "prd_ava",
                    "prd_use",
                    {"CPA_A01": "Official product", "CPA_L68A": "L68A", "CPA_L68B": "L68B"},
                ),
                {"dataset": dataset},
            )
        return (_metadata("ind_ava", "ind_use", {"A01": "Official industry"}), {"dataset": dataset})

    def fake_exio_dictionary(*, system: str, existing: pd.DataFrame, exio_year: int | None = None) -> pd.DataFrame:
        assert exio_year is None
        code = "p01.a" if system == "pxp" else "i01.a"
        rows = [{"code": code, "label": f"Official {system}", "description": f"Official {system}", "provenance_source": "test"}]
        return generator._merge_metadata_with_existing(rows, existing, output_columns=generator.DICTIONARY_COLUMNS)

    monkeypatch.setattr(generator, "_load_metadata", fake_load_metadata)
    monkeypatch.setattr(generator, "_build_exio_dictionary", fake_exio_dictionary)

    out_dir = tmp_path / "out"
    result = generator.write_sector_dictionaries(out_dir=out_dir, existing_dir=existing_dir, eurostat_storage=tmp_path)

    assert (out_dir / generator.MANIFEST_OUTPUT_NAME).exists()
    assert set(result.frames[generator.EXIO_INDUSTRIES_OUTPUT_NAME]["code"]) == {"i01.a", "i40.2.a"}
    assert result.frames[generator.CPA65_OUTPUT_NAME].loc[
        result.frames[generator.CPA65_OUTPUT_NAME]["code"].eq("CPA_A01"), "label"
    ].item() == "Official product"
