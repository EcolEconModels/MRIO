from pathlib import Path


OFFICIAL_DIRECT_CPA65 = (
    "scripts/mappings/generated/official_direct/"
    "exio_to_eurostat_cpa65_products_from_official_direct.csv"
)


def test_at_fuels_runner_preserves_year_and_parent_lineage_contracts() -> None:
    text = Path("scripts/pipelines/at_2021_workbook_reaggregation.sh").read_text(
        encoding="utf-8"
    )

    assert "IOT_Austria_${EUROSTAT_TIME}_Eurostat.xlsx" in text
    assert '"source_parent_code": str(aggregate_code)' in text
    assert '"source_parent_code",' in text
    assert "capital_stock_${EUROSTAT_TIME}.xlsx" in text
    assert "IOT_Austria_2021_Eurostat.xlsx" not in text


def test_at_fuels_runner_exposes_archive_extensions_and_eurostat_only_refresh() -> None:
    """The fuels wrapper must expose archive discovery and Eurostat refresh independently."""
    text = Path("scripts/pipelines/at_2021_workbook_reaggregation.sh").read_text(
        encoding="utf-8"
    )

    assert "ENVIRONMENTAL_ALL_EXTENSIONS" in text
    assert "--environmental-all-extensions" in text
    assert "EUROSTAT_FORCE_DOWNLOAD" in text
    assert "--eurostat-force-download" in text
    assert "--exio-force-download" not in text


def test_fuels_wrapper_defaults_to_official_direct_concordance() -> None:
    """The fuels-family production wrapper must use Method-3 topology."""
    fuels = Path("scripts/pipelines/at_2021_workbook_reaggregation.sh").read_text(
        encoding="utf-8"
    )

    assert OFFICIAL_DIRECT_CPA65 in fuels
    assert "generate_reaggregation_for_concordance.py" in fuels
    assert "scripts/mappings/exio_to_eurostat_cpa_products.csv" not in fuels
