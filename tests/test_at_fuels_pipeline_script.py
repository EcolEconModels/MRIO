from pathlib import Path


REVIEWED_OFFICIAL_DIRECT_CPA65 = (
    "scripts/mappings/"
    "exio_to_eurostat_cpa65_products_official_direct_reviewed.csv"
)
RAW_OFFICIAL_DIRECT_CPA65 = (
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


def test_fuels_wrapper_defaults_to_reviewed_official_direct_concordance() -> None:
    """Fuels/fuels-more use the reviewed production map; green/brown stays unchanged."""
    fuels = Path("scripts/pipelines/at_2021_workbook_reaggregation.sh").read_text(
        encoding="utf-8"
    )
    green_brown = Path("scripts/pipelines/at_2021_energy_green_brown.sh").read_text(
        encoding="utf-8"
    )

    assert REVIEWED_OFFICIAL_DIRECT_CPA65 in fuels
    assert RAW_OFFICIAL_DIRECT_CPA65 not in fuels
    assert "generate_reaggregation_for_concordance.py" in fuels
    assert "scripts/mappings/exio_to_eurostat_cpa_products.csv" not in fuels
    assert "scripts/mappings/exio_to_eurostat_cpa_products.csv" in green_brown
    assert REVIEWED_OFFICIAL_DIRECT_CPA65 not in green_brown


def test_non_at_workbook_requires_explicit_template() -> None:
    """EU/group workbook export cannot silently reuse the Austrian default template."""
    text = Path("scripts/pipelines/at_2021_workbook_reaggregation.sh").read_text(
        encoding="utf-8"
    )
    assert 'TEMPLATE_XLSX_EXPLICIT="${TEMPLATE_XLSX:-}"' in text
    assert text.count('TEMPLATE_XLSX_EXPLICIT="${TEMPLATE_XLSX:-}"') == 1
    assert '"$EUROSTAT_GEO" != "AT"' in text
    assert "Explicit TEMPLATE_XLSX is required" in text
