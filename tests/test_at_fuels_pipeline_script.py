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

    assert "SFC_LAYOUT_TEMPLATE_XLSX" in text
    assert "POSTGROWTH_SFC_DIR" in text
    assert "$ROOT_DIR/../PostGrowth-SFC-Austria" in text
    assert "$ROOT_DIR/tmp/IOT_Austria_2021_Eurostat.xlsx" not in text
    assert "IOT_Austria_${EUROSTAT_TIME}_Eurostat.xlsx" not in text
    assert "IOT_Austria_2021_Eurostat.xlsx" in text
    assert '"source_parent_code": str(aggregate_code)' in text
    assert '"source_parent_code",' in text
    assert "capital_stock_${EUROSTAT_TIME}.xlsx" in text


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


def test_layout_template_is_reusable_and_group_workbook_export_remains_opt_in() -> None:
    text = Path("scripts/pipelines/at_2021_workbook_reaggregation.sh").read_text(encoding="utf-8")
    assert "SFC_LAYOUT_TEMPLATE_XLSX" in text
    assert 'POSTGROWTH_SFC_DIR="${POSTGROWTH_SFC_DIR:-$ROOT_DIR/../PostGrowth-SFC-Austria}"' in text
    assert 'TEMPLATE_XLSX="$SFC_LAYOUT_TEMPLATE_XLSX"' in text
    assert "Explicit TEMPLATE_XLSX is required" not in text
    assert 'if [[ -z "$REGION_SCOPE_MAPPING_CSV" && "$EUROSTAT_GEO" == "AT" ]]' in text
    assert 'SFC_ARGS=(--write-sfc-workbook --sfc-template-xlsx "$SFC_LAYOUT_TEMPLATE_XLSX")' in text



def test_fuels_wrapper_distinguishes_environmental_run_outputs_and_validates_final_artifact() -> None:
    text = Path("scripts/pipelines/at_2021_workbook_reaggregation.sh").read_text(
        encoding="utf-8"
    )
    assert 'RUN_OUTPUT_SUFFIX="${RUN_OUTPUT_SUFFIX:-}"' in text
    assert 'RUN_OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_all_extensions"' in text
    assert "custom_extensions_${ENVIRONMENTAL_SELECTION_HASH}" in text
    assert '${RUN_OUTPUT_SUFFIX}_split_flows' in text
    assert '${RUN_OUTPUT_SUFFIX}_disaggregated_flows' in text
    assert '${RUN_OUTPUT_SUFFIX}_reaggregated_flows' in text
    assert "validate_many_many_pipeline_outputs.py" in text
    assert 'production_validation.json' in text
    # Environmental selection must not fork the mapping source-of-truth directory.
    assert '${OUTPUT_SUFFIX}_generated_mappings' in text
