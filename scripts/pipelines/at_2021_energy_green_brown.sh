#!/usr/bin/env bash
set -euo pipefail

# Reuse the shared DOM/IMP production-account reconciliation and TOTAL validator.
# The curated concordance retains C20 biofuels and reviewed C20/C21 support.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OUTPUT_SUFFIX="energy_green_brown"
export EXIO_YEAR="${EXIO_YEAR:-2021}"
export CONCORDANCE_CSV="${CONCORDANCE_CSV:-$ROOT_DIR/scripts/mappings/exio_to_eurostat_cpa_products.csv}"
export EXPECTED_REAGG_SECTORS="65"
# ETP comparison diagnostics require combustion CO2, not every extension.
export ENVIRONMENTAL_PRESET="${ENVIRONMENTAL_PRESET:-combustion-co2}"
exec bash "$ROOT_DIR/scripts/pipelines/at_2021_workbook_reaggregation.sh"
