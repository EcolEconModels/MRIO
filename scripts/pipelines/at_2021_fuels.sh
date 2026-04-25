#!/usr/bin/env bash
set -euo pipefail

# Pipeline wrapper: default to the new Step 3 workbook sheet and the "fuels"
# output suffix.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export WORKBOOK_XLSX="${WORKBOOK_XLSX:-$ROOT_DIR/tmp/IO_Exiobase_Sector_Disaggregation.xlsx}"
export WORKBOOK_SHEET="${WORKBOOK_SHEET:-Step3_Reaggregation}"
export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-fuels}"

exec "$ROOT_DIR/scripts/pipelines/at_2021_workbook_reaggregation.sh"