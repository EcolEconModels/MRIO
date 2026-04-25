#!/usr/bin/env bash
set -euo pipefail

# Pipeline wrapper: workbook Step 2 -> *_fuels_more outputs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export WORKBOOK_XLSX="${WORKBOOK_XLSX:-$ROOT_DIR/tmp/IO_Exiobase_Sector_Disaggregation.xlsx}"
export WORKBOOK_SHEET="${WORKBOOK_SHEET:-Step2_Reaggregation}"
export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-fuels_more}"

exec "$ROOT_DIR/scripts/pipelines/at_2021_workbook_reaggregation.sh"