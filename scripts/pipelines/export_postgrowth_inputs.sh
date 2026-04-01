#!/usr/bin/env bash
set -euo pipefail

# Export the PostGrowth input files from the MRIO workflow.
#
# Outputs written into POSTGROWTH_DIR:
#   - Energy_Balances_2021_Eurostat_green_brown.xlsx
#   - IOT_AT_2020_TOTAL_energy_green_brown.xlsx
#   - IOT_AT_2021_TOTAL_energy_green_brown.xlsx
#
# The legacy manual workbook Energy_Balances_2021_Eurostat.xlsx is intentionally not touched.
#
# Usage:
#   bash scripts/pipelines/export_postgrowth_inputs.sh
#
# Optional env vars:
#   POSTGROWTH_DIR=/path/to/PostGrowth-SFC-Austria
#   TEMPLATE_XLSX=/path/to/IOT_Austria_2021_Eurostat.xlsx
#   ENERGY_OUT=/path/to/Energy_Balances_2021_Eurostat_green_brown.xlsx
#   EUROSTAT_GEO=AT
#   EXIO_YEAR=2020
#   EXIO_REGION=AT

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POSTGROWTH_DIR="${POSTGROWTH_DIR:-$(cd "$ROOT_DIR/.." && pwd)/PostGrowth-SFC-Austria}"
TEMPLATE_XLSX="${TEMPLATE_XLSX:-$POSTGROWTH_DIR/IOT_Austria_2021_Eurostat.xlsx}"
ENERGY_OUT="${ENERGY_OUT:-$POSTGROWTH_DIR/Energy_Balances_2021_Eurostat_green_brown.xlsx}"
EUROSTAT_GEO="${EUROSTAT_GEO:-AT}"
EXIO_YEAR="${EXIO_YEAR:-2020}"
EXIO_REGION="${EXIO_REGION:-AT}"

if [[ ! -d "$POSTGROWTH_DIR" ]]; then
  echo "ERROR: POSTGROWTH_DIR not found: $POSTGROWTH_DIR" >&2
  exit 1
fi

if [[ ! -f "$TEMPLATE_XLSX" ]]; then
  echo "ERROR: TEMPLATE_XLSX not found: $TEMPLATE_XLSX" >&2
  exit 1
fi

echo "Exporting PostGrowth energy workbook to: $ENERGY_OUT"
export OUT_XLSX="$ENERGY_OUT"
export POSTGROWTH_SHEET_NAME="Energy_1.0"
export SPLIT_ENERGY_GREEN_BROWN="1"
bash "$ROOT_DIR/scripts/pipelines/split_energy_balances_65NACE2sectors_aggregate_energy_sources.sh"

for year in 2020 2021; do
  echo "Exporting IOT TOTAL workbook for year $year"
  export EUROSTAT_TIME="$year"
  export EXIO_YEAR="$EXIO_YEAR"
  export EXIO_REGION="$EXIO_REGION"
  export EUROSTAT_GEO="$EUROSTAT_GEO"
  export OUT_SPLIT="$ROOT_DIR/outputs/${EUROSTAT_GEO}_${year}_energy_split_flows_with_source"
  export OUT_AGG="$ROOT_DIR/outputs/${EUROSTAT_GEO}_${year}_energy_green_brown_flows_with_source"
  export TEMPLATE_XLSX="$TEMPLATE_XLSX"
  export SFC_OUTNAME="IOT_${EUROSTAT_GEO}_${year}_{variant}_energy_green_brown.xlsx"

  bash "$ROOT_DIR/scripts/pipelines/at_2021_energy_green_brown.sh"

  src_xlsx="$OUT_AGG/stk_flow=TOTAL/IOT_${EUROSTAT_GEO}_${year}_TOTAL_energy_green_brown.xlsx"
  dst_xlsx="$POSTGROWTH_DIR/IOT_${EUROSTAT_GEO}_${year}_TOTAL_energy_green_brown.xlsx"
  if [[ ! -f "$src_xlsx" ]]; then
    echo "ERROR: Expected generated workbook not found: $src_xlsx" >&2
    exit 1
  fi
  install -m 0644 "$src_xlsx" "$dst_xlsx"
  echo "Copied: $dst_xlsx"
done