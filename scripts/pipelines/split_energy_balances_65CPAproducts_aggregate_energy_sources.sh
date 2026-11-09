#!/usr/bin/env bash
set -euo pipefail

# Pipeline: AT 2021 Eurostat Energy Balances -> primary sources -> 65-product CPA-based IO targets (+ CO2 intensities)
#
# Usage:
#   bash scripts/pipelines/split_energy_balances_65CPAproducts_aggregate_energy_sources.sh
# The script name is legacy compatibility naming; the operative target classification is the repo's 65-product CPA-based IO target set.
#
# Optional env vars:
#   EUROSTAT_GEO=AT
#   EUROSTAT_TIME=2021
#   OUT_XLSX=outputs/AT_2021_energy_balances_65sectors_primary_sources.xlsx
#   MAP_CSV=scripts/mappings/energy_balance_to_cpa65_products_mapping.csv
# The default mapping file also keeps a legacy name, but it maps activity-defined Eurostat energy-balance rows into the repo's 65-product CPA-based IO rows plus households.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected venv python at $PY" >&2
  exit 1
fi

EUROSTAT_GEO="${EUROSTAT_GEO:-AT}"
EUROSTAT_TIME="${EUROSTAT_TIME:-2021}"
OUT_XLSX="${OUT_XLSX:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_energy_balances_65sectors_primary_sources.xlsx}"
MAP_CSV="${MAP_CSV:-$ROOT_DIR/scripts/mappings/energy_balance_to_cpa65_products_mapping.csv}"
POSTGROWTH_SHEET_NAME="${POSTGROWTH_SHEET_NAME:-Energy_1.0}"
SPLIT_ENERGY_GREEN_BROWN="${SPLIT_ENERGY_GREEN_BROWN:-0}"

cd "$ROOT_DIR"

BUILD_ARGS=()
if [[ "${EUROSTAT_FORCE_DOWNLOAD:-0}" == "1" ]]; then
  BUILD_ARGS+=(--force-download)
fi
if [[ "$SPLIT_ENERGY_GREEN_BROWN" == "1" ]]; then
  BUILD_ARGS+=(--split-energy-green-brown)
fi

set -x
"$PY" "$ROOT_DIR/scripts/build_energy_balances_65sectors_primary_sources.py" \
  --geo "$EUROSTAT_GEO" \
  --year "$EUROSTAT_TIME" \
  --mapping "$MAP_CSV" \
  --postgrowth-sheet-name "$POSTGROWTH_SHEET_NAME" \
  --out "$OUT_XLSX" \
  --storage-folder "$ROOT_DIR/data/eurostat" \
  "${BUILD_ARGS[@]}"
