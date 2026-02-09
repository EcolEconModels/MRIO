#!/usr/bin/env bash
set -euo pipefail

# Pipeline: AT 2021 Eurostat Energy Balances -> primary sources -> 65 sectors (+ CO2 intensities)
#
# This pipeline reuses the repo's Eurostat SDMX download+cache scaffold:
#   - `ecolecon_mrio.eurostat_iot.download_eurostat_dataset`
# and writes caches under:
#   `data/eurostat/<dataset>/<hash>/...`
#
# Usage:
#   bash scripts/pipelines/split_energy_balances_65NACE2sectors_aggregate_energy_sources.sh
#
# Optional env vars:
#   EUROSTAT_GEO=AT
#   EUROSTAT_TIME=2021
#   OUT_XLSX=outputs/AT_2021_energy_balances_65sectors_primary_sources.xlsx
#   MAP_CSV=scripts/mappings/energy_balance_to_nace2_mapping.csv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected venv python at $PY" >&2
  exit 1
fi

EUROSTAT_GEO="${EUROSTAT_GEO:-AT}"
EUROSTAT_TIME="${EUROSTAT_TIME:-2021}"
OUT_XLSX="${OUT_XLSX:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_energy_balances_65sectors_primary_sources.xlsx}"
MAP_CSV="${MAP_CSV:-$ROOT_DIR/scripts/mappings/energy_balance_to_nace2_mapping.csv}"

cd "$ROOT_DIR"

set -x
"$PY" "$ROOT_DIR/scripts/build_energy_balances_65sectors_primary_sources.py" \
  --geo "$EUROSTAT_GEO" \
  --year "$EUROSTAT_TIME" \
  --mapping "$MAP_CSV" \
  --out "$OUT_XLSX" \
  --storage-folder "$ROOT_DIR/data/eurostat"
