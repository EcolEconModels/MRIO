#!/usr/bin/env bash
set -euo pipefail

# Pipeline: AT 2021 Eurostat pxp IOT -> Energy Balances pxp split -> reaggregation.
#
# This wrapper records the CLI invocation only. It must not generate Energy Balance
# selector quantities itself; the Python split module builds quantities from the
# explicit split recipe and the official Energy Balance dictionary.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected venv python at $PY" >&2
  exit 1
fi

export MRIO_DATA_ROOT="${MRIO_DATA_ROOT:-$ROOT_DIR/data}"

EUROSTAT_GEO="${EUROSTAT_GEO:-AT}"
EUROSTAT_TIME="${EUROSTAT_TIME:-2021}"
BALANCING_SCOPE="${BALANCING_SCOPE:-local}"
OUTDIR="${OUTDIR:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_energy_balance_pxp}"

SPLIT_CSV="${SPLIT_CSV:-$ROOT_DIR/scripts/pipelines/eurostat_energybalances_split/energy_balance_pxp_split.csv}"
REAGG_CSV="${REAGG_CSV:-$ROOT_DIR/scripts/pipelines/eurostat_energybalances_split/energy_balance_pxp_reaggregation.csv}"
TECH_CSV="${TECH_CSV:-$ROOT_DIR/scripts/pipelines/eurostat_energybalances_split/energy_balance_pxp_input_technology.csv}"
PRICES_CSV="${PRICES_CSV:-$ROOT_DIR/scripts/pipelines/eurostat_energybalances_split/energy_balance_pxp_prices.csv}"
EB_DICT_CSV="${EB_DICT_CSV:-$ROOT_DIR/scripts/mappings/eurostat_energy_balances_dictionary.csv}"

set -x
"$PY" "$ROOT_DIR/scripts/split_eurostat_iot_by_energy_balances.py" \
  --geo "$EUROSTAT_GEO" \
  --year "$EUROSTAT_TIME" \
  --eurostat-dataset naio_10_cp1700 \
  --eurostat-row-dim prd_ava \
  --eurostat-col-dim prd_use \
  --eurostat-stk-flow-dim stk_flow \
  --eurostat-stk-flows '["DOM","IMP","TOTAL"]' \
  --split-csv "$SPLIT_CSV" \
  --reaggregation-csv "$REAGG_CSV" \
  --input-technology-csv "$TECH_CSV" \
  --prices-csv "$PRICES_CSV" \
  --energy-balance-dictionary-csv "$EB_DICT_CSV" \
  --outdir "$OUTDIR" \
  --balancing-scope "$BALANCING_SCOPE" \
  --write-reaggregated \
  --verbose
