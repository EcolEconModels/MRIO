#!/usr/bin/env bash
set -euo pipefail

# Canonical final run for the Eurostat pxp IOT split/reaggregation using
# Energy Balance quantities, prices, official EB metadata dictionary, and the
# committed EB split/reaggregation/technology mapping CSVs.
#
# This mirrors the fuels pipeline structure but uses the Eurostat/EB pxp CLI.
# Outputs are written under outputs/ by default.
#
# Common overrides:
#   EUROSTAT_GEO=AT
#   EUROSTAT_TIME=2021
#   OUTDIR=/path/to/output
#   STORAGE_FOLDER=/path/to/data/eurostat
#   WRITE_SFC_WORKBOOK=0|1
#   FORCE_DOWNLOAD=0|1
#   FORCE_REPARSE=0|1
#   SKIP_VALIDATE=0|1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected venv python at $PY" >&2
  exit 1
fi

EUROSTAT_GEO="${EUROSTAT_GEO:-AT}"
EUROSTAT_TIME="${EUROSTAT_TIME:-2021}"
EUROSTAT_DATASET="${EUROSTAT_DATASET:-naio_10_cp1700}"
EUROSTAT_ROW_DIM="${EUROSTAT_ROW_DIM:-prd_ava}"
EUROSTAT_COL_DIM="${EUROSTAT_COL_DIM:-prd_use}"
EUROSTAT_STK_FLOW_DIM="${EUROSTAT_STK_FLOW_DIM:-stk_flow}"
EUROSTAT_STK_FLOWS="${EUROSTAT_STK_FLOWS:-[\"DOM\",\"IMP\",\"TOTAL\"]}"
BALANCING_SCOPE="${BALANCING_SCOPE:-local}"
STORAGE_FOLDER="${STORAGE_FOLDER:-$ROOT_DIR/data/eurostat}"
OUTDIR="${OUTDIR:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_energy_balance_pxp}"
WRITE_SFC_WORKBOOK="${WRITE_SFC_WORKBOOK:-0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
FORCE_REPARSE="${FORCE_REPARSE:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

PIPELINE_DIR="$ROOT_DIR/scripts/pipelines/eurostat_energybalances_split"
SPLIT_CSV="${SPLIT_CSV:-$PIPELINE_DIR/energy_balance_pxp_split.csv}"
REAGGREGATION_CSV="${REAGGREGATION_CSV:-$PIPELINE_DIR/energy_balance_pxp_reaggregation.csv}"
INPUT_TECHNOLOGY_CSV="${INPUT_TECHNOLOGY_CSV:-$PIPELINE_DIR/energy_balance_pxp_input_technology.csv}"
PRICES_CSV="${PRICES_CSV:-$PIPELINE_DIR/energy_balance_pxp_prices.csv}"
EB_DICT_CSV="${EB_DICT_CSV:-$ROOT_DIR/scripts/mappings/eurostat_energy_balances_dictionary.csv}"

for f in "$SPLIT_CSV" "$REAGGREGATION_CSV" "$INPUT_TECHNOLOGY_CSV" "$PRICES_CSV"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required pipeline CSV missing: $f" >&2
    exit 1
  fi
done

if [[ ! -f "$EB_DICT_CSV" ]]; then
  echo "Energy Balance dictionary not found; generating $EB_DICT_CSV"
  "$PY" "$ROOT_DIR/scripts/build_eurostat_energy_balance_dictionary.py" \
    --geo "$EUROSTAT_GEO" \
    --year "$EUROSTAT_TIME" \
    --storage-folder "$STORAGE_FOLDER" \
    --out "$EB_DICT_CSV"
fi

CLI_ARGS=(
  "$ROOT_DIR/scripts/split_eurostat_iot_by_energy_balances.py"
  --geo "$EUROSTAT_GEO"
  --year "$EUROSTAT_TIME"
  --eurostat-dataset "$EUROSTAT_DATASET"
  --eurostat-row-dim "$EUROSTAT_ROW_DIM"
  --eurostat-col-dim "$EUROSTAT_COL_DIM"
  --eurostat-stk-flow-dim "$EUROSTAT_STK_FLOW_DIM"
  --eurostat-stk-flows "$EUROSTAT_STK_FLOWS"
  --split-csv "$SPLIT_CSV"
  --reaggregation-csv "$REAGGREGATION_CSV"
  --input-technology-csv "$INPUT_TECHNOLOGY_CSV"
  --prices-csv "$PRICES_CSV"
  --energy-balance-dictionary-csv "$EB_DICT_CSV"
  --outdir "$OUTDIR"
  --storage-folder "$STORAGE_FOLDER"
  --balancing-scope "$BALANCING_SCOPE"
  --write-reaggregated
  --verbose
)

if [[ "$FORCE_DOWNLOAD" == "1" ]]; then
  CLI_ARGS+=(--force-download)
fi
if [[ "$FORCE_REPARSE" == "1" ]]; then
  CLI_ARGS+=(--force-reparse)
fi
if [[ "$WRITE_SFC_WORKBOOK" == "1" ]]; then
  CLI_ARGS+=(--write-sfc-workbook)
fi

set -x
"$PY" "${CLI_ARGS[@]}"
set +x

if [[ "$SKIP_VALIDATE" != "1" ]]; then
  "$PY" "$ROOT_DIR/scripts/validate_energy_balance_pxp_outputs.py" \
    --outdir "$OUTDIR" \
    --flows "DOM,IMP,TOTAL" \
    --require-reaggregated \
    --write-summary "$OUTDIR/validation_summary.csv"
fi

cat <<RUNNER_DONE

Energy Balance pxp pipeline completed.
Output directory:
  $OUTDIR

Key outputs:
  $OUTDIR/plan_diagnostics/
  $OUTDIR/stk_flow=DOM/Z_split.csv
  $OUTDIR/stk_flow=DOM/Z_reaggregated.csv
  $OUTDIR/stk_flow=IMP/Z_split.csv
  $OUTDIR/stk_flow=IMP/Z_reaggregated.csv
  $OUTDIR/stk_flow=TOTAL/Z_split.csv
  $OUTDIR/stk_flow=TOTAL/Z_reaggregated.csv
  $OUTDIR/run_summary.csv
  $OUTDIR/validation_summary.csv
RUNNER_DONE
