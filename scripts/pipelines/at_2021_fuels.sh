#!/usr/bin/env bash
set -euo pipefail

# Pipeline: workbook-driven 2021 fuels split -> workbook-driven reaggregation
#
# Required env vars:
#   WORKBOOK_XLSX=/path/to/saved/IO_Exiobase_Sector_Disaggregation.xlsx
#
# Optional env vars:
#   WORKBOOK_SHEET=Step2_Reaggregation
#   TEMPLATE_XLSX=tmp/IOT_Austria_2021_Eurostat.xlsx
#   EUROSTAT_TIME=2021
#   EUROSTAT_GEO=AT
#   EXIO_YEAR=2020
#   EXIO_REGION=AT
#   MAPPING_CONSISTENCY_RTOL=0.2
#   MAPPING_CONSISTENCY_ATOL=1e-6
#   GENERATED_MAPPINGS_OUT=outputs/AT_2021_fuels_generated_mappings
#   OUT_SPLIT=outputs/AT_2021_fuels_split_flows
#   OUT_DISAGG=outputs/AT_2021_fuels_disaggregated_flows
#   OUT_REAGG=outputs/AT_2021_fuels_reaggregated_flows

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected venv python at $PY" >&2
  exit 1
fi

WORKBOOK_XLSX="${WORKBOOK_XLSX:-}"
if [[ -z "$WORKBOOK_XLSX" ]]; then
  echo "ERROR: Provide WORKBOOK_XLSX=/path/to/saved/workbook.xlsx" >&2
  exit 1
fi
if [[ ! -f "$WORKBOOK_XLSX" ]]; then
  echo "ERROR: Workbook not found: $WORKBOOK_XLSX" >&2
  exit 1
fi

WORKBOOK_SHEET="${WORKBOOK_SHEET:-Step2_Reaggregation}"
EUROSTAT_TIME="${EUROSTAT_TIME:-2021}"
EUROSTAT_GEO="${EUROSTAT_GEO:-AT}"
EXIO_YEAR="${EXIO_YEAR:-2020}"
EXIO_REGION="${EXIO_REGION:-$EUROSTAT_GEO}"
MAPPING_CONSISTENCY_RTOL="${MAPPING_CONSISTENCY_RTOL:-0.2}"
MAPPING_CONSISTENCY_ATOL="${MAPPING_CONSISTENCY_ATOL:-1e-6}"

GENERATED_MAPPINGS_OUT="${GENERATED_MAPPINGS_OUT:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_fuels_generated_mappings}"
OUT_SPLIT="${OUT_SPLIT:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_fuels_split_flows}"
OUT_DISAGG="${OUT_DISAGG:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_fuels_disaggregated_flows}"
OUT_REAGG="${OUT_REAGG:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_fuels_reaggregated_flows}"

SPLIT_MAPPING_CSV="$GENERATED_MAPPINGS_OUT/exio_fuels_split_from_workbook.csv"
IDENTITY_MAPPING_CSV="$GENERATED_MAPPINGS_OUT/fuels_identity_from_workbook.csv"
REAGG_MAPPING_CSV="$GENERATED_MAPPINGS_OUT/fuels_reaggregation_from_workbook.csv"
CONCORDANCE_CSV="${CONCORDANCE_CSV:-$ROOT_DIR/scripts/mappings/exio_to_eurostat_cpa.csv}"

TEMPLATE_XLSX="${TEMPLATE_XLSX:-$ROOT_DIR/tmp/IOT_Austria_2021_Eurostat.xlsx}"
if [[ ! -f "$TEMPLATE_XLSX" ]]; then
  echo "ERROR: Template workbook not found: $TEMPLATE_XLSX" >&2
  exit 1
fi

EUROSTAT_FILTERS_JSON=$(cat <<JSON
{"geo":"$EUROSTAT_GEO","time":"$EUROSTAT_TIME","freq":"A","unit":"MIO_EUR"}
JSON
)

STK_FLOWS_JSON='["DOM","IMP","TOTAL"]'

CONSISTENCY_ARGS=(
  --mapping-consistency-rtol "$MAPPING_CONSISTENCY_RTOL"
  --mapping-consistency-atol "$MAPPING_CONSISTENCY_ATOL"
)

set -x

"$PY" "$ROOT_DIR/scripts/generate_fuels_mapping_from_workbook.py" \
  --workbook "$WORKBOOK_XLSX" \
  --sheet-name "$WORKBOOK_SHEET" \
  --concordance-csv "$CONCORDANCE_CSV" \
  --outdir "$GENERATED_MAPPINGS_OUT"

"$PY" "$ROOT_DIR/scripts/split_eurostat_energy_by_exiobase.py" --verbose \
  --eurostat-dataset naio_10_cp1700 \
  --eurostat-filters "$EUROSTAT_FILTERS_JSON" \
  --eurostat-row-dim prd_ava --eurostat-col-dim prd_use \
  --eurostat-stk-flow-dim stk_flow \
  --eurostat-stk-flows "$STK_FLOWS_JSON" \
  --exio-year "$EXIO_YEAR" --exio-region "$EXIO_REGION" \
  --exio-mapping-csv "$SPLIT_MAPPING_CSV" \
  --require-mapping-coverage --coverage-tol 0 \
  --balancing-scope local \
  "${CONSISTENCY_ARGS[@]}" \
  --exio-to-eurostat-mapping-csv "$CONCORDANCE_CSV" \
  --outdir "$OUT_SPLIT"

"$PY" "$ROOT_DIR/scripts/aggregate_iot_by_mapping.py" --verbose \
  --indir "$OUT_SPLIT" \
  --outdir "$OUT_DISAGG" \
  --mapping-csv "$IDENTITY_MAPPING_CSV" \
  --require-from-contains '::' \
  --write-sfc-workbook \
  --sfc-template-xlsx "$TEMPLATE_XLSX" \
  --sfc-outname "IOT_${EUROSTAT_GEO}_${EUROSTAT_TIME}_{variant}_fuels_split.xlsx" \
  --eurostat-dataset naio_10_cp1700 \
  --eurostat-filters "$EUROSTAT_FILTERS_JSON" \
  --eurostat-row-dim prd_ava --eurostat-col-dim prd_use

"$PY" "$ROOT_DIR/scripts/aggregate_iot_by_mapping.py" --verbose \
  --indir "$OUT_SPLIT" \
  --outdir "$OUT_REAGG" \
  --mapping-csv "$REAGG_MAPPING_CSV" \
  --require-from-contains '::' \
  --write-sfc-workbook \
  --sfc-template-xlsx "$TEMPLATE_XLSX" \
  --sfc-outname "IOT_${EUROSTAT_GEO}_${EUROSTAT_TIME}_{variant}_fuels_reaggregated.xlsx" \
  --eurostat-dataset naio_10_cp1700 \
  --eurostat-filters "$EUROSTAT_FILTERS_JSON" \
  --eurostat-row-dim prd_ava --eurostat-col-dim prd_use