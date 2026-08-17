#!/usr/bin/env bash
set -euo pipefail

# Pipeline: AT 2021 Eurostat IOT -> split energy subsectors -> aggregate ENERGY_GREEN/BROWN
#
# This script is intentionally dependency-light:
# - Uses the repo virtualenv interpreter: .venv/bin/python
# - Does not require make/go-task/just
#
# Usage:
#   bash scripts/pipelines/at_2021_energy_green_brown.sh
#
# Optional env vars:
#   TEMPLATE_XLSX=tmp/IOT_Austria_2021_Eurostat.xlsx
#   OUT_SPLIT=outputs/AT_2021_energy_split_flows_with_source
#   OUT_AGG=outputs/AT_2021_energy_green_brown_flows_with_source
#   EXIO_YEAR=2020
#   EXIO_REGION=AT
#   EXIO_DOI=10.5281/zenodo.20051562
#   EUROSTAT_TIME=2021
#   EUROSTAT_GEO=AT
#   MAPPING_CONSISTENCY_RTOL=0.2
#   MAPPING_CONSISTENCY_ATOL=1e-6

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected venv python at $PY" >&2
  echo "Create it and install deps (see README/requirements)." >&2
  exit 1
fi

EXIO_YEAR="${EXIO_YEAR:-2020}"
EXIO_REGION="${EXIO_REGION:-AT}"
EXIO_DOI="${EXIO_DOI:-10.5281/zenodo.20051562}"
EUROSTAT_TIME="${EUROSTAT_TIME:-2021}"
EUROSTAT_GEO="${EUROSTAT_GEO:-AT}"
ENVIRONMENTAL_PRESET="${ENVIRONMENTAL_PRESET:-combustion-co2}"
ENVIRONMENTAL_ARGS=(--environmental-preset "$ENVIRONMENTAL_PRESET")
# Advanced escape hatch: explicit JSON replaces the preset without shell rewriting.
if [[ -n "${ENVIRONMENTAL_EXTENSIONS_JSON:-}" ]]; then
  ENVIRONMENTAL_ARGS=(--environmental-extensions-json "$ENVIRONMENTAL_EXTENSIONS_JSON")
fi

MAP_EXIO_TO_EUROSTAT="$ROOT_DIR/scripts/pipelines/energy_green_brown_split.csv"
MAP_GREEN_BROWN="$ROOT_DIR/scripts/pipelines/energy_green_brown_reaggregation.csv"
EXIO_TO_EUROSTAT_MAP="${EXIO_TO_EUROSTAT_MAP:-$ROOT_DIR/scripts/mappings/exio_to_eurostat_cpa_products.csv}"

OUT_SPLIT="${OUT_SPLIT:-$ROOT_DIR/outputs/AT_2021_energy_split_flows_with_source}"
OUT_AGG="${OUT_AGG:-$ROOT_DIR/outputs/AT_2021_energy_green_brown_flows_with_source}"
MAPPING_CONSISTENCY_RTOL="${MAPPING_CONSISTENCY_RTOL:-0.5}"
MAPPING_CONSISTENCY_ATOL="${MAPPING_CONSISTENCY_ATOL:-1e-6}"

TEMPLATE_XLSX="${TEMPLATE_XLSX:-$ROOT_DIR/tmp/IOT_Austria_2021_Eurostat.xlsx}"
if [[ -z "${SFC_OUTNAME:-}" ]]; then
  SFC_OUTNAME="IOT_${EUROSTAT_GEO}_${EUROSTAT_TIME}"'_{variant}_energy_green_brown.xlsx'
fi

EUROSTAT_FILTERS_JSON=$(cat <<JSON
{"geo":"$EUROSTAT_GEO","time":"$EUROSTAT_TIME","freq":"A","unit":"MIO_EUR"}
JSON
)

STK_FLOWS_JSON='["DOM","IMP","TOTAL"]'

set -x

# 1) Split Eurostat IOT (batch over stk_flow)
EXIO_TO_EUROSTAT_ARGS=()
if [[ -f "$EXIO_TO_EUROSTAT_MAP" ]]; then
  EXIO_TO_EUROSTAT_ARGS=(--exio-to-eurostat-mapping-csv "$EXIO_TO_EUROSTAT_MAP")
fi

CONSISTENCY_ARGS=(
  --mapping-consistency-rtol "$MAPPING_CONSISTENCY_RTOL"
  --mapping-consistency-atol "$MAPPING_CONSISTENCY_ATOL"
)

"$PY" "$ROOT_DIR/scripts/split_eurostat_energy_by_exiobase.py" --verbose \
  --eurostat-dataset naio_10_cp1700 \
  --eurostat-filters "$EUROSTAT_FILTERS_JSON" \
  --eurostat-row-dim prd_ava --eurostat-col-dim prd_use \
  --eurostat-stk-flow-dim stk_flow \
  --eurostat-stk-flows "$STK_FLOWS_JSON" \
  --exio-year "$EXIO_YEAR" --exio-region "$EXIO_REGION" \
  --exio-doi "$EXIO_DOI" \
  "${ENVIRONMENTAL_ARGS[@]}" \
  --exio-mapping-csv "$MAP_EXIO_TO_EUROSTAT" \
  --require-mapping-coverage --coverage-tol 0 \
  "${CONSISTENCY_ARGS[@]}" \
  "${EXIO_TO_EUROSTAT_ARGS[@]}" \
  --outdir "$OUT_SPLIT"

# 2) Aggregate to ENERGY_GREEN/BROWN + write Eurostat-shaped workbooks
if [[ ! -f "$TEMPLATE_XLSX" ]]; then
  echo "ERROR: Template workbook not found: $TEMPLATE_XLSX" >&2
  echo "Provide TEMPLATE_XLSX=... or create tmp/IOT_Austria_2021_Eurostat.xlsx" >&2
  exit 1
fi

"$PY" "$ROOT_DIR/scripts/aggregate_iot_by_mapping.py" --verbose \
  --indir "$OUT_SPLIT" \
  --outdir "$OUT_AGG" \
  --mapping-csv "$MAP_GREEN_BROWN" \
  --fail-on-unmapped-split-children \
  --expect-n-sectors 65 \
  --write-sfc-workbook \
  --sfc-template-xlsx "$TEMPLATE_XLSX" \
  --sfc-outname "$SFC_OUTNAME" \
  --eurostat-dataset naio_10_cp1700 \
  --eurostat-filters "$EUROSTAT_FILTERS_JSON" \
  --eurostat-row-dim prd_ava --eurostat-col-dim prd_use
