#!/usr/bin/env bash
set -euo pipefail

# Generic pipeline: canonical 2021 split -> canonical reaggregation.
#
# Required / important env vars:
#   OUTPUT_SUFFIX=fuels_more|fuels|...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected venv python at $PY" >&2
  exit 1
fi

OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-fuels}"
EUROSTAT_TIME="${EUROSTAT_TIME:-2021}"
EUROSTAT_GEO="${EUROSTAT_GEO:-AT}"
EXIO_YEAR="${EXIO_YEAR:-2020}"
EXIO_REGION="${EXIO_REGION:-$EUROSTAT_GEO}"
MAPPING_CONSISTENCY_RTOL="${MAPPING_CONSISTENCY_RTOL:-0.2}"
MAPPING_CONSISTENCY_ATOL="${MAPPING_CONSISTENCY_ATOL:-1e-6}"

GENERATED_MAPPINGS_OUT="${GENERATED_MAPPINGS_OUT:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_${OUTPUT_SUFFIX}_generated_mappings}"
OUT_SPLIT="${OUT_SPLIT:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_${OUTPUT_SUFFIX}_split_flows}"
OUT_DISAGG="${OUT_DISAGG:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_${OUTPUT_SUFFIX}_disaggregated_flows}"
OUT_REAGG="${OUT_REAGG:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_${OUTPUT_SUFFIX}_reaggregated_flows}"

IDENTITY_MAPPING_CSV="$(mktemp)"

SPLIT_MAPPING_CSV="${SPLIT_MAPPING_CSV:-$ROOT_DIR/scripts/mappings/${OUTPUT_SUFFIX}_split.csv}"
REAGG_MAPPING_CSV="${REAGG_MAPPING_CSV:-$ROOT_DIR/scripts/mappings/${OUTPUT_SUFFIX}_reaggregation.csv}"
CONCORDANCE_CSV="${CONCORDANCE_CSV:-$ROOT_DIR/scripts/mappings/exio_to_eurostat_cpa_products.csv}"

TEMPLATE_XLSX="${TEMPLATE_XLSX:-$ROOT_DIR/tmp/IOT_Austria_2021_Eurostat.xlsx}"
if [[ ! -f "$TEMPLATE_XLSX" ]]; then
  echo "ERROR: Template workbook not found: $TEMPLATE_XLSX" >&2
  exit 1
fi

for mapping_csv in "$SPLIT_MAPPING_CSV" "$REAGG_MAPPING_CSV"; do
  if [[ ! -f "$mapping_csv" ]]; then
    echo "ERROR: Canonical mapping CSV not found: $mapping_csv" >&2
    echo "Use the committed canonical fuels CSVs under scripts/mappings/." >&2
    exit 1
  fi
done

cleanup_identity_mapping() {
  rm -f "$IDENTITY_MAPPING_CSV"
}
trap cleanup_identity_mapping EXIT

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

"$PY" - "$OUT_SPLIT" "$IDENTITY_MAPPING_CSV" "$OUTPUT_SUFFIX" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

out_split = Path(sys.argv[1])
identity_csv = Path(sys.argv[2])
mapping_set = sys.argv[3]

rows: list[dict[str, object]] = []
seen: set[str] = set()
for json_path in sorted(out_split.glob("stk_flow=*/exio_selected_sectors.json")):
    with json_path.open() as handle:
        data = json.load(handle)
    for aggregate_code, labels in data.items():
        for label in labels:
            code = f"{aggregate_code}::{label}"
            if code in seen:
                continue
            seen.add(code)
            rows.append(
                {
                    "mapping_set": f"{mapping_set}_identity_generated",
                    "source_code": code,
                    "source_label": code,
                    "target_code": code,
                    "target_label": code,
                    "subtarget_code": "",
                    "subtarget_label": "",
                    "weight": 1.0,
                    "enabled": True,
                    "notes": f"Generated from {json_path.name}.",
                }
            )

fieldnames = [
    "mapping_set",
    "source_code",
    "source_label",
    "target_code",
    "target_label",
    "subtarget_code",
    "subtarget_label",
    "weight",
    "enabled",
    "notes",
]
with identity_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {identity_csv} with {len(rows)} rows")
PY

"$PY" "$ROOT_DIR/scripts/aggregate_iot_by_mapping.py" --verbose \
  --indir "$OUT_SPLIT" \
  --outdir "$OUT_DISAGG" \
  --mapping-csv "$IDENTITY_MAPPING_CSV" \
  --require-from-contains '::' \
  --write-sfc-workbook \
  --sfc-template-xlsx "$TEMPLATE_XLSX" \
  --sfc-outname "IOT_${EUROSTAT_GEO}_${EUROSTAT_TIME}_{variant}_${OUTPUT_SUFFIX}_split.xlsx" \
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
  --sfc-outname "IOT_${EUROSTAT_GEO}_${EUROSTAT_TIME}_{variant}_${OUTPUT_SUFFIX}_reaggregated.xlsx" \
  --eurostat-dataset naio_10_cp1700 \
  --eurostat-filters "$EUROSTAT_FILTERS_JSON" \
  --eurostat-row-dim prd_ava --eurostat-col-dim prd_use