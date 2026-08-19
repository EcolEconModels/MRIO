#!/usr/bin/env bash
set -euo pipefail

# Year-parameterised pipeline: canonical split -> canonical reaggregation.
#
# Required / important env vars:
#   OUTPUT_SUFFIX=fuels_more|fuels|...
# Optional overrides:
#   SPLIT_MAPPING_CSV=scripts/pipelines/fuels_split.csv
#   REAGG_MAPPING_CSV=scripts/pipelines/fuels_reaggregation.csv
#   CONCORDANCE_CSV=scripts/mappings/exio_to_eurostat_cpa_products.csv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected venv python at $PY" >&2
  exit 1
fi

OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-fuels}"
EUROSTAT_TIME="${EUROSTAT_TIME:-2021}"
EUROSTAT_GEO="${EUROSTAT_GEO:-AT}"
EXIO_YEAR="${EXIO_YEAR:-2021}"
EXIO_REGION="${EXIO_REGION:-$EUROSTAT_GEO}"
EXIO_DOI="${EXIO_DOI:-10.5281/zenodo.20051562}"
MAPPING_CONSISTENCY_RTOL="${MAPPING_CONSISTENCY_RTOL:-0.2}"
MAPPING_CONSISTENCY_ATOL="${MAPPING_CONSISTENCY_ATOL:-1e-6}"
FLOW_IDENTITY_RTOL="${FLOW_IDENTITY_RTOL:-1e-9}"
FLOW_IDENTITY_ATOL="${FLOW_IDENTITY_ATOL:-1e-6}"
EXPECTED_REAGG_SECTORS="${EXPECTED_REAGG_SECTORS:-}"
MPG4ALL_DIR="${MPG4ALL_DIR:-$(cd "$ROOT_DIR/.." && pwd)/MPG4ALL}"
CAPITAL_STOCK_SOURCE_CSV="${CAPITAL_STOCK_SOURCE_CSV:-}"
ENVIRONMENTAL_PRESET="${ENVIRONMENTAL_PRESET:-combustion-co2}"
ENVIRONMENTAL_ALL_EXTENSIONS="${ENVIRONMENTAL_ALL_EXTENSIONS:-0}"
EUROSTAT_FORCE_DOWNLOAD="${EUROSTAT_FORCE_DOWNLOAD:-0}"

_is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off|"") return 1 ;;
    *) echo "ERROR: Expected boolean env value, got: $1" >&2; exit 2 ;;
  esac
}

ENVIRONMENTAL_ARGS=(--environmental-preset "$ENVIRONMENTAL_PRESET")
if _is_true "$ENVIRONMENTAL_ALL_EXTENSIONS"; then
  if [[ -n "${ENVIRONMENTAL_EXTENSIONS_JSON:-}" ]]; then
    echo "ERROR: ENVIRONMENTAL_ALL_EXTENSIONS and ENVIRONMENTAL_EXTENSIONS_JSON are mutually exclusive." >&2
    exit 2
  fi
  ENVIRONMENTAL_ARGS=(--environmental-all-extensions)
elif [[ -n "${ENVIRONMENTAL_EXTENSIONS_JSON:-}" ]]; then
  # Advanced escape hatch: explicit JSON replaces the preset without shell rewriting.
  ENVIRONMENTAL_ARGS=(--environmental-extensions-json "$ENVIRONMENTAL_EXTENSIONS_JSON")
fi

EUROSTAT_REFRESH_ARGS=()
if _is_true "$EUROSTAT_FORCE_DOWNLOAD"; then
  # Refresh only the Eurostat API payload. EXIO source-zip download policy is untouched.
  EUROSTAT_REFRESH_ARGS=(--eurostat-force-download)
fi

GENERATED_MAPPINGS_OUT="${GENERATED_MAPPINGS_OUT:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_${OUTPUT_SUFFIX}_generated_mappings}"
OUT_SPLIT="${OUT_SPLIT:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_${OUTPUT_SUFFIX}_split_flows}"
OUT_DISAGG="${OUT_DISAGG:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_${OUTPUT_SUFFIX}_disaggregated_flows}"
OUT_REAGG="${OUT_REAGG:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_${OUTPUT_SUFFIX}_reaggregated_flows}"

IDENTITY_MAPPING_CSV="$(mktemp)"

SPLIT_MAPPING_CSV="${SPLIT_MAPPING_CSV:-$ROOT_DIR/scripts/pipelines/${OUTPUT_SUFFIX}_split.csv}"
REAGG_MAPPING_CSV="${REAGG_MAPPING_CSV:-$ROOT_DIR/scripts/pipelines/${OUTPUT_SUFFIX}_reaggregation.csv}"
CONCORDANCE_CSV="${CONCORDANCE_CSV:-$ROOT_DIR/scripts/mappings/exio_to_eurostat_cpa_products.csv}"

TEMPLATE_XLSX="${TEMPLATE_XLSX:-$ROOT_DIR/tmp/IOT_Austria_${EUROSTAT_TIME}_Eurostat.xlsx}"
if [[ ! -f "$TEMPLATE_XLSX" ]]; then
  echo "ERROR: Template workbook not found: $TEMPLATE_XLSX" >&2
  exit 1
fi

for mapping_csv in "$SPLIT_MAPPING_CSV" "$REAGG_MAPPING_CSV"; do
  if [[ ! -f "$mapping_csv" ]]; then
    echo "ERROR: Canonical mapping CSV not found: $mapping_csv" >&2
    echo "Use the committed canonical fuels CSVs under scripts/pipelines/." >&2
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

STK_FLOWS_JSON='["DOM","IMP"]'

CONSISTENCY_ARGS=(
  --mapping-consistency-rtol "$MAPPING_CONSISTENCY_RTOL"
  --mapping-consistency-atol "$MAPPING_CONSISTENCY_ATOL"
  --flow-identity-rtol "$FLOW_IDENTITY_RTOL"
  --flow-identity-atol "$FLOW_IDENTITY_ATOL"
)

EXPECT_ARGS=()
if [[ -n "$EXPECTED_REAGG_SECTORS" ]]; then
  EXPECT_ARGS=(--expect-n-sectors "$EXPECTED_REAGG_SECTORS")
fi

set -x

"$PY" "$ROOT_DIR/scripts/pipelines/check_reaggregation_coverage.py" \
  --split-mapping-csv "$SPLIT_MAPPING_CSV" \
  --reagg-mapping-csv "$REAGG_MAPPING_CSV" \
  --concordance-csv "$CONCORDANCE_CSV" \
  --report-csv "$GENERATED_MAPPINGS_OUT/reaggregation_coverage_preflight.csv"

"$PY" "$ROOT_DIR/scripts/split_eurostat_energy_by_exiobase.py" --verbose \
  --eurostat-dataset naio_10_cp1700 \
  --eurostat-filters "$EUROSTAT_FILTERS_JSON" \
  --eurostat-row-dim prd_ava --eurostat-col-dim prd_use \
  --eurostat-stk-flow-dim stk_flow \
  --eurostat-stk-flows "$STK_FLOWS_JSON" \
  "${EUROSTAT_REFRESH_ARGS[@]}" \
  --exio-year "$EXIO_YEAR" --exio-region "$EXIO_REGION" \
  --exio-doi "$EXIO_DOI" \
  "${ENVIRONMENTAL_ARGS[@]}" \
  --exio-mapping-csv "$SPLIT_MAPPING_CSV" \
  --require-mapping-coverage --coverage-tol 0 \
  "${CONSISTENCY_ARGS[@]}" \
  --exio-to-eurostat-mapping-csv "$CONCORDANCE_CSV" \
  --outdir "$OUT_SPLIT"

"$PY" "$ROOT_DIR/scripts/derive_total_iot_flow.py" \
  --outdir "$OUT_SPLIT" \
  --eurostat-dataset naio_10_cp1700 \
  --eurostat-filters "$EUROSTAT_FILTERS_JSON" \
  --eurostat-row-dim prd_ava --eurostat-col-dim prd_use \
  --eurostat-stk-flow-dim stk_flow \
  --rtol "$FLOW_IDENTITY_RTOL" \
  --atol "$FLOW_IDENTITY_ATOL"

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
    for aggregate_code, sector_codes in data.items():
        for sector_code in sector_codes:
            code = str(sector_code)
            if code in seen:
                continue
            seen.add(code)
            rows.append(
                {
                    "mapping_set": f"{mapping_set}_identity_generated",
                    "source_code": code,
                    "target_code": code,
                    "source_parent_code": str(aggregate_code),
                    "enabled": True,
                    "notes": f"Generated from {json_path.name}.",
                }
            )

fieldnames = [
    "mapping_set",
    "source_code",
    "target_code",
    "source_parent_code",
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
  --write-sfc-workbook \
  --sfc-template-xlsx "$TEMPLATE_XLSX" \
  --sfc-outname "IOT_${EUROSTAT_GEO}_${EUROSTAT_TIME}_{variant}_${OUTPUT_SUFFIX}_split.xlsx" \
  --eurostat-dataset naio_10_cp1700 \
  --eurostat-filters "$EUROSTAT_FILTERS_JSON" \
  --eurostat-row-dim prd_ava --eurostat-col-dim prd_use

"$PY" "$ROOT_DIR/scripts/derive_total_iot_flow.py" \
  --outdir "$OUT_DISAGG" --validate-only agg \
  --rtol "$FLOW_IDENTITY_RTOL" --atol "$FLOW_IDENTITY_ATOL"

"$PY" "$ROOT_DIR/scripts/aggregate_iot_by_mapping.py" --verbose \
  --indir "$OUT_SPLIT" \
  --outdir "$OUT_REAGG" \
  --mapping-csv "$REAGG_MAPPING_CSV" \
  --fail-on-unmapped-split-children \
  "${EXPECT_ARGS[@]}" \
  --write-sfc-workbook \
  --sfc-template-xlsx "$TEMPLATE_XLSX" \
  --sfc-outname "IOT_${EUROSTAT_GEO}_${EUROSTAT_TIME}_{variant}_${OUTPUT_SUFFIX}_reaggregated.xlsx" \
  --eurostat-dataset naio_10_cp1700 \
  --eurostat-filters "$EUROSTAT_FILTERS_JSON" \
  --eurostat-row-dim prd_ava --eurostat-col-dim prd_use

"$PY" "$ROOT_DIR/scripts/derive_total_iot_flow.py" \
  --outdir "$OUT_REAGG" --validate-only agg \
  --rtol "$FLOW_IDENTITY_RTOL" --atol "$FLOW_IDENTITY_ATOL"

if [[ -n "$CAPITAL_STOCK_SOURCE_CSV" ]]; then
  "$PY" "$ROOT_DIR/scripts/export_product_initialization.py" \
    --official-input-csv "$CAPITAL_STOCK_SOURCE_CSV" \
    --iot-output-csv "$OUT_REAGG/stk_flow=TOTAL/x_agg.csv" \
    --template-xlsx "$MPG4ALL_DIR/capital_stock.xlsx" \
    --output-xlsx "$MPG4ALL_DIR/capital_stock_${EUROSTAT_TIME}.xlsx" \
    --value-column capital_stock:2 \
    --share-column gross_capital_formation:3 \
    --diagnostic-csv "$OUT_REAGG/stk_flow=TOTAL/capital_stock_output_shares.csv"
fi
