#!/usr/bin/env bash
set -euo pipefail

# Refresh the existing 2021 workbooks in both model repositories.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POSTGROWTH_DIR="${POSTGROWTH_DIR:-$ROOT_DIR/../PostGrowth-SFC-Austria}"
ETP_DIR="${ETP_DIR:-$ROOT_DIR/../ETP_EnergyTransitionPlan}"
export EUROSTAT_GEO="${EUROSTAT_GEO:-AT}"
export EUROSTAT_TIME="${EUROSTAT_TIME:-2021}"
export EXIO_YEAR="${EXIO_YEAR:-$EUROSTAT_TIME}"
export OUT_REAGG="${OUT_REAGG:-$ROOT_DIR/outputs/${EUROSTAT_GEO}_${EUROSTAT_TIME}_energy_green_brown_reaggregated_flows}"
export OUT_XLSX="${ENERGY_OUT:-$ROOT_DIR/outputs/Energy_Balances_${EUROSTAT_TIME}_Eurostat_green_brown.xlsx}"
export SPLIT_ENERGY_GREEN_BROWN=1 POSTGROWTH_SHEET_NAME=Energy_1.0
bash "$ROOT_DIR/scripts/pipelines/at_2021_energy_green_brown.sh"
bash "$ROOT_DIR/scripts/pipelines/split_energy_balances_65CPAproducts_aggregate_energy_sources.sh"

# Keep required domestic/import accounts and final mappings inside the existing
# TOTAL workbook. The scientific arrays and upstream validation remain unchanged.
IOT_OUT="$ROOT_DIR/outputs/IOT_${EUROSTAT_GEO}_${EUROSTAT_TIME}_TOTAL_energy_green_brown.xlsx"
"$ROOT_DIR/.venv/bin/python" - "$ROOT_DIR" "$OUT_REAGG" "$IOT_OUT" <<'PY'
import copy
import json
import sys
from pathlib import Path
import openpyxl

root, source, output = map(Path, sys.argv[1:])
book = openpyxl.load_workbook(next((source / "stk_flow=TOTAL").glob("IOT_*.xlsx")))
for flow in ("DOM", "IMP"):
    donor = openpyxl.load_workbook(next((source / f"stk_flow={flow}").glob("IOT_*.xlsx")))
    sheets = [("Sheet 1", flow)] + ([("Re-aggregated Summary", "DOM CO2")] if flow == "DOM" else [])
    for old_name, name in sheets:
        sheet = book.create_sheet(name)
        for row in donor[old_name]:
            for cell in row:
                target = sheet.cell(cell.row, cell.column, cell.value)
                target._style = copy.copy(cell._style)
        for key, dimension in donor[old_name].column_dimensions.items():
            sheet.column_dimensions[key].width = dimension.width
info_path = source.parent / source.name.replace("reaggregated", "split") / "stk_flow=DOM/split_info.json"
info = json.loads(info_path.read_text())
for key in ("exio_doi", "exio_source_zip_sha256", "exio_to_eurostat_mapping_sha256"):
    book["Summary"].append((key, info[key]))
book.save(output)
PY
for destination in "$POSTGROWTH_DIR" "$ETP_DIR"; do
  install -m 0644 "$OUT_XLSX" "$destination/Energy_Balances_${EUROSTAT_TIME}_Eurostat_green_brown.xlsx"
  install -m 0644 "$IOT_OUT" "$destination/IOT_${EUROSTAT_GEO}_${EUROSTAT_TIME}_TOTAL_energy_green_brown.xlsx"
  install -m 0644 "$ROOT_DIR/scripts/mappings/exio_to_eurostat_cpa_products.csv" "$destination/exio_to_eurostat_cpa_products.csv"
done
