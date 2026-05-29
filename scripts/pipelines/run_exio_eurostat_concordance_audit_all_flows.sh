#!/usr/bin/env bash
set -euo pipefail

# Shell wrapper for the domestic/imports/total EXIOBASE-Eurostat concordance audit.
# The wrapper keeps the audit launch reproducible in-repo by documenting the CLI
# arguments as env vars and printing the exact command before execution.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

COUNTRY="${COUNTRY:-AT}"
YEAR="${YEAR:-2020}"
EXIO_YEAR="${EXIO_YEAR:-}"
MAPPING_CSV="${MAPPING_CSV:-$ROOT_DIR/scripts/mappings/exio_to_eurostat_cpa_products.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
REL_THRESHOLD="${REL_THRESHOLD:-0.20}"
MIN_ABS_MEUR="${MIN_ABS_MEUR:-5.0}"
IDENTITY_MODE="${IDENTITY_MODE:-diagnostic}"
EUROSTAT_DATASET="${EUROSTAT_DATASET:-naio_10_cp1700}"
EUROSTAT_UNIT="${EUROSTAT_UNIT:-MIO_EUR}"
EXIO_SYSTEM="${EXIO_SYSTEM:-pxp}"
VERBOSE="${VERBOSE:-1}"
FULL_SUBCOMPONENT_Z_SCAN="${FULL_SUBCOMPONENT_Z_SCAN:-0}"
PROGRESS_HEARTBEAT_SECONDS="${PROGRESS_HEARTBEAT_SECONDS:-30}"
MONITOR_RSS="${MONITOR_RSS:-1}"

cmd=(
  env
  PYTHONPATH=src
  "$ROOT_DIR/.venv/bin/python"
  "$ROOT_DIR/scripts/run_exio_eurostat_concordance_audit_all_flows.py"
  --country "$COUNTRY"
  --year "$YEAR"
  --mapping-csv "$MAPPING_CSV"
  --rel-threshold "$REL_THRESHOLD"
  --min-abs-meur "$MIN_ABS_MEUR"
  --identity-mode "$IDENTITY_MODE"
  --eurostat-dataset "$EUROSTAT_DATASET"
  --eurostat-unit "$EUROSTAT_UNIT"
  --exio-system "$EXIO_SYSTEM"
)

if [[ -n "$EXIO_YEAR" ]]; then
  cmd+=(--exio-year "$EXIO_YEAR")
fi

if [[ -n "$OUTPUT_ROOT" ]]; then
  cmd+=(--output-root "$OUTPUT_ROOT")
fi

if [[ "$VERBOSE" == "1" || "$VERBOSE" == "true" ]]; then
  cmd+=(--verbose)
fi

if [[ "$FULL_SUBCOMPONENT_Z_SCAN" == "1" || "$FULL_SUBCOMPONENT_Z_SCAN" == "true" ]]; then
  cmd+=(--full-subcomponent-z-scan)
fi

if (( $# )); then
  cmd+=("$@")
fi

printf 'Running concordance audit command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" &
audit_pid=$!
audit_started_at=$(date +%s)
tick=0

cleanup() {
  if kill -0 "$audit_pid" 2>/dev/null; then
    kill "$audit_pid" 2>/dev/null || true
  fi
}

trap cleanup INT TERM

while kill -0 "$audit_pid" 2>/dev/null; do
  tick=$((tick + 1))
  elapsed=$(( $(date +%s) - audit_started_at ))
  rss_text="n/a"
  if [[ "$MONITOR_RSS" == "1" || "$MONITOR_RSS" == "true" ]]; then
    rss_kb="$(ps -o rss= -p "$audit_pid" | tr -d '[:space:]' || true)"
    if [[ -n "$rss_kb" ]]; then
      rss_text="$(awk -v kb="$rss_kb" 'BEGIN { printf "%.1f", kb / 1024.0 }')MB"
    fi
  fi
  printf '[audit] tick=%s elapsed=%ss rss=%s pid=%s\n' "$tick" "$elapsed" "$rss_text" "$audit_pid"
  sleep "$PROGRESS_HEARTBEAT_SECONDS"
done

set +e
wait "$audit_pid"
status=$?
set -e
trap - INT TERM
printf '[audit] finished exit=%s\n' "$status"
exit "$status"