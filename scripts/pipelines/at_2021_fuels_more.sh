#!/usr/bin/env bash
set -euo pipefail

# Pipeline wrapper: canonical fuels_more split/reaggregation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-fuels_more}"

exec "$ROOT_DIR/scripts/pipelines/at_2021_workbook_reaggregation.sh"