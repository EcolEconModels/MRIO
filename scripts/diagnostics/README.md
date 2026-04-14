# MRIO Diagnostics Generators

This directory contains the checked-in generators for the four reviewer-facing
Austria diagnostics files written to
`../PostGrowth-SFC-Austria/mrio_energy_balance_diagnostics/`.

## Files

- `_flat_diagnostics_common.py`
  - Shared loaders and helpers for Eurostat IOT data, EXIOBASE slices,
    weighted concordance loading, energy-balance factors, and CSV writing.
- `generate_at2020_flow_mapping_details.py`
  - Regenerates the 2020 flow comparison files:
    - `at2020_dom_mapping_detail.csv`
    - `at2020_imp_mapping_detail.csv`
    - `at2020_total_mapping_detail.csv`
- `generate_at2021_co2_mapping_detail.py`
  - Regenerates the 2021 CO2 comparison file:
    - `at2021_co2_mapping_detail.csv`
- `generate_at2020_strict_zij_outliers.py`
  - Generates non-reviewer-facing debug outputs for strict one-to-one mapped
    sectors only:
    - `at2020_dom_strict_zij_outliers.csv`
    - `at2020_dom_strict_row_outliers.csv`

## Output Location

By default, both scripts write to:

`/home/agilra/EcolEconModels/PostGrowth-SFC-Austria/mrio_energy_balance_diagnostics`

This default is computed relative to the MRIO repo, so the commands do not need
to be run from a specific working directory.

The strict `z_ij` outlier generator is different. By default it writes debug
outputs to:

`/home/agilra/EcolEconModels/MRIO/data/diagnostics_debug`

## Inputs

The generators depend on four input groups.

### 1. EXIOBASE regional slices and extensions

- Loaded through the local `ecolecon_mrio` package.
- 2020 flow files use the Austria region slice for the requested flow:
  - `domestic`
  - `imports`
  - `total`
- 2021 CO2 file uses:
  - 2020 EXIO monetary output
  - EXIO energy extension
  - EXIO air-emissions extension

### 2. Eurostat IO tables

- Dataset: `naio_10_cp1700`
- Geo: `AT`
- Units: `MIO_EUR`
- Flow codes used by the scripts:
  - `DOM`
  - `IMP`
  - `TOTAL`

### 3. Concordance and mapping inputs

- `scripts/mappings/exio_to_eurostat_cpa.csv`
- `scripts/mappings/energy_balance_to_nace2_mapping.csv`
- `data/exiobase_concordance_sources/table_correspondance_nace-gics-exiobase_20230816.xlsx`

Important distinction:

- Eurostat CPA and NACE classifications are official standards.
- Eurostat NACE classifies economic activities, whereas Eurostat CPA classifies products by activity.
- The repo's EXIOBASE-to-Eurostat bridge is not an official Eurostat or EXIOBASE
  standard. It is a weighted concordance assembled in this repo, using a mix of
  retained semi-official source material plus deterministic fallback rules.
- The checked-in `scripts/mappings/energy_balance_to_nace2_mapping.csv` file keeps a legacy filename, but its operational targets are Eurostat IO `CPA_65` product rows plus households.

### 4. Austria-side energy and emissions workbooks

These are read from the PostGrowth repo:

- `Energy_Balances_2021_Eurostat.xlsx`
- `Emission_Factors_for_Cross_Sector_Tools_V2.0_0.xlsx`

## How To Run

Run from the MRIO repository root:

```bash
python scripts/diagnostics/generate_at2020_flow_mapping_details.py
python scripts/diagnostics/generate_at2021_co2_mapping_detail.py
python scripts/diagnostics/generate_at2020_strict_zij_outliers.py
python scripts/diagnostics/generate_at2020_strict_zij_outliers.py --flow all
```

Optional arguments:

```bash
python scripts/diagnostics/generate_at2020_flow_mapping_details.py \
  --geo AT \
  --region AT \
  --year 2020 \
  --output-dir /custom/output/path

python scripts/diagnostics/generate_at2021_co2_mapping_detail.py \
  --geo AT \
  --region AT \
  --year 2021 \
  --exio-year 2020 \
  --postgrowth-root /home/agilra/EcolEconModels/PostGrowth-SFC-Austria \
  --output-dir /custom/output/path

python scripts/diagnostics/generate_at2020_strict_zij_outliers.py \
  --geo AT \
  --region AT \
  --year 2020 \
  --flow dom \
  --material-threshold 10 \
  --output-dir /custom/output/path

python scripts/diagnostics/generate_at2020_strict_zij_outliers.py \
  --geo AT \
  --region AT \
  --year 2020 \
  --flow all \
  --material-threshold 10
```

## Method Notes

### 2020 flow detail files

Each row is one weighted EXIO source-to-Eurostat target allocation. The output
contains:

- the EXIO source sector value
- the split value after applying the concordance weight
- the Eurostat target-sector total
- the mapped EXIO aggregate for that Eurostat target
- the value gap and gap percentage
- provenance, certainty, and notes from the concordance row

For these 2020 flow files, the Eurostat reference is built as product use
within Austria from atomic final-demand columns only, with `P6` exports
excluded. This matches the EXIO comparator more closely, because the EXIO slice
used here is an Austria-destination use block, not Austrian total product
output including exports, and the Eurostat `prd_use` table also contains
subtotal columns such as `P3`, `P5`, `TFU`, `TOTAL`, and `TU` that would
otherwise double count final demand.

The totals row avoids double counting by:

- summing source totals over unique EXIO source codes
- summing Eurostat and mapped target totals over unique Eurostat sectors

### 2021 CO2 detail file

The 2021 CO2 file compares three sector views:

- Eurostat energy-balance-based emissions
- mapped EXIO-derived sector characteristics
- official Eurostat air-emissions totals by NACE Rev. 2

Implemented unit handling:

- EXIO energy is converted to GWh by dividing by `3.6`
- EXIO CO2 is converted to tonnes by dividing by `1000`

Implemented output definitions:

- Eurostat gross output is `TU`
- Eurostat domestic output is computed as:

  `x_dom = (I - Z / TU)^-1 * (P3 + P5)`

The comparison metric used for the MRIO-side emissions benchmark is:

- `mrio_domestic_output_emissions_tco2_2021`

This uses mapped EXIO CO2 intensity on Eurostat domestic-demand-induced output,
not on gross output.

### Strict `z_ij` outlier debug files

The strict `z_ij` generator is a debug aid, not a reviewer-facing output. It:

- keeps only strict one-to-one sectors where one Eurostat target maps to one
  EXIO source and vice versa
- compares Austria intermediate-use cells `Z_ij` directly for the selected flow
- filters the cell file to materially sized Eurostat cells using the configured
  threshold
- writes a second file with row-total mismatches for the same strict subset

## Interpretation Guardrails

- These files are designed for diagnostics and reviewer transparency, not as a
  claim that Eurostat IO sectors and EXIOBASE sectors should match exactly.
- Large gaps can remain even with a better mapping because the underlying systems
  differ in balancing, sector construction, import treatment, and national versus
  global estimation logic.
- A better official-standard bridge exists for CPA and NACE classifications, not
  for direct EXIOBASE-to-Eurostat sector matching.
