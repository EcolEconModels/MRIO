# Sector Mapping in MRIO

## Purpose

The source of truth for sector mapping in this repo is now the annotated CSV files in `scripts/mappings/`.

Legacy ad hoc mapping tables such as bare `Category,TargetSector` CSVs are no longer a supported runtime contract.

Each mapping row is both:

- machine-readable for runtime use
- human-readable through explicit `notes`, `official_source`, `provenance`, `certainty`, `source_code`, `source_label`, `target_code`, and `target_label` columns

This document intentionally does not repeat row tables. The detailed explanation for each mapping choice lives in the CSV itself.

## Canonical Mapping CSVs

These are the only mapping CSVs that should be used by code and by humans:

- `scripts/mappings/exio_to_eurostat_cpa.csv`
  - weighted EXIOBASE 3 `pxp` product to Eurostat CPA product mapping
- `scripts/mappings/exio_energy_split_cpa_with_electricity_source.csv`
  - EXIO sector split template for `CPA_C19` and `CPA_D`, preserving electricity by source
- `scripts/mappings/exio_energy_split_cpa_no_electricity_source.csv`
  - same split template, but electricity rows are grouped
- `scripts/mappings/energy_balance_to_nace2_mapping.csv`
  - Eurostat energy-balance category to Eurostat IO target-sector bridge
- `scripts/mappings/energy_green_brown_2sector_from_split_subsectors.csv`
  - analytical aggregation from split sectors to the repo’s 2-sector energy view

There should be no parallel `.mapping.csv` duplicates and no separate mapping tables in the markdown documentation.

## CSV Schema

All canonical mapping CSVs use the same base schema defined in `src/ecolecon_mrio/mapping_csv.py`:

- `kind`
- `source_dataset`, `source_system`, `source_version`, `source_level`
- `target_dataset`, `target_system`, `target_version`, `target_level`
- `source_key`, `source_code`, `source_label`
- `target_key`, `target_code`, `target_label`
- `subtarget_key`
- `weight`
- `share`
- `enabled`
- `provenance`
- `certainty`
- `transformation_rule`
- `source_name`
- `official_source`
- `notes`

Some mapping families may also carry extra columns needed by a downstream script. For example, the split templates keep `constituent` so the split pipeline can run coverage checks.

## Meaning of the Main Fields

- `kind`
  - one of `weighted`, `split`, `category-target`, `aggregate`
- `source_key`
  - the runtime source sector/category key used by code
- `target_key`
  - the runtime target sector/category key used by code
- `subtarget_key`
  - used by `split` mappings to name the emitted subsector
- `weight`
  - used by `weighted` mappings
- `share`
  - used by `category-target` mappings
- `provenance`
  - broad classification such as `semi-official` or `methodological`
- `certainty`
  - row-level confidence / construction mode such as `semi-official-retained`, `derived-single-target`, `derived-output-share-split`, `methodological-template`
- `official_source`
  - the relevant upstream source URL or metadata page when one exists
- `notes`
  - the detailed human-readable explanation of how and why that exact row exists

## Generation

There is one supported generation path.

From the repo root:

```bash
make mappings
```

This uses the MRIO virtual environment and runs:

```bash
./.venv/bin/python scripts/generate_mapping_csvs.py --geo AT --year 2020
```

You can override the benchmark year and geo used for the generated weighted mapping:

```bash
make mappings MAP_GEO=AT MAP_YEAR=2021
```

The generator script is:

- `scripts/generate_mapping_csvs.py`

It is the only supported creation path for the canonical mapping CSVs. When writing to a custom output directory, it canonicalizes the checked-in rule files from `scripts/mappings/` and writes the resulting CSVs there; it does not expect seed mapping files to already exist in the target directory.

## Validation

Run the focused mapping test suite with:

```bash
make mappings-test
```

This runs the tests that cover:

- the canonical mapping CSV loader
- weighted mapping generation behavior
- split pipeline mapping consumption
- aggregation mapping consumption
- diagnostic concordance helpers

## Runtime Use

All runtime loading of canonical mapping CSVs now goes through:

- `src/ecolecon_mrio/mapping_csv.py`

That module is the only supported loader surface for mapping CSVs.

### Split pipeline

- `scripts/split_eurostat_energy_by_exiobase.py`

Relevant CSVs:

- `scripts/mappings/exio_energy_split_cpa_with_electricity_source.csv`
- `scripts/mappings/exio_energy_split_cpa_no_electricity_source.csv`
- optional CO2 re-mapping input: `scripts/mappings/exio_to_eurostat_cpa.csv`

### Aggregation pipeline

- `scripts/aggregate_iot_by_mapping.py`

Relevant CSV:

- `scripts/mappings/energy_green_brown_2sector_from_split_subsectors.csv`

### Diagnostics

- `scripts/diagnose_at_co2_mismatch.py`

Relevant CSVs:

- `scripts/mappings/exio_to_eurostat_cpa.csv`
- `scripts/mappings/energy_balance_to_nace2_mapping.csv`

## Provenance Policy

The repo still follows the same source priority:

1. official machine-readable classification or concordance
2. semi-official concordance or workbook when a direct official bridge is missing
3. deterministic repo-authored method when upstream does not provide a complete operational bridge

For the actual row-level provenance, use the CSV columns themselves rather than this markdown file.

Important upstream sources referenced inside the CSVs include:

- ACPR / Banque de France EXIOBASE-to-NACE workbook
  - `https://acpr.banque-france.fr/system/files/2025-03/table_correspondance_nace-gics-exiobase_20230816.xlsx`
- Eurostat correspondence tables repository
  - `https://raw.githubusercontent.com/eurostat/correspondenceTables/main/inst/extdata/NACE.csv`
  - `https://raw.githubusercontent.com/eurostat/correspondenceTables/main/inst/extdata/NACE_CPA21.csv`
- NACE Rev. 1.1 to NACE Rev. 2 correspondence PDF
  - `https://circabc.europa.eu/sd/a/acbc9240-03b9-43df-97b5-146d102fc18e/NACE%20Rev.%201.1%20-%20NACE%20Rev.%202%20-%20short%20version.pdf`
- Eurostat energy balance metadata
  - `https://ec.europa.eu/eurostat/cache/metadata/en/nrg_bal_esms.htm`

## What Was Removed

The repo no longer treats these as supported public paths:

- duplicate `.mapping.csv` files
- the old `mapping_api.py` registry/spec compatibility layer
- the old per-format CSV readers in `weighted_mapping.py`, `exio_share_split.py`, and `iot_aggregation.py`
- separate markdown tables duplicating information that now lives in the CSV rows

## Practical Rule

If you want to understand why a row exists, open the CSV.

If you want to regenerate the mappings, run `make mappings`.

If you want to load mappings in code, use `src/ecolecon_mrio/mapping_csv.py`.
