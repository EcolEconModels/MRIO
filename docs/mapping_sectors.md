# Sector Mapping in MRIO

## Purpose

The source of truth for sector mapping in this repo is split by artifact type: static concordances stay in `scripts/mappings/`, while user-editable split and reaggregation helper tables live in `scripts/pipelines/`.

Legacy ad hoc mapping tables such as bare `Category,TargetSector` CSVs are no longer a supported runtime contract.

Each mapping row is both:

- machine-readable for runtime use
- human-readable through explicit `notes`, `official_source`, `provenance`, `certainty`, `source_code`, `source_label`, `target_code`, and `target_label` columns

This document intentionally does not repeat row tables. The detailed explanation for each mapping choice lives in the CSV itself.

## Canonical Mapping CSVs

These are the mapping CSVs used by code and by humans:

- `scripts/mappings/exio_to_eurostat_cpa65_products_official_direct_reviewed.csv`
  - preferred fuels/fuels-more production EXIOBASE3 `pxp` product to Eurostat CPA65
    topology; a checked-in reviewed derivative of the official-direct incidence map
- `scripts/mappings/generated/official_direct/exio_to_eurostat_cpa65_products_from_official_direct.csv`
  - unchanged reproducible EXIO official-direct CPA65 provenance baseline
- `scripts/mappings/exio_to_eurostat_cpa_products.csv`
  - older curated/statistical C20-only route retained by the unchanged green/brown
    wrapper and available for explicit comparison and diagnostics
- `scripts/mappings/exio_energy_split_cpa_with_electricity_source.csv`
  - EXIO sector split template for `CPA_C19` and `CPA_D`, preserving electricity by source
- `scripts/mappings/exio_energy_split_cpa_no_electricity_source.csv`
  - same split template, but electricity rows are grouped
- `scripts/mappings/energy_balance_to_cpa65_products_mapping.csv`
  - legacy filename retained for compatibility; the bridge itself maps Eurostat energy-balance rows, whose official row definitions are often activity-based, to the repo's 65-product CPA-based IO targets plus `Households`, including explicit direct-combustion transformation rows for `CPA_C19` and `CPA_D`
- `scripts/mappings/eurostat_cpa64_to_cpa21_divisions.csv`
  - curated machine-readable bridge from the repo's CPA_64 IO target dictionary to CPA 2.1 two-digit divisions
- `scripts/mappings/eurostat_cpa65_to_cpa21_divisions.csv`
  - curated machine-readable bridge from the repo's CPA_65 compatibility target dictionary to CPA 2.1 two-digit divisions; division 68 deliberately maps to both real-estate split targets
- `scripts/pipelines/fuels_split.csv`
  - split manifest for the fuels pipeline
- `scripts/pipelines/fuels_more_split.csv`
  - expanded split manifest for the fuels-more pipeline
- `scripts/pipelines/fuels_reaggregation.csv`
  - fuels reaggregation table
- `scripts/pipelines/fuels_more_reaggregation.csv`
  - expanded fuels reaggregation table
- `scripts/pipelines/energy_green_brown_split.csv`
  - split manifest for the analytical green/brown energy view
- `scripts/pipelines/energy_green_brown_reaggregation.csv`
  - reaggregation table for the analytical green/brown energy view
The split and reaggregation helper tables are single-sourced under `scripts/pipelines/`; do not mirror them into `scripts/mappings/`.

### Design notes

- `docs/2026_06_24_fuels_split_reaggregation_edge_aware_plan.md`
  - phased design note for reaggregation coverage preflights and later edge-aware many-to-many split semantics

There should be no parallel `.mapping.csv` duplicates and no separate mapping tables in the markdown documentation.

Terminology note: under Eurostat's statistical classifications, NACE classifies economic activities, while CPA classifies products by activity. In this repo, the direct energy-balance bridge uses NACE only on the source-definition side of some correspondence logic; operationally it is a mapping into the repo's 65-product CPA-based IO targets. The retained `nace2` filename reflects legacy naming only, not the bridge's target classification.

## Classification vs analytical energy taxonomy

The preferred fuels/fuels-more production EXIO→Eurostat concordance is the
reviewed derivative of the official-direct topology. Its three explicit target
choices are mapping-data decisions, not code branches: `p23.3` — Nuclear fuel ->
`CPA_C24` — Basic metals; `p24.d` — Chemicals nec retains both `CPA_C20` —
Chemicals and chemical products and `CPA_C21` — Basic pharmaceutical products and
pharmaceutical preparations while the remaining `p24.*` products map only to
`CPA_C20`; and `p33` — Medical, precision and optical instruments, watches and
clocks -> `CPA_C26` — Computer, electronic and optical products only. The runtime
resolver still partitions the retained multi-target `p24.d` source from the
flow-specific Eurostat target-side reference, with source weights summing to one.
The untouched official-direct file remains the provenance baseline.

The green/brown taxonomy remains a separate analytical pipeline using the older
C20-only concordance. In that taxonomy, p24.g (biogasoline), p24.h (biodiesel),
and p24.i (other liquid biofuels) map to `ENERGY_GREEN`; other C20 products remain
`CPA_C20_others`. Its concordance and reaggregation recipe are intentionally
unchanged.

Ambiguous carriers such as nuclear, biomass/waste, electricity network services, and gas distribution should be documented in the reaggregation notes rather than silently mixed into the concordance layer.

## CSV Schemas

The checked-in CSVs fall into a small number of schema families.

### 1. Weighted concordance rows

These files keep the weighted concordance shape but intentionally omit the old
key/share columns that were removed from the EXIO bridge:

- `scripts/mappings/exio_to_eurostat_cpa65_products_official_direct_reviewed.csv`
- `scripts/mappings/generated/official_direct/exio_to_eurostat_cpa65_products_from_official_direct.csv`
- `scripts/mappings/exio_to_eurostat_cpa_products.csv`

Columns:

- `order`
- `kind`
- `source_dataset`, `source_system`, `source_version`, `source_level`
- `target_dataset`, `target_system`, `target_version`, `target_level`
- `source_code`, `source_label`
- `target_code`, `target_label`
- `enabled`
- `provenance`
- `certainty`
- `allocation_rule`
- `source_name`
- `official_source`
- `notes`
- `source_balance_role`
- `target_sector_role`
- `include_in_direct_energy`

The older curated comparison file additionally retains a legacy `weight` column;
the official-direct map deliberately has no numerical-share column.

For EXIO concordances, split rows are rule-driven: the CSV carries
`allocation_rule=split-by-output-share`, while the loader derives actual shares
from the year- and flow-specific target reference at runtime. The older curated
file may carry explicit legacy `weight` cells for its single-target rows.

### 2. Full Canonical Mapping Rows

These files use the canonical mapping schema implemented in `src/ecolecon_mrio/mapping_csv.py`:

- `scripts/mappings/energy_balance_to_cpa65_products_mapping.csv`
- `scripts/mappings/exio_energy_split_cpa_with_electricity_source.csv`
- `scripts/mappings/exio_energy_split_cpa_no_electricity_source.csv`
- `scripts/mappings/energy_primary_source_buckets.csv`
- `scripts/mappings/siec_to_primary_energy_sources.csv`

Canonical columns:

- `kind`
- `source_dataset`, `source_system`, `source_version`, `source_level`
- `target_dataset`, `target_system`, `target_version`, `target_level`
- `source_code`, `source_label`
- `target_code`, `target_label`
- `subtarget_code`
- `weight`
- `share`
- `enabled`
- `provenance`
- `certainty`
- `transformation_rule`
- `source_name`
- `official_source`
- `notes`
- `source_balance_role`
- `target_sector_role`
- `include_in_direct_energy`

File-specific extras:

- `scripts/mappings/exio_energy_split_cpa_with_electricity_source.csv`
- `scripts/mappings/exio_energy_split_cpa_no_electricity_source.csv`
  - adds `constituent` so the split pipeline can trace the source subsector that was retained

### 3. Lightweight Split / Reaggregation Tables

These files are narrower helper tables used by the split/reaggregation pipelines:

- `scripts/pipelines/fuels_split.csv`
- `scripts/pipelines/fuels_more_split.csv`
- `scripts/pipelines/energy_green_brown_split.csv`
  - columns: `source_code`, `source_label`

- `scripts/pipelines/fuels_reaggregation.csv`
- `scripts/pipelines/fuels_more_reaggregation.csv`
- `scripts/pipelines/energy_green_brown_reaggregation.csv`
  - columns: `source_code`, `source_label`, `target_code`, `target_label`, `enabled`, `notes`

For these tables, `source_code` and `target_code` are the operational keys used by the loaders. The label columns are carried only as descriptive metadata.

These tables are intentionally small; they are not the same schema as the full canonical mapping files.

### 4. Dictionary Files

These are pure lookup tables for labels and descriptions:

- `scripts/mappings/exiobase_products_dictionary.csv`
- `scripts/mappings/exiobase_industries_dictionary.csv`
- `scripts/mappings/eurostat_products_dictionary.csv`
- `scripts/mappings/eurostat_industries_dictionary.csv`
- `scripts/mappings/eurostat_cpa65_products_dictionary.csv`

Columns:

- `code`
- `label`
- `description`

Some dictionary snapshots also carry `provenance_source` so the curated aggregate labels can be traced back to the detailed mined source tables.

## Main Field Meanings

The meanings below apply to the full-schema family. The reduced EXIO concordance intentionally omits `share`.

- `kind`
  - identifies the mapping family, for example `weighted`, `split`, `category-target`, `aggregate`, or `source-bucket`
- `source_code`
  - the stable sector code used as the primary key in the dictionary-backed files
- `target_code`
  - the stable target code used as the primary key in the dictionary-backed files
- `subtarget_code`
  - used by split mappings to name the emitted subsector
- `weight`
  - used by weighted mappings and some helper ordering tables
- `share`
  - used by category-target mappings
- `enabled`
  - marks whether the row is active
- `provenance`
  - broad classification such as `semi-official`, `methodological`, or `repo-authored`
- `certainty`
  - row-level confidence / construction mode such as `semi-official-retained`, `derived-single-target`, `derived-output-share-split`, `methodological-template`
- `transformation_rule`
  - the operational rule used by the loader; for the weighted EXIO concordance this now marks split rows, for example `split-by-output-share`
- `official_source`
  - the relevant upstream source URL or metadata page when one exists
- `notes`
  - the detailed human-readable explanation of how and why that exact row exists
- `source_balance_role`
  - classifies the Eurostat energy-balance row as `final_use`, `transformation_input`, `energy_sector_own_use`, `transformation_output`, or `non_energy_use`
- `target_sector_role`
  - classifies the Eurostat IO target as `end_use_sector`, `transformation_sector`, or `households`
- `include_in_direct_energy`
  - marks whether the row belongs in the repo's direct-combustion energy accounting object

For the weighted EXIO concordance, split rows are rule-driven: the CSV now carries the split rule in `transformation_rule`, while the loader derives the actual shares from the year-specific output reference at runtime.

## Generation and curated target topology

Curated mappings are checked in under `scripts/mappings/`. The preferred direct
official outputs are also checked in under `scripts/mappings/generated/official_direct/`
and reproduce byte-for-byte from their committed source and builder; normal pipeline
runs consume them rather than regenerating them.

The CPA_64 and CPA_65 top-level product dictionaries are also curated operational dictionaries.
Their `description` fields are human-readable documentation only and must not be parsed by code
to infer CPA division coverage.  Machine-readable target topology lives in the curated bridge
files `eurostat_cpa64_to_cpa21_divisions.csv` and `eurostat_cpa65_to_cpa21_divisions.csv`.

If you need to inspect or edit the canonical rows, work directly with the checked-in CSVs and keep
the dictionaries and target-division bridge files in lock-step.

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
For weighted concordances, the loader can also resolve split rows from year-specific output references when a row is marked with the split rule; that lets the same rule file adapt to the year being analyzed instead of baking one year's split shares into the CSV.

### Split pipeline

- `scripts/split_eurostat_energy_by_exiobase.py`

Relevant CSVs:

- `scripts/mappings/exio_energy_split_cpa_with_electricity_source.csv`
- `scripts/mappings/exio_energy_split_cpa_no_electricity_source.csv`
- production EXIO-to-Eurostat input:
  `scripts/mappings/generated/official_direct/exio_to_eurostat_cpa65_products_from_official_direct.csv`

### Aggregation pipeline

- `scripts/aggregate_iot_by_mapping.py`

Relevant CSV:

- `scripts/pipelines/energy_green_brown_reaggregation.csv`

### Diagnostics

- `scripts/diagnose_at_co2_mismatch.py`

Relevant CSVs:

- `scripts/mappings/exio_to_eurostat_cpa_products.csv` (explicit retained
  comparison/diagnostic route)
- `scripts/mappings/energy_balance_to_cpa65_products_mapping.csv`

For the energy-balance bridge specifically, the runtime contract is now broader than a final-use concordance. It includes final-use rows, refinery direct-use and refinery own-use rows for `CPA_C19`, main-activity electricity and heat transformation-input rows for `CPA_D`, and the `Households` analytical row. The CSV remains the single source of truth for which Eurostat energy-balance rows enter direct-combustion accounting. Despite the legacy filename, the operative targets in that CSV are the repo's 65-product CPA-based IO rows rather than NACE activity classes.

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
