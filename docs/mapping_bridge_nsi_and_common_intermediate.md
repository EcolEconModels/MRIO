# Official-concordance EXIOBASE and Energy Balance mappings

This workflow keeps Eurostat Energy Balance quantities/prices as the numerical
driver while using source concordance files only for topology. The EXIOBASE raw
concordance ZIP remains uncommitted and is read directly; RAMON/Eurostat
correspondence files are committed under `data/eurostat_concordance_sources/`.

## EXIOBASE3 -> CPA64/CPA65 generator

`scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_official_concordances.py`
builds the EXIOBASE3 product -> Eurostat CPA target map from this source chain:

```text
data/exiobase_concordance_sources/exiobase_concordances.zip
  Concordances/Different aggregations/EXIOBASE20p_ESA59p.txt
    EXIOBASE2 product -> ESA59 old product group / old CPA2002-style division
  Concordances/EXIOBASE/EXIOBASE2 products comparison with EXIOBASE3 products.xlsx
    EXIOBASE2 product -> EXIOBASE3 product
data/eurostat_concordance_sources/CPA_2002_to_CPA_2008.csv
  CPA 2002 -> CPA 2008
data/eurostat_concordance_sources/CPA_2008_to_CPA_21.csv
  CPA 2008 -> CPA 2.1
scripts/mappings/eurostat_cpa64_to_cpa21_divisions.csv or
scripts/mappings/eurostat_cpa65_to_cpa21_divisions.csv
  CPA2.1 two-digit division -> target aggregate
repo CPA64/CPA65 target dictionaries
  target labels and human-readable descriptions only
```

No Layer-2 normalized source-concordance files are written as generated mapping outputs: matrix-style and workbook-style source concordances are parsed internally and discarded after the Layer-3 EXIOBASE3 -> CPA mapping is built. For auditability, the internal EXIOBASE3/ESA59/CPA2.1 edge table is written under `scripts/mappings/generated/official_concordances/diagnostics/` by default, and `--edge-audit-csv` can override that path.

There is no hard-coded `NSI_PRODUCT_TO_CPA64` table. CPA64/CPA65 aggregation is
derived from two-digit CPA2.1 divisions and the committed curated target-division
bridge files, not by parsing target dictionary descriptions.  For CPA65, division
68 is deliberately emitted to both real-estate split targets in
`eurostat_cpa65_to_cpa21_divisions.csv`; runtime target-side output/use values
must resolve the split.

## Route-B composition

The common-intermediate composer remains the second stage:

```text
Eurostat Energy Balance row -> EXIOBASE3 product -> CPA64/CPA65 target
```

The direct EB -> CPA output is a current-schema `category-target` mapping. It is
suitable for the direct Energy Balance workbook path when generated with the
final-use/direct-energy filters, but it is not a silent replacement for the
technology-specific `energy_balance_pxp` split/reaggregation recipes.

## Example commands

Build the official-concordance EXIOBASE3 -> CPA65 mapping:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_official_concordances.py \
  --concordances-zip data/exiobase_concordance_sources/exiobase_concordances.zip \
  --cpa2002-cpa2008 data/eurostat_concordance_sources/CPA_2002_to_CPA_2008.csv \
  --cpa2008-cpa21 data/eurostat_concordance_sources/CPA_2008_to_CPA_21.csv \
  --target-version cpa65 \
  --target-division-map scripts/mappings/eurostat_cpa65_to_cpa21_divisions.csv \
  --out-dir scripts/mappings/generated/official_concordances
```

This writes:

```text
scripts/mappings/generated/official_concordances/exio_to_eurostat_cpa65_products_from_official_concordances.csv
scripts/mappings/generated/official_concordances/exio_to_eurostat_cpa65_products_from_official_concordances_diagnostics.csv
scripts/mappings/generated/official_concordances/exio_to_eurostat_cpa65_products_from_official_concordances_manifest.json
scripts/mappings/generated/official_concordances/diagnostics/exio_to_eurostat_cpa65_products_from_official_concordances_edges.csv
```

Compose EB -> EXIOBASE3 and EXIOBASE3 -> CPA65 into a direct final-use EB -> CPA
map for the direct workbook path:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/compose_mapping_via_common_intermediate.py \
  --left-csv data/exiobase_concordance_sources/derived/eurostat_energy_balance_to_exiobase3/eb_to_exio3_products_collapsed.csv \
  --right-csv scripts/mappings/generated/official_concordances/exio_to_eurostat_cpa65_products_from_official_concordances.csv \
  --out-csv scripts/mappings/generated/compare_routes/official_concordances/energy_balance_to_cpa65_products.csv \
  --edge-out-csv scripts/mappings/generated/compare_routes/official_concordances/energy_balance_to_cpa65_products_edges.csv \
  --diagnostics-csv scripts/mappings/generated/compare_routes/official_concordances/energy_balance_to_cpa65_products_diagnostics.csv \
  --manifest-json scripts/mappings/generated/compare_routes/official_concordances/energy_balance_to_cpa65_products_manifest.json \
  --target-version CPA_65 \
  --source-balance-role-filter final_use \
  --include-direct-energy-only
```

## Comparing old A64/compat and new official-concordance EXIOBASE -> CPA routes

`scripts/mappings/compare_exiobase3_to_cpa_mapping_routes.py` compares active
source-target pairs from the old A64/compat generated map and the new
official-concordance generated map.  It writes per-pair and per-source diffs,
plus multi-target source summaries, without declaring either route correct.

Example for CPA65:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python   scripts/mappings/compare_exiobase3_to_cpa_mapping_routes.py   --target-version cpa65   --old-csv scripts/mappings/generated/exio_to_eurostat_cpa65_products_from_exiobase_concordances.csv   --new-csv scripts/mappings/generated/official_concordances/exio_to_eurostat_cpa65_products_from_official_concordances.csv   --out-dir scripts/mappings/generated/compare_routes/exiobase3_to_cpa
```

When comparing the `direct-plus-curated-gaps` new map, pass either the generated
CSV explicitly with `--new-csv` or use `--new-anchor-source-mode
direct-plus-curated-gaps` with the default generated filename convention.
