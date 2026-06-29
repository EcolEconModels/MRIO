# Official-concordance EXIOBASE and Energy Balance mappings

This workflow keeps Eurostat Energy Balance quantities/prices as the numerical
driver while using source concordance files only for topology. The EXIOBASE raw
concordance ZIP remains uncommitted and is read directly; RAMON/Eurostat
correspondence files are committed under `data/eurostat_concordance_sources/`.

## EXIOBASE3 -> CPA64/CPA65 generator

`scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_bridge_nsi.py`
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
repo CPA64/CPA65 target dictionaries
  CPA2.1 two-digit division -> target aggregate
```

No Layer-2 normalized concordance files are written as generated outputs by default: matrix-style and workbook-style source concordances are parsed internally and discarded after the Layer-3 EXIOBASE3 -> CPA mapping is built. This avoids accidentally reusing intermediate parse artifacts as source truth. For local debugging, an optional `--edge-audit-csv` can be pointed under `tmp/`, but that audit file is not part of the committed generated mapping contract.

There is no hard-coded `NSI_PRODUCT_TO_CPA64` table. CPA64/CPA65 aggregation is
derived from two-digit CPA2.1 divisions and the committed target dictionaries.
For CPA65, division 68 is deliberately emitted to both real-estate split targets
when the CPA65 dictionary contains `CPA_L68A` and `CPA_L68B`; runtime target-side
output/use values must resolve the split.

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
  scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_bridge_nsi.py \
  --concordances-zip data/exiobase_concordance_sources/exiobase_concordances.zip \
  --cpa2002-cpa2008 data/eurostat_concordance_sources/CPA_2002_to_CPA_2008.csv \
  --cpa2008-cpa21 data/eurostat_concordance_sources/CPA_2008_to_CPA_21.csv \
  --target-version cpa65 \
  --out-dir scripts/mappings/generated/official_concordances
```

This writes:

```text
scripts/mappings/generated/official_concordances/exio_to_eurostat_cpa65_products_from_official_concordances.csv
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
