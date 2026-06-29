# Official-concordance EXIOBASE and Energy Balance mappings

This workflow keeps Eurostat Energy Balance quantities/prices as the numerical
driver while using source concordance files only for topology.  The EXIOBASE raw
concordance ZIP remains uncommitted, but the generator now reads only files under
`data/exiobase_concordance_sources/` and `data/eurostat_concordance_sources/`.

## EXIOBASE3 -> CPA64/CPA65 generator

`scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_bridge_nsi.py`
now builds the EXIOBASE3 product -> Eurostat CPA target map from this source
chain:

```text
EXIOBASE20p_ESA59p.txt
  EXIOBASE2 product -> ESA59 old product group / old CPA2002-style division
EXIOBASE2 products comparison with EXIOBASE3 products.xlsx
  EXIOBASE2 product -> EXIOBASE3 product
RAMON CPA 2002 -> CPA 2008 correspondence
RAMON CPA 2008 -> CPA 2.1 correspondence
  old CPA division -> current CPA2.1 division
repo CPA64/CPA65 target dictionaries
  CPA2.1 two-digit division -> target aggregate
```

There is no hard-coded `NSI_PRODUCT_TO_CPA64` table.  CPA64/CPA65 aggregation is
derived from two-digit CPA2.1 divisions and the committed target dictionaries.
For CPA65, division 68 is deliberately emitted to both real-estate split targets
when the CPA65 dictionary contains `CPA_L68A` and `CPA_L68B`; runtime target-side
output/use values must resolve the split.

## Route-B composition

The common-intermediate composer remains the second stage:

```text
Eurostat Energy Balance row -> EXIOBASE3 product -> CPA64/CPA65 target
```

The direct EB -> CPA output is a current-schema `category-target` mapping.  It is
suitable for the direct Energy Balance workbook path when generated with the
final-use/direct-energy filters, but it is not a silent replacement for the
technology-specific `energy_balance_pxp` split/reaggregation recipes.

## Example commands

Build the official-concordance EXIOBASE3 -> CPA65 mapping:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python   scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_bridge_nsi.py   --target-version cpa65   --out-dir scripts/mappings/generated
```

Compose EB -> EXIOBASE3 and EXIOBASE3 -> CPA65 into a direct final-use EB -> CPA
map for the direct workbook path:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python   scripts/mappings/compose_mapping_via_common_intermediate.py   --left-csv data/exiobase_concordance_sources/derived/eurostat_energy_balance_to_exiobase3/eb_to_exio3_products_collapsed.csv   --right-csv scripts/mappings/generated/exio_to_eurostat_cpa65_products_from_official_concordances.csv   --out-csv scripts/mappings/generated/energy_balance_to_cpa_from_common_intermediate/energy_balance_to_cpa65_products_from_official_concordances_final_use.csv   --edge-out-csv scripts/mappings/generated/energy_balance_to_cpa_from_common_intermediate/energy_balance_to_cpa65_products_from_official_concordances_final_use_edges.csv   --diagnostics-csv scripts/mappings/generated/energy_balance_to_cpa_from_common_intermediate/energy_balance_to_cpa65_products_from_official_concordances_final_use_diagnostics.csv   --manifest-json scripts/mappings/generated/energy_balance_to_cpa_from_common_intermediate/energy_balance_to_cpa65_products_from_official_concordances_final_use_manifest.json   --target-version CPA_65   --source-balance-role-filter final_use   --include-direct-energy-only
```
