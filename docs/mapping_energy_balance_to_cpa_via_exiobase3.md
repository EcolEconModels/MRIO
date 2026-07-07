# Eurostat Energy Balance to CPA via EXIOBASE3

This route composes two independent mappings:

```text
Eurostat Energy Balance row -> EXIOBASE3 product -> CPA64/CPA65 target
```

It is different from the EXIOBASE3 -> CPA official-concordance generator. The
left-hand side is an Energy Balance row map built from Energy Balance source
definitions and EXIOBASE concordances; the right-hand side is a selected
EXIOBASE3 -> Eurostat CPA map.

The direct EB -> CPA output is a current-schema `category-target` mapping. It is
suitable for the direct Energy Balance workbook path when generated with the
final-use/direct-energy filters, but it is not a silent replacement for the
technology-specific `energy_balance_pxp` split/reaggregation recipes.

## Inputs

Build the left-hand EB -> EXIOBASE3 map first:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/build_energy_balance_to_exiobase3_mapping.py \
  --concordances-zip data/exiobase_concordance_sources/exiobase_concordances.zip \
  --rules-csv scripts/mappings/energy_balance_sector_rules_for_exiobase3.csv \
  --out-dir data/exiobase_concordance_sources/derived/eurostat_energy_balance_to_exiobase3
```

Build or select the right-hand EXIOBASE3 -> CPA map separately. For the
official-concordance route, see
`docs/mapping_exiobase_eurostat_using_official_concordances.md`.

## Compose command

Compose EB -> EXIOBASE3 and EXIOBASE3 -> CPA65 into a direct final-use EB -> CPA
map for the direct workbook path:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/compose_mapping_via_common_intermediate.py \
  --left-csv data/exiobase_concordance_sources/derived/eurostat_energy_balance_to_exiobase3/eb_to_exio3_products_collapsed.csv \
  --right-csv scripts/mappings/generated/official_concordances/exio_to_eurostat_cpa65_products_from_official_concordances_direct_plus_curated_gaps.csv \
  --out-csv scripts/mappings/generated/compare_routes/official_concordances/energy_balance_to_cpa65_products.csv \
  --edge-out-csv scripts/mappings/generated/compare_routes/official_concordances/energy_balance_to_cpa65_products_edges.csv \
  --diagnostics-csv scripts/mappings/generated/compare_routes/official_concordances/energy_balance_to_cpa65_products_diagnostics.csv \
  --manifest-json scripts/mappings/generated/compare_routes/official_concordances/energy_balance_to_cpa65_products_manifest.json \
  --target-version CPA_65 \
  --source-balance-role-filter final_use \
  --include-direct-energy-only
```

The composed output inherits the economic limits of both inputs. In particular,
unweighted relational links are topology, not empirical allocation shares, and
unresolved diagnostics in either stage should remain visible in the manifest and
diagnostics CSV.
