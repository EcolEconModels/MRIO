# Bridge-NSI and common-intermediate Energy Balance mappings

This patch adds two generator scripts for the Route-B workflow in which Eurostat
Energy Balance quantities/prices remain the numerical driver, while EXIOBASE
concordances provide the mapping topology.

## Scripts

1. `scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_bridge_nsi.py`

   Reads `Bridge_NSI_AT.xlsx` from the local concordance ZIP and writes a
   current-schema EXIOBASE3 product -> Eurostat CPA weighted mapping.  The
   default product bridge sheet is `AT_r_sup`.  The generated rows do not store
   static split weights; many-target EXIO products use
   `allocation_rule=split-by-output-share` so the existing mapping resolver can
   compute weights from the selected Eurostat country/year/flow target-side
   value reference.

2. `scripts/mappings/compose_mapping_via_common_intermediate.py`

   Composes two mappings through a common intermediate.  The default Route-B use
   is:

   ```text
   Eurostat Energy Balance row -> EXIOBASE3 product -> Eurostat CPA target
   ```

   The output is a direct canonical `category-target` mapping.  The existing
   Energy Balance mapping code can consume this direct EB -> CPA map without
   carrying EXIOBASE through the runtime numerical pipeline.  The intermediate
   edge table is written separately for audit.

## Mathematics

Let `s` index Energy Balance source rows, `e` index common intermediate products
(EXIOBASE3 in the default use), and `c` index Eurostat CPA targets.  The two
input relations are

```text
L_{s,e} in {0,1}
R_{e,c} in {0,1}
```

The composed topology is

```text
T_{s,c} = 1[ sum_e L_{s,e} R_{e,c} > 0 ].
```

For each source row `s`, the direct generator emits all `c` with `T_{s,c}=1`.
If there is more than one target, it marks the row with
`allocation_rule=split-by-energy-product-prd-use`.  The existing runtime resolver
then computes weights from the selected Eurostat target-side value reference
`v_c`:

```text
w_{s,c} = v_c / sum_{c': T_{s,c'}=1} v_{c'}.
```

The Energy Balance quantity/price side should remain separate.  If a source row
has a resolved quantity `q_s` and price `p_s`, its monetary value is

```text
m_s = q_s p_s.
```

The mapped value assigned to target `c` is then

```text
m_{s,c} = m_s w_{s,c}.
```

Economically, this is valid as a relational concordance plus runtime allocation
scheme when the intermediate relation is used to define admissible targets and
Eurostat country/year/flow values resolve many-many splits.  It should not be
interpreted as an empirical EXIOBASE weight unless an explicit empirical weight
column is later added and documented.

## Example commands

Build the bridge-derived EXIOBASE3 -> CPA65 mapping:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_bridge_nsi.py \
  --target-version cpa65
```

Compose EB -> EXIOBASE3 and EXIOBASE3 -> CPA65 into a direct EB -> CPA map:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/compose_mapping_via_common_intermediate.py \
  --source-balance-role-filter final_use \
  --include-direct-energy-only
```

Use the generated direct map with the existing direct Energy Balance workbook
pipeline:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/build_energy_balances_65sectors_primary_sources.py \
  --mapping scripts/mappings/generated/energy_balance_to_cpa_from_common_intermediate/energy_balance_to_cpa_products_from_common_intermediate.csv \
  --with-diagnostics
```

The `energy_balance_pxp` split/reaggregation pipeline is more restrictive: its
current split plan is a technology/product split for selected producer rows.  The
direct EB -> CPA map generated here is therefore an input to the existing Energy
Balance mapping APIs, not a silent replacement for technology-specific SIEC and
price selector recipes.

For the direct EB workbook path, filter to `final_use` rows and direct-energy rows.
The workbook builder constructs a final-use-by-sector table; transformation inputs,
transformation outputs, and energy-sector own-use rows should not be mixed into
that direct final-use workbook map.
