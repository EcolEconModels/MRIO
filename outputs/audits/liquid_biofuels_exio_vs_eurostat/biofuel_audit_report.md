# Liquid biofuels audit: EXIOBASE vs Eurostat

## Executive summary

- Main finding: **strong red flag**. EXIOBASE Austria 2020 PXP places 26.580 MEUR of `p24.g` Biogasoline into `p51` wholesale, while Eurostat Austria 2020 `CPA_C20 -> CPA_G46` is 11.450 MEUR.
- EXIO `p24.g -> p51` exceeds Eurostat `CPA_C20 -> CPA_G46`: **True**. EXIO liquid biofuels total `p24.g/h/i -> p51` is 29.000 MEUR and also exceeds the Eurostat parent cell: **True**.
- Scope/year/valuation do not explain the central red flag: the central comparison is AT, 2020, total flow, product-by-product IOT, million EUR. DOM/IMP and 2021 sensitivity rows are written separately in `biofuel_exio_vs_eurostat_parent_check.csv`.
- Recommended treatment: For transition-direction analysis, p24.g/h/i should be classified statistically under CPA_C20 and analytically as ENERGY_GREEN where appropriate, but their EXIOBASE monetary PXP destination columns should not be interpreted as physical sectoral combustion. Physical use/emissions should be handled via a separate Eurostat Energy Balance satellite account.

## Data and comparability

- EXIOBASE: country/region `AT`, year `2020`, table type `pxp`, flow scopes `domestic`, `imports`, and `total`.
- Eurostat: `naio_10_cp1700`, years `2020` and `2021`, `stk_flow` values `DOM`, `IMP`, and `TOTAL`; row dimension `prd_ava`, column dimension `prd_use`.
- Unit: million EUR for EXIO and Eurostat IOT checks. Energy-balance plausibility uses GWh and is qualitative, not converted to monetary terms.
- Valuation: Eurostat product-by-product IOT is treated as basic-price product-by-product data. EXIO values are current-price monetary IOT flows from cached EXIOBASE PXP slices.
- Row/column mappings: EXIO `p24.g`, `p24.h`, and `p24.i` map statistically to `CPA_C20`; EXIO `p51` maps to `CPA_G46`.

## EXIOBASE Z/Y allocation

- `p24.g` Biogasoline total use in the AT 2020 total-use slice is 134.615 MEUR: intermediate use 129.648 MEUR and final demand 4.966 MEUR.
- `p24.g -> p51` wholesale is 26.580 MEUR, equal to 19.7% of total biogasoline use and 20.5% of intermediate use.
- Transport-like EXIO destination columns receive 0.425 MEUR of `p24.g`, or 0.33% of intermediate use.
- The top-destination table is written to `biofuel_exio_top_destinations.csv`.

## EXIO C20 parent consistency check

- EXIO products mapped to `CPA_C20`: 10.
- EXIO `CPA_C20`-mapped products to `p51`: 149.138 MEUR.
- Liquid biofuels to `p51`: 29.000 MEUR, 19.4% of EXIO's own C20-to-wholesale parent aggregate.
- Biogasoline alone is 17.8% of EXIO's own C20-to-wholesale parent aggregate.

## EXIO vs Eurostat CPA_C20->G46 check

- Central ratio `EXIO C20 -> p51 / Eurostat C20 -> G46`: 13.03.
- Central ratio `EXIO p24.g -> p51 / Eurostat C20 -> G46`: 2.32.
- Central ratio `EXIO p24.g/h/i -> p51 / Eurostat C20 -> G46`: 2.53.
- Decision rule result: **strong red flag**, because a subset product (`p24.g`) and the liquid-biofuel subset (`p24.g/h/i`) exceed the Eurostat parent aggregate after the like-for-like 2020 total-flow check.

## PXP vs IXI / SUT check

- PXP shows the high `p24.g -> p51` allocation.
- IXI comparison: unavailable in this local cache. No `IOT_2020_ixi.zip` or cached `slices/AT/ixi` object was found, so the audit cannot distinguish a PXP transformation artefact from a deeper EXIO allocation issue using IXI.
- Underlying SUT/product-by-industry check: no EXIO SUT or use-table object was found in the repo/data cache. The decisive upstream object is not available in this repo.
- Decision rule result: **not resolved**. Even in this unresolved case, EXIO `p24.g` destination columns should not be treated as physical combustion allocation.

## Energy-balance plausibility

- Eurostat `nrg_bal_c` Austria 2020 liquid-biofuel final-use rows show road transport receives 94.1% of liquid-biofuel final energy use across the selected final-use categories.
- This is a physical energy-balance concept, not a monetary PXP concept. The qualitative mismatch is nevertheless material: physical liquid-biofuel use is concentrated in road transport, while EXIO PXP biogasoline is concentrated in non-transport monetary destination columns including wholesale, R&D, and real estate.
- If EXIO is used as an embodied-input monetary trace, this can be defensible as a monetary construction. If it is used as direct combustion allocation, the mismatch invalidates that interpretation.

## Interpretation

- Wholesale and retail output are trade-margin services, not the full value of goods resold.
- Goods purchased for resale should not be read as wholesale/retail intermediate consumption in the same way as own fuel use.
- If `Z[p24.g,p51]` is interpreted literally, it means wholesale's own intermediate use of biogasoline, not biogasoline being resold to households.
- Therefore, a large `p24.g -> p51` cell is credible as sectoral fuel use only if wholesale plausibly burns that fuel itself. The available physical energy-balance evidence does not support wholesale as the major direct user.
- This audit does not claim EXIOBASE violated SNA/ESA accounting. The upstream SUT/use table needed to prove that is unavailable.

## Recommendation

- Quarantine EXIO liquid-biofuel PXP destination columns for direct combustion/emissions allocation.
- Use EXIO liquid-biofuel monetary PXP data only as an embodied-input allocation unless separately validated from upstream EXIO SUT/use-table evidence.
- For transition-direction analysis, p24.g/h/i should be classified statistically under CPA_C20 and analytically as ENERGY_GREEN where appropriate, but their EXIOBASE monetary PXP destination columns should not be interpreted as physical sectoral combustion. Physical use/emissions should be handled via a separate Eurostat Energy Balance satellite account.

## Output files

- `biofuel_exio_z_y_summary.csv`
- `biofuel_exio_top_destinations.csv`
- `biofuel_exio_c20_parent_check.csv`
- `biofuel_exio_vs_eurostat_parent_check.csv`
- `biofuel_pxp_vs_ixi_check.csv`
- `biofuel_year_scope_valuation_check.md`
- `biofuel_timeseries_country_sanity.csv`
- `biofuel_energy_balance_plausibility.csv`
