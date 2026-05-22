# Liquid biofuels audit settings

- country / region: AT
- EXIOBASE year: 2020
- Eurostat years: 2020, 2021
- EXIO table type: pxp
- EXIO flow scopes: domestic, imports, total when cached slices are available
- Eurostat table: product-by-product IOT `naio_10_cp1700`
- Eurostat stk_flow scopes: DOM, IMP, TOTAL
- valuation: Eurostat product-by-product IOT at basic prices; EXIOBASE monetary IOT values are treated as million EUR current-price monetary flows from the cached EXIOBASE table
- unit: million EUR for IOT checks; GWh for energy-balance plausibility
- row dimensions: EXIO product rows `p24.g`, `p24.h`, `p24.i`; Eurostat row `CPA_C20`
- column dimensions: EXIO product destination columns, especially `p51`; Eurostat column `CPA_G46`
- central comparison: EXIO 2020 total-use slice versus Eurostat 2020 `TOTAL`
- sensitivity checks: EXIO 2020 domestic/import scopes and Eurostat 2021 DOM/IMP/TOTAL

The audit labels any year or scope mismatch explicitly. It does not compare EXIO `total` to Eurostat `DOM` as a central result.
