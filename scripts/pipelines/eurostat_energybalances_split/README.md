# Eurostat Energy Balances pxp split pipeline maps

Place this folder in the repo at:

```text
EcolEconModels/MRIO/scripts/pipelines/eurostat_energybalances_split/
```

Files:

- `energy_balance_pxp_split.csv` — enriched disaggregation recipe. It constructs EB-derived child sectors for the only allowed split parents: `CPA_C19`, `CPA_C20`, `CPA_D`.
- `energy_balance_pxp_reaggregation.csv` — pure reaggregation map from EB child sectors to final target sectors. This should remain compatible with the existing aggregation API (`source_code -> target_code`).
- `energy_balance_pxp_input_technology.csv` — required supplier-specific technology rules for column allocation. Use these where matched and fall back to child shares for unmatched supplier rows, with diagnostics.
- `energy_balance_pxp_prices.csv` — price definitions only. No quantities or subtraction rules live here.

The advanced/conventional liquid-biofuel subtraction logic lives in `energy_balance_pxp_split.csv` using `advanced_external_quantity` and `selector_residual` rows with a shared `subsplit_group`.

Conventional liquid-biofuel prices are placeholders. Replace rows with status `placeholder_to_replace` when better values are available.
