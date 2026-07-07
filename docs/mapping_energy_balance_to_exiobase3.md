# Eurostat Energy Balances to EXIOBASE3 concordance generator

This document describes the independent generator for Eurostat Energy Balance row to EXIOBASE3 product links.

The generator is intentionally separate from the repo-authored manual Energy-Balance-to-CPA mapping. It does not read that file. Its inputs are:

- `scripts/mappings/energy_balance_sector_rules_for_exiobase3.csv`
- `data/exiobase_concordance_sources/exiobase_concordances.zip`

The generated chain is:

```text
Eurostat Energy Balance row
  -> definition-level NACE Rev. 2 selector or explicit non-production diagnostic
  -> EXIOBASE2 product, via NACE2_EXIOBASE20p_list.csv in the ZIP
  -> EXIOBASE3 product, via EXIOBASE2 products comparison with EXIOBASE3 products.xlsx in the ZIP
```

## Why this is a generator, not a manual final CSV

The EXIOBASE concordance archive does not contain a direct current Eurostat `nrg_bal` row to EXIOBASE3 sector map. The small rule CSV therefore records only Energy Balance source definitions as NACE selectors or explicit non-production cases. The generator performs the EXIO joins from the ZIP and writes reproducible audit files.

Generated links are relational and unweighted. They should not be treated as empirical allocation shares. Rows with broad residual definitions, households, or process semantics must be reviewed in the diagnostics before numerical use.

## Build command

From the MRIO repository root, after placing the EXIOBASE concordance archive at the expected local path:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/build_energy_balance_to_exiobase3_mapping.py \
  --concordances-zip data/exiobase_concordance_sources/exiobase_concordances.zip \
  --rules-csv scripts/mappings/energy_balance_sector_rules_for_exiobase3.csv \
  --out-dir data/exiobase_concordance_sources/derived/eurostat_energy_balance_to_exiobase3
```

Use `--check-only` to build in memory and print coverage without writing generated outputs.

## Generated outputs

The default output directory is ignored because it is derived from external EXIOBASE concordance material:

```text
data/exiobase_concordance_sources/derived/eurostat_energy_balance_to_exiobase3/
```

The generator writes:

- `eb_to_exio3_products_long.csv`
- `eb_to_exio3_products_collapsed.csv`
- `eb_unmatched_or_ambiguous.csv`
- `manifest.json`
- `README.md`

To compose this output with an EXIOBASE3 -> CPA map, see
`docs/mapping_energy_balance_to_cpa_via_exiobase3.md`.

## Caveats

- This is a product-level EXIOBASE3 `pxp` candidate concordance, not an industry-technology model.
- Household final use is recorded as a non-production diagnostic instead of being forced into an EXIO production sector.
- Process rows such as refinery own use and electricity or heat generation are represented by activity selectors. Technology/source splits are not inferred by this concordance.
- Runtime numerical allocation should use documented country/year reference data. Do not use equal splits as empirical shares.
