# Mapping Update Audit

This note records the mapping cleanup performed in this pass.

## Updated artifacts

- `scripts/mappings/eurostat_products_dictionary.csv`
- `scripts/mappings/eurostat_industries_dictionary.csv`
- `scripts/mappings/exio_to_eurostat_cpa_products.csv`
- `scripts/mappings/exio_energy_split_cpa_with_electricity_source.csv`
- `scripts/mappings/exio_energy_split_cpa_no_electricity_source.csv`
- `scripts/pipelines/energy_green_brown_reaggregation.csv`
- `tests/test_mapping_integrity.py`
- `docs/mapping_sectors.md`

## Intentional semantic changes

- EXIO p24.g, p24.h, and p24.i are kept out of `CPA_C19` in the static energy split templates and now map to `CPA_C20` as liquid-biofuel chemistry rows.
- The analytical green/brown layer still groups p24.g, p24.h, and p24.i under `ENERGY_GREEN`.
- The Eurostat dictionary snapshots now carry `provenance_source` so the curated aggregate lookups can be traced back to the detailed source tables.
- Malformed EXIO concordance notes were normalized so the source code and code prefix are explicit again.

## Validation notes

- The new regression tests check the dictionary provenance column, the static split-template biofuel target codes, and the updated green/brown note wording.
- Existing statistical concordance semantics for p24.g, p24.h, and p24.i remain `CPA_C20`.
