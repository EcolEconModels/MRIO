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

## Final cleanup pass

- Cleaned 15 malformed active EXIO concordance fallback notes that previously used empty-division wording. The rewritten notes now identify the source code, source label, EXIO prefix, target code, target label, and the substantive fallback rationale.
- Fossil extraction and mining-related fallback rows mapped to `CPA_B` now state that `CPA_B` is the Eurostat aggregate for mining and quarrying.
- Gas and energy-network fallback rows mapped to `CPA_D` now state that `CPA_D` is the Eurostat aggregate for electricity, gas, steam and air conditioning.
- Active Eurostat product dictionary inspection: 5,522 rows; 0 blank descriptions; 0 blank provenance values; 21 rows where `description == label`; all 21 are explicitly marked as fallback.
- Active Eurostat industry dictionary inspection: 996 rows; 0 blank descriptions; 0 blank provenance values; 21 rows where `description == label`; all 21 are explicitly marked as fallback.
- Liquid biofuels remain statistically mapped to `CPA_C20` in the EXIO-to-Eurostat concordance: p24.g Biogasoline, p24.h Biodiesels, and p24.i Other Liquid Biofuels.
- Liquid biofuels remain analytically mapped to `ENERGY_GREEN` in green/brown reaggregation.
- Other C20 rows remain outside green/brown as `CPA_C20_others`: p24.a, p24.a.w, p24.b, p24.c, p24.d, p24.e, and p24.f.
- Temporary/local checks run: dictionary inspection for products and industries; malformed-note phrase scan over `scripts/**/*.csv` and `scripts/**/*.md`; biofuel mapping grep over `scripts/mappings` and `scripts/pipelines`; existing lightweight pytest check `tests/test_mapping_integrity.py`.
