# Generated EXIOBASE concordance mappings

This directory is for locally generated current-schema mapping CSVs derived from
EXIOBASE/XIO concordance sources. Generated CSVs are ignored by default because
redistribution terms for EXIOBASE-derived concordance tables must be checked
before publication.

Create the source A64 concordance first:

```bash
../MRIO/.venv/bin/python scripts/mappings/build_exiobase3_to_eurostat_iot_a64_mapping.py
```

Then create compatibility mappings:

```bash
../MRIO/.venv/bin/python scripts/mappings/build_exiobase_concordance_compat_mappings.py
```

Expected local outputs:

```text
scripts/mappings/generated/exio_to_eurostat_iot_a64_products_from_exiobase_concordances.csv
scripts/mappings/generated/exio_to_eurostat_cpa65_products_from_exiobase_concordances.csv
scripts/mappings/generated/exiobase_concordance_compat_manifest.json
```

The A64 file is the canonical 64-sector projection and keeps a single
`CPA_L68` real-estate target. The CPA_65 file is a compatibility projection that
splits only `CPA_L68` into `CPA_L68A` and `CPA_L68B` for older workflows.
Neither file stores static split weights; rows requiring a split use
`allocation_rule=split-by-output-share` so runtime code can compute shares from
the selected Eurostat target-side value reference.
