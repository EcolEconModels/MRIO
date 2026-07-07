# Generated EXIOBASE mapping outputs

This directory is for locally generated current-schema mapping CSVs derived from
EXIOBASE/XIO concordance sources. Generated CSVs are ignored by default because
redistribution terms for EXIOBASE-derived concordance tables must be checked
before publication.

## Old A64/compatibility route

This route is retained for comparison and backward compatibility. Create the
source A64 concordance first:

```bash
../MRIO/.venv/bin/python scripts/mappings/build_exiobase3_to_eurostat_iot_a64_mapping.py
```

Then create compatibility mappings. The compatibility converter reads labels
from the committed CPA_64 and CPA_65 dictionaries:

```bash
../MRIO/.venv/bin/python scripts/mappings/build_exiobase_concordance_compat_mappings.py
```

Expected local outputs:

```text
scripts/mappings/generated/exio_to_eurostat_cpa64_products_from_exiobase_concordances.csv
scripts/mappings/generated/exio_to_eurostat_cpa65_products_from_exiobase_concordances.csv
scripts/mappings/generated/exiobase_concordance_compat_manifest.json
```

The CPA_64 file is the 64-sector projection and keeps a single `CPA_L68`
real-estate target. The CPA_65 file is a compatibility projection that splits
only `CPA_L68` into `CPA_L68A` and `CPA_L68B` for older workflows.
Neither file stores static split weights; rows requiring a split use
`allocation_rule=split-by-output-share` so runtime code can compute shares from
the selected Eurostat target-side value reference.

## Official-concordance route outputs

The newer EXIOBASE3 -> CPA route is generated separately from detailed
EXIOBASE2 -> CPA2002 anchors and official CPA2002 -> CPA2008 -> CPA2.1
correspondence tables:

```bash
../MRIO/.venv/bin/python \
  scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_official_concordances.py \
  --target-version cpa65 \
  --anchor-source-mode direct-plus-curated-gaps \
  --allow-diagnostics \
  --out-dir scripts/mappings/generated/official_concordances
```

The `direct-plus-curated-gaps` mode uses curated CPA2002 product anchors only
for EXIOBASE products absent from the direct CPA2002 workbook. It does not map
the unresolved `p33`/`p34` downstream CPA2002 -> CPA2008 gaps; those remain
diagnostics, so these outputs are exploratory unless and until those diagnostics
are resolved by an explicit production decision.

Compare old and new route outputs with:

```bash
../MRIO/.venv/bin/python \
  scripts/mappings/compare_exiobase3_to_cpa_mapping_routes.py \
  --target-version cpa65 \
  --new-anchor-source-mode direct-plus-curated-gaps \
  --out-dir scripts/mappings/generated/compare_routes/exiobase3_to_cpa
```
