# EXIOBASE concordance compatibility mappings

The canonical EXIOBASE-to-Eurostat concordance source is generated from the
EXIOBASE/XIO concordance archive through the A64 builder. The compatibility
converter in `scripts/mappings/build_exiobase_concordance_compat_mappings.py`
then converts the generated collapsed A64 table into the current MRIO mapping CSV
schema used by the existing disaggregation and audit scripts.

## Inputs

The converter expects the collapsed A64 file at:

```text
data/exiobase_concordance_sources/derived/exiobase_eurostat_iot_a64/exiobase3_products_to_eurostat_iot_a64_collapsed.csv
```

That file is generated from the EXIOBASE concordance chain:

```text
NACE Rev. 2 class -> EXIOBASE2 product -> EXIOBASE3 product -> Eurostat IOT A*64
```

## Outputs

The converter reads labels from the committed CPA_64 and CPA_65 dictionaries; it does not write dictionary files.


The converter reads the committed CPA_64/CPA_65 target dictionaries and writes local generated outputs to:

```text
scripts/mappings/generated/
```

The two mapping projections are:

```text
exio_to_eurostat_cpa64_products_from_exiobase_concordances.csv
exio_to_eurostat_cpa65_products_from_exiobase_concordances.csv
```

The CPA_64 output is the canonical 64-sector version and keeps `CPA_L68` as one
real-estate sector. The CPA_65 output is a compatibility projection for older
workflows: it splits only `CPA_L68` into `CPA_L68A` and `CPA_L68B`.

## Runtime weights

The generated compatibility mappings do not store static split weights. For
one-to-many source mappings, rows use:

```text
allocation_rule=split-by-output-share
```

The existing mapping API can resolve these rows from a supplied Eurostat target
output/value reference. This keeps the official concordance relational and lets
the selected country/year/flow determine numerical split weights.

## Licensing note

Generated CSVs are ignored by default. They derive from EXIOBASE/XIO concordance
sources, so redistribution should be checked against the upstream terms before
public release.
