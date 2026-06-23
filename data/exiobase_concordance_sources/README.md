# EXIOBASE concordance source archive

Download the EXIOBASE/XIO concordance bundle from the EXIOBASE concordance
folder used for this project:

```text
https://xio.app.box.com/s/4jowwu1vb70sq7k3rrwdq05v6z07filh/folder/355593503896
```

Rename the full-folder download to `exiobase_concordances.zip` and place it at:

```text
data/exiobase_concordance_sources/exiobase_concordances.zip
```

This zip file is intentionally ignored by git. It contains external EXIOBASE
concordance material and should not be redistributed in this repository unless
the EXIOBASE licence explicitly permits it. This directory is for local
reproducibility sources, not for vendoring upstream EXIOBASE files.

The mapping builder reads the following files inside the archive:

```text
Concordances/Archive/International classifications/NACE2_EXIOBASE20p_list.csv
Concordances/EXIOBASE/EXIOBASE2 products comparison with EXIOBASE3 products.xlsx
```

The generated mapping is an unweighted relational concordance. Equal-split
weights, where written, are placeholders for downstream users who explicitly
choose such an assumption; they are not empirical allocation shares.

<!-- BEGIN EXIOBASE A64 AUDIT TRAIL -->

## Recreating the EXIOBASE3 to Eurostat IOT A*64 mapping

The raw EXIOBASE concordance archive is a licensed external source and should not
be committed to this repository. Download the concordance folder from the
EXIOBASE Box share used for the paper audit trail, archive it as
`exiobase_concordances.zip`, and place it at:

```text
data/exiobase_concordance_sources/exiobase_concordances.zip
```

The mapping builder is:

```text
scripts/mappings/build_exiobase3_to_eurostat_iot_a64_mapping.py
```

Run the builder from the MRIO repository root with:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/build_exiobase3_to_eurostat_iot_a64_mapping.py \
  --concordances-zip data/exiobase_concordance_sources/exiobase_concordances.zip \
  --out-dir data/exiobase_concordance_sources/derived/exiobase_eurostat_iot_a64
```

The generated derived files are written under:

```text
data/exiobase_concordance_sources/derived/exiobase_eurostat_iot_a64
```

These derived outputs are reproducible audit artifacts, but they remain ignored
by default because they are produced from EXIOBASE concordance inputs. Commit the
script, documentation, tests, and licence/provenance notices; do not commit the
raw concordance archive unless EXIOBASE explicitly permits redistribution.

<!-- END EXIOBASE A64 AUDIT TRAIL -->
