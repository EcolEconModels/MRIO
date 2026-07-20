# EXIOBASE concordance source archive

<!-- BEGIN OFFICIAL DIRECT CPA64 EXIOBASE3 BUILDER -->
## Preferred direct official concordance

The committed upstream source `cpasut64_exiobase3p.csv` is the preferred direct
CPA64-to-EXIOBASE3 topology. Its `weight=1.0` values are binary incidence markers,
not allocation shares. Build both canonical runtime orientations with:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/build_exiobase3_to_cpa_from_official_direct.py
```

The builder reverses the edges to EXIOBASE3-to-CPA, normalizes upstream `CPA_L`
to the repository CPA64 code `CPA_L68`, and emits the CPA65 variant by replacing
that one target with `CPA_L68A` and `CPA_L68B`. Multi-target EXIOBASE3 products
retain no static numerical weight; the existing runtime output/use-share resolver
allocates them while enforcing source weights that sum to one.
<!-- END OFFICIAL DIRECT CPA64 EXIOBASE3 BUILDER -->

Offical EXIOBASE concordances are at:
https://github.com/EXIOBASE/00-concordances-public 
with more concordances and better organisation than the obsolete one below.

The direct Eurostat CPA64 to Exiobase3 concordance map is taken from URL below and copied in this directory:  
https://github.com/EXIOBASE/00-concordances-public/blob/main/concordances/international/cpasut64_exiobase3p.csv  

There is also a "wide" concordance map with the map as a matrix of 1s and 0s with CPA64 codes as rows and EXIOBASE3 codes as columns, in the "_wide" directory, i.e. at:  
https://github.com/EXIOBASE/00-concordances-public/blob/main/concordances/international/_wide/cpasut64_exiobase3p.csv  


## OBSOLETE SOURCE:

Since this source didn't have a direct CPA64 to Exiobase3 mapping, we needed to use a chain of concordance maps.  
This is now obsolete as we have the official direct concordance map as above!

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

## OBSOLETE -- Recreating the EXIOBASE3 to Eurostat IOT A*64 mapping

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
