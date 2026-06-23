# EXIOBASE3 products to Eurostat IOT A*64 concordance

This repository keeps the mapping logic and audit trail in git, while keeping
the raw EXIOBASE concordance archive out of git.

## External source file location

Download the EXIOBASE/XIO concordance archive from the project concordance
folder:

```text
https://xio.app.box.com/s/4jowwu1vb70sq7k3rrwdq05v6z07filh/folder/355593503896
```

Download the full folder, rename the zip file to `exiobase_concordances.zip`,
and place it here:

```text
data/exiobase_concordance_sources/exiobase_concordances.zip
```

The archive is ignored by `.gitignore` because it is external licence-controlled
data. Do not commit the raw zip, extracted concordance sheets, or generated
EXIOBASE-derived outputs unless redistribution has been explicitly cleared.

## Mapping chain

The builder constructs an unweighted relational concordance:

```text
NACE Rev. 2 class
  -> EXIOBASE2 product
  -> EXIOBASE3 product
  -> Eurostat IOT A*64 product/activity aggregate
```

The source files inside the archive are:

```text
Concordances/Archive/International classifications/NACE2_EXIOBASE20p_list.csv
Concordances/EXIOBASE/EXIOBASE2 products comparison with EXIOBASE3 products.xlsx
```

The A*64 target codelist is stored in:

```text
data/external/eurostat/eurostat_iot_a64_codelist.csv
```

## Build command

Run the builder from the MRIO repository root after placing the source archive
at the path above:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python   scripts/mappings/build_exiobase3_to_eurostat_iot_a64_mapping.py   --concordances-zip data/exiobase_concordance_sources/exiobase_concordances.zip   --out-dir data/derived/concordances/exiobase_eurostat_iot_a64
```

The script can also be used with its defaults once the zip is in the expected
location:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python   scripts/mappings/build_exiobase3_to_eurostat_iot_a64_mapping.py
```

## Outputs

The script writes generated files under:

```text
data/derived/concordances/exiobase_eurostat_iot_a64/
```

Generated outputs are intentionally ignored by git by default. Commit them only
if the project has explicitly decided that redistribution of the derived
concordance is permitted under the applicable EXIOBASE terms.

## Important caveats

- The mapping is unweighted. Equal-split weights are placeholders only.
- The 65-sector real-estate/imputed-rent split is not inferred from these source
  concordances. The target here is the 64-sector A*64 aggregate with one `L68`
  real-estate category.
- If this is used for numerical reallocation, add empirically justified weights
  from SUT/product output/PRODCOM or another documented source.
