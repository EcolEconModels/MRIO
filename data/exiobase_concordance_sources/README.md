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
