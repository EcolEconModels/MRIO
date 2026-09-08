# LicenseRef-EXIOBASE-concordance-derived-NOTICE

This repository may contain scripts that generate mapping tables derived from
EXIOBASE/XIO concordance sources. The generated mapping CSVs are not committed
by default.

Expected upstream source archive location:

```text
data/exiobase_concordance_sources/exiobase_concordances.zip
```

Expected generated compatibility directory:

```text
scripts/mappings/generated/
```

Before redistributing raw EXIOBASE concordance files, extracted spreadsheets, or
derived concordance tables, check the applicable EXIOBASE/XIO and Zenodo licence
terms. This notice is an audit-trail pointer, not legal advice and not a
replacement for upstream licence terms.

## Committed EXIOBASE3rx regional bridge

The following committed mapping is a documented exception with an identified
redistribution licence:

`src/ecolecon_mrio/mappings/exiobase3rx_to_exiobase3_regions.csv`

Source attribution: Bjelle, E.L., Többen, J., Stadler, K. et al. (2020), "Adding
country resolution to EXIOBASE: impacts on land use embodied in trade", Additional
file 2, `40008_2020_182_MOESM2_ESM.xlsx`, DOI `10.1186/s40008-020-0182-y`.
The article and its supplementary information are distributed under the Creative
Commons Attribution 4.0 International licence (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/ .

Local derivative/change notice: the committed four-column CSV is an
extracted/reformatted machine-readable derivative of the published 214-by-49
`Countries` regional concordance. It preserves the published source-country and
target-membership assignments and unit weights; the recovery audit retained the
available ISO3 annotations, with Netherlands Antilles and Zanzibar deliberately
left blank rather than assigned invented modern codes. The exact committed bytes
are pinned by SHA-256
`92e9b95d240b338c0a6a8c06f2c4768149d81092839ef4581d780a1ce119f972`.

Reuse or redistribution of this derived CSV must preserve appropriate attribution,
a reference or link to CC BY 4.0, and an indication that the CSV is an
extracted/reformatted derivative. This notice does not relicense unrelated
EXIOBASE/XIO material under CC BY 4.0.
