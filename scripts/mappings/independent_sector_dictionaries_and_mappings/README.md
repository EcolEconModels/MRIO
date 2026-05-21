# Independent sector dictionaries and mappings

Generated without using any MRIO repository mapping rows. The MRIO repository was not used as a data source for labels, descriptions, or target assignments.

## Files

Dictionaries, all with columns: `code,label,description,provenance_source`:

- eurostat_products_detailed_dictionary.csv
- eurostat_products_cpa_dictionary.csv
- eurostat_industries_detailed_dictionary.csv
- eurostat_industries_nace_dictionary.csv
- exiobase_products_dictionary.csv
- exiobase_industries_dictionary.csv

Mappings, using the requested mapping-schema shape:

- eurostat_products_detailed_to_cpa.csv
- eurostat_industries_detailed_to_nace.csv
- exiobase_products_to_cpa.csv
- exiobase_industries_to_nace.csv


## Important methodological caveat

The Eurostat detailed product dictionary is a CPA class-level product-by-activity proxy at the 4-digit CPA/NACE-aligned level, because CPA is structurally aligned with NACE down to the fourth level. It is not a full six-digit CPA-CN product dictionary. Rows are labelled as class-level and provenance is explicit.

EXIOBASE mappings are independent prefix/label rules and include `certainty=derived-review-required` where a single EXIO row spans multiple Rev.2/IO64 aggregates, e.g. post+telecom, support transport+travel agency, health+social work, and cultural+sporting services.

## Counts

- Eurostat detailed product rows: 630
- Eurostat CPA aggregate rows: 64
- Eurostat detailed industry rows: 630
- Eurostat NACE aggregate rows: 64
- EXIOBASE product rows: 200
- EXIOBASE industry rows: 164
- Eurostat product detailed-to-CPA mapping rows: 630
- Eurostat industry detailed-to-NACE mapping rows: 630
- EXIOBASE product-to-CPA mapping rows: 200
- EXIOBASE industry-to-NACE mapping rows: 164
