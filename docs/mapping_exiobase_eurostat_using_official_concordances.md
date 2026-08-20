# EXIOBASE3 to Eurostat CPA using official concordances

<!-- BEGIN DIRECT OFFICIAL ROUTE -->
## Preferred direct official route

The preferred route now starts from the official direct file
`data/exiobase_concordance_sources/cpasut64_exiobase3p.csv`. One small builder,
`scripts/mappings/build_exiobase3_to_cpa_from_official_direct.py`, validates full
CPA64 and EXIOBASE3 coverage and writes compatible CPA64 and CPA65 canonical maps
under `scripts/mappings/generated/official_direct/`.

The official file's unit weights describe membership only. The builder therefore
does not preserve them as numerical allocation shares. It reverses the topology,
marks EXIOBASE3 products with multiple CPA targets as `split-by-output-share`, and
leaves the established flow-specific Eurostat output/use resolver unchanged. The
CPA65 output differs from CPA64 only by the repository's existing real-estate
refinement: upstream `CPA_L` is normalized to `CPA_L68`, then emitted as
`CPA_L68A` and `CPA_L68B` for CPA65.

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/build_exiobase3_to_cpa_from_official_direct.py
```

The longer chained-concordance workflow documented below is retained separately
for comparison and can be deleted later without affecting this direct route.
<!-- END DIRECT OFFICIAL ROUTE -->

This workflow builds an EXIOBASE3 product -> Eurostat CPA64/CPA65 target map
from detailed product concordances. It is the official-concordance route used to
test whether the fuels pipeline performs better than the older A64/compatibility
route.

The EXIOBASE raw concordance ZIP remains uncommitted and is read directly.
RAMON/Eurostat correspondence files are committed under
`data/eurostat_concordance_sources/`.

## Generator

`scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_official_concordances.py`
builds the EXIOBASE3 product -> Eurostat CPA target map from this source chain:

```text
data/exiobase_concordance_sources/exiobase_concordances.zip
  Concordances/Different aggregations/EXIOBASE20p_ESA59p.txt
    EXIOBASE2 product -> ESA59 old product group / old CPA2002-style division
  Concordances/EXIOBASE/EXIOBASE2 products comparison with EXIOBASE3 products.xlsx
    EXIOBASE2 product -> EXIOBASE3 product
data/eurostat_concordance_sources/CPA 2002 - CPA 2008 (CSV format).csv
  CPA 2002 -> CPA 2008
data/eurostat_concordance_sources/CPA 2008 - CPA 2.1.csv
  CPA 2008 -> CPA 2.1
scripts/mappings/eurostat_cpa64_to_cpa21_divisions.csv or
scripts/mappings/eurostat_cpa65_to_cpa21_divisions.csv
  CPA2.1 two-digit division -> target aggregate
repo CPA64/CPA65 target dictionaries
  target labels and human-readable descriptions only
```

No Layer-2 normalized source-concordance files are written as generated mapping outputs: matrix-style and workbook-style source concordances are parsed internally and discarded after the Layer-3 EXIOBASE3 -> CPA mapping is built. For auditability, the internal EXIOBASE3/ESA59/CPA2.1 edge table is written under `scripts/mappings/generated/official_concordances/diagnostics/` by default, and `--edge-audit-csv` can override that path.

There is no hard-coded `NSI_PRODUCT_TO_CPA64` table. CPA64/CPA65 aggregation is
derived from two-digit CPA2.1 divisions and the committed curated target-division
bridge files, not by parsing target dictionary descriptions.  For CPA65, division
68 is deliberately emitted to both real-estate split targets in
`eurostat_cpa65_to_cpa21_divisions.csv`; runtime target-side output/use values
must resolve the split.

The generator has two CPA2002 anchor modes:

```text
direct-only
  Uses only direct detailed EXIOBASE2 -> CPA2002 workbook anchors.
  Products absent from that workbook are reported as diagnostics.

direct-plus-curated-gaps
  Uses direct detailed workbook anchors plus the small curated gap file
  scripts/mappings/exiobase2_cpa2002_product_anchor_conciliation.csv,
  but only for EXIOBASE products absent from the direct workbook.
```

Neither mode maps the unresolved `p33`/`p34` direct workbook anchors through a
curated fallback. Those anchors are present in the direct workbook but fail
later because the official CPA2002 -> CPA2008 table has no exact or numeric
ancestor path for `33.30.90`, `34.10.25`, and `34.10.43`. Outputs that retain
those unresolved diagnostics must be written with `--allow-diagnostics` and are
marked exploratory in the manifest.

## Example commands

Build the official-concordance EXIOBASE3 -> CPA65 mapping:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_official_concordances.py \
  --concordances-zip data/exiobase_concordance_sources/exiobase_concordances.zip \
  --cpa2002-cpa2008 "data/eurostat_concordance_sources/CPA 2002 - CPA 2008 (CSV format).csv" \
  --cpa2008-cpa21 "data/eurostat_concordance_sources/CPA 2008 - CPA 2.1.csv" \
  --target-version cpa65 \
  --target-division-map scripts/mappings/eurostat_cpa65_to_cpa21_divisions.csv \
  --out-dir scripts/mappings/generated/official_concordances
```

For the current fuels-pipeline comparison candidate, use the curated direct-gap
mode and allow the three downstream `p33`/`p34` diagnostics explicitly:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_official_concordances.py \
  --target-version cpa65 \
  --anchor-source-mode direct-plus-curated-gaps \
  --allow-diagnostics \
  --out-dir scripts/mappings/generated/official_concordances
```

This writes:

```text
scripts/mappings/generated/official_concordances/exio_to_eurostat_cpa65_products_from_official_concordances_direct_plus_curated_gaps.csv
scripts/mappings/generated/official_concordances/exio_to_eurostat_cpa65_products_from_official_concordances_direct_plus_curated_gaps_diagnostics.csv
scripts/mappings/generated/official_concordances/exio_to_eurostat_cpa65_products_from_official_concordances_direct_plus_curated_gaps_manifest.json
scripts/mappings/generated/official_concordances/diagnostics/exio_to_eurostat_cpa65_products_from_official_concordances_direct_plus_curated_gaps_edges.csv
```

Energy-balance routes are separate. The EB -> EXIOBASE3 generator is documented
in `docs/mapping_energy_balance_to_exiobase3.md`; composing EB -> EXIOBASE3 with
this official EXIOBASE3 -> CPA map is documented in
`docs/mapping_energy_balance_to_cpa_via_exiobase3.md`.

## Comparing old A64/compat and new official-concordance EXIOBASE -> CPA routes

`scripts/mappings/compare_exiobase3_to_cpa_mapping_routes.py` compares active
source-target pairs from the old A64/compat generated map and the new
official-concordance generated map.  It writes per-pair and per-source diffs,
plus multi-target source summaries, without declaring either route correct.

Example for CPA65:

```bash
/home/agilra/EcolEconModels/MRIO/.venv/bin/python \
  scripts/mappings/compare_exiobase3_to_cpa_mapping_routes.py \
  --target-version cpa65 \
  --new-anchor-source-mode direct-plus-curated-gaps \
  --out-dir scripts/mappings/generated/compare_routes/exiobase3_to_cpa
```

When comparing the `direct-plus-curated-gaps` new map, pass either the generated
CSV explicitly with `--new-csv` or use `--new-anchor-source-mode
direct-plus-curated-gaps` with the default generated filename convention.

The route comparison is a structural source-target comparison. It reports pair
and target-set differences between the old A64/compat route and the new
official-concordance route, but it does not compare runtime output-share weights,
mapped Eurostat use totals, or fuel-flow results. Those are evaluated by running
the fuels pipeline with each map and comparing the emitted
`flow_specific_exio_to_eurostat_weights.csv`, `mapped_output_consistency.csv`,
and split/reaggregated matrices.

## Fuels pipeline route comparison

The fuels split/reaggregation wrapper uses `CONCORDANCE_CSV` as the
EXIOBASE3 -> Eurostat CPA weighted mapping. Keep the split and reaggregation
recipes fixed, and change only `CONCORDANCE_CSV`.

The active production default is the preferred direct official CPA65 file under
`scripts/mappings/generated/official_direct/`. The commands below document older
A64 and chained-official comparison experiments; neither is the default production
route. Runtime flow-specific target shares, not binary incidence values, partition
multi-target EXIO sources.

Old A64/compat route:

```bash
cd /home/agilra/EcolEconModels/MRIO
TEMPLATE_XLSX=/home/agilra/EcolEconModels/PostGrowth-SFC-Austria/IOT_Austria_2021_Eurostat.xlsx \
OUTPUT_SUFFIX=fuels_old_a64 \
CONCORDANCE_CSV=scripts/mappings/generated/exio_to_eurostat_cpa65_products_from_exiobase_concordances.csv \
bash scripts/pipelines/at_2021_workbook_reaggregation.sh
```

New official-concordance route:

```bash
cd /home/agilra/EcolEconModels/MRIO
TEMPLATE_XLSX=/home/agilra/EcolEconModels/PostGrowth-SFC-Austria/IOT_Austria_2021_Eurostat.xlsx \
OUTPUT_SUFFIX=fuels_official_cpa \
CONCORDANCE_CSV=scripts/mappings/generated/official_concordances/exio_to_eurostat_cpa65_products_from_official_concordances_direct_plus_curated_gaps.csv \
bash scripts/pipelines/at_2021_workbook_reaggregation.sh
```

Both commands use the same committed `scripts/pipelines/fuels_split.csv` and
`scripts/pipelines/fuels_reaggregation.csv` by default. Compare the corresponding
output trees under:

```text
outputs/AT_2021_fuels_old_a64_split_flows/
outputs/AT_2021_fuels_old_a64_reaggregated_flows/
outputs/AT_2021_fuels_official_cpa_split_flows/
outputs/AT_2021_fuels_official_cpa_reaggregated_flows/
```
