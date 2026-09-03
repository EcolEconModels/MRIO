# MRIO inverted pyramids

Workflow scaffold to replicate Cahen-Fourot et al. (2020) inverted pyramid analysis using multi-regional input-output data, then extend it to environmentally extended MRIO (CO2, energy, water, land).

## Layout
- `src/ecolecon_mrio/`: reusable functions (Leontief, embodied pressure allocation, dependency rankings).
- `notebooks/01_replicate_cahen_fourot.ipynb`: monetary inverted pyramid replication.
- `notebooks/02_env_pressure_pyramids.ipynb`: environmental pressure extensions.
- `tests/`: lightweight unit tests for core transforms.
- `data/`: place downloaded MRIO datasets here (gitignored).
- `scripts/run_replication.py`: scriptable replication (downloads/caches WIOD 2013, computes Leontief/Ghosh, prunes graph, detects communities).
- `scripts/split_eurostat_energy_by_exiobase.py` + `scripts/aggregate_iot_by_mapping.py`: optional `*_value_per_physical_input.csv` unit-value proxies (monetary output / physical quantity).
- `scripts/README.md`: script-by-script workflow and concordance documentation for the Eurostat/EXIO/PostGrowth pipelines.

## Quickstart
1) Create/activate the local venv and install deps:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   (optional but recommended) register the venv as a Jupyter kernel so VSCode/notebooks pick it up:
   ```
   source .venv/bin/activate
   python -m ipykernel install --user --name=mrio-venv --display-name "Python (.venv)"
   ```
2) Download or point to an MRIO dataset (EXIOBASE3 or WIOD). For low-RAM EXIOBASE table-level work, use `load_mrio_iot_matrices("exiobase", year, matrices=("Z", "x", "Y"))`; requested members are streamed from the DOI-checked source zip when download is allowed. Full-object EXIO loading is intentionally not the table-level API.
3) Run notebooks (headless) with local paths on the Python path and a writable IPython dir:
   ```
   source .venv/bin/activate
   export PYTHONPATH=$PWD/src
   export IPYTHONDIR=$PWD/.ipython
   jupyter nbconvert --to notebook --execute notebooks/01_replicate_cahen_fourot.ipynb
   jupyter nbconvert --to notebook --execute notebooks/02_env_pressure_pyramids.ipynb
   ```
   Or open interactively via `jupyter lab`/`jupyter notebook` after exporting the same environment variables. In VSCode, pick the kernel named `Python (.venv)` (kernelspec `mrio-venv`).
   Defaults now use `DATASET = "exiobase"` (public EXIOBASE3, YEAR = 2011) for automated runs. To replicate Cahen-Fourot’s WIOD setup exactly, switch to `DATASET = "wiod"` and supply the WIOD IO table in `data/wiod13_<year>/` if it is not publicly downloadable in your environment.
4) Run tests:
   ```
   source .venv/bin/activate
   PYTHONPATH=src pytest
   ```
5) Use the library from scripts or an interpreter:
   ```
   source .venv/bin/activate
   export PYTHONPATH=$PWD/src
   python -c "import ecolecon_mrio as mp; print(mp)"
   ```

## Replication run (Cahen-Fourot et al.)
- Open `notebooks/01_replicate_cahen_fourot.ipynb`, defaults `DATASET = "oecd"`, `YEAR = 2018`, and `REGION` can be set to a country code (e.g., DEU/FRA/GBR/ITA). For quick demos, switch to `DATASET = "test"` (tiny built-in MRIO). For EXIOBASE, set `DATASET = "exiobase"` (downloads pxp for the chosen year). To mirror the paper’s WIOD setup, switch to `DATASET = "wiod"` and ensure the WIOD IO table is present in `data/wiod13_<year>/`. The notebook builds Leontief + Ghosh inverses and adds a pruned Ghosh-based graph with community detection.
- Open `notebooks/02_env_pressure_pyramids.ipynb` to extend with environmental satellites; replace the placeholder satellite reference with the correct table from the MRIO object. Includes a Ghosh-based dependency graph demo for environmental lenses.
- Prefer scripts? Run the replication without notebooks:
  ```
  source .venv/bin/activate
PYTHONPATH=src python scripts/run_replication.py --dataset oecd --year 2018 --region DEU --base-sector B --top 15 --threshold 0.01 --top-k 3
  ```
  Switch `--dataset test` for a tiny demo, `--dataset exiobase` for public EXIOBASE3 (heavier), or `--dataset wiod` only if you have the WIOD IO tables available locally.

## VS Code notebook refresh tip
VS Code does not auto-reload notebooks when the file changes; close/reopen or use the Python script (`scripts/run_replication.py`) for iterative runs. The notebooks include a bootstrap to add `src` to `sys.path`, so imports work without manual `PYTHONPATH` tweaking when the correct kernel is selected.

## Replication run (Cahen-Fourot et al.)
- Open `notebooks/01_replicate_cahen_fourot.ipynb`, keep `DATASET = "wiod"`, `YEAR = 2011`. The loader downloads WIOD 2013 tables (if not cached), parses with `pymrio.parse_wiod`, caches at `data/wiod13_2011/mrio_system.pkl`, then builds both Leontief and Ghosh inverses for the inverted pyramid analysis.
- Open `notebooks/02_env_pressure_pyramids.ipynb` to extend with environmental satellites; replace the placeholder satellite reference with the correct table from the MRIO object.

## Next steps
- Wire in plotting utilities (bar/treemap/network) under `src/ecolecon_mrio/`.
- Add data-loading helpers for WIOD and reproducible sample fixtures.
- Flesh out the pyramid construction to mirror the exact algorithmic choices in Cahen-Fourot (layer definitions, thresholds) and adapt them to environmental intensities.

## Pipeline: AT 2021 ENERGY_GREEN / ENERGY_BROWN

This repo includes a reproducible 3-step pipeline to:

1) Generate an EXIO → Eurostat energy split mapping CSV (no network).
2) Split Eurostat AT 2021 IOT into energy subsectors (batch over `stk_flow`).
3) Re-aggregate split subsectors into `ENERGY_GREEN` / `ENERGY_BROWN` and write one Eurostat-shaped workbook per `stk_flow`.

The pipeline is code-driven end to end: `source_code` is the runtime key for split and reaggregation mappings, and labels are retained for human readability only.

Recommended entrypoint (no extra dependencies):

```
bash scripts/pipelines/at_2021_energy_green_brown.sh
```

You can run the script from any working directory; it resolves the repo root internally.

Optional convenience (if `make` is available):

```
make at-2021-energy-green-brown
```

The script uses `.venv/bin/python` and supports overrides via environment variables:
`TEMPLATE_XLSX`, `OUT_SPLIT`, `OUT_AGG`, `EXIO_YEAR`, `EXIO_REGION`, `EUROSTAT_TIME`, `EUROSTAT_GEO`.

Detailed documentation for the scripts and mapping files used by this pipeline lives in `scripts/README.md`.

## Exporting PostGrowth Inputs

The exact CLI workflow for the current PostGrowth inputs is now wrapped in:

```
bash scripts/pipelines/export_postgrowth_inputs.sh
```

By default this writes the following files into the sibling repo `../PostGrowth-SFC-Austria/`:

- `Energy_Balances_2021_Eurostat_green_brown.xlsx`
- `IOT_AT_2020_TOTAL_energy_green_brown.xlsx`
- `IOT_AT_2021_TOTAL_energy_green_brown.xlsx`

Detailed behavior:

1. Restores nothing and does not touch the legacy manual workbook `Energy_Balances_2021_Eurostat.xlsx`.
2. Rebuilds the 2021 model-aligned energy-balance workbook from the Eurostat download/cache path using the declared mapping CSVs.
3. Forces the upstream brown/green split for the energy balance export, so PostGrowth no longer needs to do that split in R.
4. Rebuilds the 2020 and 2021 `TOTAL` brown/green IO workbooks from the MRIO split-and-aggregate workflow.
5. Copies the generated `TOTAL` workbooks into the PostGrowth repo under the filenames expected by the model.
6. Writes `mapped_output_consistency.csv` for any `TOTAL` flow where the EXIO-to-Eurostat output sanity check is evaluated, so the scaled sector-level comparison remains inspectable.

For classification terminology, Eurostat NACE is the activity classification and Eurostat CPA is the product classification by activity. The direct energy-balance export in this repo targets a project-specific 65-product CPA-based IO classification used downstream in PostGrowth. Retained `NACE2` tokens in one shell-script filename and one mapping filename are legacy compatibility names only: NACE enters on the activity-definition side of parts of the concordance logic, while the operative bridge targets are CPA-based product rows plus households.

Useful overrides:

```
POSTGROWTH_DIR=/abs/path/to/PostGrowth-SFC-Austria \
TEMPLATE_XLSX=/abs/path/to/IOT_Austria_2021_Eurostat.xlsx \
ENERGY_OUT=/abs/path/to/Energy_Balances_2021_Eurostat_green_brown.xlsx \
bash scripts/pipelines/export_postgrowth_inputs.sh
```

Low-level commands are still available if you only want one artifact:

```
OUT_XLSX=/abs/path/to/Energy_Balances_2021_Eurostat_green_brown.xlsx \
POSTGROWTH_SHEET_NAME=Energy_1.0 \
SPLIT_ENERGY_GREEN_BROWN=1 \
bash scripts/pipelines/split_energy_balances_65CPAproducts_aggregate_energy_sources.sh
```

Despite the legacy script name, that command exports a workbook mapped to the repo's 65-product CPA-based IO rows plus households.

```
EUROSTAT_TIME=2021 \
TEMPLATE_XLSX=/abs/path/to/IOT_Austria_2021_Eurostat.xlsx \
SFC_OUTNAME=IOT_AT_2021_{variant}_energy_green_brown.xlsx \
bash scripts/pipelines/at_2021_energy_green_brown.sh
```

The `TOTAL` flow run now keeps the EXIO-to-Eurostat mapping sanity check enabled for both 2020 and 2021. The hard failure criterion is total-scale consistency, while `mapped_output_consistency.csv` records the scaled sector-by-sector comparison for inspection.

<!-- wp4-region-pipeline -->
### Region-aware fuels/fuels-more production path

WP4 wires the existing fuels/fuels-more sector mappings to the generic geography and
environmental APIs. AT is a singleton donor scope and uses no region mapping CSV.
`EU27_2020` is compiled from `scripts/pipelines/region_scopes/eu27_2020.csv` against
the actual EXIO source-region basis; RoW is its complement. The target monetary EU
IOT is the directly published Eurostat `geo=EU27_2020` table, not a sum of national
Eurostat IOTs. DOM, IMP/RoW and TOTAL environmental sidecars keep distinct boundaries
and are reaggregated by summing physical numerators and their matching monetary
denominators before deriving intensity. Aggregate exports exclude intra-scope trade.
