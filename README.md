# MRIO inverted pyramids

Workflow scaffold to replicate Cahen-Fourot et al. (2020) inverted pyramid analysis using multi-regional input-output data, then extend it to environmentally extended MRIO (CO2, energy, water, land).

## Layout
- `src/mrio_pyramids/`: reusable functions (Leontief, embodied pressure allocation, dependency rankings).
- `notebooks/01_replicate_cahen_fourot.ipynb`: monetary inverted pyramid replication.
- `notebooks/02_env_pressure_pyramids.ipynb`: environmental pressure extensions.
- `tests/`: lightweight unit tests for core transforms.
- `data/`: place downloaded MRIO datasets here (gitignored).
- `scripts/run_replication.py`: scriptable replication (downloads/caches WIOD 2013, computes Leontief/Ghosh, prunes graph, detects communities).

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
2) Download or point to an MRIO dataset (EXIOBASE3 or WIOD). `mrio_pyramids.data.load_exiobase3` can download and parse EXIOBASE3 when network is available.
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
   python -c "import mrio_pyramids as mp; print(mp)"
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
- Wire in plotting utilities (bar/treemap/network) under `src/mrio_pyramids/`.
- Add data-loading helpers for WIOD and reproducible sample fixtures.
- Flesh out the pyramid construction to mirror the exact algorithmic choices in Cahen-Fourot (layer definitions, thresholds) and adapt them to environmental intensities.
