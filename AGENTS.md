# AI Agent Instructions

## Environment Configuration
> [!IMPORTANT]
> **ALWAYS** use the Python interpreter located in the local virtual environment: `.venv/bin/python`.
> Do NOT use `python` or `python3` directly from the path, as it may not have the required dependencies installed.

### Execution Example
```bash
.venv/bin/python scripts/my_script.py
```


## EXIO sparse loading and strict MultiIndex contract

> [!IMPORTANT]
> EXIOBASE loaders and diagnostics must obey `docs/exio_sparse_multiindex_contract.md`.

- `load_mrio_matrices()` reads `x.txt`, `Z.txt`, and `Y.txt` from the source zip and derives sparse `A`; it must not require `A.txt`.
- Returned EXIO vectors use `pd.MultiIndex(names=["region", "sector"])`.
- Sparse EXIO matrices may carry `.index` and `.columns` metadata; diagnostics must not densify EXIO-scale CSR matrices.
- Preserve tab fields when parsing EXIO extension tables; do not use whitespace parsing that drops blank numeric cells.

## Git Safety
> [!IMPORTANT]
> Do **NOT** run `git commit`, `git reset`, `git rebase`, or `git push` unless the user explicitly requests it.

## Project Structure
- `src/`: Core library code (`ecolecon_mrio`)
- `scripts/`: Replication and analysis scripts
- `tests/`: Unit tests
- `.venv/`: Local virtual environment (Primary Python)

## Mapping Source Of Truth
- Split and reaggregation helper CSVs live under `scripts/pipelines/` and are the single source of truth.
- Do not mirror those helper CSVs into `scripts/mappings/` or maintain parallel checked-in copies.
