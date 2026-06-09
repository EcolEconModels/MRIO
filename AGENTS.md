# AI Agent Instructions

<!-- ecolecon-agent-memory-patch-template-single-source -->
## Shell patch-script invariant

> [!IMPORTANT]
> Before producing or editing any `.sh` patch script, agents must read the single
> source of truth in repository `ecolecon-agent-memory` at
> `prompt-templates/patch-script-task.md`. If that file is missing or unreadable, stop and report that the patch-script template could not be found and fail.

Do not duplicate the patch-script rules in this `AGENTS.md` or in repo-local
patch-delivery docs; update the template in `ecolecon-agent-memory` instead.

## EXIO sparse loading and strict MultiIndex contract

> [!IMPORTANT]
> EXIOBASE loaders and diagnostics must obey `docs/exio_sparse_multiindex_contract.md`.

- `load_mrio_matrices()` reads `x.txt`, `Z.txt`, and `Y.txt` from the source zip and derives sparse `A`; it must not require `A.txt`.
- Returned EXIO vectors use `pd.MultiIndex(names=["region", "sector"])`.
- Sparse EXIO matrices may carry `.index` and `.columns` metadata; diagnostics must not densify EXIO-scale CSR matrices.
- Preserve tab fields when parsing EXIO extension tables; do not use whitespace parsing that drops blank numeric cells.

## EXIO low-RAM zip-stream contract

> [!IMPORTANT]
> EXIOBASE loaders and diagnostics must obey `docs/exio_lowram_zip_stream_contract.md` and `docs/exio_sparse_multiindex_contract.md`.

- The raw EXIOBASE source zip is the only default durable data cache.
- Load only requested zip members; do not parse a full `pymrio.IOSystem` for table-level needs.
- Derive sparse `A` from `Z.txt` and `x.txt`; do not require or create `A.txt`.
- Do not write full-object pickle caches, extracted text caches, or derived matrix caches by default.
- Keep full-basis EXIO matrices sparse and avoid dense conversions.
- Release large temporary arrays and matrices promptly in long-running year loops.

## Disk-space cache policy

> [!IMPORTANT]
> The raw EXIOBASE source zip is the only default durable EXIO cache. Do not write full `pymrio` pickles, extracted text caches, generated `A.txt`, or persistent `Z/A/x/y` matrix caches by default.

- Optional development caches must be explicit opt-in, ignored by git, and deleted after the experiment.
- Production loaders read only required zip members and derive sparse `A` from `Z` and `x`.
- Never delete `IOT_*_pxp.zip` or `IOT_*_ixi.zip` unless the user explicitly requests a redownload.

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

## Resource-bounded/resumable MRIO analyses

Follow `../ecolecon-agent-memory/coding/resource-bounded-resumable-analysis.md` for progress diagnostics, 4 GB default memory discipline, source-zip-only cache discipline, sparse/streaming loading, and manifest-based resumability.  Local application notes are in `docs/ai/resource_bounded_resumable_analysis.md`.

## EcolEcon durable coding rules

Follow `../ecolecon-agent-memory/coding/scientific-code-documentation-specs.md` for scientific code documentation.  In particular, scientific analysis code needs file/function/class/attribute documentation and line/block comments for every scientifically relevant loop, branch, indexing step, normalization, projection, edge-case guard, and output write.

Follow `../ecolecon-agent-memory/coding/pylance-compliant-python.md` before producing or committing Python patches.  Patch scripts must run py_compile, focused tests, Ruff when available, Pyright when available, and no-suppression/static guardrails so Pylance/Ruff errors such as unused imports, undefined names, and unknown external imports are not handed to the user.

## Cross-repo Python analysis validation

Before handing over Python patch scripts, follow the durable guidance in
`ecolecon-agent-memory/coding/pylance-compliant-python.md` and
`ecolecon-agent-memory/coding/resource-bounded-resumable-analysis.md`.
Run Ruff `F401/F821`, Pyright, focused tests, and `git diff --check`; then check
Pylance in VS Code for the changed files.

## MRIO index and strict SDA API contract

For MRIO index/SDA API rules, see `docs/mrio_index_and_sda_contract.md`. New
vector-returning APIs must preserve `MultiIndex(region, sector)` labels and must
not stringify sector keys. Strict SDA pair APIs must raise if a requested forward
or backward polar solve fails.
