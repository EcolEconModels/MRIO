# AI Agent Instructions

## Mandatory patch-script workflow

<!-- patch-script-template-must-read-2026-06-07 -->
Before producing any shell patch script, agents must read and follow
`../ecolecon-agent-memory/prompt-templates/patch-script-task.md`.  This is mandatory for every `.sh` patch-script
deliverable, even when the user does not explicitly mention the template.

For EcolEconModels patch scripts, use `/home/agilra/EcolEconModels/MRIO/.venv/bin/python`
unless the user explicitly overrides it.  The script must be provided as a
download link, not pasted into chat, and the user-facing command must assume the
script was downloaded to `tmp/` in the relevant repo.  Patch scripts must support
an explicit `--commit` option; when commits are requested, show the run command
with `--commit`.


## Environment Configuration


<!-- ecolecon-patch-script-template-must-read -->
## Shell patch-script tasks — must-read durable template

> [!IMPORTANT]
> Before producing any `.sh` patch script, read
> `../ecolecon-agent-memory/prompt-templates/patch-script-task.md`.  The user
> expects a downloadable script, not an inline script body.

- Give the run command from the repository root as `bash tmp/<script>.sh --commit` when committing is requested; do not tell the user to `cd tmp` first.
- Every patch script must support an explicit `--commit` option; without it, scripts may validate and stage intended files but must not commit.
- Use `/home/agilra/EcolEconModels/MRIO/.venv/bin/python` for EcolEconModels Python validation unless the user explicitly overrides it.
- Stage only intended files, run validation, show status/diff summaries, and commit with a detailed message only when `--commit` is passed.
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

<!-- ecol-econ-shell-patch-must-read-start -->
## MUST READ before shell patch scripts

Before producing any shell patch-script response for this repo, every agent must read and follow `../ecolecon-agent-memory/prompt-templates/patch-script-task.md`. This shared template is the highest-priority source of truth for patch-script delivery, validation, staging, and commit behavior.

The required user command format is from the repository root, not from inside `tmp/`:

```bash
bash tmp/<script-name>.sh
bash tmp/<script-name>.sh --commit
```

Do not paste the script body in chat when a downloadable shell script is requested. Provide only the download link, the `bash tmp/...` command, and a short note.

All EcolEconModels patch scripts must support `--commit`. Without `--commit`, they must not commit. With `--commit`, they may commit only intended staged files and must exit without committing if there are no changes.

Use `/home/agilra/EcolEconModels/MRIO/.venv/bin/python` for Python patching and validation unless the user explicitly overrides it.
<!-- ecol-econ-shell-patch-must-read-end -->
