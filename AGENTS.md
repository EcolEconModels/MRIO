# AI Agent Instructions

<!-- ecolecon-agent-memory-patch-template-single-source -->
## Shell patch-script invariant

[!IMPORTANT]
Before producing or editing any `.sh` patch script, agents must read fully and follow the single source of truth in repository `ecolecon-agent-memory` at
`prompt-templates/patch-script-task.md`. If not, stop and report to user, and do not produce a .sh script.

Do not duplicate the patch-script rules in this `AGENTS.md` or in repo-local
patch-delivery docs; update the template in `ecolecon-agent-memory` instead.

<!-- ecolecon-scientific-method-reasoning -->
## Scientific-method reasoning

For any scientific or methodological MRIO question, plan, review, or change whose
answer depends on economics, mathematics, accounting, environmental boundaries,
optimisation, modelling, or source-table semantics, read and follow
`../ecolecon-agent-memory/prompt-templates/scientific-method-reasoning.md` before
settling the answer. Inspect the actual MRIO loader/API and source-table semantics,
not only downstream variable names. When a downstream paper or analysis supplies
the scientific question, reconcile the MRIO boundary with that repository's
scientific authority rather than reasoning about the library in isolation.
<!-- /ecolecon-scientific-method-reasoning -->

## Current MRIO IOT, geography, and environmental API contract

> [!IMPORTANT]
> EXIOBASE loaders and diagnostics must obey `docs/exio_sparse_multiindex_contract.md`,
> `docs/exio_lowram_zip_stream_contract.md`, and `docs/mrio_api_spec.md`.

- `load_mrio_iot_matrices(dataset, year, matrices=(...))` is the lightweight complete-IOT source-account boundary. The only selectable primitive accounts are `Z`, `x`, and `Y`; it reads only requested source tables and returns `(matrices, basis_index, metadata)`.
- `A` is never a source-account selector. Derive it explicitly with `derive_A_from_Z_and_x(Z, x, basis_index=...)`. The transition non-export final-demand vector is likewise an explicit derivation from `Y`.
- `load_mrio(...)` is the distinct heavy `pymrio.IOSystem` boundary for providers where full-object loading is supported. Do not route EXIOBASE table-level work through a full object.
- `aggregate_mrio_regions(...)` performs complete many-to-one monetary geography aggregation on primitive `Z`, `x`, and `Y` only. Aggregate extensive accounts first and derive coefficients afterward.
- `load_mrio_iot_scope_slice(...)` is the one low-memory destination-use scope reader. Its `internal`, `external_imports`, and `total` `Z`/`Y` blocks are use boundaries; `x_scope` is member production and must not be combined with arbitrary import-use `Z` to derive `A`.
- Environmental geography is extensive-first. `aggregate_scope_production_extension(...)`, `aggregate_external_import_extension_direct(...)`, `aggregate_scope_final_demand_extension(...)`, and `combine_total_supply_extension_direct(...)` keep physical numerators and monetary denominators separate until after the required geography/product transformation.
- External-import direct environmental accounting must retain external origin through product projection before forming an origin-specific production coefficient. `read_exiobase3_scope_external_import_use_by_origin(...)` provides that origin-product monetary-use primitive; do not substitute the already-collapsed external-import scope slice.
- `F_Y` stays on the destination final-demand-category axis and never receives a product projection or a RoW-import counterpart.
- Obsolete loader/result APIs, singleton/plural EXIO region-slice wrappers, legacy source/result types, and derived scope caches remain deleted. Do not recreate compatibility aliases without an explicit external contract.

<!-- ecolecon-contract-first-production-debugging -->
## Contract-first production debugging

For a real-data MRIO production failure, follow
`../ecolecon-agent-memory/coding/contract-first-scientific-debugging.md` before
changing scientific semantics, mappings, denominators, missingness, tolerances, or
validators. Search Methods/specifications, tests, current call sites, and repository
history for the same invariant first. If it was already accepted elsewhere, treat
the failure as a cross-layer regression and add a spanning contract test rather than
inventing a local rule.

For broad heterogeneous domains such as all environmental extensions, multiple
geographies, or multiple years, do not use the fail-fast production path as the
primary discovery mechanism. Run the non-fail-fast support/state census first.
Environmental extension work must follow
`docs/ai/environmental_support_census.md` and use
`scripts/audit_environmental_support.py` before changing donor-support semantics.
Do not add geography-specific branches unless the published accounting or physical
boundary itself is geography-specific.

Census provenance is part of the scientific gate: the audit must verify the exact
EXIO source ZIP hash and scope provenance recorded by the successful monetary split,
and diagnostic edge metadata must never override the authoritative persisted mapping.
<!-- /ecolecon-contract-first-production-debugging -->

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
>
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

Follow `../ecolecon-agent-memory/coding/pylance-compliant-python.md` before producing or committing Python patches. That file is the authoritative cross-repository validation policy; do not duplicate or weaken it here.

## MRIO index and strict SDA API contract

For MRIO index/SDA API rules, see `docs/mrio_index_and_sda_contract.md`. New
vector-returning APIs must preserve `MultiIndex(region, sector)` labels and must
not stringify sector keys. Strict SDA pair APIs must raise if a requested forward
or backward polar solve fails.

## Region-aware fuels/fuels-more pipeline

The fuels/fuels-more target geography and EXIO donor geography are separate. A
singleton target such as AT uses `RegionScope(code="AT", members=("AT",))` and
requires no region mapping file. A multi-member donor scope such as `EU27_2020`
must be compiled from `scripts/pipelines/region_scopes/eu27_2020.csv` against the
actual EXIO release. RoW is always the source-release complement, never a hard-coded
country list. The target EU IOT remains Eurostat's directly published `geo=EU27_2020`
table.

DOM donor flows are origins and destinations inside the scope. IMP/RoW donor flows
are origins outside the scope into destinations inside it. Aggregate-scope exports
are origins inside the scope to destinations outside it; intra-scope trade is not an
export. Environmental DOM uses scope-production coefficients; IMP uses the WP3
origin-retaining external-import coefficient; TOTAL is the exact target DOM+IMP
extensive combination. Direct `F_Y` remains a final-demand-category account and is
not given a RoW product decomposition.
