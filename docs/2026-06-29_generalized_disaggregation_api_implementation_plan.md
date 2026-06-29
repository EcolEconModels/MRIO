# 2026-06-29 implementation plan: generalized archive API and dis/re-aggregation pipeline

## Objective

Implement the generalized source-archive API and modular disaggregation /
reaggregation pipeline specified in:

- `docs/generalized_data_archive_api_spec.md`
- `docs/generalized_disaggregation_reaggregation_pipeline_spec.md`

The work should preserve the current Energy Balance and EXIOBASE routes as
compatibility wrappers while introducing common source-archive, evidence,
split-plan, execution, and diagnostics contracts.

## Scope boundaries

In scope:

1. source-archive cache API with zip-only durable storage;
2. generic dataset request and parse specs;
3. profile-driven dictionaries and recipe validation;
4. common evidence-provider interface;
5. common split-plan compiler;
6. two execution kernels: row/column-specific and full donor-block;
7. common diagnostics and reaggregation validation;
8. compatibility wrappers for existing Energy Balance and EXIOBASE scripts.

Out of scope for the first implementation:

1. replacing all existing scripts in one patch;
2. downloading every possible EU/national dataset;
3. implementing every provider at production quality immediately;
4. changing scientific split recipes without explicit profile-file changes;
5. writing derived matrix caches.

## Phase 0: preflight and inventory

1. Read `AGENTS.md`, the low-RAM zip-stream contract, sparse MultiIndex
   contract, existing Energy Balance split code, existing EXIO donor-block code,
   and current pipeline CSVs.
2. Inventory existing route-specific files and identify compatibility wrapper
   boundaries.
3. List all current hard-coded sector/product restrictions in code and decide
   which profile CSV should own them.
4. Add no behavior changes in this phase.

Deliverable:

```text
docs/audits/generalized_split_inventory.md
```

## Phase 1: source-archive API

Create:

```text
src/ecolecon_mrio/source_archives.py
tests/test_source_archives.py
```

Core objects:

```python
DatasetRequestSpec
SourceArchiveRef
ParseSpec
ParsedDataset
SourceArchiveStore
```

Implementation requirements:

1. write raw source archives as zip files with `request.json`,
   `source_manifest.json`, and `payload/...` members;
2. open members from zip without extracting to disk;
3. compute deterministic request hashes;
4. support existing raw EXIO source zip references without rewrapping;
5. support wrapping Eurostat JSON/CSV responses into repo-created source zips;
6. provide a compatibility reader for existing Eurostat loose `raw.json` caches
   but do not write new loose caches;
7. add tests proving no loose raw files, extracted folders, parsed pickles, or
   matrix caches are created by default.

Validation:

```bash
$PYTHON -m py_compile src/ecolecon_mrio/source_archives.py
$PYTHON -m pytest -q tests/test_source_archives.py
$PYTHON -m ruff check --select F401,F821 src/ecolecon_mrio/source_archives.py tests/test_source_archives.py
git diff --check
```

## Phase 2: generic dictionaries and profile validator

Create:

```text
src/ecolecon_mrio/split_profile.py
tests/test_split_profile.py
```

Responsibilities:

1. load `split_children.csv`, `reaggregation.csv`, `provider_priority.csv`,
   `evidence_selectors.csv`, `column_allocation_rules.csv`, `donor_mapping.csv`,
   `residual_policy.csv`, and dictionary files;
2. validate required columns;
3. validate duplicate child codes and duplicate reaggregation rows;
4. validate that enabled selector systems and units exist in dictionaries;
5. validate that no code path requires hard-coded split parents;
6. expose a `SplitProfileSpec` dataclass for compilers.

Deliverable profile skeleton:

```text
scripts/pipelines/generalized_fuels_template/
  README.md
  dataset_registry.csv
  split_children.csv
  reaggregation.csv
  provider_priority.csv
  evidence_selectors.csv
  column_allocation_rules.csv
  donor_mapping.csv
  residual_policy.csv
  validation_rules.csv
  dictionaries/
```

The template may be minimal and illustrative; it must not duplicate existing
Energy Balance mapping files as a parallel source of truth.

## Phase 3: evidence-provider interface

Create:

```text
src/ecolecon_mrio/split_evidence.py
tests/test_split_evidence.py
```

Core interface:

```python
class EvidenceProvider:
    name: str
    def load_evidence(...)-> pd.DataFrame: ...
```

Initial providers:

1. `ManualOverrideProvider` for profile-supplied small CSV evidence;
2. `ExistingEnergyBalancePxpProvider` adapter around current Energy Balance
   quantity and price resolution;
3. `ExistingExioDonorProvider` adapter around current EXIO donor profile logic.

Provider output schema must match the spec. Tests should prove deterministic
priority resolution and explicit fallback diagnostics.

## Phase 4: common split-plan compiler

Create:

```text
src/ecolecon_mrio/split_plan.py
tests/test_split_plan.py
```

Responsibilities:

1. resolve evidence by priority;
2. compute child component values;
3. create parent residual children according to profile rules;
4. build `SplitProfiles`;
5. build optional `column_allocator`;
6. build optional donor context;
7. write `evidence_resolution`, `component_values`, `parent_residuals`, and
   coverage diagnostics.

This phase should not change the numerical behavior of existing routes unless
called through the new API.

## Phase 5: execution kernels

Create:

```text
src/ecolecon_mrio/split_execute.py
tests/test_split_execute.py
```

Move or wrap existing kernels:

1. row/column-specific route delegates to existing `disaggregate_and_balance()`;
2. full donor-block route wraps the EXIO donor-block allocation logic in a donor-
   agnostic function;
3. common `GenericSplitResult` replaces route-specific result classes or provides
   adapters;
4. common reaggregation validation is shared by both routes.

Critical guardrails:

- no full-basis sparse densification;
- identity supplementary rows use derived split identities, not proxy shares;
- donor fallback diagnostics are preserved;
- local balancing remains the default for row/column-specific runs.

## Phase 6: compatibility wrappers

Update existing scripts only after the generic API is tested:

```text
scripts/split_eurostat_iot_by_energy_balances.py
scripts/split_eurostat_energy_by_exiobase.py
```

The scripts should keep their CLI behavior but internally call the generic API or
adapters. Outputs should remain backward compatible while adding common
diagnostics.

Do not delete existing route-specific functions until the wrappers are stable.

## Phase 7: provider expansion

Add providers incrementally:

1. PEFA supply/use provider;
2. PEFA bridge provider;
3. PRODCOM value/quantity provider;
4. trade value/quantity provider;
5. SBS/STS or national activity-statistics provider.

Each provider requires:

- dictionary-driven selectors;
- source-archive cache support;
- unit conversion tests;
- coverage diagnostics;
- no durable derived caches.

## Phase 8: documentation and examples

Add user-facing docs:

```text
docs/generalized_disaggregation_user_guide.md
docs/generalized_disaggregation_provider_authoring.md
docs/generalized_disaggregation_diagnostics.md
```

Examples should use placeholders or template profile files. They must not teach
users to hard-code sector codes in Python.

## Phase 9: final validation profile

Run at least two toy profiles and one real profile:

1. toy row/column-specific profile;
2. toy donor-block profile;
3. existing Energy Balance profile through compatibility wrapper;
4. existing EXIO route through compatibility wrapper.

Required checks:

```bash
$PYTHON -m pytest -q tests/test_source_archives.py tests/test_split_profile.py tests/test_split_evidence.py tests/test_split_plan.py tests/test_split_execute.py
$PYTHON -m ruff check --select F401,F821 src/ecolecon_mrio tests
git diff --check
```

Run Pyright when available with standard type checking.

## Implementation order summary

1. `source_archives.py`
2. `split_profile.py`
3. `split_evidence.py`
4. `split_plan.py`
5. `split_execute.py`
6. compatibility wrappers
7. PEFA/PRODCOM/trade/SBS providers
8. docs and real profile validation

## Success criteria

The implementation is complete when:

1. every durable downloaded source is a zip archive or an original upstream zip;
2. no default loader writes parsed caches, extracted directories, full-object
   pickles, generated `A.txt`, or derived matrix caches;
3. the Energy Balance and EXIO routes can run through the common API;
4. all sector/product/fuel/final-demand semantics are in profile files and
   dictionaries;
5. both execution routes reaggregate to the original target tables within
   tolerance;
6. diagnostics reveal provider choices, fallbacks, residuals, and validation
   status;
7. EXIOBASE is clearly one donor provider rather than a hard-coded general
   method.
