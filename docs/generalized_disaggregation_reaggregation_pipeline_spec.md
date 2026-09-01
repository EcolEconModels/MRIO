# Generalized disaggregation and reaggregation pipeline specification

## Purpose

This document specifies a small, modular pipeline for disaggregating and
reaggregating monetary IOT/SUT/MRIO tables using external evidence. It unifies
two existing families of methods:

1. full donor-block disaggregation using EXIOBASE or another MRIO/IOT donor; and
2. row-, column-, and final-demand-specific disaggregation using Energy Balances,
   PEFA, PRODCOM, trade, activity statistics, national statistical-office data,
   and explicit manual overrides.

The pipeline must be general across countries, years, classification systems,
and split profiles. No sector code, product code, Energy Balance code, PEFA code,
EXIO sector, NACE/CPA/PRODCOM/CN code, or final-demand code may be hard-coded in
Python. All such information must come from dictionaries, mapping files, split
recipes, and reaggregation recipes.

## Economic methods dependency

The scientific/economic method is specified independently in
`docs/economic_methods/generalized_energy_disaggregation_methods.md`. This API
specification remains the software-facing orchestration contract: it defines the
profile files, provider interfaces, execution routes, diagnostics, validation
hooks, and migration rules needed to implement that methods document. The API
spec must not weaken, replace, or silently alter the economic equations and
accounting identities in the standalone methods spec.

The independent region-scope API described below is an accepted **vNext design**,
not an implemented API. Existing singleton EXIO loaders and current split-pipeline
interfaces remain authoritative until the corresponding source-and-test patch is
merged.

## Existing repo abstractions to preserve

The existing low-level abstraction `SplitProfiles` is the common contract. A
profile defines one aggregate sector and its child subsectors with four share
vectors:

```text
output_share
intermediate_sales_share
intermediate_input_share
final_demand_share
```

A profile may also carry an optional child-to-donor-sector map. Existing generic
expansion code already consumes `SplitProfiles` and an optional
`column_allocator`. Existing EXIO donor-block code already builds split profiles
from donor blocks and validates that split outputs reaggregate back to the
original table. The new design should reuse these concepts instead of creating a
third split representation.

## Design principles

1. Evidence resolution is separate from IO execution.
2. Split children are defined by profile files, not by source-specific code.
3. Parent monetary totals from the target IOT are authoritative unless an
   override policy explicitly says otherwise.
4. Energy/product quantity evidence and monetary value evidence are different
   evidence kinds and must be resolved separately.
5. Column-use evidence and producer-technology evidence are different allocation
   problems and must not be merged implicitly.
6. EXIOBASE is one donor/evidence provider, not the default scientific truth for
   country-level disaggregation.
7. Manual prices, equal splits, and EXIO fallbacks are permitted only when the
   profile declares them and diagnostics record them.
8. Reaggregation validation is mandatory for every run.
9. Every output cell created by a split must have a deterministic allocation
   method and a diagnostic trail.
10. Split and reaggregation files under `scripts/pipelines/<profile>/` are the
    source of truth for the sector profile.
11. Region-scope mappings are independent of sector profiles. Extension ratios
    follow the methods specification: fixed linear extensive accounts may be
    geography/sector transformed before division, while external-import direct
    accounts retain origin through product projection and form the origin-specific
    coefficient only on that common projected basis.

## Independent geography contract

Geography aggregation is an upstream donor operation, not a fuels/fuels-more or
green/brown profile feature. Checked-in region-scope helpers are proposed under
`scripts/pipelines/region_scopes/` (or may be supplied explicitly), consistent
with the repository-wide `scripts/pipelines/` source-of-truth rule. The directory
is created by the future implementation patch, not by this documentation-only
change. Region-scope mappings use:

~~~text
scope_code,scope_label,member_region,member_order,enabled,notes
~~~

The target geography and donor scope are independent:

- when Eurostat publishes the target geography, load it directly through the
  `geo` filter (for example `EU27_2020`);
- aggregate the EXIOBASE donor with a versioned region-scope mapping;
- never infer the EXIO scope from the Eurostat code; and
- never sum national target IOTs to replace an official consolidated table.

The generic region API returns a donor context with `internal`,
`external_imports`, and `total` destination-use boundaries plus the separate
scope-production output account. A singleton-region compatibility wrapper maps
the old `domestic` and `imports` names to the neutral boundaries.

Sector plan compilation consumes this donor context but does not read or validate
region membership itself. Consequently, region aggregation can be run and
inspected without a sector split, and every sector route can operate on either a
singleton or a group donor.

## Profile file layout

Recommended profile directory:

```text
scripts/pipelines/<profile>/
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
    classification_codes.csv
    dimensions.csv
    units.csv
    final_demand_codes.csv
    provider_datasets.csv
```

Only `split_children.csv` and `reaggregation.csv` are mandatory for an empty
profile. Other files are required when their provider or rule type is enabled.

## Required recipe files

### `split_children.csv`

Defines target child sectors independently of evidence sources.

```text
order,enabled,parent_code,parent_label,child_code,child_label,child_family,
child_role,required,allow_zero,default_residual_policy,notes
```

`parent_code` and `child_code` are profile data, not Python constants.

### `reaggregation.csv`

Defines how child sectors aggregate back to target sectors.

```text
source_code,source_label,target_code,target_label,weight,enabled,notes
```

For pure reaggregation, `weight` is normally 1. Weighted reaggregation is allowed
only when a profile explicitly declares it.

### `provider_priority.csv`

Defines source ranking by evidence kind.

```text
evidence_kind,rank,provider,dataset_role,required,allow_fallback,notes
```

Standard `evidence_kind` values:

```text
physical_quantity
monetary_output
unit_value
row_sales_share
column_input_share
final_demand_share
column_use_prior
producer_technology_prior
full_donor_block
emission_intensity
bridge_adjustment
manual_override
```

### `evidence_selectors.csv`

Maps children or families to provider selectors.

```text
enabled,evidence_kind,provider,dataset_role,child_code,child_family,
selector_system,selector_code,flow_role,unit,aggregation_rule,notes
```

Selectors are opaque strings interpreted by provider plugins. They may refer to
classification dictionaries but must not be interpreted by hard-coded sector
branches.

### `column_allocation_rules.csv`

Defines row/column/final-demand allocation behavior for the row/column-specific
route.

```text
order,enabled,parent_code,child_code,allocation_target,buyer_selector_system,
buyer_selector,provider,dataset_role,evidence_kind,rule,fallback_method,notes
```

`allocation_target` examples:

```text
Z_COLUMNS
Z_ROWS
Y_ROWS
V_COLUMNS
M_COLUMNS
```

The valid values are dictionary-driven table roles.

### `donor_mapping.csv`

Defines the relation between emitted child labels and donor sectors for the full
donor-block route.

```text
enabled,parent_code,child_code,donor_provider,donor_dataset_role,
donor_selector_system,donor_selector,edge_target_code,weight,notes
```

Many-to-many mappings must be represented explicitly. Edge labels such as a
source-target pair are allowed as emitted child labels only if produced from this
mapping file and recorded in diagnostics.

### `residual_policy.csv`

Defines how unresolved parent mass is handled.

```text
enabled,parent_code,residual_child_code,residual_method,negative_policy,
tolerance,notes
```

Standard residual methods:

```text
parent_output_minus_declared_children
zero_if_no_parent_mass
explicit_external_residual
forbidden
```

Standard negative policies:

```text
fail
clamp_with_warning_within_tolerance
scale_declared_children
send_to_diagnostic_only
```

The default must be `fail` beyond tolerance.

## Common dataclasses

`RegionScope` is owned by the proposed generic MRIO geography API in
`docs/mrio_api_spec.md`; the split pipeline reuses that type rather than defining
a parallel geography model. Its proposed pipeline-specific adapter is:

~~~python
@dataclass(frozen=True)
class DonorScopeContext:
    scope: RegionScope
    internal: object
    external_imports: object
    total: object
    x_scope: pd.Series
    environmental_accounts: pd.DataFrame | None
    final_demand_environmental_accounts: pd.DataFrame | None
    metadata: dict[str, object]
~~~

```python
@dataclass(frozen=True)
class SplitChild:
    parent_code: str
    child_code: str
    child_label: str
    child_family: str
    required: bool
    allow_zero: bool
```

```python
@dataclass(frozen=True)
class SplitEvidence:
    evidence_kind: str
    provider: str
    dataset_role: str
    parent_code: str
    child_code: str | None
    child_family: str | None
    value: float | None
    unit: str | None
    selector_system: str | None
    selector_code: str | None
    confidence_rank: int
    notes: str
```

```python
@dataclass(frozen=True)
class GenericSplitPlan:
    split_profiles: dict[str, SplitProfiles]
    reaggregation_mapping: dict[str, str]
    component_values: pd.DataFrame
    evidence_resolution: pd.DataFrame
    column_allocator: dict[str, dict[str, pd.Series]] | None
    donor_context: object | None
    diagnostics: dict[str, pd.DataFrame]
```

```python
@dataclass(frozen=True)
class GenericSplitResult:
    Z: pd.DataFrame
    Y: pd.DataFrame
    x: pd.Series
    A: pd.DataFrame
    split_profiles: dict[str, SplitProfiles]
    balancing_method: str
    balancing_stats: BalanceStats
    diagnostics: dict[str, pd.DataFrame]
    V: pd.DataFrame | None = None
    M: pd.DataFrame | None = None
    io_system: object | None = None
```

## Public API

Region membership is compiled once by the generic
`compile_region_aggregation(...)` API from `docs/mrio_api_spec.md`. The split
pipeline does not add a second `compile_region_scope(...)` validator. Its proposed
adapter only packages the generic scope slices and environmental accounts needed
by sector-plan compilation:

~~~python
def load_donor_scope(
    *,
    source: MrioSource,
    scope: RegionScope,
    flows: Sequence[str] = ("internal", "external_imports", "total"),
    environmental_selection: object | None = None,
) -> DonorScopeContext:
    """Assemble pipeline donor context from generic MRIO geography results."""
~~~

```python
def compile_split_plan(
    *,
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    profile_dir: Path,
    target_geo: str,
    year: int,
    providers: Mapping[str, EvidenceProvider],
    archive_store: SourceArchiveStore,
    donor_context: DonorScopeContext | None = None,
) -> GenericSplitPlan:
    """Resolve evidence, shares, residuals, and allocators from profile files."""
```

```python
def run_split_plan(
    *,
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    plan: GenericSplitPlan,
    method: Literal["row_col_specific", "full_donor_block"],
    balancing_scope: Literal["local", "global"] = "local",
    validate_reaggregation: bool = True,
    create_io_system: bool = False,
) -> GenericSplitResult:
    """Execute a compiled plan with the selected generic route."""
```

```python
def reaggregate_split_result(
    result: GenericSplitResult,
    reaggregation_mapping: Mapping[str, str],
    *,
    keep_unmapped: bool = True,
) -> GenericSplitResult:
    """Aggregate emitted child sectors to target sectors using the profile map."""
```

```python
def write_split_diagnostics(
    outdir: Path,
    *,
    plan: GenericSplitPlan,
    result: GenericSplitResult,
) -> None:
    """Write reproducibility, evidence, allocation, residual, and validation outputs."""
```

## Route 1: full donor-block disaggregation

Use this route when a donor MRIO/IOT/SUT has enough child-sector structure to
allocate full submatrices.

Inputs:

```text
target Z,Y,V,M
donor scope context with selected flow boundary and optional extension tables
SplitProfiles
donor_mapping.csv
final-demand dictionary
```

Algorithm:

1. Build emitted child labels and donor-sector weights from `donor_mapping.csv`.
2. For every target intermediate cell `(i, j)`, identify emitted row children of
   `i` and emitted column children of `j`.
3. Build the donor submatrix over those row and column children.
4. If the donor submatrix total is positive, allocate the target cell
   proportionally to the donor submatrix and rescale exactly to the target cell
   total.
5. If the donor submatrix total is unavailable or zero, fall back according to
   `provider_priority.csv` and `column_allocation_rules.csv`; record a donor
   fallback diagnostic.
6. Split final demand using donor final-demand-category shares where a dictionary
   maps target final-demand codes to donor categories; otherwise use the profile
   final-demand share fallback.
7. Split supplementary rows with explicit identity-aware rules from dictionaries.
8. Recompute `x` and `A` from split `Z` and `Y`.
9. Validate that aggregating split `Z`, `Y`, and `x` returns the original target
   table within tolerance.

This route is appropriate for EXIOBASE, a previous-year detailed MRIO, a national
SUT/IOT, or any donor that can provide a complete block structure. EXIOBASE
extension-derived unit values or emission intensities are optional evidence
outputs, not primary country totals.

### Environmental extension sidecar contract

The fuels/fuels-more production EXIOBASE3-to-CPA65 concordance is the reviewed
CSV `scripts/mappings/exio_to_eurostat_cpa65_products_official_direct_reviewed.csv`,
derived from the unchanged EXIO official-direct CPA64/EXIOBASE3 topology. It
narrows only the project-reviewed `p23.3`, `p24.*`, and `p33` aggregate
correspondences: Nuclear fuel (`p23.3`) -> `CPA_C24` Basic metals; only Chemicals
nec (`p24.d`) retains the `CPA_C20` Chemicals and chemical products / `CPA_C21`
Basic pharmaceutical products and pharmaceutical preparations split while all
other `p24.*` products go to `CPA_C20`; and `p33` Medical, precision and optical
instruments, watches and clocks -> `CPA_C26` Computer, electronic and optical
products. Multi-target `p24.d` carries no fixed or equal share: the common runtime
resolver uses the flow-specific Eurostat target reference to partition the source
with unit total weight. Fuels reaggregation remains downstream from this
statistical concordance. The separate green/brown wrapper retains its older
C20-only concordance and unchanged analytical reaggregation recipe.

The scientific equations are owned by
`docs/economic_methods/generalized_energy_disaggregation_methods.md`. The EXIO
full-donor route exposes their result as two long-form tables through one generic
pair of `SplitResult` fields rather than adding environment-specific fields.
Reaggregation stays outside the monetary `AggregationResult` and is handled by one
total-first satellite operator.

#### Geography-aware satellite boundaries

The generic satellite operator carries extensive physical values and monetary
denominators using:

~~~text
axis_position,axis_label,boundary,denominator_role,extension,stressor,unit,
value,monetary_denominator,intensity
~~~

Initial boundaries are:

| boundary | axis | denominator role |
|---|---|---|
| `scope_production` | product/sector | member-region production output |
| `external_imports_direct` | product/sector | imports used by the destination scope |
| `total_supply_direct` | product/sector | target domestic output plus target import supply |
| `scope_final_demand_direct` | final-demand category | all-origin final demand in the scope |

`scope_production` is formed by summing member-region `F` and `x` before
division. `external_imports_direct` weights origin-specific `F/x` by actual EXIO
`Z+Y` flows from origins outside the scope to destinations inside it.
`total_supply_direct` combines the first two coefficients with the authoritative
target DOM and IMP monetary components. `F_Y` is summed over destination members
and divided by final demand by category; it never receives a product position or
a RoW-import counterpart.

The existing model-facing
`sector_position,sector,extension,stressor,unit,value,monetary_output,intensity`
file remains a temporary compatibility view of `scope_production` only. New
multi-boundary APIs return the generic table above and a separate category-axis
table for `F_Y`. Missing evidence remains missing, not zero.

Raw donor evidence uses:

```text
source_label,extension,kind,stressor,unit,value,monetary_denominator,intensity
```

The temporary legacy production-side compatibility view uses:

```text
sector_position,sector,extension,stressor,unit,value,monetary_output,intensity
```

`value` is always the extensive physical total on the table's current sector
basis. A blank selected donor satellite cell is unavailable and must fail closed;
it is never converted to physical zero. Input `monetary_output` is an integrity
copy that must match the canonical IOT output sector by sector. Reaggregation sums
`value`, independently aggregates
the canonical IOT output, and derives `intensity` afterward. Every positive-output
split child whose coefficient is used must have donor support at projection. Each
supported extension/stressor/unit must cover every positive-output source sector
contributing to every aggregate target represented by that sidecar group. This
target-local closure rejects a partial physical numerator against a full
canonical-output denominator.

Sparse reaggregation mappings describe only sectors that change. Unmapped sectors
retain their identifiers and supported environmental estimates. Once scientific
reaggregation and its integrity checks finish, the final model-facing production
sidecar is completed in long form to exactly the final IOT sector axis for every
selected raw production-`F` extension/stressor/unit group. This raw metadata domain
preserves even a selected stressor with no supported final-sector estimate anywhere;
`F_Y` groups do not enter it. `sector_position` is one-based and follows
the authoritative final `Z`/`x`/sector-contract order; it is not an independent
classification registry. `monetary_output` is populated from canonical final IOT
output for every row.

The value field records whether a scientific estimate is defined. A supported
zero-output sector has `value=0` and missing `intensity`; a supported physical zero
at positive output has `value=0` and zero `intensity`. A sector without a valid donor
coefficient has missing `value` and `intensity`, even when its monetary output is
zero. Missing `value` is not a physical zero or an absent sector, and `intensity`
missingness alone does not diagnose support. No availability/status Boolean is part
of the schema. Such rows preserve the model axis but are not imputed observations and
must not enter physical sums. The canonical artifact remains
one long-format `environmental_extensions.csv`, not a wide or per-extension family
of files.

Production `F` is transported as an extensive numerator together with its EXIO
monetary-output denominator and is divided only after projection. A raw production
donor may therefore have nonzero `F` with zero `x`: its raw `intensity` is unavailable,
but its physical `value` remains in the mapped numerator. A required positive-output
split child still needs a positive mapped donor denominator before a coefficient is used.

`F_Y` rows remain final-demand-category sidecars because they are not a
product-sector axis and therefore do not pass through the EXIO-to-Eurostat projection
`P`. Their monetary denominator is destination-region final demand by category summed
across all EXIO producer origins/products. Because these category coefficients are
reported directly rather than sector-projected, a nonzero `F_Y` observation with a zero
category monetary denominator is inconsistent and fails closed. Every selected extension
must provide both `F.txt` and `F_Y.txt` plus `unit.txt`; missing selected source evidence
fails closed. Sector reaggregation copies these `F_Y` rows unchanged rather than
manufacturing a sector-by-final-demand satellite matrix.

The raw `exio_environmental_extensions.csv` remains source-oriented and retains the
unchanged `source_label,extension,kind,stressor,unit,value,monetary_denominator,intensity`
schema for both `F` and `F_Y`. It receives neither final-sector completion nor
`sector_position` presentation fields.

The routine CLI uses `--environmental-preset combustion-co2` by default and
`--environmental-preset none` to disable the satellite. Custom selections use the
repeatable official EXIO extension-directory form
`--environmental-extension EXTENSION[=PATTERN]`; `EXTENSION` is the exact directory
name in the selected EXIO archive, so the code does not maintain aliases such as
release-specific singular/plural spellings. A bare extension selects all
indicator/stressor rows, while each pattern selects every matching row. A pattern
may therefore deliberately select multiple indicators. The JSON form is retained
only as the advanced `--environmental-extensions-json` escape hatch.
`--environmental-all-extensions` discovers every archive directory containing the
required `F.txt`, `F_Y.txt`, and `unit.txt` triplet and selects all stressors. The
small `combustion-co2`/`none` presets remain convenience selections, not an extension
name registry.

DOM, IMP, and TOTAL outputs use distinct satellite boundaries. A scope-production
coefficient is valid only for DOM/scope production. IMP uses the origin-weighted
external-import account, and TOTAL combines target DOM and IMP extensive components;
no coefficient is copied between these boundaries.

## Route 2: row/column-specific evidence disaggregation

Use this route when evidence is richer by row, column, product family, or
technology input than as a complete donor matrix.

Inputs:

```text
target Z,Y,V,M
split_children.csv
provider_priority.csv
evidence_selectors.csv
column_allocation_rules.csv
residual_policy.csv
reaggregation.csv
```

Algorithm:

1. Resolve physical quantities for each child from the highest-priority enabled
   provider with coverage.
2. Resolve monetary child values independently from product values, unit values,
   or quantity times price according to `provider_priority.csv`.
3. Compare declared child values to the parent target output.
4. Create required residual children according to `residual_policy.csv`.
5. Convert child values into `output_share`, `intermediate_sales_share`,
   `intermediate_input_share`, and `final_demand_share` candidates.
6. Resolve column allocators from column-use evidence, producer-technology
   evidence, or declared fallbacks.
7. Expand `Z` and `Y` using `SplitProfiles` and optional `column_allocator`.
8. Locally or globally balance `[Z|Y]` to the selected row/column/final-demand
   targets.
9. Recompute `x` and `A` from the balanced split tables.
10. Validate reaggregation and write diagnostics.

This route is appropriate for Energy Balances, PEFA, PRODCOM, trade, SBS/STS,
national statistical-office tables, and manual overrides.

## Evidence-provider responsibilities

Providers return normalized evidence tables. They do not mutate IOTs.

```python
class EvidenceProvider:
    name: str

    def load_evidence(
        self,
        *,
        geo: str,
        year: int,
        profile_dir: Path,
        archive_store: SourceArchiveStore,
        selectors: pd.DataFrame,
    ) -> pd.DataFrame:
        ...
```

Provider output schema:

```text
provider,dataset_role,geo,year,evidence_kind,parent_code,child_code,
child_family,selector_system,selector_code,flow_role,value,unit,
quality_status,confidence_rank,source_archive,source_member,notes
```

Standard providers:

```text
source_archive_provider
energy_balance_quantity_provider
pefa_supply_use_provider
pefa_bridge_provider
prodcom_value_quantity_provider
trade_value_quantity_provider
activity_statistics_provider
full_mrio_donor_provider
exio_extension_provider
manual_override_provider
```

## Evidence resolution rules

Resolution is deterministic:

1. Filter enabled evidence by `geo`, `year`, child/family, and evidence kind.
2. Validate units or convert with a declared rule in `dictionaries/units.csv`.
3. Sort by `provider_priority.csv.rank`, then by profile-declared confidence.
4. Select the first passing evidence row unless the recipe declares an aggregate
   combination rule.
5. Record every candidate and selected row in `evidence_resolution.csv`.
6. If no evidence passes and fallback is not allowed, fail.
7. If fallback is allowed, record the fallback method and provider.

No equal split, EXIO fallback, manual price, or residual absorption may occur
without a diagnostic row.

## Column allocation contract

The pipeline distinguishes two allocation problems.

### Delivered-use allocation

This allocates purchases of a delivered product family across buyer columns. It
uses PEFA, target IOT rows, trade, activity statistics, or other use evidence.
For product families that are physically homogeneous at delivery, all source
children may share the same buyer pattern unless a profile declares otherwise.

### Producer-technology allocation

This allocates inputs into split producer columns. It uses transformation-input
evidence, donor technologies, PEFA production-side use, national SUTs, or manual
technology rules.

The two allocation types must be separate rows in `column_allocation_rules.csv`.

## Balancing contract

The default balancing scope is `local`:

- cells touching emitted split rows or columns may adjust;
- untouched original blocks remain fixed;
- final-demand split rows are not changed unless the profile declares them
  adjustable;
- global balancing is an explicit option.

RAS is acceptable for non-negative targets. GRAS fallback is acceptable only if
implemented and recorded in balancing diagnostics.

## Supplementary row contract

Supplementary rows such as value added, output identities, imports, supply
identities, margins, or tax rows must be split by dictionary-driven rules. The
pipeline must recognize accounting identities from a supplementary-row dictionary
rather than hard-coded row names.

Examples of rule roles:

```text
output_identity_from_split_x
intermediate_use_identity_from_split_Z_columns
supply_identity_from_output_plus_imports
value_added_proxy_from_output_share
imports_proxy_from_trade_or_input_share
copy_unaffected
```

The exact source row codes for those rules live in dictionaries.

## Reaggregation validation

Every run must validate:

```text
aggregate(Z_split) == Z_original
aggregate(Y_split) == Y_original
aggregate(x_split) == x_original
optional aggregate(V_split) == V_original
optional aggregate(M_split) == M_original
```

Validation outputs:

```text
reaggregation_validation.csv
largest_gap_by_table.csv
failed_validation_cells.csv
```

Failure beyond tolerance must stop the pipeline before writing final approved
outputs or committing generated artifacts.

## Diagnostics outputs

Required diagnostics:

```text
component_values.csv
split_profiles.csv
evidence_resolution.csv
quantity_resolution.csv
price_resolution.csv
column_allocator_matches.csv
column_allocator_fallbacks.csv
donor_block_fallbacks.csv
parent_residuals.csv
negative_residual_flags.csv
provider_coverage.csv
reaggregation_validation.csv
run_manifest.json
```

`run_manifest.json` must include the profile path, git commit if available,
provider versions, source archive hashes, selected route, balancing method,
manual fallback count, EXIO fallback count, unresolved children, and validation
status.

## Testing contract

Minimum tests for the generalized pipeline:

1. A toy row/column-specific split preserves parent totals after reaggregation.
2. A toy donor-block split uses donor submatrices when positive and records
   fallback diagnostics when the donor block is zero.
3. Provider priority selects the highest-ranked covered evidence and records all
   rejected candidates.
4. Missing required evidence fails unless fallback is declared.
5. Negative residuals beyond tolerance fail by default.
6. No parser writes extracted files or derived caches by default.
7. Supplementary identity rows are derived from split `x` or split `Z`, not from
   proxy shares.
8. Reaggregation recovers `Z`, `Y`, and `x` within tolerance.
9. Full-MRIO paths do not densify sparse matrices.
10. Profile validation catches duplicate child codes, duplicate reaggregation
    targets, and unknown dictionary codes.
11. A singleton scope reproduces current region-slice results; all-region scope has
    zero external imports; internal plus external imports equals total.
12. Streamed scope aggregation agrees with `pymrio.IOSystem.aggregate` on a small
    fixture for `Z`, `Y`, `x`, `F`, and `F_Y`.
13. Environmental tests sum numerators and denominators before division and verify
    that target TOTAL uses target DOM/IMP monetary components.

## Migration from current routes

1. Keep existing route-specific scripts as compatibility wrappers initially.
2. Move Energy Balance recipe parsing behind generic profile/evidence-provider
   interfaces.
3. Move EXIO donor-block allocation into the generic `full_donor_block` kernel.
4. Replace route-specific result classes with `GenericSplitResult` or adapters.
5. Preserve existing diagnostics while adding common evidence and validation
   outputs.
6. Remove hard-coded split-parent restrictions from code after equivalent
   restrictions are represented in profile files.
7. Introduce the region-scope loader independently, then make the existing
   single-region loader a singleton wrapper.
8. Replace DOM-only extension branches with one boundary-aware extensive-account
   builder; keep the old production-sidecar schema only as a compatibility view.


### Local EXIOBASE beta provenance caveat (2026-08-19)

A development checkout may temporarily replace the DOI-downloaded yearly ZIP with a
Box-distributed EXIOBASE beta before that beta has an official DOI. The Austria 2021
beta archive used during this extension audit has SHA-256
`ef1a379366a230b85883428e36322fb1a1d99ea45c8e3672077bad9414e1e2eb`. In that
manual-substitution state, the existing `download_log.json`/`exio_doi` value still
describes the configured download locator and **must not be cited as byte-level
provenance for the beta archive**. Do not invent a beta DOI. Until an official release
identifier is available, record the local archive hash in analysis provenance and do
not use EXIO force-download when the intention is to retain the manually supplied beta.
