# MRIO index and strict SDA API contract

This document records the local MRIO API contract used by the EcolEconModels
analysis repos.

## Canonical index

New vector-returning MRIO APIs must preserve a canonical pandas
`MultiIndex(region, sector)`.

Single-region datasets should still use a two-level index, for example
`region="AT"` or `region="TEST"` and `sector=<sector label>`. Do not stringify
`(region, sector)` labels in new APIs that return vectors for downstream basis
alignment.

## Loader boundary

`load_mrio_iot_matrices(...)` is the canonical low-RAM primitive IOT loader used by SDA and other complete-basis analyses. Its selectable source accounts are only `Z`, `x`, and `Y`, and its return is `(matrices, basis_index, metadata)`.

SDA callers explicitly derive the quantities they need:

- sparse `A` from `Z` and `x` using `derive_A_from_Z_and_x(...)`;
- the established non-export final-demand vector from complete `Y` using `derive_nonexport_final_demand_from_Y(...)`.

The loader does not require `A.txt`, does not return a compatibility tuple `(A, x, y, ...)`, and does not write derived matrix caches. A caller needing only Y or x should request only that primitive account.

Complete region aggregation uses `aggregate_mrio_regions(...)` on primitive Z/x/Y and derives A afterward. Low-memory destination-scope work uses `load_mrio_iot_scope_slice(...)`; its external-import use block must not be treated as a complete IOT denominator boundary.

Production extension rows continue to align to the canonical `MultiIndex(region, sector)` basis. Geography-aware environmental aggregation keeps extensive physical numerators and monetary denominators separate until transformation is complete; direct `F_Y` remains a separate final-demand-category account.

## Strict SDA pair result

`compute_output_sda_pair_result(...)` returns compact vectors for one year pair:

- observed output change;
- average-polar intermediate-use / technical-coefficient component;
- average-polar final-demand component;
- residual.

When `average_identities=True`, both forward and backward polar solves must
succeed. The strict API must raise a clear error if either solve fails; it must
not silently skip the pair or fall back to forward-only decomposition.

## Units

The MRIO API exposes unit labels and unit provenance for reporting, but it does
not perform automatic conversion. Analysis and plotting code should save units in
CSV outputs and should only apply conversions through explicit, separately tested
code.

## Production-side extension SDA

`compute_extension_sda_pair_result_from_states(...)` decomposes production-side
extension rows indexed by `MultiIndex(region, sector)`. The economic identity is
`f_t = q_t * x_t = q_t * L_t y_t`, where `q_t` is a direct extension intensity.
The public API initially supports only extension matrix kind `F`; `F_Y` is a
final-demand-side matrix and must be handled by a separate additive API.

The extension SDA has three factors: stabilized direct intensity, production
structure, and final demand. The implementation averages all six polar orderings
of `(intensity, structure, demand)`. The components decompose the stabilized
`q_tilde * L_t y_t` identity, while reported stabilization-gap vectors reconcile
back to raw observed extension rows.

The implementation must preserve the existing sparse/low-RAM contract: no explicit
Leontief inverse, no stored solver factorization, no persistent derived matrix
cache, and no full extension matrix load for the default row-level path.

The pair result separates three reconciliation layers. First, the six-polar
components plus `sda_residual` reconstruct only `leontief_extension_delta`, so
`sda_residual` is a numerical residual for the `q_tilde * L y` identity. Second,
`reported_x_identity_gap_delta` reconciles `leontief_extension_delta` to the
stabilized direct row change `q_tilde * x` when reported output is not exactly the
solved Leontief output. Third, `stabilization_gap_delta` reconciles the stabilized
row change back to raw observed extension rows.

Plain pandas Series are accepted only when they carry explicit `attrs["unit"]` and
`attrs["unit_source"]` metadata; otherwise public callers should pass
`ExtensionRowResult`. Selected all-zero production extension rows are rejected by
default when MRIO output has positive mass, because such rows usually indicate an
incorrect stressor/year selection or an incomplete extension source.

`load_mrio_extension_year_state(...)` returns `MrioExtensionYearState`, containing
the `MrioYearState`, raw `ExtensionRowResult`, and computed `ExtensionIntensityState`.
Pair callers can therefore compose convenience-loaded years through
`compute_extension_sda_pair_result_from_extension_year_states(...)` without
reloading extension rows or losing unit/stressor provenance. By default, that pair
helper uses the stored `ExtensionIntensityState` objects directly; it recomputes
from raw rows only when the caller explicitly overrides stabilization, tau, or the
all-zero-row rejection policy.

## MrioYearState source controls

`load_mrio_year_state(...)` forwards the same resumable-download/progress/chunk,
timeout and retry controls needed by low-memory source-ZIP callers. Downstream
analysis should therefore reuse this strict compact state rather than reloading
Z/x/Y and rebuilding A/y with local reindex or fill-zero logic.

## Monetary valuation of strict year states

`load_mrio_year_state(...)` has one optional price-sensitivity mode. The default
`price_base_year=None` is the existing current-price state. With a base year, MRIO
loads primitive `Z`, `x`, and `Y`, applies the validated EXIOBASE3
country-GDP/common-EUR row factors, and only then derives sparse `A` and non-export
`y`. The state records `valuation_basis`, `price_base_year`, the common factor-panel
year domain, and canonical factor-table SHA; strict output and extension SDA pair
boundaries reject mismatched valuation signatures.

Production-side physical extension rows remain unchanged. A price-adjusted
`MrioExtensionYearState` therefore recomputes its direct coefficient `q=F/x`
against the revalued monetary output while preserving the source physical `F`
values and units.

<!-- ecolecon-valued-primitive-loader -->
## Selective valued primitive accounts

`load_mrio_iot_matrices(...)` remains the raw provider/source-account API and never
changes valuation. EXIOBASE callers that need only selected monetary primitives in
the common-EUR sensitivity use
`load_exiobase3_iot_matrices_in_price_basis(...)`. The latter preserves selective
`Z`/`x`/`Y` loading and the source-ZIP-only cache policy, but centralizes the
optional row revaluation before any caller derives coefficients or aggregates final
demand categories.

This distinction is required for analyses that need the full final-demand matrix
`Y` (for example consumption/GFCF versus inventories/valuables partitions) and for
x-only historical stages: neither should reconstruct price factors downstream or
load a full SDA state merely to obtain one primitive account. `load_mrio_year_state`
therefore reuses the valued primitive helper for `Z/x/Y` and derives `A/y` only from
its returned accounts.

Extension SDA pair results preserve the checked `valuation_basis` and
`price_base_year` in result metadata. Physical production extension `F` remains
untransformed; only its direct intensity changes through the selected monetary
`x` denominator.

Panel analyses must pass one shared `price_factor_years` sequence through every
valued primitive, `load_mrio_year_state(...)`, and `load_mrio_extension_year_state(...)`
call. Returned common-EUR states record the full factor-panel year domain and the
canonical factor-table SHA. Strict monetary and extension pair APIs compare that
full signature, so matching base years alone cannot combine independently scoped
or differently sourced factor tables. This is a valuation/provenance contract only;
MRIO does not choose downstream analysis or figure output directories.
<!-- /ecolecon-valued-primitive-loader -->

<!-- public-year-state-leontief-rhs-v1 -->
## Public Leontief solve on an already-loaded year state

`solve_mrio_year_state_leontief_rhs(state, rhs, ...)` is the narrow public boundary
for downstream analyses that need to apply the same `(I-A_t)^{-1}` operator as the
strict SDA APIs to one or more caller-defined right-hand sides. `rhs` is a pandas
DataFrame on the state's exact canonical `MultiIndex(region, sector)`; its columns
are preserved in the returned DataFrame. Multiple columns are solved together so
the existing sparse multi-RHS implementation factorizes the year system once.

The helper validates the year-state valuation metadata and matrix/index boundary,
forms no explicit Leontief inverse, stores no solver factorization, performs no unit
conversion, and creates no derived matrix cache. It is an operator boundary only:
the downstream caller owns the economic meaning and dimensional consistency of the
RHS columns. Strict SDA identities continue to be owned by the existing output and
extension pair APIs.
