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

`load_mrio_matrices(...)` remains the low-RAM EXIOBASE source-zip loader for
SDA inputs. It streams `x.txt`, `Z.txt`, and `Y.txt`, derives sparse technical
coefficients `A`, and returns `A, x, y, basis_index, metadata`. It does not write
derived matrix caches or return raw `Z`.

The current SDA average uses `A`, `x`, and `y`. A raw-`Z` return mode should be
added only when a separate analysis stage explicitly needs `Z` without `A`.

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
reloading extension rows or losing unit/stressor provenance.
