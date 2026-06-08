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
