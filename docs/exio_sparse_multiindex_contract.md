# EXIO sparse loading and strict MultiIndex contract

This document defines the reusable MRIO-side basis and sparsity contract.

## Primitive loader invariant

For EXIOBASE zip-stream loading, `load_mrio_iot_matrices(...)` reads only requested primitive `Z`, `x`, and `Y` accounts from the raw source zip. It does not expose `A` as a source selector and does not require `A.txt`.

`A` is derived explicitly from Z and x:

```text
A_ij = Z_ij / x_j
```

with the repository's zero-output-column convention.

## Canonical basis

Complete-IOT primitive vectors and rows use:

```python
pd.MultiIndex(names=["region", "sector"])
```

For a bundle containing x/Z/Y:

- `x.index.equals(basis_index)`;
- `Z.shape == (len(basis_index), len(basis_index))`;
- `Y.index.equals(basis_index)`;
- a subsequently derived sparse `A` uses the same row/column basis.

Do not stringify `(region, sector)` tuples into authoritative alignment keys.

## Sparse boundary rule

Full EXIOBASE Z and derived A are sparse. Diagnostics and transforms must accept sparse matrices where feasible and must not densify EXIO-scale matrices.

Forbidden on full EXIO matrices unless guarded by an explicitly small fixture:

```python
pd.DataFrame(A)
A.values
np.asarray(A)
A.toarray()
A.todense()
```

Use sparse reductions, CSR operations, coordinate remapping, and `.data` diagnostics instead.

## Geography transformations

Complete region aggregation preserves the sector basis and applies the region mapping extensively to Z/x/Y before any coefficient derivation. Low-memory scope slices are destination-use donor blocks, not complete balanced IOTs.

Environmental production extensions preserve the same `(region, sector)` basis until the accepted geography/product transformation is complete. Direct `F_Y` uses a destination-region/final-demand-category axis instead.

## Extension parsing

EXIO extension parsers must preserve tab fields and align numeric columns intentionally. Blank selected cells are unavailable evidence and must fail where required; whitespace parsing must not silently drop them.

See `docs/exio_lowram_zip_stream_contract.md` for cache and memory policy and `docs/mrio_api_spec.md` for the public API boundary.
