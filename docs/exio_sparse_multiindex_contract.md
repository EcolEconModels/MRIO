# EXIO sparse loading and strict MultiIndex contract

This document defines the reusable MRIO-side contract used by the eigenmode paper pipeline.

## Loader invariant

For EXIOBASE zip-stream loading:

- read `x.txt`, `Z.txt`, and `Y.txt` from the raw source zip;
- compute sparse technical coefficients `A` from `Z` and `x`;
- do not require `A.txt`;
- do not write derived full-object caches by default;
- keep the source zip as the durable local data cache.

## Basis invariant

The returned basis must be:

```python
pd.MultiIndex(names=["region", "sector"])
```

Return values from `load_mrio_matrices()` must satisfy:

- `x.index.equals(basis_index)`;
- `y.index.equals(basis_index)`;
- sparse `A.shape == (len(basis_index), len(basis_index))`;
- sparse `A` should carry `.index = basis_index` and `.columns = basis_index` when possible.

## Sparse boundary rule

MRIO diagnostics must accept sparse matrices where feasible. Do not densify EXIO-scale matrices.

Forbidden in full EXIO code paths unless guarded by a small-matrix check:

```python
pd.DataFrame(A)
A.values
np.asarray(A)
```

Safe alternatives:

- `scipy.sparse.issparse(A)`;
- `csr_matrix(A)`;
- `A.sum(axis=0)` / `A.sum(axis=1)`;
- `A.data` for nonzero-entry diagnostics.

## Extension tables

EXIO extension tables read from zip may have header and blank-field quirks. Parsers must preserve tab fields, infer documented row-label width, and align numeric columns to the canonical basis. Blank numeric fields should be handled intentionally, not accidentally dropped by whitespace splitting.

## Compatibility

Legacy outputs may encode labels as stringified tuples. New MRIO APIs should not produce those labels as authoritative alignment keys.
