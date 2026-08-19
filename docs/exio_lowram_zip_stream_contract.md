# EXIO low-RAM zip-stream contract

This is the MRIO-side contract for reusable EXIOBASE loading under the eigenmode paper pipeline.

## Durable cache policy

The raw source zip is the only default durable EXIOBASE data cache:

```text
data/exiobase3_YYYY/IOT_YYYY_pxp.zip
```

Default loaders must not create:

- full `pymrio` pickle caches;
- extracted text-directory caches;
- `A.txt`;
- derived matrix caches such as `.npz` or parquet;
- any cache that becomes an authoritative source instead of the raw zip.

Optional caches must be profile/script opt-in, documented, and safely regenerable from the source zip.

## Table-level loading policy

Load only the tables requested by the caller.

For `load_mrio_matrices()` this means:

- `x.txt`;
- `Z.txt`;
- `Y.txt`;
- metadata needed to identify the source zip and basis.

Selected extension readers may read tables such as `air_emissions/F.txt`, but only when the caller actually needs that extension.

All requested monetary source values are fail-closed. Blank, malformed, truncated, or
non-finite observations in `x.txt`, `Z.txt`, or `Y.txt` must raise before coefficients
or donor profiles are constructed. The region-slice `Z` and `Y` row bases must match;
a missing `Y` row is not an implicit all-zero row. Explicit numerical zeros remain
valid zeros.

## Sparse construction policy

`A` is derived from `Z` and `x`:

```text
A_ij = Z_ij / x_j
```

The loader must not require `A.txt`.

`Z` and `A` must be sparse for full EXIOBASE basis work. Diagnostics must accept sparse input and must not convert sparse matrices to dense objects unless guarded by a small-test-matrix check.

Forbidden on full EXIO matrices:

```python
pd.DataFrame(A)
A.values
np.asarray(A)
A.toarray()
A.todense()
```

## Memory and cleanup

MRIO loaders and diagnostics used by the paper pipeline must remain compatible with a 4 GB RSS ceiling.

Required practices:

- stream zip members rather than extracting full text directories;
- construct sparse rows in bounded passes;
- avoid keeping both dense and sparse copies of the same matrix;
- delete temporary arrays and intermediate matrices when no longer needed;
- call `gc.collect()` after large year-level work when running long loops;
- preserve progress messages for long-running loads.

## Basis contract

The basis returned by EXIO loaders is:

```python
pd.MultiIndex(names=["region", "sector"])
```

`x`, `y`, sparse `A.index`, and sparse `A.columns` should all align to that same basis.

## Disk-space and derived-cache policy

The MRIO repo must not create large derived EXIOBASE caches by default. The raw source zip is the only default durable cache:

```text
data/exiobase3_YYYY/IOT_YYYY_pxp.zip
data/exiobase3_YYYY/IOT_YYYY_ixi.zip
```

Default EXIO loaders must not write:

- full `pymrio.IOSystem` pickle caches;
- extracted text-cache directories;
- generated `A.txt`;
- generated `Z.npz`, `A.npz`, `x.npz`, `y.npz`;
- parquet/feather equivalents of `Z`, `A`, `x`, or `y`;
- hidden matrix-cache directories.

Development-only caches are permitted only behind explicit opt-in flags. Such flags must:

1. default to off;
2. state expected disk use;
3. write under an ignored scratch/development path;
4. never be used by production paper profiles;
5. be safely deletable without changing scientific results;
6. have cleanup instructions.

Do not remove source zips during cleanup. They avoid expensive redownloads and are the authoritative input.

Suggested inspection command before cleanup:

```bash
find data -type f \( -name '*.pkl' -o -name '*.pickle' -o -name '*.npz' -o -name 'A.txt' \) -print
```

Inspect the list before deleting. Keep all `IOT_*_pxp.zip` and `IOT_*_ixi.zip` files.
