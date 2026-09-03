# EXIO low-RAM zip-stream contract

This is the MRIO-side contract for reusable EXIOBASE source loading under data-heavy analysis workflows.

## Durable cache policy

The DOI-checked raw source zip is the only default durable EXIOBASE data cache:

```text
data/exiobase3_YYYY/IOT_YYYY_pxp.zip
data/exiobase3_YYYY/IOT_YYYY_ixi.zip
```

Default loaders must not create full `pymrio` pickle caches, extracted text directories, generated `A.txt`, derived matrix caches, or hidden alternative authoritative sources.

## Complete-IOT primitive loading

`load_mrio_iot_matrices(dataset, year, matrices=(...))` loads only requested primitive source accounts. The supported account names are exactly `Z`, `x`, and `Y`.

For EXIOBASE:

- `Z` comes from `Z.txt` and remains sparse at full basis;
- `x` comes from `x.txt`;
- `Y` comes from `Y.txt`;
- metadata identifies the source zip and canonical basis;
- a Y-only request must not read x/Z, and analogous selective requests must avoid unrelated members.

Technical coefficients are an explicit post-load derivation:

```text
A_ij = Z_ij / x_j
```

using `derive_A_from_Z_and_x(...)`. The loader does not expose `A` or summed `y` as source-account selectors and does not require `A.txt`.

## Geography boundaries

`aggregate_mrio_regions(...)` transforms complete primitive monetary accounts extensively:

```text
Z^R = R Z R^T
x^R = R x
Y^R = R Y H^T
```

and callers derive `A` only afterward.

`load_mrio_iot_scope_slice(...)` is the low-memory destination-use boundary. It streams only destination columns for a declared `RegionScope` and returns `internal`, `external_imports`, or `total` Z/Y use together with separate member-production `x_scope`. These use blocks are not interchangeable with a balanced complete IOT.

For direct environmental imports, origin must remain explicit until after product projection. `read_exiobase3_scope_external_import_use_by_origin(...)` therefore streams origin-product Z+Y use into a scope without forming an intensity.

## Fail-closed source semantics

Requested monetary values must be finite. Blank, malformed, truncated, duplicate, basis-mismatched, or non-finite observations in requested x/Z/Y members raise before coefficients or donor profiles are constructed. Explicit numerical zero remains a valid zero; missing evidence is never converted to zero.

Region-scope streaming must prove identical ordered destination bases across members, identical ordered producer-sector bases across origins, and exact x/Z sector equality.

## Sparse and memory policy

Full-basis EXIO Z and derived A must remain sparse. Do not convert EXIO-scale sparse matrices to dense arrays or DataFrames unless a small-fixture guard proves the operation is bounded.

Long loops should release large temporary matrices promptly and remain compatible with the repository's resource-bounded analysis policy.

## Development caches

Optional development-only caches are permitted only behind explicit opt-in controls, under ignored scratch paths, with documented disk use and safe regeneration from the source zip. They must never become production source-of-truth artifacts.

Never remove source zips during routine cleanup; they are the authoritative reproducibility cache and avoid unnecessary redownloads.

### Provider-independent primitive result validation

The public `load_mrio_iot_matrices(...)` boundary applies the same result checks
after every provider adapter: requested Z/Y values are finite, x is aligned,
finite and non-negative, and Y uses the declared producer basis. Provider-specific
parsers may be heavier internally, but they may not weaken the public missingness
or denominator semantics.
