
# MRIO API specification draft

## Goals

The MRIO API should separate generic MRIO concepts from dataset-specific source mechanics. EXIOBASE DOI handling, zip layout, and `pxp`/`ixi` system choices belong to the EXIOBASE provider layer, not to the generic API semantics.

## Source model

A future generic source descriptor should be shaped like:

```python
@dataclass(frozen=True)
class MrioSource:
    dataset: str
    year: int
    version: str | None = None
    system: str | None = None
    data_root: Path | None = None
```

Dataset-specific interpretation examples:

- EXIOBASE 3: `dataset="exiobase3"`, `version="10.5281/zenodo.20051562"`, `system="pxp"` or `"ixi"`.
- WIOD 2016: `dataset="wiod16"`, `version` names the release/source archive policy.
- OECD ICIO: `dataset="oecd_icio"`, `version` names the OECD release, such as `v2018`.

The generic layer should pass `version` through. Providers decide whether it is a DOI, release tag, file bundle version, or local-source identifier.

## Source-boundary contract

Every dataset provider must implement a source-boundary check before reading data.

For EXIOBASE 3 Phase 1:

- The source artifact is `IOT_<year>_<system>.zip`.
- The provenance artifact is `download_log.json` in the same year directory.
- Existing zips are accepted only when the recorded DOI/version matches the requested normalized DOI/version.
- Existing zips with missing or mismatched DOI/version fail loudly unless the caller forces replacement.
- Integrity checking must include at least the existing source-structure check (`file_parameters.json`) and DOI/version provenance checking.

## Public API layers

### Full MRIO object

```python
load_mrio(...)
```

Purpose: load a full MRIO object for datasets where that is practical and explicitly requested.

Policy:

- This should stay generic.
- EXIOBASE full-object loading must not be silently invoked by region-slice or sparse-matrix workflows.
- If EXIOBASE full-object loading is restored, it should require an explicit large-load opt-in.

### Backward-compatible sparse matrix tuple

```python
load_mrio_matrices(...) -> tuple[A, x, y, basis_index, metadata]
```

Current compatibility contract:

- Returns sparse technical coefficients `A`, output vector `x`, summed final demand vector `y`, basis index, and metadata.
- Default behavior must not break existing SDA callers.
- For EXIOBASE, this remains zip-streaming and does not write derived matrix caches.

### Matrix result object

```python
load_mrio_matrix_result(...) -> MrioMatrixResult
```

Draft result:

```python
@dataclass(frozen=True)
class MatrixResult:
    Z: csr_matrix | None
    A: csr_matrix | None
    x: pd.Series | None
    y: pd.Series | None
    Y: pd.DataFrame | csr_matrix | None
    index: pd.Index | pd.MultiIndex
    metadata: dict[str, Any]
```

Request options should include:

- matrix selection may request `"A"`, `"Z"`, `"both"`, or `"none"`; `"none"` reads no Z/A matrix.
- `final_demand="summed"`, `"matrix"`, or `"none"`.

`load_mrio_matrices()` should become a wrapper around this object only when doing so does not break memory behavior or return semantics.

### Region-slice result

```python
load_mrio_region_slice(source, request) -> RegionSliceResult
```

Region slices are not merely matrix selections. They represent destination-region use blocks with flow semantics:

- `flow="domestic"`: origin region equals destination region.
- `flow="imports"`: origin region differs from destination region, aggregated by source sector.
- `flow="total"`: all origin regions aggregated by source sector.

For EXIOBASE, region slices should stream directly from the DOI-checked source zip by default. Persistent slice caches are optional and must be off by default.

## Dataset-specific wrappers

Dataset-specific wrappers may remain for readability:

```python
load_exiobase3_region_slice(...)
load_exiobase3_extension_row_for_year(...)
```

They should call the same provider/source-boundary machinery as generic APIs and must not duplicate divergent version-checking logic.

## Non-goals for Phase 1

- No full generic provider registry.
- No change to region-slice implementation yet.
- No change to the `load_mrio_matrices()` return shape.
- No EXIOBASE full-object re-enable.

## EXIOBASE region slice contract

`load_exiobase3_region_slice(...)` is the EXIOBASE-specific implementation of the future generic region-slice API. Its default behavior is direct source-zip streaming: it ensures the requested EXIOBASE zip through the DOI/version-checked source boundary, reads only `x.txt`, `Z.txt`, and `Y.txt`, and returns an in-memory `ExioRegionSlice`. It must not call the full EXIOBASE `load_mrio`/`pymrio.IOSystem` path in the default region-slice workflow.

Derived region-slice pickle caches are optional and disabled by default. When enabled, cache metadata must include `year`, `system`, `region`, `flow`, and `exio_doi`, and a request with mismatched metadata must fail rather than silently reusing stale slices.


## Phase 3 EXIOBASE matrix-result contract

`load_mrio_matrix_result(...)` is the Phase 3 structured result API for EXIOBASE zip-stream reads. It accepts `matrix="A"`, `"Z"`, `"both"`, or `"none"` and `final_demand="summed"`, `"matrix"`, or `"none"`. `matrix="none"` is the Y-only path: it reads `x.txt` and the requested final-demand representation without parsing `Z.txt` or deriving `A`. The existing `load_mrio_matrices(...)` API remains stable and delegates to this object with `matrix="A"` and `final_demand="summed"`.

The Phase 3 implementation is intentionally not yet a full provider registry. It keeps EXIOBASE DOI/system handling in the EXIOBASE source boundary while making matrix-result requests explicit and testable.
