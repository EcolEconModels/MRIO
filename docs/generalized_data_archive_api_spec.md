# Generalized source-archive download and parse API specification

## Purpose

This document specifies a small, pymrio-like data access layer for MRIO-side
inputs used by disaggregation, reaggregation, and diagnostics. The design
extends the existing `download_*`, `parse_*`, and `load_*` pattern while
adopting a strict source-archive cache policy.

The API must support EXIOBASE, Eurostat IOT/SUT-style datasets, Eurostat Energy
Balances, PEFA tables, PRODCOM, international trade, SBS/STS-style activity
data, and national statistical-office datasets without making any dataset code,
sector code, product code, or country-specific rule a Python constant.

All dataset names, dimensions, code lists, selectors, and mappings must be read
from dictionaries and recipe files.

## Non-negotiable invariants

1. The only durable on-disk cache is the raw source archive.
2. Default loaders must not write extracted text directories, parsed pickle
   caches, matrix caches, parquet/feather caches, generated `A.txt`, or full
   `pymrio.IOSystem` pickles.
3. When an upstream source is already a zip archive, keep that archive as the
   authoritative cache.
4. When an upstream source returns JSON, CSV, TSV, SDMX, gzip, or another
   single response format, wrap the unmodified response plus request metadata in
   a repo-created source zip. Do not persist loose raw files next to it.
5. Parsers must read only the requested zip members into memory and must never
   extract whole archives to disk as part of normal operation.
6. Derived objects returned by `load_*` are in-memory products of the source
   archive and the supplied parse spec; they are not authoritative cached data.
7. Full-basis MRIO matrices must remain sparse where applicable. Dense pandas
   DataFrames are allowed only for country-scale or explicitly small table
   outputs.
8. EXIO-style MRIO vector and matrix APIs must preserve the strict
   `MultiIndex(region, sector)` basis when returning full MRIO vectors or sparse
   matrices.
9. Source archive filenames must be deterministic from provider, dataset,
   request filters, vintage/version, and system/profile fields, not from Python
   call-site names.
10. Every cached archive must include enough metadata to reproduce the request
    and identify the parser contract used to read it.

## Relationship to existing repo contracts

This spec generalizes the existing low-RAM EXIO zip-stream rule:

- keep raw source zips as the durable input;
- read only requested members;
- derive sparse `A` from `Z` and `x`;
- do not create full-object pickles or extracted text caches by default.

The same rule is extended to all other relevant datasets. The current Eurostat
`download_*` utilities cache raw SDMX-JSON responses and intentionally avoid
parsed pickle caches. The generalized API keeps that good behavior but changes
the durable cache unit from loose raw JSON files to source zip archives.

## Pymrio-like API contract

Each provider implements three layers.

```python
def download_dataset_archive(
    request: DatasetRequestSpec,
    *,
    archive_store: SourceArchiveStore | None = None,
    force_download: bool = False,
    timeout_s: int = 120,
) -> SourceArchiveRef:
    """Download or reuse a raw source archive only."""
```

```python
def parse_dataset_archive(
    archive: SourceArchiveRef,
    parse_spec: ParseSpec,
) -> ParsedDataset:
    """Read only requested members from the archive and return in-memory tables."""
```

```python
def load_dataset(
    request: DatasetRequestSpec,
    parse_spec: ParseSpec,
    *,
    archive_store: SourceArchiveStore | None = None,
    force_download: bool = False,
    force_reparse: bool = False,
) -> ParsedDataset:
    """Convenience wrapper: download/reuse source archive and parse in memory."""
```

`force_reparse` is retained for API compatibility but must not trigger creation
of a derived parsed cache. It only forces the in-memory parser to run again.

## Core dataclasses

```python
@dataclass(frozen=True)
class DatasetRequestSpec:
    provider: str
    dataset: str
    filters: Mapping[str, tuple[str, ...]]
    version: str | None = None
    format_hint: str | None = None
    source_url: str | None = None
    profile: str | None = None
```

```python
@dataclass(frozen=True)
class SourceArchiveRef:
    provider: str
    dataset: str
    request_hash: str
    archive_path: Path
    manifest_member: str
    request_member: str
```

```python
@dataclass(frozen=True)
class ParseSpec:
    parser: str
    table_role: str
    row_dim: str | None = None
    col_dim: str | None = None
    value_dim: str | None = None
    filters: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    member_selectors: tuple[str, ...] = ()
```

```python
@dataclass(frozen=True)
class ParsedDataset:
    tables: Mapping[str, pd.DataFrame | pd.Series]
    dimensions: Mapping[str, pd.DataFrame]
    metadata: Mapping[str, object]
    archive: SourceArchiveRef
```

The dataclasses above are contracts. Implementations may add fields, but callers
must not need provider-specific return classes for ordinary disaggregation use.

## Source archive layout

Default path pattern:

```text
data/source_archives/<provider>/<dataset>/<request_hash>.zip
```

Required zip members:

```text
request.json
source_manifest.json
payload/<upstream-response-filename>
```

Optional zip members:

```text
dimensions/<dimension-id>.json
checksums/<member>.sha256
licenses/<source-license>.txt
```

`source_manifest.json` must contain:

```json
{
  "provider": "...",
  "dataset": "...",
  "request_hash": "...",
  "created_utc": "...",
  "source_url": "...",
  "filters": {},
  "version": null,
  "format_hint": "sdmx-json|tsv|csv|zip|xlsx|other",
  "payload_members": ["payload/raw.json"],
  "parser_contract": "source-archive-v1",
  "sha256": {}
}
```

The archive may contain metadata extracted from the response for faster
inspection, but the raw payload remains the source of truth.

## Provider registry

Provider behavior must be data-driven by a registry file, for example:

```text
scripts/pipelines/<profile>/dataset_registry.csv
```

Required columns:

```text
provider,dataset,enabled,format_hint,base_url,default_unit,default_parser,
cache_policy,member_selector,dimension_dictionary,notes
```

The registry may list Eurostat, EXIOBASE, national statistical-office, or local
manual-file providers. Python code must never branch on a specific dataset code
to decide scientific meaning. It may branch only on generic parser types such as
`sdmx_json`, `zip_table`, `exio_zip`, `csv`, or `xlsx_member`.

## Dictionary files

All classifications and selectors must be external files. Recommended files:

```text
scripts/pipelines/<profile>/dictionaries/provider_datasets.csv
scripts/pipelines/<profile>/dictionaries/dimensions.csv
scripts/pipelines/<profile>/dictionaries/classification_codes.csv
scripts/pipelines/<profile>/dictionaries/units.csv
scripts/pipelines/<profile>/dictionaries/final_demand_codes.csv
scripts/pipelines/<profile>/dictionaries/table_roles.csv
```

No sector, product, flow, stressor, NACE, CPA, SIEC, PRODCOM, CN, PEFA, or EXIO
code may be hard-coded in provider logic. Selectors must be looked up by
semantic roles in dictionaries.

## Relevant dataset roles

The generalized downloader must support dataset roles, not hard-coded dataset
names:

```text
iot_monetary
energy_balance_quantity
pefa_supply_use
pefa_bridge
prodcom_value_quantity
trade_value_quantity
activity_statistics
full_mrio_donor
manual_override
```

A profile maps these roles to concrete datasets. For example, an EU profile can
point the energy-balance role to Eurostat Energy Balances, while a national
profile can point the same role to a national energy-statistics archive.

## Parser responsibilities

### EXIO/full MRIO zip parser

- Read only requested members such as `Z`, `Y`, `x`, and selected extension
  tables.
- Preserve row/column labels and strict `MultiIndex(region, sector)` when
  returning full MRIO matrices.
- Derive sparse `A` from `Z` and `x`; do not read or generate `A.txt` as a
  durable artifact.
- Never create full-object pickles by default.

### Eurostat SDMX parser

- Read `payload/raw.json` from the source archive.
- Convert sparse SDMX observations to a `Series` or selected 2D tables.
- Validate requested dimensions against payload dimensions and emit useful
  dimension-code diagnostics.
- Persist no parsed data.

### Generic zip-table parser

- Open the source archive, select only matching member names, and read those
  members directly with pandas or streaming parsers.
- Do not extract the archive.
- Fail if a member selector matches multiple files unless the parse spec declares
  that multiple matches are expected.

### Dictionary parser

- Read small CSV/JSON dictionaries from the repo, not from data caches.
- Validate required columns and duplicate keys.
- Return normalized code-label tables with explicit classification-system names.

## Archive-store implementation rules

The archive store must provide:

```python
class SourceArchiveStore:
    def archive_path(self, request: DatasetRequestSpec) -> Path: ...
    def exists(self, request: DatasetRequestSpec) -> bool: ...
    def write_archive(self, request: DatasetRequestSpec, payload: bytes, metadata: Mapping[str, object]) -> SourceArchiveRef: ...
    def open_member(self, archive: SourceArchiveRef, member: str) -> BinaryIO: ...
```

`write_archive()` must write atomically via a temporary file and rename after the
zip and manifest are complete. It must not delete existing source archives unless
`force_download=True` is supplied by the caller.

## Provenance and reproducibility

Every parsed table returned to the disaggregation compiler must carry:

```text
provider
dataset
request_hash
archive_path
source_member
parse_spec_hash
unit
classification_system
filters
```

Diagnostics must propagate this provenance into `evidence_resolution.csv` and
`manifest.json` for each split run.

## Failure behavior

The API must fail loudly for:

- missing archive members;
- non-singleton dimensions in a parser that expects a 2D table;
- duplicate dictionary keys;
- unit mismatches without a conversion rule;
- missing required selectors for enabled recipe rows;
- full-matrix densification attempts in full MRIO paths;
- attempts to write derived caches under default settings.

The API may degrade gracefully only when the profile declares a fallback provider
and the fallback is recorded in diagnostics.

## Compatibility and migration notes

1. Existing EXIO zip-stream loaders already satisfy most of this spec and should
   be treated as the reference implementation for archive-member streaming.
2. Existing Eurostat loaders already follow the `download_*`, `parse_*`, and
   `load_*` shape and avoid parsed caches. They should be migrated from loose
   `raw.json` caches to zipped source archives with a compatibility reader for
   old raw-cache directories during a transition period.
3. Existing Energy Balance split code should stop reading dataset-specific
   selectors directly from code paths. The selectors should remain in recipe CSVs
   and dictionaries, and the compiler should consume those files through generic
   provider roles.
4. The downloader layer must not know which sectors are being split. Sector and
   product meaning belongs to the disaggregation profile.
