# MRIO API specification draft

## Goals

The MRIO API separates four concerns:

1. source acquisition and provenance;
2. geography selection or aggregation;
3. sector disaggregation/reaggregation; and
4. derivation of coefficients or intensities from extensive monetary and physical accounts.

Geography and sector transformations are independent. Dataset-specific mechanics
(EXIOBASE DOI and ZIP layout, Eurostat SDMX dimensions, or `pxp`/`ixi` choices)
belong to providers and must not leak into the generic transformation APIs.

## Source model

~~~python
@dataclass(frozen=True)
class MrioSource:
    dataset: str
    year: int
    version: str | None = None
    system: str | None = None
    data_root: Path | None = None
~~~

Provider interpretations include EXIOBASE 3 DOI releases, WIOD releases, and OECD
ICIO releases. The generic layer passes `version` through; the provider decides
whether it is a DOI, release tag, bundle version, or local-source identifier.

## Target geography selection is not donor aggregation

A published aggregate geography must be selected directly when the target provider
offers it. Eurostat compiles consolidated European symmetric IOTs and
`naio_10_cp1700` exposes them through the ordinary `geo` dimension. For example:

~~~python
eu_iot = load_eurostat_iot(
    dataset="naio_10_cp1700",
    filters={
        "geo": "EU27_2020",
        "time": "2021",
        "freq": "A",
        "unit": "MIO_EUR",
        "stk_flow": "DOM",
    },
)
~~~

The same request pattern applies to `IMP` and `TOTAL`. The loader must not sum
national Eurostat tables to recreate a geography that Eurostat publishes. The
published consolidated table owns its boundary treatment, including intra-group
flows and imports from outside the group.

A region mapping is still required for the EXIOBASE donor because EXIOBASE remains
multi-regional, and for a custom target group that its target provider does not
publish. Target geography selection and donor geography aggregation are separate
request fields and must never be inferred from one another.

Eurostat source references:

- [ESA supply, use and input-output table information](https://ec.europa.eu/eurostat/web/esa-supply-use-input-tables/information-data)
- [`naio_10_cp1700` Data Browser](https://ec.europa.eu/eurostat/databrowser/view/NAIO_10_CP1700/defa)

## Region-scope model

A scope is a named set of source regions. Membership is one-to-one from source
region to aggregate region; this operation never disaggregates geography.

~~~python
@dataclass(frozen=True)
class RegionScope:
    code: str
    members: tuple[str, ...]
    label: str | None = None

@dataclass(frozen=True)
class RegionAggregationPlan:
    source_regions: tuple[str, ...]
    scopes: tuple[RegionScope, ...]
    source_to_scope: pd.Series
~~~

The checked-in mapping format is:

~~~text
scope_code,scope_label,member_region,member_order,enabled,notes
EU27_2020,European Union - 27 countries (from 2020),AT,1,true,
EU27_2020,European Union - 27 countries (from 2020),BE,2,true,
...
~~~

Rules:

- `member_region` uses the source provider's canonical region code.
- each enabled member occurs at most once within a plan;
- membership is unweighted and exhaustive for each declared scope;
- `member_order` is deterministic metadata, not an economic weight;
- the complement is derived against the regions actually present in the selected
  source release; and
- time-varying political definitions use versioned scope codes, such as
  `EU27_2020`, rather than a mutable `EU` alias.

Scope mappings belong under a geography mapping directory, not inside a fuels,
fuels-more, or green/brown sector profile.

## Public API layers

### Full MRIO object

~~~python
load_mrio(...)
~~~

This loads a full MRIO object only when explicitly requested and practical. An
EXIOBASE full-object load must not be silently invoked by scope-slice or sparse
matrix workflows.

### Backward-compatible sparse matrix tuple

~~~python
load_mrio_matrices(...) -> tuple[A, x, y, basis_index, metadata]
~~~

This compatibility API retains sparse `A`, `x`, summed `y`, basis index, and
metadata. For EXIOBASE it remains ZIP-streaming and writes no derived matrix cache
by default.

### Matrix result object

~~~python
@dataclass(frozen=True)
class MrioExtensionResult:
    F: pd.DataFrame | None
    F_Y: pd.DataFrame | None
    unit: pd.DataFrame | None

@dataclass(frozen=True)
class MrioMatrixResult:
    Z: csr_matrix | None
    A: csr_matrix | None
    x: pd.Series | None
    y: pd.Series | None
    Y: pd.DataFrame | csr_matrix | None
    index: pd.Index | pd.MultiIndex
    extensions: Mapping[str, MrioExtensionResult] | None
    metadata: dict[str, Any]
~~~

Matrix selection accepts `"A"`, `"Z"`, `"both"`, or `"none"`. Final-demand
selection accepts `"summed"`, `"matrix"`, or `"none"`.

### Geography aggregation

~~~python
def compile_region_aggregation(
    *,
    source_regions: Sequence[str],
    mapping: Path | pd.DataFrame,
    scopes: Sequence[str] | None = None,
    include_complement: bool = False,
    complement_code: str = "RoW",
) -> RegionAggregationPlan:
    """Validate a many-regions-to-one-or-more-regions aggregation."""

def aggregate_mrio_regions(
    result: MrioMatrixResult,
    plan: RegionAggregationPlan,
) -> MrioMatrixResult:
    """Aggregate extensive core and extension accounts; then recalculate coefficients."""
~~~

This is the complete-object/sparse-matrix API. Sector aggregation is neither
required nor implied.

### Low-memory destination-scope slice

~~~python
@dataclass(frozen=True)
class MrioScopeSlice:
    scope: RegionScope
    Z: pd.DataFrame
    Y: pd.DataFrame
    x_scope: pd.Series
    flow: Literal["internal", "external_imports", "total"]
    metadata: dict[str, Any]

def load_mrio_scope_slice(
    source: MrioSource,
    *,
    scope: RegionScope,
    flow: Literal["internal", "external_imports", "total"],
) -> MrioScopeSlice:
    """Stream a destination-scope use block without loading the full MRIO."""
~~~

`x_scope` is production output of members of the scope and therefore does not
change with the requested destination-use flow. Flow-specific import-use
denominators are separate named accounts; they must not be stored as `x_scope`.

The existing EXIO singleton API remains as a wrapper:

~~~python
load_exiobase3_region_slice(region="AT", flow="domestic")
~~~

For a singleton, `domestic` aliases `internal` and `imports` aliases
`external_imports`. New generic code uses the boundary-neutral names.

## Scope-flow accounting

Let \(G\) be a set of EXIO regions, \(o\) an origin, \(d\) a destination, \(k\)
an origin product, \(j\) a using product, and \(c\) a final-demand category.

The consolidated internal blocks are

\[
Z^{\mathrm{int},G}_{kj}
 = \sum_{o\in G}\sum_{d\in G} Z_{ok,dj},
\qquad
Y^{\mathrm{int},G}_{kc}
 = \sum_{o\in G}\sum_{d\in G} Y_{ok,dc}.
\]

External imports are

\[
Z^{\mathrm{imp},G}_{kj}
 = \sum_{o\notin G}\sum_{d\in G} Z_{ok,dj},
\qquad
Y^{\mathrm{imp},G}_{kc}
 = \sum_{o\notin G}\sum_{d\in G} Y_{ok,dc}.
\]

`total` is the cellwise sum of those blocks. Scope production output is

\[
x^G_k = \sum_{o\in G}x_{ok}.
\]

Required invariants are

\[
Z^{\mathrm{total},G}=Z^{\mathrm{int},G}+Z^{\mathrm{imp},G},
\qquad
Y^{\mathrm{total},G}=Y^{\mathrm{int},G}+Y^{\mathrm{imp},G}.
\]

These equations are a destination-use aggregation. They do not manufacture a
balanced standalone IOT for an arbitrary group unless all required production,
trade, valuation, and supplementary accounts are also consolidated and validated.

## Environmental extensions

Environmental transformations are extensive-first. Raw `F` and its monetary
denominator are aggregated or projected before division. `F_Y` is transformed on
the destination-region/final-demand axis and is never projected through a product
mapping.

### Scope production

For a production stressor,

\[
F^G_k=\sum_{o\in G}F_{ok},
\qquad
e^{\mathrm{prod},G}_k =
\frac{F^G_k}{x^G_k}.
\]

This is a territorial/scope-production coefficient. It is appropriate for the
published target domestic-output control after it is projected to the target
product basis. It is not an import coefficient.

### External-import direct production

Let the monetary use of origin-product \(ok\) by destinations in \(G\) be

\[
u_{ok\to G}
 = \sum_{d\in G,j}Z_{ok,dj}
 + \sum_{d\in G,c}Y_{ok,dc}.
\]

For \(o\notin G\), the direct production stressor associated with imports is

\[
F^{\mathrm{imp},G}_k
 = \sum_{o\notin G}\frac{F_{ok}}{x_{ok}}u_{ok\to G},
\qquad
m^{\mathrm{imp},G}_k
 = \sum_{o\notin G}u_{ok\to G},
\]

\[
e^{\mathrm{imp},G}_k =
\frac{F^{\mathrm{imp},G}_k}{m^{\mathrm{imp},G}_k}.
\]

Terms with positive use and zero or missing origin output fail closed. This is a
gross-import-weighted direct production coefficient. It is preferable to an
unweighted coefficient for a single collapsed RoW region. It is not a Leontief
footprint or a full embodied-import intensity.

### Total target supply

Do not copy the scope-production coefficient onto `TOTAL`. When the target
provider supplies domestic-output and import-supply components, combine implied
extensive totals using those authoritative target denominators:

\[
\widehat F^{\mathrm{total}}_a =
e^{\mathrm{prod},G}_a x^{\mathrm{target,dom}}_a
+e^{\mathrm{imp},G}_a m^{\mathrm{target,imp}}_a,
\]

\[
e^{\mathrm{total}}_a =
\frac{\widehat F^{\mathrm{total}}_a}
{x^{\mathrm{target,dom}}_a+m^{\mathrm{target,imp}}_a}.
\]

The target identity against total supply must be validated. Falling back to EXIO
domestic/import monetary shares is allowed only as an explicit lower-quality
policy when the target components are unavailable.

### Direct final demand

For a final-demand stressor,

\[
F^{Y,G}_c=\sum_{d\in G}F^Y_{dc},
\qquad
y^G_c=\sum_{d\in G}\sum_o\sum_kY_{ok,dc},
\qquad
e^{Y,G}_c=\frac{F^{Y,G}_c}{y^G_c}.
\]

`F_Y` remains a category sidecar (households, government, and so on). There is no
RoW-import version of `F_Y` because the account records direct emissions at the
destination's final-demand activity, not emissions of foreign producers.

Generic long-form environmental accounts use explicit boundaries and denominator
roles:

~~~text
axis_position,axis_label,boundary,denominator_role,extension,stressor,unit,
value,monetary_denominator,intensity
~~~

Allowed initial boundaries are `scope_production`,
`external_imports_direct`, `total_supply_direct`, and
`scope_final_demand_direct`. Missing evidence is not zero.

## Independence of geography and sector transformations

Let \(R\) be a region concordance and \(P\) a sector concordance/projection.
Region aggregation and sector transformation are separate operators. For fixed
linear mappings, extensive accounts may be composed in either order; the combined
core concordance is \(R\otimes P\). For example, a full MRIO uses

\[
Z'=(R\otimes P)Z(R\otimes P)^\top.
\]

The implementation should nevertheless use the order that minimizes memory:
EXIOBASE normally streams and collapses the region scope first, then applies the
sector projection to the much smaller slice.

This commutation applies only to extensive numerators and denominators under fixed
mappings. It does not apply to ratios, target-dependent share resolvers, balancing,
or fallbacks. Intensities must always be derived after all relevant extensive
transformations.

The sector API consumes a donor context and does not parse region mappings:

~~~python
donor = load_mrio_scope_slice(source, scope=eu_scope, flow="internal")
plan = compile_split_plan(..., donor_context=donor)
result = run_split_plan(..., plan=plan)
~~~

The same `donor` can be inspected or used without any sector split, and the same
sector pipeline can consume a singleton or aggregated donor context.

## Relation to pymrio

[pymrio `IOSystem.aggregate`](https://pymrio.readthedocs.io/en/latest/api_references.html)
already supports independent `region_agg` and `sector_agg` concordances. Its
implementation resets extensions to flows, aggregates `Z` with the combined
region-sector concordance, aggregates `Y` with a separate
region/final-demand-category concordance, sums `x`, aggregates `F` and `F_Y` on
their correct axes, and then recalculates extension coefficients.

That is the semantic reference for full-object validation. It is not the default
EXIOBASE production path here because loading a full EXIOBASE `IOSystem` violates
the repository's low-memory scope-slice contract. Tests should compare a small
streamed scope result against pymrio aggregation of the same fixture.

## Source-boundary and cache contract

Every provider checks source provenance before reading. EXIOBASE uses
`IOT_<year>_<system>.zip` and its adjacent `download_log.json`; DOI/version
mismatches fail unless replacement is explicitly requested. Integrity checking
includes source structure and version provenance.

EXIOBASE scope slices stream directly from the checked ZIP and read only required
members. Persistent derived caches are disabled by default. If enabled, cache
metadata includes year, system, scope code, ordered members, flow, source
DOI/version, and mapping hash; mismatches fail rather than silently reusing a
stale slice.

## Compatibility and migration

1. Keep `load_exiobase3_region_slice` and current single-region CLI flags as
   singleton wrappers.
2. Add a mutually exclusive scope-mapping argument; do not add EU-specific code.
3. Move shared extensive-first extension projection/reaggregation into one
   satellite-account operator.
4. Keep the legacy production sidecar adapter temporarily for
   `scope_production`; new multi-boundary callers use the generic schema.
5. Delete flow-specific DOM/IMP/TOTAL extension branches after they delegate to
   the boundary-aware account builder.
