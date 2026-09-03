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

## Status and compatibility

The implemented public source boundaries are deliberately small:

- `load_mrio(...)` for practical full `pymrio.IOSystem` providers;
- `load_mrio_iot_matrices(...)` for selected primitive square-IOT accounts `Z`, `x`, and `Y`;
- explicit `derive_A_from_Z_and_x(...)` and `derive_nonexport_final_demand_from_Y(...)`;
- `compile_region_aggregation(...)` for validated many-to-one region membership;
- `aggregate_mrio_regions(...)` for complete primitive monetary region aggregation; and
- `load_mrio_iot_scope_slice(...)` for low-memory destination-scope use blocks.

The earlier `MrioMatrixResult`, tuple wrappers, `MrioSource`, EXIO singleton/plural
region-slice loaders, and derived slice pickle cache are deleted rather than kept as
compatibility aliases. Environmental extension region aggregation remains a later
work package and must build on extensive physical and monetary accounts.

## Source selection

Source selection is expressed directly on the loader call: dataset, year and the
provider-specific source controls required to locate an authoritative release.
The public transformation functions do not carry a second `MrioSource` descriptor.
EXIOBASE interprets `doi` as its source version and `exio_system` as `pxp`/`ixi`;
other providers may use their own full-object adapters under the same primitive
`Z`/`x`/`Y` result contract.

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

Scope mappings are independent of fuels, fuels-more, and green/brown sector
profiles. Checked-in region-scope helpers live under
`scripts/pipelines/region_scopes/`, consistent with the repository rule that
pipeline helper CSVs live under `scripts/pipelines/`. The initial implementation
ships the versioned `EU27_2020` membership there; custom mappings may still be
supplied explicitly.

## Public API layers

### Full MRIO object

~~~python
mrio = load_mrio(dataset="wiod16", year=2014, ...)
~~~

`load_mrio(...)` returns one full `pymrio.IOSystem` for providers where that is
practical. Full EXIOBASE loading remains deliberately disabled; EXIOBASE analysis
uses the source-ZIP primitive loader below.

### Primitive square-IOT accounts

~~~python
matrices, basis_index, metadata = load_mrio_iot_matrices(
    "exiobase",
    2022,
    matrices=("Z", "x", "Y"),
    ...,
)
~~~

`matrices` is a dictionary containing exactly the requested primitive source
accounts. Allowed requests are `"Z"`, `"x"`, and `"Y"`. The loader does not expose
`A` or summed `y` as source selectors. For EXIOBASE, unrequested ZIP members are
not read.

Derived objects are explicit:

~~~python
A = derive_A_from_Z_and_x(
    matrices["Z"], matrices["x"], basis_index=basis_index
)
y_nonexport = derive_nonexport_final_demand_from_Y(matrices["Y"])
~~~

`derive_A_from_Z_and_x` uses the column-output convention `A_ij = Z_ij / x_j`,
requires exact basis alignment, maps zero-output columns to zero coefficients, and
does not mutate caller-owned `Z` by default. The non-export final-demand helper is
a model boundary, not a source account; callers that need all final demand can use
`Y.sum(axis=1)` explicitly.

### Geography aggregation

`compile_region_aggregation(...)` validates an unweighted many-to-one membership plan.
`aggregate_mrio_regions(...)` applies a **complete** plan to the canonical primitive
`Z`/`x`/`Y` bundle returned by `load_mrio_iot_matrices(...)`:

~~~python
matrices_R, basis_R, metadata_R = aggregate_mrio_regions(
    matrices,
    basis_index=basis_index,
    plan=plan,
    metadata=metadata,
)
~~~

The operation requires every source region to be assigned exactly once. A partial
scope is valid for `load_mrio_iot_scope_slice(...)`, but not for a complete-IOT
aggregation; callers normally request `include_complement=True` when compiling a
single named scope such as `EU27_2020`.

The monetary transformation is extensive-first:

$$
Z^R = R Z R^\mathsf{T},\qquad x^R = R x,\qquad Y^R = RYH^\mathsf{T}.
$$

`R` maps `(region, sector)` rows/columns while preserving the sector classification;
`H` maps `(region, final-demand-category)` columns while preserving final-demand
categories. Every source region must expose the same unique ordered sector basis and
the same unique ordered final-demand-category basis. Malformed or incomplete bases
fail closed rather than being filled with zero or silently dropped.

The result contains only primitive `Z`, `x`, and `Y`, on a new `(scope, sector)`
basis and `(scope, category)` final-demand columns. It does **not** aggregate or
return `A`. Only after the extensive transformation may callers derive

$$
A^R_{ij}=\frac{Z^R_{ij}}{x^R_j}
$$

with `derive_A_from_Z_and_x(...)`. Environmental extensions are not part of WP2;
their extensive-first region transformation is a later work package. The full
EXIOBASE production path therefore remains source-ZIP based and does not create a
full `pymrio.IOSystem` merely to aggregate geography.

### Low-memory destination-scope slice

~~~python
@dataclass(frozen=True)
class MrioScopeSlice:
    scope: RegionScope
    Z: pd.DataFrame
    Y: pd.DataFrame
    x_scope: pd.Series
    flow: Literal["internal", "external_imports", "total"]
    metadata: dict[str, object]

slice_ = load_mrio_iot_scope_slice(
    "exiobase",
    2022,
    scope=RegionScope(code="AT", members=("AT",)),
    flow="internal",
    ...,
)
~~~

This is a low-memory destination-use boundary, not a second full-IOT loader.
`Z` and `Y` contain use by destinations in the scope; origin rows are selected by
the requested boundary. `x_scope` is production of scope members and therefore is
identical across `internal`, `external_imports`, and `total` requests. A caller
must not combine an external-import `Z` with `x_scope` to derive `A`.

The EXIO-specific scope streaming function is a private provider implementation.
There are no public singleton-country or plural EXIO slice wrappers and no derived
slice pickle cache; the DOI-checked source ZIP remains the reproducibility cache.

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

For external imports, a many-to-many product projection must not transport a
raw source intensity. Let \(P_{ak}\) be the same donor-to-target product mapping
used by the sector route. Retain origin \(o\) until the product transformation has
been applied, and form the extensive origin/target-product accounts

\[
\bar F_{oa}=\sum_k P_{ak}F_{ok},
\qquad
\bar x_{oa}=\sum_k P_{ak}x_{ok},
\qquad
\bar u_{oa\to G}=\sum_k P_{ak}u_{ok\to G}.
\]

Then the direct production stressor associated with imports is

\[
F^{\mathrm{imp},G}_a
 = \sum_{o\notin G}\frac{\bar F_{oa}}{\bar x_{oa}}
   \bar u_{oa\to G},
\qquad
m^{\mathrm{imp},G}_a
 = \sum_{o\notin G}\bar u_{oa\to G},
\]

\[
e^{\mathrm{imp},G}_a =
\frac{F^{\mathrm{imp},G}_a}{m^{\mathrm{imp},G}_a}.
\]

A positive \(\bar u_{oa\to G}\) with zero or missing \(\bar x_{oa}\) fails
closed. This preserves the existing numerator-first many-to-many extension rule:
\(F\), \(x\), and import use are projected to the same target-product basis before
an origin-specific coefficient is formed. Only then are origins weighted by their
actual imports into the scope. The result is a direct origin-production import
account, not a Leontief footprint or full embodied-import intensity.

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

WP3 implements these boundaries as independent geography/satellite operators:
`aggregate_scope_production_extension(...)`,
`aggregate_external_import_extension_direct(...)`,
`aggregate_scope_final_demand_extension(...)`, and
`combine_total_supply_extension_direct(...)`. The production/import operators accept
a complete product projection whose non-negative weights sum to one for every source
product; both extensive numerator and denominator traverse that projection before any
ratio is formed. The EXIO provider exposes
`read_exiobase3_scope_external_import_use_by_origin(...)` so the external origin is
retained through product projection. `F_Y` has no product-projection argument.

These operators do not migrate the fuels/fuels-more split pipeline. They produce the
boundary-aware accounts that a later pipeline adapter consumes; CLI wiring and
DOM/IMP/TOTAL output integration remain WP4.

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
or fallbacks. Scope production can therefore collapse geography before applying
\(P\). The external-import direct account is the important exception to an
unqualified "collapse geography first" rule: origin must be retained through the
product projection because the import account forms an origin-specific production
coefficient before weighting origins by actual use. No raw `F/x` intensity is
projected through a many-to-many product mapping.

The generic geography API owns `RegionScope`, `RegionAggregationPlan`, region
validation, and low-memory scope slices. The generalized split pipeline consumes
those objects; any `DonorScopeContext` is a pipeline adapter assembled from the
generic geography results, not a second geography implementation. The sector API
does not parse region mappings:

~~~python
donor = load_mrio_iot_scope_slice(
    "exiobase", 2022, scope=eu_scope, flow="internal"
)
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
members. Persistent derived scope-slice caches are not part of the public API; the
DOI-checked source ZIP is the reproducibility cache.

## Compatibility and migration

The breaking API cleanup is complete for monetary loading and geography:

1. `load_exiobase3_region_slice(s)`, `load_exiobase3_scope_slice`, `MrioSource`,
   `ExioRegionSlice`, and derived region-slice caches remain deleted; no singleton
   compatibility wrappers are retained.
2. `load_mrio_iot_scope_slice(...)` is the one low-memory destination-scope reader.
3. `aggregate_mrio_regions(...)` is the one complete primitive monetary region
   aggregation, and requires complete source-region coverage.
4. Environmental extension geography is implemented by the boundary-aware WP3
   satellite operators. Production and import coefficients are formed only after their
   required extensive aggregations/projections; direct final demand remains on its
   category axis, and target total supply requires authoritative DOM/IMP controls.
5. Split-pipeline donor-context assembly and CLI/output wiring remain WP4 and must
   consume the generic geography/satellite APIs rather than reimplement them.

> **Current loader contract:** `load_mrio_iot_matrices(dataset, year, matrices=(...))` loads only requested primitive square-IOT accounts `Z`, `x`, and `Y`. `A` and non-export `y` are explicit derivations; obsolete compatibility APIs were deleted.

<!-- wp4-region-pipeline -->
### Region-aware fuels/fuels-more production path

WP4 wires the existing fuels/fuels-more sector mappings to the generic geography and
environmental APIs. AT is a singleton donor scope and uses no region mapping CSV.
`EU27_2020` is compiled from `scripts/pipelines/region_scopes/eu27_2020.csv` against
the actual EXIO source-region basis; RoW is its complement. The target monetary EU
IOT is the directly published Eurostat `geo=EU27_2020` table, not a sum of national
Eurostat IOTs. DOM, IMP/RoW and TOTAL environmental sidecars keep distinct boundaries
and are reaggregated by summing physical numerators and their matching monetary
denominators before deriving intensity. Aggregate exports exclude intra-scope trade.
