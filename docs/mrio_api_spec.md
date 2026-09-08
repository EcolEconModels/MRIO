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
requires exact basis alignment, maps a zero-output column to zero coefficients only when that Z buyer column has zero intermediate-input mass; otherwise it fails, and
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

### Total target transaction use

Do not copy the scope-production coefficient onto `TOTAL`. Combine domestic output
with the signed target IMP transaction-use component; do not substitute supply-side
P7 merely because it shares the product classification:

\[
\widehat F^{\mathrm{total}}_a =
e^{\mathrm{prod},G}_a x^{\mathrm{target,dom}}_a
+e^{\mathrm{imp},G}_a u^{\mathrm{target,imp}}_a,
\]

\[
e^{\mathrm{total}}_a =
\frac{\widehat F^{\mathrm{total}}_a}
{x^{\mathrm{target,dom}}_a+u^{\mathrm{target,imp}}_a}.
\]

The target identity against TOTAL transaction use must be validated. The public
function name `combine_total_supply_extension_direct(...)` is retained for API
compatibility, but its import/total target keywords carry import-use/total-use
semantics. Supply-side P7 is not a fallback for signed IMP use.

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
`external_imports_direct`, `total_use_direct`, and
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

<!-- ecolecon-price-index-api -->
## Country-level price indices and monetary price adjustment

MRIO owns the reusable acquisition and numerical transformation boundary for
country-level price-adjustment sensitivities.  Downstream analyses must not fetch
macroeconomic deflators independently or deflate already-derived `A`, SDA, or
extension results after the fact.

### Official source and cache

The initial provider is the United Nations Statistics Division (UNSD) **Analysis
of Main Aggregates (AMA)** REST API:

- `GET /api/Country?countriesOnly=true` supplies the official M49 country domain
  and currency metadata;
- `GET /api/Series` supplies the current series ids and names;
- `POST /api/Data/basic/{serieId}` supplies observations for requested countries
  and years; and
- `GET /api/Data/lastupdated` supplies source-release provenance.

`download_unsd_ama_gdp_deflators(...)` discovers the GDP implicit-price-deflator
series from the official series metadata instead of hard-coding a mutable series
or spreadsheet-file id.  Small raw JSON responses are cached under
`MRIO_DATA_ROOT/price_indices/unsd_ama/` (or an explicitly supplied storage
folder).  `load_unsd_ama_gdp_deflators(...)` returns the validated long-form table
plus provenance metadata.  Source rows retain M49 code, country name, year,
currency basis, price-index value, source series/item ids and names, unit, and
observation note.

The parser recognizes the two AMA country-profile deflator bases:
`national_currency` and `usd`.  `price_adjustment_factors(...)` requires the
currency basis explicitly and computes

\[
\phi_{r,t}^{(b)} = \frac{P_{r,b}}{P_{r,t}},
\]

so multiplying a current-price value from year \(t\) by \(\phi_{r,t}^{(b)}\)
expresses it at the chosen base-year macro price level under that deflator basis.
When an explicit area list is supplied, missing year or base-year observations
fail rather than being interpolated or filled.

The two UNSD bases have different meanings.  The national-currency GDP deflator
tracks the domestic aggregate price level.  The USD deflator also reflects the
conversion of current and constant-price GDP into US dollars and can therefore
incorporate exchange-rate movements.  **Neither basis is silently declared to be
an exact constant-EUR transformation for an EUR-valued MRIO.**  A later analysis
must choose and document the appropriate currency treatment before claiming
constant-EUR results.

### Extensive-account transformation

`adjust_mrio_iot_prices(...)` accepts only a non-empty subset of the primitive
monetary accounts `x`, `Z`, and `Y`, plus a full region-sector row-factor Series.
For

\[
D_t = \operatorname{diag}(\phi_{i,t}),
\]

it applies

\[
x_t^{(b)} = D_t x_t, \qquad
Z_t^{(b)} = D_t Z_t, \qquad
Y_t^{(b)} = D_t Y_t.
\]

The same factor therefore scales every monetary supply row across intermediate
and final uses and preserves the row accounting identity.  Sparse matrices are
scaled row-by-row in CSR storage without densification or an nnz-length repeated
factor vector.  Inputs are copied rather than mutated.

Technical coefficients are deliberately outside this API.  They must be derived
*after* the extensive accounts are adjusted, using the existing MRIO coefficient
helper.  Algebraically,

\[
A_t^{(b)}
= Z_t^{(b)}\operatorname{diag}(x_t^{(b)})^{-1}
= D_t A_t D_t^{-1}.
\]

If one country-level factor is applied to every product in a region, its domestic
block of \(A\) is unchanged exactly.  The sensitivity therefore removes aggregate
country price-level effects but **does not remove within-country product-relative
price movements**.  It must not be described as a reconstruction of
product-by-country constant-price EXIOBASE tables.

`expand_region_price_factors(...)` expands factors already defined on the exact
MRIO region labels to the canonical `MultiIndex(region, sector)`.  It performs no
country-name matching and no Rest-of-World aggregation.  Construction of the five
EXIOBASE Rest-of-World factors and any exact common-currency treatment are separate
scientific mapping tasks and are intentionally not guessed in this first API
patch.

Physical environmental extensions (`F`, `F_Y`, material, energy, emissions, and
other non-monetary rows) are never price-adjusted.  Their monetary intensities may
be recomputed later against adjusted monetary denominators only when the consuming
analysis explicitly defines that sensitivity.
<!-- /ecolecon-price-index-api -->

<!-- ecolecon-exiobase-eur-price-factors -->
### Complete EXIOBASE3 common-EUR macro factor construction

The country-level sensitivity has one higher-level EXIOBASE3 mapping owner:
`build_exiobase3_eur_price_factors(...)`. It returns a long table containing the
requested years plus the base year and exactly 49 legacy EXIOBASE3 regions per
year (44 explicit countries and `WA`, `WL`, `WE`, `WF`, `WM`). Missing source
members, missing years, duplicate observations, non-positive source weights, or an
incomplete 49-region result fail; no source member is silently dropped or
interpolated.

For ordinary countries the production builder uses paired UNSD AMA GDP levels in US
dollars: GDP at current prices $C_{c,t}$ and GDP at the provider's common
constant-price reference $K_{c,t}$ (currently 2020 prices).  If $u_t$ is annual
USD per EUR, the common-base-EUR factor is

\[
\phi^{EUR}_{c,t\to b}
=\frac{C_{c,b}/K_{c,b}}{C_{c,t}/K_{c,t}}\frac{u_t}{u_b}.
\]

The constant-price reference normalization cancels from the ratio for one country.
The direct published USD implicit-deflator API remains available as an independent
audit path rather than a second production implementation; the completed source
audit found the direct and $C/K$ factors identical to numerical precision over
2,299 overlapping country-years (maximum relative discrepancy $6.66\times10^{-16}$).
Raw fiscal-year notes remain provenance; the API does not silently calendarize
fiscal-year countries.

The five EXIOBASE Rest-of-World factors use the packaged recovered EXIOBASE3rx
214-to-49 regional bridge from Bjelle et al. (2020), Additional file 2
(`40008_2020_182_MOESM2_ESM.xlsx`, "Concordance matrices for regions, land use
types and industries"). The bridge is the sole RoW-membership authority. Its 44
explicit targets are published as ISO3 labels; the pinned EXIOBASE `00-class`
`country_coverage_matrix.csv` at commit
`768ff12b3c1e449e89d4310a44db9c5dc2266652` is used only as the explicit
ISO3-to-runtime-`EXIO3` crosswalk (for example `AUT -> AT` and `TWN -> TW`). Its
`EXIO3_3L`/RoW classification fields are never used to redefine membership.  The committed file
`src/ecolecon_mrio/mappings/exiobase3rx_to_exiobase3_regions.csv` is the exact
four-column audit extraction (`source_country`, `source_iso3`,
`target_exio_region`, `weight`), SHA-256 `92e9b95d240b338c0a6a8c06f2c4768149d81092839ef4581d780a1ce119f972`.  It was created by
extracting the published `Countries` one-hot 214-by-49 regional concordance,
preserving the source-country labels and target membership, and retaining the ISO3
annotations established in the recovery audit; Netherlands Antilles and Zanzibar
intentionally have blank `source_iso3` cells rather than invented modern ISO3
replacements.

Springer Nature lists Additional file 2 under the article's Supplementary
Information immediately above the article's CC BY 4.0 rights statement.  The
derived CSV is redistributed with attribution to Bjelle et al., DOI
`10.1186/s40008-020-0182-y`, under CC BY 4.0; this documentation identifies the
extraction/reformatting change.  The static bridge contains 170 former-RoW source
identities: `WA=45`, `WL=43`, `WE=14`, `WF=53`, `WM=15`.
A separate year-bounded identity mapping resolves provider statistical units and
historical aggregate aliases.  For the audited 2012--2022 sensitivity, Mainland
Tanzania uses UNSD 835, Zanzibar uses 836, and the EXIOBASE3rx `ANT` row is inactive
throughout because it overlaps the separately published Curaçao and Sint Maarten
rows; BES is not introduced.

For a RoW bucket $R$, current-USD and constant-price-USD GDP are extensive and are
summed over the exact effective member set before the implicit deflator is formed,

\[
C_{R,t}=\sum_{c\in R} C_{c,t},\qquad
K_{R,t}=\sum_{c\in R} K_{c,t},\qquad
\phi^{EUR}_{R,t\to b}
=\frac{C_{R,b}/K_{R,b}}{C_{R,t}/K_{R,t}}\frac{u_t}{u_b}.
\]

The implementation requires the same audited effective member set in every
requested/base year and complete positive finite $C$ and $K$ observations.  It
never averages country factors, drops a missing member, interpolates, or
renormalizes around missing data.

Taiwan (`TW`) remains an explicit EXIOBASE country and is not substituted by the
UNSD China observation. For the project's fixed 2012--2022 sensitivity, its
annual aggregate GDP implicit-price deflator is a packaged, audited snapshot of
the official Taiwan DGBAS Macro Database table `A018102050`, measure `fldid=3`
(implicit price deflator, 2021=100), item `code1=14` (GDP), annual frequency.
The snapshot records the official [National Accounts Yearbook Table 16,
"4-1. Implicit Price Deflators"](https://ws.dgbas.gov.tw/001/Upload/464/relfile/10866/235856/nie.pdf)
publication and [DGBAS table metadata](https://nstatdb.dgbas.gov.tw/dgbasAll/webMain.aspx?sdmx/A018102050/all).
It is deliberately bounded to 2012--2022 and fails closed outside that domain.
This replaces live DGBAS XML/data.gov discovery: the available machine transports
could not be fetched reproducibly with normal certificate/API verification, so
there is no DGBAS runtime HTTP request or transport fallback. Its
annual-average NTD/USD series is read from the Central Bank of the Republic of
China. With `n_t` denoting NTD per USD, Taiwan uses

\[
\phi^{EUR}_{TW,t\to b}
=\frac{P^{NTD}_{TW,b}}{P^{NTD}_{TW,t}}
 \frac{n_t u_t}{n_b u_b}.
\]

All common-EUR base-year factors must equal one to numerical precision. The
returned factors can then be expanded over the exact MRIO `(region, sector)` basis
and passed to `adjust_mrio_iot_prices(...)`; `A` is still derived only afterward.
Physical environmental extensions remain outside the transformation.

UNSD AMA is described here as a global national-accounts series, not as a claim
that every country-year observation is a direct national-statistical submission:
UNSD supplements official submissions with secondary sources and estimates where
needed. Source URLs, the pinned EXIOBASE explicit-country crosswalk commit, exact source hashes, fiscal-year notes,
and the Taiwan pinned-source identity are retained in returned provenance metadata.

This remains a **country-GDP-deflator sensitivity**, not a reconstruction of
product-by-country constant-price EXIOBASE. A common factor within an explicit
country leaves that country's domestic block of `A` unchanged and therefore does
not remove within-country relative-price changes between products.


### Valuation-aware strict year states

`load_mrio_year_state(...)` remains the single compact SDA state boundary. With
`price_base_year=None` it preserves current-price behavior. Supplying a base year
selects the EXIOBASE3 country-GDP-deflator/common-EUR sensitivity: the builder
constructs the exact 49 runtime-region factors, `expand_region_price_factors(...)`
expands them over the canonical `(region, sector)` basis, and
`adjust_mrio_iot_prices(...)` transforms primitive `Z`, `x`, and `Y` before
`derive_A_from_Z_and_x(...)` and `derive_nonexport_final_demand_from_Y(...)` run.
Derived `A` or `y` are never deflated after construction.

The returned `MrioYearState.metadata` records `valuation_basis`, `price_base_year`,
source/factor provenance, and the canonical factor-table SHA. Strict SDA pair
helpers reject mixed valuation bases or different common-EUR base years.
`load_mrio_extension_year_state(...)` forwards the monetary valuation controls to
that same year-state loader but never transforms physical `F`; direct extension
intensity is recomputed as physical `F` divided by the selected monetary `x`.
<!-- /ecolecon-exiobase-eur-price-factors -->

<!-- ecolecon-valued-primitive-loader -->
### Selective valued primitive EXIOBASE accounts

`load_exiobase3_iot_matrices_in_price_basis(...)` is the shared selective monetary
boundary for analyses that need primitive `x`, `Y`, or `Z` rather than a compact
SDA state. With `price_base_year=None` it delegates unchanged to the canonical raw
EXIOBASE `load_mrio_iot_matrices(...)` source-account loader and records
`valuation_basis=current_price`. With a base year it builds the same audited
country-GDP/common-EUR factors used by `load_mrio_year_state(...)`, requires their
runtime region labels to match the loaded MRIO basis exactly, expands them over the
producer rows, and applies `adjust_mrio_iot_prices(...)` only to the requested
primitive accounts.

The helper does not derive `A`, sum final demand to `y`, load physical extensions,
or create a derived MRIO matrix cache. This makes x-only historical projections and Y-category
analyses use the same valuation owner as SDA while preserving the source-ZIP-only,
selective low-memory contract. `load_mrio_year_state(...)` composes this helper for
`Z/x/Y` and derives `A` and non-export `y` afterward; it no longer implements a
parallel factor-construction/row-scaling path.
<!-- /ecolecon-valued-primitive-loader -->

## Price-factor mapping data contract

Scientific geography and source-selection facts for macro price factors are mapping
data, not country-name branches in Python.  Installed-library mappings therefore
live once under `src/ecolecon_mrio/mappings/` and are included in the wheel; they
are not mirrored under `scripts/mappings/`.  Provider algorithms and validation
remain Python code, while the mapping rows choose which algorithm/source identity
applies.

`exiobase_price_factor_source_policy.csv` maps an EXIOBASE explicit-region code and
inclusive validity interval to a price-factor method.  An enabled `*` row supplies
the ordinary UNSD+ECB default and an exact region row overrides it.  The current
Taiwan choice is therefore a mapping row rather than an `if iso3 == "TWN"` rule.
The pinned DGBAS observations remain reference data because observations are not a
mapping.

`exiobase3rx_to_exiobase3_regions.csv` is the immutable recovered extraction of
the published regional bridge used by the installed library; runtime validation
normalizes its raw four-column schema in memory but does not rewrite the mapping.
`exiobase_price_factor_source_identity.csv` is a separate mapping layer for
provider statistical identities and audited historical aliases; the code does not
contain country-name branches for these choices.

The source-identity mapping resolves Mainland Tanzania to UNSD 835 and Zanzibar to
836.  It marks `ANT` inactive for the full 2012--2022 sensitivity domain.  That
choice is not inferred from a generic zero-GDP rule: MATLAB exports of the official
EXIOBASE3rx files show that in 2009 `ANT` GDP is essentially exactly Curaçao plus
Sint Maarten, the overlap persists in 2012, `ANT` is zero in 2013/2015, and BES is
absent.  Other zero-GDP 3rx rows are therefore not automatically removed.  The
runtime resolver applies only explicit mapping rows and fails closed for any
non-current source identity lacking an audited provider rule.
