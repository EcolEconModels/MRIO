# Eurostat IOT source-role inventory

## Purpose

This inventory records the current repository representation of Eurostat
symmetric IOT data and identifies the evidence required before implementing the
comprehensive validator. It is an implementation audit, not a claim that every
current row interpretation is economically settled.

Baseline inspected: `e65fe712b8b5743802880c7b9b13d0e025b1eacf`.

## Current parsed object

`src/ecolecon_mrio/eurostat_iot.py` defines:

```python
@dataclass(frozen=True)
class ParsedEurostatIOT:
    Z: pd.DataFrame
    Y: pd.DataFrame
    meta: dict[str, Any]
    V: pd.DataFrame | None = None
    M: pd.DataFrame | None = None
```

The object is a convenient file-layout representation:

- `Z`: product rows by intermediate-use product columns;
- `Y`: product rows by final-use columns and published display controls;
- `V`: supplementary rows under product/intermediate-use columns;
- `M`: supplementary rows under final-use columns;
- `meta`: source and parser metadata.

`V` and `M` are not complete economic domains. A single economic control, such
as use of imported products, may span both blocks. Primary inputs, supply
controls, identity rows, and valuation bridges may coexist in `V` or `M` and
must be classified by role before validation.

## Current computational final-demand selection

There are currently two relevant selectors:

1. `select_eurostat_atomic_final_demand_columns()` in `eurostat_iot.py`;
2. `atomic_final_demand_columns()` in `iot_flow_semantics.py`.

Both seek a non-overlapping P3/P5/P6 basis, but they are separate
implementations. The comprehensive validator should use one authoritative role
mapping and one selection algorithm. Until migration, both call sites must be
inventoried and tested for equivalent source cases.

Current candidate families are:

| Family | Aggregate | Detail candidates | Current caution |
|---|---|---|---|
| Final consumption | `P3` | `P3_S13`, `P3_S14`, `P3_S15` | Complete detail should replace aggregate computationally; published controls may be rounded independently. |
| Capital formation | `P5` | `P51G`, `P52`, `P53` | Complete detail should replace aggregate computationally. |
| Exports | `P6` | `P6_*` candidates | Detail exhaustiveness is not generically established. |
| Total final use | `TFU` | derived | Never additive with components. |
| Total use | `TU` | derived from `Z` and atomic final use | Never additive with components. |

## Current supplementary role recognition

`src/ecolecon_mrio/iot_identities.py` and
`src/ecolecon_mrio/iot_flow_semantics.py` currently recognise labels or codes
for these roles:

| Current role | Examples currently recognised | Current constructor behaviour | Status for comprehensive validator |
|---|---|---|---|
| Output | `P1`, `OUTPUT`, `output` | Replaced with supplied output vector in `V` | Economic role valid; the supplied vector must be flow-role specific. |
| Intermediate-use control | `P2_ADJ` and label variants | Replaced with `Z` column sums | Valuation meaning unresolved; do not generalise until source metadata is confirmed. |
| Gross value added | `B1G`, `B1GQ`, `VALUE_ADDED_GROSS` | For `TOTAL`, replaced with output minus `Z` column sums | Potential valuation defect if the control includes product-tax or margin adjustments. |
| Imports | `P7`, `P7_CIF`, `P7_FOB`, imports labels | Resolved as one complete finite product vector | Must be bridged to complete `IMP` row uses, not interpreted as only final imports. |
| Total supply at basic prices | `TS_BP` and label variants | For `TOTAL`, replaced with output plus imports | Must also equal total use at a common valuation. |

Current label matching is useful compatibility code but must not remain the
long-term economics dictionary. The role map should eventually be sourced from
profile/dictionary data and validated against source metadata.

## Current block-by-block interpretation

### Z

Current expected orientation:

- rows: supplied products;
- columns: producing/using products;
- cell: intermediate use of row product by column product.

Required inventory fields before validation:

| Field | Required answer |
|---|---|
| Row classification | Exact Eurostat dimension and classification version |
| Column classification | Exact Eurostat dimension and classification version |
| Symmetry | Whether rows and columns are guaranteed identical |
| Unit | Monetary unit and scale |
| Price basis | Basic or purchasers' prices |
| Stock-flow role | `DOM`, `IMP`, or `TOTAL` |
| Missingness | Whether absent SDMX cells mean zero, missing, confidential, or not applicable |
| Sign convention | Treatment of negative inventories, margins, and adjustments |

### Y

Current expected orientation:

- rows: supplied products;
- columns: final-use categories plus possibly aggregate/display controls.

Required role classification for every present column:

- primitive final-use category;
- exhaustive aggregate control;
- non-exhaustive aggregate control;
- generated total final use;
- generated total use;
- territorial adjustment;
- other published control.

The validator must not sum every `Y` column.

### V

Current expected orientation:

- rows: heterogeneous primary-input, supply, valuation, and control rows;
- columns: the product/intermediate-use domain.

Required classification for every present row:

- primitive primary input;
- value-added component;
- gross value-added control;
- output control;
- imports control;
- total-supply control;
- intermediate-consumption control;
- imported-use control;
- product-tax/subsidy control;
- trade/transport-margin control;
- other valuation adjustment;
- generated identity;
- structurally non-applicable row.

### M

Current expected orientation:

- rows: heterogeneous supplementary controls;
- columns: the complete published `Y` column domain.

Required classification for every present row and column combination:

- imported final use;
- intermediate/final-use identity control;
- valuation adjustment applying to final use;
- source observation;
- generated identity;
- structurally non-applicable cell.

Rows such as output, gross value added, and total supply are currently blanked
under final-use columns because they are non-applicable. The validator must
preserve this distinction from zero.

## Current source-rounding path

`infer_source_rounding_quantum()` in `eurostat_iot.py` infers the reporting
quantum retained by the SDMX payload. `rounded_source_control_diagnostic()` in
`iot_flow_semantics.py` compares independently rounded controls with sums of
independently rounded components.

This is the authoritative direction for published source controls. The future
validator should call or reuse this logic rather than implementing a second
rounding formula.

Generated rows created after splitting or reaggregation must not use this
source-rounding allowance.

## Current generated-identity constructors

### Final-demand display identities

`enforce_final_demand_display_identities()` currently:

- rebuilds `P3` from complete P3 details;
- rebuilds `P5` from complete P5 details;
- derives `TFU` from the atomic final-demand basis; and
- derives `TU` from row sums of `Z` plus `TFU`.

These are constructors. The comprehensive validator should independently check
the resulting equations but should not create a second constructor.

### Supplementary rows

`refresh_supplementary_rows()` currently:

- assigns output rows from a supplied vector;
- assigns recognised intermediate-use rows from `Z` column sums;
- derives gross value added for `TOTAL` as output minus `Z` column sums; and
- derives total supply for `TOTAL` as output plus a strict imports vector.

The output, imports, and total-supply directions have explicit recent tests.
The intermediate-consumption and value-added equations require source-valuation
confirmation before becoming validator invariants.

### Final-use supplementary rows

`refresh_m_identity_rows()` currently:

- derives applicable intermediate/final-use controls from `Y` column sums and
  total-use identities; and
- blanks output, value-added, and total-supply rows under final-use columns.

The future role dictionary must state which cells are applicable and which
identity each applicable cell represents.

## Current strict contracts already established

The following contracts are already deliberate and should be preserved:

1. one explicit non-overlapping computational final-demand basis;
2. no missing product in an authoritative imports vector;
3. no non-finite imports value;
4. no ambiguity from multiple recognised imports rows;
5. generated total supply at basic prices equals supplied output plus imports;
6. generated P3/P5/TFU/TU controls are derived once from primitives;
7. source rounded controls use quantisation-aware diagnostics; and
8. local and global balancing have distinct output-share semantics.

## Known missing or incomplete checks

The current pipeline lacks one central report covering all of the following:

- product-level supply equals use;
- cellwise `TOTAL = DOM + IMP` for `Z`;
- cellwise `TOTAL = DOM + IMP` for atomic `Y`;
- imports supply equals complete imported intermediate and final uses;
- imported-use lower rows equal `IMP` column sums;
- row-oriented domestic output equals the output control;
- intermediate consumption plus value added equals output at compatible
  valuations;
- gross value-added component decomposition;
- valuation-matrix row, column, and total controls;
- GDP production, expenditure, and income consistency;
- conservation of every primitive block under disaggregation;
- conservation of every primitive block under reaggregation;
- applicability-mask preservation; and
- a pymrio conversion gate comparing validated output and coefficients.

## Pymrio boundary

Pymrio already owns the analytical IO representation and calculations:

- transactions `Z`;
- final demand `Y`;
- output `x`;
- coefficients `A`;
- Leontief inverse `L`; and
- extensions and accounts.

The repository validator should not duplicate that framework. It should validate
published/source accounting semantics, flow-role decomposition, valuation,
missingness, and transformation conservation before creating a pymrio system.

## Source evidence required for the first production audit

For Austria 2021, capture and inspect for each of `TOTAL`, `DOM`, and `IMP`:

1. Eurostat dataset code;
2. complete request filters;
3. dimension code and label dictionaries;
4. complete `Z`, `Y`, `V`, and `M` labels;
5. unit and scale;
6. price basis;
7. source rounding quantum;
8. missing/flagged observation semantics;
9. exact rows corresponding to output, imports, total supply, intermediate
   consumption, value added, taxes, subsidies, and imported use;
10. applicability patterns under intermediate and final-use columns; and
11. whether all three flow roles share identical product and final-use domains.

The first read-only audit command should print these facts before attempting any
uncertain numerical invariant.

## Schema-resolution audit outputs

The read-only Austria audit now emits two additional evidence tables before any
new source-role policy is adopted:

- `iot_source_schema.csv` inventories every parsed row and column code, source
  label, dimension, block, axis and current compatibility-role hint for each
  `DOM`, `IMP` and `TOTAL` table;
- `iot_supplementary_row_profiles.csv` records positional supplementary-row
  statistics separately for `V` and `M`, including finite, zero, non-zero and
  non-finite counts and finite sums/ranges.

These files are evidence, not authoritative economics dictionaries. Existing
label recognisers are exposed only as `compatibility_hint_only` values. They do
not settle `P2_ADJ`, value-added, imports valuation, tax/margin roles, or
applicability. Because `parse_eurostat_iot()` currently fills sparse SDMX cells
with zero, zero/non-zero profiles cannot distinguish a reported economic zero
from a missing or structurally non-applicable source observation.

The Austria 2021 production audit must be reviewed before converting any hint
into a validator policy or integrating validation into Energy-Balance, EXIO or
pymrio routes.


### Schema-resolution question table

The audit also writes `iot_schema_resolution_questions.csv`. This is a compact
review index over the source-schema and supplementary-row evidence. It records
candidate codes, labels, blocks and flow coverage for the unresolved questions
that must be settled before shadow integration: `P2_ADJ` valuation, common
supply/use valuation, final-demand exhaustiveness, `V`/`M` orientation,
tax/margin roles, imports valuation and source missingness.

The table is deliberately descriptive. A row with
`evidence_collected_not_resolved` means relevant candidates were found; it does
not approve a role, equation or valuation. A row with
`no_candidate_found_not_resolved` is likewise not evidence that the economic
concept is absent from the source. The actual Austria 2021 output must be
reviewed together with Eurostat metadata before any validator policy is added.

## Decisions deferred by this patch

This documentation patch deliberately does not decide:

- whether `CanonicalIOT` should be retained, renamed, narrowed, or removed;
- whether `P2_ADJ` includes allocated product taxes or other valuation terms;
- whether export details are exhaustive;
- whether source-sparse absent cells are economic zeros;
- the complete value-added component set;
- the exact GDP territorial adjustments;
- where economic roles will ultimately be stored; or
- which current distributed identity checks can be deleted.

Those decisions require the source evidence listed above. Encoding them before
that audit would create another parallel and potentially incorrect economics
path.

## Planned validator modules after this inventory

The next implementation stages should be:

1. common `InvariantCheck` and `InvariantReport` result types;
2. structural, domain, missingness, unit, and final-demand-basis checks;
3. row identities with flow-role-specific meaning;
4. column, value-added, and supply/use identities after valuation confirmation;
5. cross-flow `TOTAL`/`DOM`/`IMP` checks;
6. disaggregation and reaggregation conservation checks;
7. a read-only Austria source audit command;
8. shadow integration in Energy-Balance and EXIO routes;
9. blocking generated-table validation; and
10. removal of duplicate economics checks after numerical equivalence is proven.

## Reviewed Austria 2021 source profile

The production audit at commit `eccc63de229a3dfeac80650545bb8482f3aa1794`
confirmed one identical ordered 65-product domain for `DOM`, `IMP`, and `TOTAL`,
one identical ordered 17-column final-demand domain, complete rounded P3 and P5
partitions, aggregate-authoritative P6, and source-rounded `TOTAL = DOM + IMP`
for Z and the atomic final-demand basis.

`src/ecolecon_mrio/eurostat_iot_source_policy.py` records those facts as the
reviewed `AT_2021_NAIO_10_CP1700_PROFILE`. It also records source-code roles for
P1, B1G, P7, and TS_BP while authorizing numerical equations only where the
existing accounting contract and production audit provide sufficient evidence.

The profile deliberately excludes unresolved interpretations from numerical
policy: P2_ADJ valuation, TS_BP/TU valuation equivalence, regional P6 detail
exhaustiveness, a complete B1G decomposition, tax/margin bridges, the DOM `IMP`
row versus IMP-flow bridge, and source applicability or missingness. Presence in
the reviewed schema is not authorization to use such a row in an accounting
equation.

The profile validator checks exact source domains and positional supplementary
row schemas. It does not transform source data, infer missingness, or validate
any excluded economic equation. Shadow integration must use these exclusions
unchanged until further source metadata or independent methodological evidence
resolves them.
