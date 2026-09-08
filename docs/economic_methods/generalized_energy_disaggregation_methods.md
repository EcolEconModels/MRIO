# Economic methods for generalized energy-sector disaggregation and reaggregation

## Status and scope

This document specifies the economic and accounting method for disaggregating and
reaggregating monetary input-output, supply-use, and MRIO tables using external
physical, product, activity, trade, and donor-table evidence. It is written as a
stand-alone methods specification. A correct implementation can be built from
this document plus the referenced mapping, dictionary, selector, and
reaggregation files, without access to any existing source code.

The method covers two scientific routes:

1. **row/column-specific evidence disaggregation**, where physical quantities,
   prices, product values, use priors, technology priors, final-demand priors,
   and residual rules are combined to split selected aggregate sectors; and
2. **full donor-block disaggregation**, where a detailed donor table supplies a
   complete child-by-child flow structure that is rescaled to the aggregate
   target table.

The method is classification-neutral. No energy product, sector, activity,
final-demand category, donor sector, or supplementary row is part of the method
itself. All such semantics enter through mappings and dictionaries.

## Required mapping objects

The method requires a mapping set for each scientific split definition. The
mapping set must define the following objects.

| Object | Economic role |
|---|---|
| Parent-child map | Defines each aggregate parent and its emitted children. |
| Reaggregation map | Maps each emitted child back to aggregate target sectors. |
| Evidence selectors | Map children or child families to physical, monetary, price, use, technology, trade, donor, bridge, and manual evidence. |
| Evidence-priority table | Gives deterministic source priority by evidence kind. |
| Unit dictionary | Defines admissible units and conversion factors. |
| Classification dictionaries | Define all sector, product, activity, flow, geography, and final-demand codes used by selectors. |
| Child-family map | Groups children that share a delivered product family, technology family, or residual family. |
| Eligibility map | Declares impossible child-buyer, supplier-child, child-final-demand, or child-child combinations. |
| Compatibility map | Gives optional multiplicative weights for allowed child-child pairs. |
| Residual policy | Declares how unresolved parent mass is represented and when negative residuals fail. |
| Supplementary-row role map | Classifies lower rows such as output identities, imports, margins, taxes, value added, and totals. |
| Region-scope map | Defines a versioned many-source-regions-to-one-scope aggregation independently of all sector mappings. |

The mapping set is authoritative. A classification code absent from the
dictionaries is unknown. A child absent from the parent-child map does not exist.
A residual child absent from the residual policy must not be created.

## Target table and notation

Let the target monetary intermediate block be

\[
Z \in \mathbb{R}^{I \times I},
\]

and let the target final-demand block be

\[
Y \in \mathbb{R}^{I \times F},
\]

where \(I\) is the set of aggregate target sectors and \(F\) is the set of
final-demand categories. Unless a signed balancing method is explicitly declared,
all entries are non-negative monetary flows.

The target output of aggregate sector \(i\) is

\[
x_i = \sum_{j \in I} Z_{ij} + \sum_{f \in F} Y_{if}.
\]

Let \(P \subseteq I\) be the set of aggregate parents to split. Each parent
\(p \in P\) has a non-empty child set \(C_p\). Unsplitted sectors are represented
as singleton child sets:

\[
C_i = \{i\} \quad \text{for } i \notin P.
\]

The emitted sector set is

\[
I' = \bigcup_{i \in I} C_i.
\]

The parent map \(\pi : I' \to I\) maps every emitted child to its original target
sector. The default reaggregation matrix \(G \in \{0,1\}^{I \times I'}\) is

\[
G_{ia} =
\begin{cases}
1, & \pi(a)=i,\\
0, & \text{otherwise.}
\end{cases}
\]

Weighted reaggregation is allowed only if the reaggregation map explicitly
declares weights. Otherwise reaggregation is pure summation.

A valid disaggregation produces \(Z' \in \mathbb{R}^{I' \times I'}\) and
\(Y' \in \mathbb{R}^{I' \times F}\) satisfying

\[
G Z' G^{\top} = Z,
\]

\[
G Y' = Y,
\]

and

\[
G x' = x,
\quad
x'_a = \sum_{b \in I'} Z'_{ab} + \sum_{f \in F} Y'_{af}.
\]

These are accounting identities, not optional diagnostics. Failure beyond the
declared tolerance invalidates the split.

## Evidence kinds and deterministic selection

The method distinguishes evidence kinds because they answer different economic
questions.

| Evidence kind | Symbol | Economic meaning |
|---|---|---|
| Physical quantity | \(q_h\) | Physical amount of child \(h\), usually energy or product quantity. |
| Monetary child value | \(m_h\) | Observed monetary production, output, or sales value of child \(h\). |
| Unit value or price | \(p_h\) | Monetary value per physical unit for child \(h\) or family. |
| Delivered-use prior | \(u_{g,k}\) | Use of delivered product family \(g\) by buyer activity/product group \(k\). |
| Producer-technology prior | \(t_{r,h}\) | Use of supplier/input group \(r\) by producer child \(h\). |
| Final-demand prior | \(d_{h,f}\) | Allocation prior for child \(h\) to final-demand category \(f\). |
| Donor block | \(D^Z,D^Y\) | Detailed donor intermediate and final-demand flows. |
| Bridge adjustment | \(b\) | Reconciliation adjustment between accounting conventions. |
| Manual override | profile value | Explicit scientific assumption used only when declared. |

For a given child, evidence kind, geography, year, and selector, admissible
evidence rows are sorted deterministically by:

1. evidence-priority rank;
2. quality status declared by the evidence source or mapping set;
3. coverage score, descending;
4. mapping-set order;
5. provider name and selector code, lexicographically.

The first passing row is selected unless the selector declares an explicit
combination rule such as sum, weighted mean, or ratio of sums. Every selected and
rejected candidate must be reportable. Equal splits, donor fallbacks, manual
values, residual absorption, and placeholder prices must never occur implicitly.

## Child value resolution

For each child \(h \in C_p\), resolve a non-negative monetary child value
\(\tilde v_h\) using the highest-priority admissible rule.

### Direct monetary value

If an observed monetary value \(m_h\) is available and passes unit and coverage
checks,

\[
\tilde v_h = m_h.
\]

This is preferred when the source directly measures the child product's monetary
production or sales value in the target country and year.

### Quantity times unit value

If a physical quantity \(q_h\) and an observed unit value \(p_h\) are available in
compatible units,

\[
\tilde v_h = q_h p_h.
\]

If \(p_h\) is observed only for a broader family \(g(h)\), the family unit value
may be transferred to \(h\) only if the mapping set declares that transfer valid.

### Quantity times fallback price

If only a physical quantity and fallback price are available,

\[
\tilde v_h = q_h p^{fallback}_{g(h)}.
\]

The diagnostic must distinguish observed unit values, harmonised prices, donor
prices, manual prices, and placeholders.

### Declared share

If the mapping set declares an admissible monetary share \(\sigma_h\) and no
higher-priority value evidence is available,

\[
\tilde v_h = \sigma_h x_p.
\]

Declared shares must be non-negative. They are normalized within the parent only
if the mapping set declares normalization.

### Missing values

If no admissible evidence exists for a required child, the method fails unless
the mapping set explicitly declares an allowed zero, residual-only, or fallback
rule. A missing optional child is assigned zero only if the child is declared
zero-allowing.

## Residuals and negative residuals

Let \(D_p\) be the set of declared non-residual children of parent \(p\). Let
\(R_p\) be the residual child if the residual policy declares one. The unresolved
raw parent mass is

\[
\tilde r_p = x_p - \sum_{h \in D_p} \tilde v_h.
\]

The default residual rule is

\[
v_{R_p} = \tilde r_p
\]

when \(R_p\) exists and \(\tilde r_p \ge -\epsilon_p\), where \(\epsilon_p\) is
the parent tolerance. If \(\tilde r_p < -\epsilon_p\), the default is failure.
Alternative negative-residual policies are admissible only if declared:

1. **tolerance clamp**: set \(v_{R_p}=0\) only when \(|\tilde r_p| \le \epsilon_p\);
2. **scale declared children**: multiply declared child values by
   \(x_p / \sum_{h \in D_p}\tilde v_h\);
3. **diagnostic-only residual**: write the residual to diagnostics and stop before
   approved output;
4. **explicit external residual**: use an external residual value and validate
   that the parent identity still holds.

After residual treatment, the approved child values \(v_h\) must satisfy

\[
\sum_{h \in C_p} v_h = x_p
\]

within tolerance. The output share is

\[
s^x_h = \frac{v_h}{x_p}
\]

for \(x_p>0\). If \(x_p=0\), all child values must be zero unless the mapping set
declares a special zero-output treatment.

## Row/column-specific allocation route

The row/column-specific route constructs every split intermediate cell as a share
of one original target cell. For \(i,j \in I\), \(a \in C_i\), and \(b \in C_j\):

\[
Z'_{ab} = Z_{ij} W^{ij}_{ab},
\]

where

\[
\sum_{a \in C_i}\sum_{b \in C_j} W^{ij}_{ab} = 1,
\quad
W^{ij}_{ab} \ge 0.
\]

The weight \(W^{ij}_{ab}\) is built from a row score, a column score, and an
optional pair-compatibility score.

### Delivered-use row allocation

Delivered-use allocation answers: **which emitted child product supplies the
purchase recorded in row parent \(i\) and buyer column \(j\)?**

Let \(g(a)\) be the delivered product family of child \(a\). Let \(k(j)\) be the
buyer group assigned to target buyer \(j\) by the mapping set. Let
\(u_{g(a),k(j)}\) be delivered-use evidence for product family \(g(a)\) and buyer
group \(k(j)\). Let \(e^R_{a j}\in\{0,1\}\) be row-side eligibility.

The row score is

\[
R_{ij,a} = s^x_a\; u_{g(a),k(j)}\; e^R_{a j}.
\]

If no delivered-use evidence is admissible for the family, the declared fallback
row score is used. The default fallback is

\[
R_{ij,a} = s^{sales}_a e^R_{a j},
\]

where \(s^{sales}_a\) is the declared or derived intermediate-sales share. Equal
shares are allowed only if explicitly declared.

For physically homogeneous delivered products, such as delivered electricity,
source children may share the same buyer pattern:

\[
u_{g(a),k(j)} = u_{g(a'),k(j)}
\quad \text{for all source children } a,a' \text{ in the same delivered family,}
\]

unless source-specific procurement evidence is declared. This means buyer-side
use evidence can allocate total delivered electricity across buyers without
claiming that each buyer purchased source-specific coal, wind, or solar
electricity.

### Producer-technology column allocation

Producer-technology allocation answers: **which emitted producer child \(b\)
uses the input supplied by row parent \(i\)?**

Let \(r(i)\) be the supplier/input family of target row \(i\). Let
\(t_{r(i),b}\) be technology evidence for input family \(r(i)\) used by producer
child \(b\). Let \(s^{in}_b\) be the declared or derived intermediate-input share
of producer child \(b\). Let \(e^C_{i b}\in\{0,1\}\) be column-side eligibility.

The column score is

\[
C_{ij,b} = s^{in}_b\; t_{r(i),b}\; e^C_{i b}.
\]

If no technology evidence is admissible, the declared fallback is used. The
default fallback is

\[
C_{ij,b} = s^{in}_b e^C_{i b}.
\]

Producer-technology evidence is separate from delivered-use evidence. For
example, delivered electricity may be homogeneous for buyers, while electricity
producer children have different input technologies for coal, gas, hydro, wind,
solar, nuclear, biomass, transmission, or residual services.

### Pair compatibility and normalization

Let \(M^{ij}_{ab}\ge0\) be a declared compatibility multiplier. If no multiplier
is declared, \(M^{ij}_{ab}=1\) for eligible pairs. The raw cell score is

\[
S^{ij}_{ab}=R_{ij,a}\; C_{ij,b}\; M^{ij}_{ab}.
\]

If

\[
\sum_{a\in C_i}\sum_{b\in C_j} S^{ij}_{ab} > 0,
\]

then

\[
W^{ij}_{ab} =
\frac{S^{ij}_{ab}}
{\sum_{a'\in C_i}\sum_{b'\in C_j} S^{ij}_{a'b'}}.
\]

If the denominator is zero, the method uses the next declared fallback in order:

1. row-only allocation: \(W^{ij}_{ab}\propto R_{ij,a}s^{in}_b\);
2. column-only allocation: \(W^{ij}_{ab}\propto s^{sales}_a C_{ij,b}\);
3. parent child-output product: \(W^{ij}_{ab}\propto s^x_a s^x_b\);
4. declared equal-share fallback;
5. failure.

The fallback actually used must be recorded for the affected original cell
\((i,j)\).

## Final-demand allocation

For each target final-demand cell \(Y_{if}\) and child \(a\in C_i\),

\[
Y'_{af}=Y_{if}H^{if}_a,
\quad
\sum_{a\in C_i}H^{if}_a=1.
\]

If child-final-demand evidence \(d_{a,f}\) is available,

\[
H^{if}_a =
\frac{s^x_a d_{a,f} e^Y_{a f}}
{\sum_{a'\in C_i} s^x_{a'} d_{a',f} e^Y_{a' f}}.
\]

If the denominator is zero, use declared fallbacks in order:

1. final-demand prior share;
2. intermediate-sales share;
3. output share;
4. declared equal-share fallback;
5. failure.

Final-demand categories and mappings to donor or evidence categories are defined
only by the final-demand dictionary.

## Full donor-block route

The full donor-block route is used when a donor table contains enough detail to
construct child-by-child flows. Let donor intermediate and final-demand blocks be

\[
D^Z \in \mathbb{R}^{K\times K},
\quad
D^Y \in \mathbb{R}^{K\times L}.
\]

Let \(P_{a k}\ge0\) map donor sector \(k\in K\) to emitted child \(a\in I'\).
The mapping is defined by donor mappings and may include edge weights. For each
target intermediate cell \((i,j)\), form donor scores

\[
B^{ij}_{ab} =
\sum_{k\in K}\sum_{\ell\in K}
P_{a k} D^Z_{k\ell} P_{b\ell}.
\]

If

\[
\sum_{a\in C_i}\sum_{b\in C_j} B^{ij}_{ab} > 0,
\]

then

\[
Z'_{ab} = Z_{ij}
\frac{B^{ij}_{ab}}
{\sum_{a'\in C_i}\sum_{b'\in C_j}B^{ij}_{a'b'}}.
\]

If the donor denominator is zero, use the row/column-specific fallback sequence.
Do not invent donor shares. The donor fallback diagnostic must record the target
cell, donor denominator, emitted rows, emitted columns, and fallback type.

For final demand, let \(Q_{f\ell}\ge0\) map target final-demand category \(f\) to
donor category \(\ell\). Donor final-demand scores are

\[
B^Y_{a f} = \sum_{k\in K}\sum_{\ell\in L} P_{a k}D^Y_{k\ell}Q_{f\ell}.
\]

If the denominator over \(a\in C_i\) is positive,

\[
Y'_{af}=Y_{if}\frac{B^Y_{af}}{\sum_{a'\in C_i}B^Y_{a'f}}.
\]

Otherwise use the final-demand fallback sequence.

A donor table is a structural prior, not an authority for target-country totals.
The target aggregate cell totals \(Z_{ij}\) and \(Y_{if}\) remain authoritative
unless a separate override policy is explicitly declared.

## Geographic scope aggregation

Geographic aggregation is a first-class operation and is independent of sector
disaggregation/reaggregation. The low-memory source implementation supports checked
multi-region destination scopes for monetary \(Z\), \(Y\), and scope production
output \(x^G\); complete primitive monetary IOT aggregation is also implemented for
full \(Z\), \(x\), and \(Y\) source accounts. The boundary-aware WP3 environmental
operators now implement scope production, origin-retaining direct external imports,
direct final demand, and target total-supply combination without changing the sector
split pipeline. Let \(G\) be the set of member regions in a checked, versioned
region-scope mapping. The mapping is unweighted: each source region is either a member
or not a member. The complement \(G^c\) is derived from the actual region basis in
the selected donor release.

A published target table for the desired geography is authoritative. In
particular, Eurostat publishes consolidated European `naio_10_cp1700` tables, so
`geo=EU27_2020` must be selected directly for the target EU IOT. National tables
must not be summed to recreate that published target. The region-scope map is
still applied to the multi-regional donor and to custom groups for which no
published target exists.

For EXIO origin \(o\), destination \(d\), origin product \(k\), using product
\(j\), and final-demand category \(c\), the destination-scope use blocks are

\[
Z^{\mathrm{int},G}_{kj}
 = \sum_{o\in G}\sum_{d\in G}Z_{ok,dj},
\qquad
Y^{\mathrm{int},G}_{kc}
 = \sum_{o\in G}\sum_{d\in G}Y_{ok,dc},
\]

\[
Z^{\mathrm{imp},G}_{kj}
 = \sum_{o\notin G}\sum_{d\in G}Z_{ok,dj},
\qquad
Y^{\mathrm{imp},G}_{kc}
 = \sum_{o\notin G}\sum_{d\in G}Y_{ok,dc}.
\]

The total-use blocks are the cellwise sums of internal and external-import
blocks. Scope production output is a separate account,

\[
x^G_k=\sum_{o\in G}x_{ok}.
\]

It is not replaced by an import-use total when the requested flow is imports.


For a complete source-region partition, let `R` map region-sector rows/columns and
`H` map region-final-demand-category columns without changing sector or category
classification. The primitive monetary IOT is aggregated as

\[
Z^R=RZR^\top,\qquad x^R=Rx,\qquad Y^R=RYH^\top.
\]

This operation requires complete source-region coverage; partial destination scopes
remain low-memory donor slices rather than complete IOTs. Technical coefficients are
formed only after the extensive aggregation,

\[
A^R_{ij}=Z^R_{ij}/x^R_j.
\]

Thus a source `A` is never geographically averaged or summed. The zero-output rule is
inherited from the explicit coefficient derivation helper after region aggregation.

All environmental aggregation is extensive-first. For production extension
\(F\),

\[
F^G_k=\sum_{o\in G}F_{ok},
\qquad
e^{\mathrm{prod},G}_k=\frac{F^G_k}{x^G_k}.
\]

This is a scope-production coefficient for the target domestic-output account.
It includes member production irrespective of whether that output is used inside
the scope or exported; this matches the production denominator and must not be
confused with an internal-use weighting.

A collapsed Rest-of-World production average is not the coefficient of imports
to \(G\). Define the use of origin-product \(ok\) by the scope as

\[
u_{ok\to G}
 = \sum_{d\in G,j}Z_{ok,dj}
 + \sum_{d\in G,c}Y_{ok,dc}.
\]

For a many-to-many donor-to-target product projection \(P_{ak}\), do not
transport the raw source intensity \(F_{ok}/x_{ok}\) through \(P\). Retain each
external origin until \(F\), \(x\), and the import-use denominator have been
projected to the same target-product basis:

\[
\bar F_{oa}=\sum_k P_{ak}F_{ok},
\qquad
\bar x_{oa}=\sum_k P_{ak}x_{ok},
\qquad
\bar u_{oa\to G}=\sum_k P_{ak}u_{ok\to G}.
\]

The direct production extension attributable to external imports and its
denominator are then

\[
F^{\mathrm{imp},G}_a
 = \sum_{o\notin G}\frac{\bar F_{oa}}{\bar x_{oa}}
   \bar u_{oa\to G},
\qquad
m^{\mathrm{imp},G}_a
 = \sum_{o\notin G}\bar u_{oa\to G},
\]

\[
e^{\mathrm{imp},G}_a
 = \frac{F^{\mathrm{imp},G}_a}{m^{\mathrm{imp},G}_a}.
\]

A positive \(\bar u_{oa\to G}\) with zero or missing \(\bar x_{oa}\) is
inconsistent and fails closed. This is the import analogue of the existing
numerator-first extension rule: mapped extensive numerator and denominator are
formed before division, then origin coefficients are weighted by actual imports
to the selected scope. A nonzero raw \(F_{ok}\) with zero raw \(x_{ok}\) is not
rejected merely because the raw ratio is undefined; it fails only if the required
mapped origin/target-product denominator remains non-positive where import use is
positive. The account is direct origin production, not a full embodied footprint;
upstream burdens require a separately declared Leontief calculation.

The combined direct-use extension must not reuse the domestic coefficient. The
target IMP transaction row total is imported-product use, not the supply-side P7
control, and it can be signed. Form

\[
\widehat F^{\mathrm{total,use}}_a
 = e^{\mathrm{prod},G}_a x^{\mathrm{target,dom}}_a
 + e^{\mathrm{imp},G}_a u^{\mathrm{target,imp}}_a,
\]

\[
e^{\mathrm{total,use}}_a
 = \frac{\widehat F^{\mathrm{total,use}}_a}
 {x^{\mathrm{target,dom}}_a+u^{\mathrm{target,imp}}_a}.
\]

Here \(a\) is on the transformed target product basis. The denominator must
match the target TOTAL transaction-use vector. It is not a reconstruction of
supply-side P7/TS_BP. EXIO shares remain donor evidence only.

If all origin-level consistency checks pass but the collapsed donor coefficient
\(e^{\mathrm{imp},G}_a\) is still undefined (for example because the mapped net
import-use denominator is zero), a nonzero authoritative target import-use
control does not justify inventing a coefficient. The target-sector/stressor
estimate is then unavailable evidence: retain its monetary denominator and store
physical value and intensity as `NaN`, never zero and never the domestic coefficient.
A TOTAL estimate is likewise unavailable whenever any required nonzero-mass DOM or IMP
component required for that TOTAL is unavailable. An unavailable component with
zero target monetary mass does not invalidate a supported positive-mass counterpart.
Subsequent sector reaggregation may form a finite ratio only from complete
nonzero-denominator support; partial support remains a fail-closed condition.

Direct final-demand extensions remain on their own axis:

\[
F^{Y,G}_c=\sum_{d\in G}F^Y_{dc},
\qquad
y^G_c=\sum_{d\in G}\sum_o\sum_kY_{ok,dc},
\qquad
e^{Y,G}_c=\frac{F^{Y,G}_c}{y^G_c}.
\]

There is no imported-RoW version of \(F_Y\): it records direct emissions of final
demand in the destination region, not foreign production. It remains a category
sidecar and is never projected through a product concordance.

Let \(R\) be a region concordance and \(P\) a product concordance/projection.
With fixed linear mappings, region and product transforms commute for genuinely
extensive accounts; a full-system core concordance is \(R\otimes P\). Ratios,
target-dependent share resolution, balancing, and fallback policies do not
commute. Scope-production \(F\) and \(x\) may therefore collapse geography
before product transformation for memory efficiency. External-import direct
production is different: because origins are weighted by an origin-specific
production coefficient, the origin axis must be retained through the product
projection and only collapsed after the mapped \(\bar F/\bar x\) coefficient
has been applied to mapped import use. In no case is a raw intensity projected
through a many-to-many product mapping.

Limiting cases required by tests are:

- a singleton scope reproduces the current single-region domestic/import/total
  matrices;
- a scope containing every source region has zero external imports;
- internal plus external imports equals total cellwise for \(Z\) and \(Y\);
- complete monetary region aggregation conserves the explicit \(RZR^\top\), \(Rx\),
  and \(RYH^\top\) sums and agrees with `pymrio.IOSystem.aggregate(region_agg=...)`
  on a small complete fixture;
- summing scope-production \(F\) and \(x\) separately before division matches
  complete monetary region aggregation once environmental geography is implemented;
- external-import extensions retain origin through product projection and agree
  with an origin-resolved reference calculation under a many-to-many mapping;
- an all-source scope yields an empty external-origin use domain and therefore a
  zero-valued/zero-denominator direct-import account rather than fabricated imports;
- direct `F_Y` aggregation has no product-projection path and rejects nonzero direct
  physical values over a zero category monetary denominator; and
- changing only the sector mapping does not change scope membership, while
  changing only the scope mapping does not change sector topology.

## Donor environmental extension transport

For the fuels/fuels-more production EXIOBASE3-to-CPA65 route, (P)'s structural
topology comes from the checked-in reviewed production concordance
`scripts/mappings/exio_to_eurostat_cpa65_products_official_direct_reviewed.csv`.
It is derived from the EXIO official-direct CPA64/EXIOBASE3 incidence topology but
narrows only three aggregate-correspondence choices after clarification with the
EXIO support team: `p23.3` (Nuclear fuel) is assigned to `CPA_C24` (Basic metals);
`p24.d` (Chemicals nec) alone retains the `CPA_C20` (Chemicals and chemical
products) / `CPA_C21` (Basic pharmaceutical products and pharmaceutical
preparations) split while the remaining `p24.*` products map only to `CPA_C20`;
and `p33` (Medical, precision and optical instruments, watches and clocks) maps
only to `CPA_C26` (Computer, electronic and optical products). The raw official
concordance remains unchanged as provenance. The only retained multi-target choice
among these reviewed rows, `p24.d`, carries no fixed or equal share: the
flow-specific Eurostat target reference partitions that raw EXIO source across its
targets and the resulting source weights sum to one. Green/brown remains a separate
analytical pipeline on its existing concordance and is unchanged by this choice.

Production-side donor environmental extensions are optional evidence, not
authoritative target-country totals. For one donor stressor let
\(f^D \in \mathbb{R}^{K}\) be the donor physical extension vector and let
\(x^D \in \mathbb{R}^{K}\) be donor monetary output. The same donor-to-emitted
projection \(P\) used by the donor-block route defines

\[
\bar f_a = \sum_k P_{ak} f^D_k,
\qquad
\bar x_a = \sum_k P_{ak} x^D_k.
\]

Where \(\bar x_a>0\), the transferable donor evidence is the ratio

\[
e^D_a = \frac{\bar f_a}{\bar x_a}.
\]

The environmental satellite total attached to the transformed target table is
then

\[
f'_a = e^D_a x'_a,
\]

where \(x'_a\) is the authoritative transformed target output. Thus \(f'_a\) is
an **EXIO-coefficient-implied target satellite estimate**, not an independently
observed official target-country environmental total. This construction is
deliberate: when several emitted children are later reaggregated, their coefficients
are weighted by the actual transformed target-output composition. If an authoritative
target-country environmental parent total exists, it supersedes this donor-derived
estimate and donor evidence may only be used for the declared within-parent allocation.

After \(f'\) has been constructed, it is an extensive satellite quantity.
Reaggregation therefore sums physical totals and monetary output separately:

\[
f^{agg}=Gf',
\qquad
x^{agg}=Gx',
\qquad
e^{agg}_i=\frac{f^{agg}_i}{x^{agg}_i}.
\]

Child intensities must never be arithmetically averaged or independently summed.
For every selected EXIO extension, both production ``F`` and direct-final-demand
``F_Y`` must be present, together with ``unit.txt``. A missing selected matrix, unit,
or physical observation is unavailable evidence, not an economic zero, and must fail
before an intensity is derived. For production ``F``, the extensive numerator and
EXIO monetary denominator are projected through \(P\) before division. A nonzero raw
``F`` observation with zero raw sector output is therefore retained as donor evidence,
while its raw intensity is undefined. It fails only if a required positive-output split
child has no positive mapped donor denominator. The target-side availability state is
therefore explicit: a defined mapped coefficient applied to zero target output gives
``value=0`` and missing intensity; an undefined mapped coefficient with zero target
output gives missing ``value`` and intensity rather than a fabricated physical zero;
and an undefined mapped coefficient with positive target output is a fail-closed donor-
support error. The raw/mapped physical numerator remains donor provenance in the
undefined-coefficient cases and is never overwritten merely to create a target ratio.
For direct final-demand ``F_Y``, the category coefficient is reported without sector
projection, so a nonzero physical observation with zero category monetary denominator
is inconsistent and must fail. At projection, every positive-output split child whose
coefficient is used must have usable donor support. At reaggregation, each supported
extension/stressor/unit must cover every positive-output source sector contributing to
each aggregate target it represents. Zero-output unavailable children do not enter the
physical sum and do not poison a supported positive-output counterpart. This target-
local closure prevents a partial physical numerator from being divided by the full
canonical monetary-output denominator.

Scientific availability and the final model sector axis are separate contracts.
Unsplit target sectors are singleton sectors and retain their identity. A sparse
reaggregation map specifies only transformations: any sector absent from the map
carries through under its existing identifier. After scientific projection,
reaggregation, and numerical validation have operated only on supported estimates,
the model-facing long-form production sidecar is completed to the exact final IOT
sector axis for every selected production ``F`` extension/stressor/unit group,
including a selected group for which no sector coefficient is defined. The group
domain and order come from raw ``F`` metadata, never from ``F_Y`` or a release-specific
registry. ``sector_position`` is
the one-based position in the authoritative final IOT order, and
``monetary_output`` is the canonical final IOT output for that sector.

Grid completion does not impute an environmental observation. A sector without a
valid donor coefficient is retained with missing ``value`` and ``intensity``;
missing evidence is never physical zero. A supported coefficient applied to zero
target output instead has ``value=0`` and an undefined intensity. Likewise, a
supported physical zero at positive output retains ``value=0`` and zero intensity.
Thus finite ``value`` denotes an actual projected/reaggregated estimate, while
missing ``value`` denotes undefined environmental evidence; no separate availability
flag is used. The raw EXIO provenance sidecar is not grid-completed:
production ``F`` remains source-product evidence and ``F_Y`` remains category-level
evidence outside the sector-axis contract. Production-side model extensions use the explicit boundaries defined in
`Geographic scope aggregation`. A domestic/scope-production coefficient is never
transferred to external imports or total transaction use; those accounts use the
import-weighted and target-component equations above.

Direct final-demand extensions \(F_Y\) are a separate accounting boundary and are
not sector-disaggregated through \(P\). For destination region \(r\) and final-demand
category \(f\), the EXIO final-demand coefficient uses the category's monetary
final demand across all producer origins/products supplying \(r\):

\[
 e^{Y,D}_{rf}=
 \frac{F^Y_{rf}}{\sum_o\sum_k Y^D_{ok,rf}}.
\]

Using only domestically produced rows in that denominator would overstate the
coefficient whenever the destination imports products for final demand. The resulting \(F_Y\) coefficient remains EXIO donor evidence by final-demand
category for a separate use-phase proxy; it is not an authoritative target-country
final-demand satellite total and is not allocated across product/sector rows. Sector
reaggregation therefore leaves these category-level donor rows unchanged.

A destination scope's production extension must not be reused as an import-production
proxy. Imported-product and combined total-use extensions are defined only by the
origin-weighted and target-component equations in `Geographic scope aggregation`;
unweighted complement production is a different descriptive account and is not a
valid substitute.

## Balancing and reconciliation

The split construction above preserves each original cell when weights are
normalized exactly. If additional row, column, final-demand, import, supply,
margin, or supplementary constraints are imposed, a reconciliation step may be
used.

Let \(T^0=[Z'\;Y']\) be the unbalanced split table. Let \(\mathcal{A}\) be the
set of adjustable cells and \(\mathcal{F}\) the set of fixed cells. Local
balancing means

\[
T_{uv}=T^0_{uv}\quad \text{for }(u,v)\in \mathcal{F},
\]

and only cells in \(\mathcal{A}\) may change. The default adjustable set contains
only cells touching emitted children of split parents, unless the mapping set
declares a wider scope.

For non-negative tables, the preferred reconciliation is biproportional scaling:

\[
T_{uv}=\alpha_u T^0_{uv}\beta_v
\quad \text{for }(u,v)\in \mathcal{A},
\]

with \(\alpha_u,\beta_v>0\) chosen so declared row and column targets are met.
If signed flows are allowed, a signed reconciliation method may be used only if
the mapping set declares it and diagnostics report the method.

Balancing must not violate the reaggregation identities unless the run is marked
as failed. A balancing method is acceptable only if it reports convergence,
maximum row gap, maximum column gap, number of iterations, adjustable scope, and
any infeasible targets.

## Supplementary rows and accounting identities

Supplementary rows include value added, imports, margins, taxes, subsidies,
output identities, total supply, total use, and diagnostic totals. They are not
ordinary economic intermediate rows unless the supplementary-row role dictionary
says so.

Identity rows must be computed from split tables, not proxied from donor or
output shares. Examples:

\[
\text{output identity}_a = x'_a,
\]

\[
\text{intermediate input identity}_a = \sum_{u\in I'} Z'_{u a},
\]

\[
\text{total supply}_a = x'_a + \text{imports}_a + \text{declared supply adjustments}_a.
\]

Rows that are not identities, such as value added or margins, require declared
proxy evidence or balancing constraints. If no rule is declared, the row is not
split.

## Reaggregation and approval tests

The approved output must satisfy all declared reaggregation tests:

\[
\max_{i,j}\left|(GZ'G^{\top})_{ij}-Z_{ij}\right| \le \tau_Z,
\]

\[
\max_{i,f}\left|(GY')_{if}-Y_{if}\right| \le \tau_Y,
\]

\[
\max_i\left|(Gx')_i-x_i\right| \le \tau_x.
\]

If supplementary blocks are split, analogous tests must hold for all declared
supplementary identities and reaggregated lower rows. Runs that fail these tests
must not be labelled approved outputs.

## Required diagnostics

A complete run must be auditable from diagnostics. At minimum, the following
quantities must be reportable:

1. child monetary values \(v_h\), physical quantities \(q_h\), unit values
   \(p_h\), selected provider, rejected providers, and quality status;
2. residuals \(\tilde r_p\), approved residual values, negative residual actions,
   and tolerances;
3. delivered-use row scores \(R_{ij,a}\), producer-technology column scores
   \(C_{ij,b}\), compatibility scores \(M^{ij}_{ab}\), normalized weights
   \(W^{ij}_{ab}\), and fallback type for every affected target cell or an
   aggregated equivalent that can reconstruct the rule;
4. donor denominators and fallback rows for the full donor-block route;
5. final-demand allocation weights \(H^{if}_a\);
6. balancing method, adjustable set, convergence, row gaps, column gaps, and
   infeasible targets;
7. reaggregation gaps for \(Z\), \(Y\), \(x\), and any supplementary blocks;
8. manual, equal-share, donor, placeholder-price, and residual fallback counts.

## Interpretation of source families

The source families used by the method have distinct scientific roles.

| Source family | Proper use | Limitation |
|---|---|---|
| Energy balances | Physical fuel/source quantities, transformation flows, electricity by source, fuel families. | Usually not sufficient for monetary values or detailed buyer columns. |
| Product production statistics | Monetary product values and observed unit values. | Often limited by confidentiality and product coverage. |
| Trade statistics | Import/export values, quantities, and unit-value proxies. | Trade prices may not equal domestic production or use prices. |
| PEFA or physical energy accounts | Activity/user allocation, resident-accounting energy use, bridge diagnostics. | Usually too broad for detailed source-specific delivered products. |
| Business/activity statistics | NACE/activity weights for output, turnover, employment, or production. | Activity weights are not product-flow matrices. |
| Donor MRIO/IOT/SUT | Complete structural submatrix prior. | Not authoritative for target-country aggregate totals. |
| Manual rules | Explicit last-resort assumptions. | Must be declared, counted, and scientifically justified. |

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

<!-- ecolecon-environmental-support-census-contract -->
## Environmental support state-space preflight

Broad extension selections must be audited as a complete support state space before
changing donor-support rules. The read-only census is specified in
`../ai/environmental_support_census.md` and implemented by
`../../scripts/audit_environmental_support.py`.

For production `F`, physical numerator evidence and the EXIO monetary-output
denominator are extensive primitives and are projected before division. A nonzero
raw or mapped `F` at zero donor `x` therefore means the donor coefficient is
undefined; it is not by itself invalid source evidence and must not be silently
zeroed. Under the current strict method, a **positive-output target** still requires
a positive mapped donor denominator before a coefficient can be transferred. A
zero-output target with nonzero `F` and zero donor `x` is a valid diagnostic state
with unavailable intensity, not a new physical zero.

`F_Y` remains a separate direct-final-demand boundary: because no product projection
creates a later denominator, nonzero `F_Y` at zero monetary final demand remains a
fail-closed source/support state. External-import direct coefficients likewise
require positive mapped production output for every origin/target carrying nonzero
mapped import use.

The census must enumerate all such states and their source lineage without stopping
at the first stressor. If it reveals a positive-target unsupported state whose
scientific treatment is not already specified, implementation must stop for method
review rather than weaken the validator.
<!-- /ecolecon-environmental-support-census-contract -->
