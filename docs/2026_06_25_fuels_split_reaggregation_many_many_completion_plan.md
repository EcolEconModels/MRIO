# Fuels split/reaggregation many-many completion plan — 2026-06-25

This plan supersedes the completion-status claims in the 2026-06-24 plan where implementation and documentation diverged. It keeps the original five-phase structure, but makes the source-cache boundary, many-to-many projection contract, CO2/value/price mapping, lower-row `V/M` handling, and reaggregation checks explicit.

## Cache policy

We keep a **source cache** only:

```text
data/eurostat/<dataset>/<request_hash>/raw.json
data/eurostat/<dataset>/<request_hash>/request.json
```

We do **not** keep or trust derived parsed caches such as:

```text
parsed/Z.pkl
parsed/Y.pkl
parsed/meta.json
```

`load_eurostat_iot()` must always parse from the cached source payload. This avoids stale parsed-cache bugs where `parse_eurostat_iot()` supports `V/M`, but a legacy parsed cache silently returns only `Z/Y/meta`.

## One mapping after Phase 4

The user-authored mapping CSV/concordance remains an **input** used to build split profiles and edge children. After Phase 4 materializes edge children, downstream output tables must use the emitted mapping:

```text
aggregate Eurostat sector -> emitted child label -> raw EXIO member(s)
```

Example:

```text
CPA_C20 -> p24.d@CPA_C20 -> p24.d
```

The original raw requested mapping must not be used as a downstream fallback for value/price or CO2 tables, because it does not contain emitted edge-child labels.

## Mathematical contract

For a flow \(f\), let raw EXIO sectors be \(E\), Eurostat targets be \(C\), and emitted split labels be \(R^f\). Let \(W^f_{ce}\ge 0\) be the EXIO-source-to-Eurostat-target concordance weight.

A many-target EXIO source \(e\) selected inside target \(c\) is emitted as an edge child:

\[
r=(e,c)
\]

represented in CSV labels as:

```text
e@c
```

The projection from raw EXIO sectors to emitted split labels is:

\[
P^f_{re} =
\begin{cases}
W^f_{ce}, & r=(e,c),\\
1, & r=e \text{ and no edge split is required},\\
0, & \text{otherwise}.
\end{cases}
\]

The EXIO donor blocks used for allocation are:

\[
\tilde Z^{X,f}=P^f Z^{X,f}(P^f)^\top,
\qquad
\tilde Y^{X,f}=P^fY^{X,f}.
\]

Eurostat intermediate cells are split by donor shares:

\[
Z'^{f}_{rs}
=
Z^{E,f}_{ij}
\frac{\tilde Z^{X,f}_{rs}}
{\sum_{r'\in R_i}\sum_{s'\in R_j}\tilde Z^{X,f}_{r's'}},
\qquad
r\in R_i,\ s\in R_j.
\]

Eurostat final demand cells are split by donor shares:

\[
Y'^{f}_{rk}
=
Y^{E,f}_{ik}
\frac{\tilde Y^{X,f}_{rk}}
{\sum_{r'\in R_i}\tilde Y^{X,f}_{r'k}}.
\]

If the donor denominator is zero or invalid, the implementation must use explicit fallback shares and write diagnostics.

## CO2, price, and value-per-physical-input mapping

Extraction from EXIO extensions must use **raw EXIO sector codes**, never emitted edge labels. For emitted child \(r\), ratios must be computed as weighted ratio-of-sums:

\[
\tilde m^f_r=\sum_e P^f_{re}m^f_e,
\qquad
\tilde x^f_r=\sum_e P^f_{re}x^f_e,
\qquad
\iota^f_r=\frac{\tilde m^f_r}{\tilde x^f_r}.
\]

This applies to both direct CO2 intensity and value/price proxies. For a single raw EXIO source split across multiple targets, the numerator and denominator both receive the same edge weight, so the intensity remains unchanged while totals are correctly attributed. For multiple raw EXIO members, the ratio is a true weighted ratio-of-sums.

## Reaggregation contract

Let \(H_{gr}=1\) if emitted split label \(r\) maps to aggregate group \(g\). Then:

\[
Z^{agg}=HZ'H^\top,
\qquad
Y^{agg}=HY',
\qquad
x^{agg}=Hx',
\qquad
V^{agg}=V'H^\top,
\qquad
M^{agg}=M'.
\]

Edge labels such as `p24.d@CPA_C20` must map through their target side, not as if the raw source `p24.d` belonged wholly to every target.

## Phase tracker

| Phase | Scope | Status after this patch | Completion condition |
|---|---|---:|---|
| 0 | Source-cache-only Eurostat loading | Implemented | `load_eurostat_iot()` persists/uses raw source payloads only and ignores legacy parsed caches. |
| 1 | Canonical many-many edge projection \(P^f\) | Partially implemented | Add a public helper or module that materializes \(P^f\) and makes the projection contract directly testable. |
| 2 | CO2/value/price mapping under many-many edges | Implemented for selected split children | Extension extraction uses raw EXIO members; disaggregated ratios use edge weights and emitted child labels. |
| 3 | `split_child_edges.csv` metadata | Implemented | Materialized edge children parse into raw source, target, edge weight, and `phase4_edge_output`. |
| 4 | Explicit \(PZP^\top\), \(PY\) toy tests | Partially implemented | Add a toy IO fixture comparing implementation allocations exactly against explicit matrix projection. |
| 5 | Lower-row `V/M` split hierarchy | Pending | Implement documented \(q_{\ell s}^f\) hierarchy or downgrade documentation to the current intermediate-input-share-only method. |
| 6 | Reaggregation with edge labels and `V/M` | Needs audit | Reaggregate `Z/Y/x/V/M` with \(H\) and verify exact recovery or documented tolerances. |
| 7 | End-to-end validation on Eurostat/EXIOBASE data | Pending | Run split, CO2/value/price extraction, reaggregation, and audit reports for representative flows and years. |

## Tests added by this patch

- Legacy parsed-cache files are ignored; fresh parsed `V/M` from the raw source path are returned.
- Emitted mapping helper returns `aggregate -> emitted child -> raw EXIO member(s)`.
- Edge-child CO2 intensity maps raw EXIO member `p24.d` into emitted child `p24.d@CPA_C20` using the concordance edge weight.
- `split_child_edges.csv` metadata now records `source_code=p24.d`, `concordance_target_code=CPA_C20`, and `phase=phase4_edge_output` for a materialized edge child.

## Remaining high-priority work

1. Add an explicit `build_projection_matrix_for_split_profiles()` or equivalent public helper so tests and docs can refer to the actual \(P^f\).
2. Add a toy `PZP^T/PY` allocation test with one raw EXIO source split over two Eurostat targets.
3. Implement the full lower-row `V` split hierarchy:
   \[
   V'^{f}_{\ell s}
   =
   V^{E,f}_{\ell j}
   \frac{q^f_{\ell s}}{\sum_{s'\in R_j}q^f_{\ell s'}}
   \]
   using row-specific production/value-added proxies where available, projected EXIO column-use where appropriate, and equal fallback only as a last resort.
4. Audit reaggregation outputs for edge-child labels and `V/M`.
