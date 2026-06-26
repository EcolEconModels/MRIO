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

`load_eurostat_iot()`, `load_eurostat_table_2d()`, and `load_eurostat_series_1d()` must always parse from the cached source payload. This avoids stale parsed-cache bugs where parser behaviour changes but a legacy parsed cache silently returns an older derived representation.

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
| 0 | Source-cache-only Eurostat loading | Implemented for IOT/2D/1D loaders | Eurostat loaders persist/use raw source payloads only and ignore legacy parsed caches. |
| 1 | Canonical many-many edge projection \(P^f\) | Implemented for split profiles | Public helper materializes \(P^f\) from emitted split profiles and raw EXIO members. |
| 2 | CO2/value/price mapping under many-many edges | Implemented for selected split children | Extension extraction uses raw EXIO members; disaggregated ratios use edge weights and emitted child labels. |
| 3 | `split_child_edges.csv` metadata | Implemented | Materialized edge children parse into raw source, target, edge weight, and `phase4_edge_output`. |
| 4 | Explicit \(PZP^\top\), \(PY\) toy tests | Implemented for selected-edge toy case | Toy fixture compares donor projection and split allocations against explicit matrix projection. |
| 5 | Lower-row `V/M` split hierarchy | Implemented row-specific hierarchy | Supplementary `V` rows choose output-share proxy for value-added-like rows, projected EXIO column-use proxy for other lower rows, and equal fallback with row-level diagnostics. |
| 6 | Reaggregation with edge labels and `V/M` | Implemented for library helper and CLI edge metadata augmentation | Reaggregate `Z/Y/x/V/M` with \(H\), verify exact toy recovery, and auto-augment CLI mappings from `split_child_edges.csv`. |
| 7 | End-to-end validation on Eurostat/EXIOBASE data | Validation tool implemented; representative data runs still required | Run split, CO2/value/price extraction, reaggregation, and audit reports for representative flows and years. |

## Tests added by this patch

- Legacy parsed-cache files are ignored for IOT, 2D, and 1D Eurostat loaders; fresh parsed values from the raw source path are returned.
- Emitted mapping helper returns `aggregate -> emitted child -> raw EXIO member(s)`.
- Edge-child CO2 intensity maps raw EXIO member `p24.d` into emitted child `p24.d@CPA_C20` using the concordance edge weight.
- `split_child_edges.csv` metadata now records `source_code=p24.d`, `concordance_target_code=CPA_C20`, and `phase=phase4_edge_output` for a materialized edge child.
- Projection matrix helper builds explicit emitted-label by raw-EXIO `P`, and the toy test checks `PZP^T`, `PY`, and resulting Eurostat split allocations.
- Reaggregation helper builds explicit emitted-child to aggregate-parent `H` mappings from split profiles, and the toy test checks exact `Z/Y/x/V/M` recovery for edge labels.
- Aggregation CLI auto-augments missing edge-child mapping rows from `split_child_edges.csv` and writes `aggregation_edge_mapping_audit.csv`.

## Remaining high-priority work

1. Run the Phase-7 validator on representative Eurostat/EXIOBASE flow/year outputs after a full split and reaggregation run, for example:

   ```bash
   /home/agilra/EcolEconModels/MRIO/.venv/bin/python \
     scripts/validate_many_many_pipeline_outputs.py \
     --split-dir <split-output-dir> \
     --agg-dir <aggregation-output-dir> \
     --mapping-csv <aggregation-mapping.csv> \
     --out-json <validation-report.json>
   ```

2. Inspect generated `validation-report.json`, `aggregation_edge_mapping_audit.csv`, `mapped_output_consistency.csv` for each flow, `mapped_output_flow_identity.csv`, value/price ratio checks, CO2 ratio checks, and exact recomputed `Z/Y/x/V/M` reaggregation gaps.

Supplementary `V` splitting now writes row-level diagnostics with row code, proxy method, denominator, fallback reason, and mass-before/mass-after checks.
- Phase-7 output validator recomputes edge-aware aggregation from split outputs and checks exact `Z/Y/x/V/M` recovery plus CO2/value ratio files.
- Boundary-correct mapped-output diagnostics now run for `DOM`, `IMP`, and `TOTAL`; cross-dataset EXIO-vs-Eurostat level gaps are diagnostic by default, while `mapped_output_flow_identity.csv` hard-checks `DOM + IMP` against `TOTAL`.
