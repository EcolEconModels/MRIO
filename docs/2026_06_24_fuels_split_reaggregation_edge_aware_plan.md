# Fuels split/reaggregation: coverage preflight and edge-aware plan

## Status and rule for use

This document is the working plan for making the fuels, fuels-more, and
green/brown split pipelines safe under official-derived EXIOBASE-to-Eurostat CPA
concordances.

Do **not** treat outputs from this pipeline as scientifically final until the
edge-aware split and reaggregation phases are implemented and validated. Phase 1
is operational safety. Phase 2 is diagnostics only. The scientifically clean
version needs at least the edge-aware numerical split and edge-aware
reaggregation phases.

The key problem is that a split child is not always a raw EXIOBASE product. With
one-EXIO-to-many-Eurostat or true many-to-many concordances, the correct child is
an EXIO source to Eurostat target edge, for example `p24.d@CPA_C20`.

## Recommended patch series

This should remain a multi-phase patch series on one branch, not one giant
patch. The contracts touched are split-manifest expansion, flow-specific
concordance weights, output sector naming, reaggregation manifests, diagnostics,
Eurostat supplementary IO rows, and tests.

### Phase 1 -- Coverage safety and manifest repair

Purpose: fix the immediate `p10.g` failure and prevent future expensive runs
from failing only at the final reaggregation step.

Changes:

1. Add `p10.g` to:
   - `scripts/pipelines/fuels_reaggregation.csv`,
   - `scripts/pipelines/fuels_more_reaggregation.csv`,
   - `scripts/pipelines/energy_green_brown_reaggregation.csv`.
2. Add:
   - `scripts/pipelines/check_reaggregation_coverage.py`.
3. The preflight expands the split manifest through the selected concordance and
   checks:

```text
expanded split children subset_of active reaggregation source codes
```

4. Wire the preflight into:
   - `scripts/pipelines/at_2021_workbook_reaggregation.sh`,
   before the expensive EXIO split call.
5. Add tests for fuels, fuels-more, green/brown, and generated CPA_65 coverage
   when the generated concordance exists locally.

This phase is safe and useful, but it does **not** fix the `p24.d` edge issue.

### Phase 2 -- Edge-child metadata without numerical-output changes

Purpose: add the internal data model and diagnostics while keeping old numerical
behavior available.

Add diagnostics with these conceptual fields:

```text
flow
parent_code
current_child_code
proposed_edge_child_code
source_code
concordance_target_code
source_n_targets
edge_weight_current_reference
is_many_target_source
is_edge_child_candidate
source_label
target_label
```

For one-target sources:

```text
proposed_edge_child_code = source_code
```

For multi-target sources:

```text
proposed_edge_child_code = source_code + "@" + target_code
```

Example:

```text
p24.g          # one-target child
p24.d@CPA_C20  # edge child
```

Mathematically, for each flow `f`, `W^f_{ce}` is the flow-specific
EXIO-to-Eurostat concordance weight from EXIO source `e` to Eurostat target `c`.
For a selected split parent `c in S`, the future edge-aware emitted child set is:

```text
A_c = {(e, c): W^f_{ce} > 0}
```

not merely:

```text
{e: e maps to c}
```

This phase writes diagnostics such as:

```text
split_child_edges.csv
```

It must **not** change `Z_split`, `Y_split`, `x_split`, reaggregation, or
flow-specific weighting. It lets us inspect `p24.d@CPA_C20` before changing
matrices.

### Phase 3 -- Flow-specific concordance weights

Purpose: stop using one `TOTAL`-based concordance weighting for `DOM`, `IMP`, and
`TOTAL`.

For each flow:

```text
f in {DOM, IMP, TOTAL}
```

compute the Eurostat target-use reference:

```math
u^f_c = \sum_j Z^{E,f}_{c j} + \sum_{k \notin exports} Y^{E,f}_{c k}.
```

Then resolve one-to-many weights as:

```math
W^f_{c e} = \frac{u^f_c}{\sum_{d \in C(e)} u^f_d}.
```

For `p24.d`:

```math
W^f_{CPA\_C20,p24.d}
= \frac{u^f_{CPA\_C20}}{u^f_{CPA\_C20}+u^f_{CPA\_C21}},
```

```math
W^f_{CPA\_C21,p24.d}
= \frac{u^f_{CPA\_C21}}{u^f_{CPA\_C20}+u^f_{CPA\_C21}}.
```

This phase can still output raw children, but it must prove the flow-specific
weights are available and tested. The audit boundary should be use-side and
exclude export final-demand columns from the destination-use reference.


#### Phase 3 implementation note

Phase 3 resolves the EXIO-to-Eurostat weighted concordance separately for each
`stk_flow` variant. The pipeline must no longer resolve the full concordance once
from `TOTAL` and then reuse those weights for `DOM` and `IMP`. Each flow writes a
`flow_specific_exio_to_eurostat_weights.csv` diagnostic beside the split outputs.

This phase still preserves the Phase 2 raw-child output contract: multi-target
sources are diagnosed with proposed edge-child codes, but `Z_split`, `Y_split`,
and `x_split` are not renamed to edge children until Phase 4.

### Phase 4 -- Edge-aware split output

Purpose: change the actual split output so multi-target sources are not emitted
as whole raw EXIO children.

For selected split sectors, output:

```text
p24.d@CPA_C20
```

instead of:

```text
p24.d
```

when splitting `CPA_C20`.

The projection matrix must be built over the **full** EXIO-to-Eurostat
concordance, not only selected split sectors.

Define `P^f` from EXIO sources to output split sectors.

For unsplit Eurostat sectors `c`:

```math
P^f_{c e} = W^f_{c e}.
```

For edge child `a=(e,c)`:

```math
P^f_{a e'} =
\begin{cases}
W^f_{c e}, & e'=e,\\
0, & e'\neq e.
\end{cases}
```

Then donor matrices are:

```math
\widetilde{Z}^{X,f} = P^f Z^{X,f} (P^f)^\top,
```

```math
\widetilde{Y}^{X,f} = P^f Y^{X,f}.
```

The Eurostat split must preserve Eurostat mass exactly. For each Eurostat cell
`Z^{E,f}_{ij}`, split over the child block:

```math
B_{ij} = \{r:g(r)=i\} \times \{s:g(s)=j\}
```

using:

```math
Z'^{f}_{rs}
= Z^{E,f}_{ij}
\cdot
\frac{\widetilde{Z}^{X,f}_{rs}}
{\sum_{(r',s')\in B_{ij}} \widetilde{Z}^{X,f}_{r's'}}.
```

For final demand:

```math
Y'^{f}_{rk}
= Y^{E,f}_{ik}
\cdot
\frac{\widetilde{Y}^{X,f}_{rk}}
{\sum_{r':g(r')=i}\widetilde{Y}^{X,f}_{r'k}}.
```

This is the first phase that changes actual output sector names and numerical
donor shares.

### Phase 5 -- Eurostat supplementary IO rows, columns, and value added

Purpose: split not just the square `Z` block and final-demand rows, but also the
other rows/columns that belong to a full Eurostat IO table/workbook.

The current core parser returns only:

```text
Z = product/product intermediate block
Y = product-row final demand block
```

It restricts rows to codes that also appear as intermediate-demand columns. That
means lower rows such as value added, taxes less subsidies, compensation of
employees, gross operating surplus, net taxes on products, and other
supplementary IO rows are not currently represented as first-class split arrays.
Workbook-template code may preserve or reshape workbook layout, but that is not
a tested core split/reaggregation object for all supplementary rows.

Required design:

1. Preserve a full parsed IO-table representation in addition to `Z` and `Y`:

```text
Z      product rows x product columns
Y      product rows x final-demand columns
V      supplementary/lower rows x product columns
M      optional supplementary rows x final-demand columns, if present and meaningful
meta   row/column classifications and original ordering
```

2. Split product rows and product columns with the same output child set used for
   `Z`, `Y`, and `x`.
3. Split lower/supplementary rows column-wise. For a lower row `ell` and original
   product column `j`, distribute over child columns `s` with `g(s)=j`:

```math
V'_{\ell s}
= V_{\ell j}
\cdot
\frac{q^f_s}{\sum_{s':g(s')=j} q^f_{s'}}.
```

4. The column-side weights `q^f_s` must be documented and tested. Candidate
   hierarchy:
   - use EXIO/Eurostat child output or production-side value-added proxies when
     available and appropriate for the row type;
   - otherwise use the projected EXIO column-use profile from the same
     edge-aware concordance;
   - otherwise use deterministic equal-share fallback and write diagnostics.
5. Reaggregation must exactly recover the original lower-row values:

```math
\sum_{s:g(s)=j} V'_{\ell s} = V_{\ell j}.
```

6. Reaggregation of edge children must apply to `Z`, `Y`, `x`, `V`, and any
   workbook/full-table outputs with one consistent child-to-target mapping.
7. Add tests that fail if value added or any supplementary row is silently
   dropped during split or reaggregation.

This phase is required before the full workbook output can be considered
consistent with the disaggregation/reaggregation of the monetary IO table.

Implementation status, 2026-06-24:

- `parse_eurostat_iot()` now returns `V` and `M` supplementary arrays alongside
  `Z`, `Y`, and metadata.
- `split_iot_using_exiobase_*()` passes `V` and `M` through the split pipeline.
  `V` is split column-wise over product children using edge-aware EXIO
  column-input donor shares and writes diagnostics when a fallback is needed.
  `M` is preserved as the supplementary-row by final-demand block.
- `aggregate_iot()` and `aggregate_iot_by_mapping.py` reaggregate/write `V` and
  `M` as first-class outputs.
- The SFC workbook exporter consumes the aggregated `V` and `M` outputs when
  available, falling back to the older template identity/share reconstruction
  only for rows or columns that are absent.
- Focused tests cover parser retention, edge-aware `V` splitting,
  reaggregation of supplementary rows, and workbook export from provided `V/M`.

### Phase 6 -- Reaggregation manifest update for edge children

Purpose: make reaggregation operate on the new output child codes.

Keep existing raw rows for backward compatibility where harmless, but add edge
rows where needed:

```csv
p24.d@CPA_C20,Chemicals nec allocated to CPA_C20,CPA_C20_others,CPA_C20: Chemicals and chemical products_others,True,Edge child for multi-target EXIO source p24.d; only the CPA_C20 share is reaggregated here.
```

Also keep:

```csv
p10.g,BKB/Peat Briquettes,CPA_C19_COAL,CPA_C19: Coal,True,...
```

The reaggregation loader can keep enforcing one source to one target because
`p24.d@CPA_C20` and `p24.d@CPA_C21` are different source codes.

Implementation status, 2026-06-24:

- The split pipeline emits edge children such as `p24.d@CPA_C20` for selected
  multi-target EXIO sources while retaining the raw EXIO source as the donor
  member.
- The fuels, fuels-more, and energy green/brown reaggregation manifests include
  the `p24.d@CPA_C20` edge row.
- The reaggregation coverage preflight expands selected multi-target sources to
  their emitted edge-child code before checking coverage.

### Phase 7 -- Boundary-correct consistency checks

Purpose: remove the wrong hard check and replace it with diagnostics matching the
audit boundary.

The split pipeline should not hard-fail on:

```text
Eurostat TOTAL including exports vs EXIO producer-output x_region
```

Instead, it should report:

1. use-side comparison:

```math
Z + Y_{non-export}
```

for:

```text
DOM
IMP
TOTAL
```

2. flow identity:

```math
DOM + IMP \approx TOTAL
```

3. export-specific caveat:

```text
P6 export columns are preserved, but donor profiles are domestic-use based unless export-specific EXIO donor profiles are enabled.
```

The conceptual boundary is that `DOM`, `IMP`, and `TOTAL` are separate flow modes
and `TOTAL` means products used by the selected country, not production-side
output including exports.

Implementation status, 2026-06-26:

- The mapped-output consistency check now runs for `DOM`, `IMP`, and `TOTAL` variants.
- The Eurostat side uses the same `Z + non-export Y` destination-use boundary used for flow-specific concordance weights.
- The EXIO side uses the selected flow's destination-use slice (`Z` rows for the requested flow into the destination plus destination final demand), not producer-side `x_region`.
- The pipeline writes and hard-checks `mapped_output_flow_identity.csv` so `DOM + IMP` is compared against `TOTAL` for both Eurostat and mapped-EXIO totals, using strict flow-identity tolerances independent of the looser cross-dataset mapping-consistency diagnostics.
- Accounting identity lower rows such as `P1`, `P2_ADJ`, and `TS_BP` are derived from split `x`, split `Z` column sums, and split imports where available, not from donor proxy shares.
- `IMP` extension CO2/value proxy files are suppressed until origin-weighted import intensities are implemented; export/import-specific donor profiles remain Phase 8 optional work.

### Phase 8 -- Optional export-specific donor profile

Purpose: improve export splitting.

For final-demand columns:

```text
P3/P5/domestic final demand -> destination-country EXIO final-demand donor profile
P6 exports                  -> selected-country origin to foreign final-demand donor profile
```

This requires loading an EXIO export slice or deriving selected-country origin
rows to foreign final-demand columns. It is conceptually correct but not needed
to fix the current `p24.d` and `p10.g` issues.

## Should mapping files change?

### Split manifests

No. Keep them simple. Current split files such as `fuels_split.csv` and
`fuels_more_split.csv` list Eurostat sectors to split. That remains the right
user-facing contract.

### Reaggregation manifests

Yes. They need to allow edge-child source codes.

Add rows like:

```text
p10.g -> CPA_C19_COAL / ENERGY_BROWN
p24.d@CPA_C20 -> CPA_C20_others
```

Possibly keep old raw `p24.d` rows for compatibility with old outputs.

### Concordance files

No structural change is needed. The generated CPA_65 concordance can stay as
source-target rows. Edge children are derived runtime/output artifacts, not
committed concordance rows.

### Output metadata

Yes. Add and preserve detailed metadata:

```text
split_child_edges.csv
```

with at least:

```text
flow
parent_code
current_child_code
proposed_edge_child_code
source_code
concordance_target_code
edge_weight_current_reference
is_edge_child_candidate
source_n_targets
target_label
source_label
```

Later numerical phases should also write metadata for value-added/supplementary
row splitting, including row type, allocation denominator, fallback method, and
mass-preservation diagnostics.

## Tests required

At minimum:

1. **Coverage preflight test**

Input:

```text
split: CPA_C19
concordance: p10.g -> CPA_C19
reaggregation missing p10.g
```

Expected: preflight fails before split.

2. **Coverage success test**

Add `p10.g`; preflight passes.

3. **Edge child expansion test**

Input:

```text
p24.d -> CPA_C20, CPA_C21
split manifest contains CPA_C20
```

Expected future numerical output:

```text
p24.d@CPA_C20
```

not raw:

```text
p24.d
```

Phase 2 only diagnoses this; Phase 4 changes the actual output.

4. **Flow-specific weight test**

If:

```text
DOM:   C20=90, C21=10
IMP:   C20=40, C21=60
TOTAL: C20=130, C21=70
```

then:

```text
DOM p24.d@CPA_C20 weight   = 0.9
IMP p24.d@CPA_C20 weight   = 0.4
TOTAL p24.d@CPA_C20 weight = 0.65
```

5. **Mass preservation test for Z and Y**

After splitting:

```math
\sum_{r:g(r)=i,\ s:g(s)=j} Z'_{rs} = Z_{ij}
```

and:

```math
\sum_{r:g(r)=i} Y'_{rk} = Y_{ik}
```

for every original Eurostat parent `i,j,k`.

6. **Mass preservation test for supplementary rows/value added**

For every lower/supplementary row `ell` and product parent column `j`:

```math
\sum_{s:g(s)=j} V'_{\ell s} = V_{\ell j}.
```

The test should fail if supplementary rows are silently dropped.

7. **Reaggregation test**

Edge child:

```text
p24.d@CPA_C20
```

maps to:

```text
CPA_C20_others
```

and the final aggregate is mass-preserving for `Z`, `Y`, `x`, supplementary rows,
and any full-workbook output.

## Practical commit grouping

Use multi-phase patches. A practical grouping is:

```text
Patch 1: coverage preflight + p10.g manifest rows
Patch 2: edge-child data model + diagnostics, no numerical output change
Patch 3: flow-specific weights + edge-aware split output for Z/Y/x
Patch 4: supplementary IO rows/value added split + reaggregation consistency
Patch 5: reaggregation manifests/tests/docs updated for edge children
Patch 6: boundary-correct consistency diagnostics
Patch 7: optional export-specific donor profiles
```

Do not run final paper/results pipelines after only Patch 1 or Phase 2. Phase 1
is operational safety. Phase 2 is diagnostic. Scientific reruns need the
edge-aware numerical phases and supplementary-row consistency.
