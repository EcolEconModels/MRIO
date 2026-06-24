# Fuels split/reaggregation: coverage preflight and edge-aware plan

## Status

This document records the patch plan for making the fuels, fuels-more, and
green/brown split pipelines safe under the official-derived EXIOBASE to
Eurostat CPA mappings.

Phase 1 is intentionally limited to operational safety:

- add missing reaggregation coverage for `p10.g` (`BKB/Peat Briquettes`),
- fail early when a split manifest expanded through the selected concordance
  contains children that the reaggregation manifest does not cover,
- document the later many-to-many edge-aware design.

Phase 1 does not change the numerical split semantics. In particular, it does
not yet replace raw many-target EXIO children such as `p24.d` with edge children
such as `p24.d@CPA_C20`.

## Why Phase 1 is needed

The lightweight split manifests list Eurostat parent sectors to split, for
example `CPA_C19`, `CPA_C20`, and `CPA_D`. The split code expands those parent
sectors through the selected EXIOBASE-to-Eurostat concordance. When a newer
concordance assigns an additional EXIO product to one of those parents, the
reaggregation manifest must cover that new child.

The official-derived CPA 65 concordance can select `p10.g` (`BKB/Peat
Briquettes`) under `CPA_C19`. The fuels reaggregation taxonomy treats this as a
solid coal/peat fuel:

- `fuels_reaggregation.csv`: `p10.g -> CPA_C19_COAL`,
- `fuels_more_reaggregation.csv`: `p10.g -> CPA_C19_COAL`,
- `energy_green_brown_reaggregation.csv`: `p10.g -> ENERGY_BROWN`.

## Phase 1 algorithm

Given:

- split manifest parent sectors `S`,
- active concordance rows `(source_code=e, target_code=c)`,
- active reaggregation rows keyed by `source_code`,

compute the selected raw children for the current non-edge-aware pipeline:

```text
selected_children = {e | (e -> c) is active in concordance and c in S}
```

Then require:

```text
selected_children subset_of active_reaggregation_source_codes
```

If not, the preflight prints the missing child codes and exits non-zero before
EXIOBASE streaming starts.

## Full edge-aware design

The long-run correction is to stop treating every split child as a raw EXIOBASE
product. For one-to-many and many-to-many concordance rows, the split child
should be the EXIO source to Eurostat target edge.

Let:

- `E` be EXIOBASE products,
- `C` be Eurostat CPA sectors,
- `S` be the subset of Eurostat sectors requested for splitting,
- `f` be a flow in `{DOM, IMP, TOTAL}`.

Resolve a flow-specific concordance matrix:

```math
W^f_{c e} >= 0,      sum_{c in C(e)} W^f_{c e} = 1.
```

For one-to-one rows, `W^f_{ce}=1`. For one-to-many rows without official static
weights, use the flow-specific Eurostat use reference:

```math
W^f_{c e} = u^f_c / sum_{d in C(e)} u^f_d,
```

where:

```math
u^f_c = sum_j Z^{E,f}_{c j} + sum_{k notin exports} Y^{E,f}_{c k}.
```

For example, if `p24.d` maps to both `CPA_C20` and `CPA_C21`, then splitting
`CPA_C20` should emit `p24.d@CPA_C20`, not the full raw `p24.d`.

The edge-aware output child set for selected parent `c` is:

```text
A_c = {(e, c) | W^f_{c e} > 0 and c in S}
```

Raw EXIO product codes are only a shorthand for one-target edges.

## Future patch phases

1. **Coverage safety**: implemented by this phase.
2. **Edge-child metadata**: emit `split_child_edges.csv` with parent, child,
   source, target, flow, and edge weight, without changing numerical output.
3. **Flow-specific weights**: resolve concordance weights separately for `DOM`,
   `IMP`, and `TOTAL`; do not reuse a `TOTAL` reference for all flows.
4. **Edge-aware split output**: emit edge children such as `p24.d@CPA_C20` for
   multi-target sources and project EXIO donor matrices through the full
   concordance.
5. **Edge-aware reaggregation manifests**: allow rows such as
   `p24.d@CPA_C20 -> CPA_C20_others`, while retaining raw rows where needed for
   backward compatibility.
6. **Boundary-correct diagnostics**: compare use-side `Z + non-export Y` for
   `DOM`, `IMP`, and `TOTAL`, and report export split caveats separately from
   producer-output checks.

## Export boundary note

Destination-use checks should exclude export columns from the Eurostat final
demand reference. Export columns in the split table are still preserved by the
split, but a later patch should add export-specific donor profiles if export
composition is analytically important.
