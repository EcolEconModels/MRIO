# Environmental support census

## Purpose

`scripts/audit_environmental_support.py` is the read-only preflight for broad
EXIOBASE environmental-extension work. It enumerates denominator/support states
before an all-extension production run can discover them serially one stressor at
a time.

The census is **not** a production estimator and must not be used to make an
unsupported extension pass. It reads a previously successful monetary split,
reconstructs the persisted emitted-sector projection, streams the EXIOBASE
extension archive, and writes audit artifacts under an explicit audit directory.
It never rewrites the split or canonical environmental outputs.

## Required scientific distinction

For production-side `F`, the accepted method projects the extensive physical
numerator and monetary production output before division. Therefore a nonzero
physical donor observation at zero raw or mapped output is not automatically
invalid. Its coefficient is unavailable until the projected denominator is
positive.

The census classifies production target rows as:

- `SUPPORTED`: mapped donor output is positive;
- `ZERO_ZERO`: mapped `F=0` and mapped `x=0`;
- `NONZERO_F_ZERO_X`: mapped `F!=0` and mapped `x=0`;
- `NO_PROJECTION_SUPPORT`: the persisted projection has no donor edge;
- `NEGATIVE_MAPPED_OUTPUT` or `NONFINITE`: invalid monetary/source states.

Each row is also classified by target output. Under the current method, a
positive-output target with no positive mapped donor denominator is a scientific
support blocker. A zero-output target with `NONZERO_F_ZERO_X` is diagnostic donor
evidence with an undefined coefficient; it is not by itself a reason to discard
physical source evidence.

Direct final-demand `F_Y` remains a separate boundary. Because it is not projected
through the product concordance, nonzero `F_Y` over a zero final-demand denominator
remains fail-closed under the current method.

External-import support is extension-independent at the denominator boundary: any
origin/target with nonzero mapped import use and non-positive mapped production
output is reported as a blocker before extension-specific coefficients are formed.

## Inputs

The census requires:

- one EXIOBASE source zip (or an existing local year resolved without download);
- the scope code and optional region-scope mapping;
- a **successful monetary split root** containing `stk_flow=DOM` and
  `stk_flow=IMP`;
- the persisted split provenance:
  `flow_specific_exio_to_eurostat_weights.csv`, `exio_selected_sectors.json`,
  optional `split_child_edges.csv`, and the canonical emitted-sector vectors.

The persisted projection must reconstruct as a complete partition of the raw EXIO
product axis. If a selected child cannot be proven from those artifacts, the audit
fails instead of guessing its donors.


## Provenance verification

Before environmental rows are scanned, the census proves that the selected EXIO
source bytes and geography scope are the ones persisted by the successful monetary
split. DOM and IMP `split_info.json` must agree with the census on the exact EXIO
source ZIP SHA-256, scope code, ordered scope members, and region-scope mapping
SHA-256 where applicable, and both splits must be marked reconstruction-validated.

`flow_specific_exio_to_eurostat_weights.csv` is the authoritative persisted
projection-weight source. `split_child_edges.csv` is lineage/topology metadata; an
edge weight recorded there is accepted only when it agrees with the authoritative
flow-specific mapping. Duplicate persisted source-target mapping edges fail rather
than using row order as an implicit precedence rule.

Every output CSV has a stable schema even when it contains zero rows. An archive
with no complete `F.txt`/`F_Y.txt`/`unit.txt` extension directories is a
source-integrity blocker, not an empty successful census.

## Outputs

The output directory contains:

- `source_extension_completeness.csv`;
- `production_F_support.csv`;
- `production_F_exception_lineage.csv`;
- `external_import_support_exceptions.csv`;
- `final_demand_FY_support.csv`;
- `source_row_errors.csv`;
- `census_summary.json`.

`production_F_exception_lineage.csv` traces every exceptional mapped target to the
raw scope-level donor sectors, projection weights, and weighted `F/x`
contributions. This is the first artifact to inspect for a case such as Gas Coke.

<!-- environmental-magnitude-diagnostic-v1 -->
## Magnitude and signed-cancellation trace

`scripts/diagnose_environmental_extension_magnitudes.py` is a second-stage,
read-only diagnostic for a successful split whose environmental sidecars are
algebraically valid but numerically implausible. It reuses this census's exact
source-ZIP provenance check, persisted product projection, coded EXIO output, and
scope helpers rather than creating a second mapping or source-loading authority.
It never rewrites canonical split, aggregate, or environmental artifacts.

For DOM and every selected production `F` stressor already present in the target
sidecar, the trace records raw global and scope physical totals, mapped `F`, mapped
EXIO `x`, the mapped donor coefficient, the authoritative target flow denominator,
and the implied versus persisted target physical value and intensity. It also reads
the same `F` table through the hardened variable-offset parser in `temporal.py` and
compares the resulting numeric row against the production extension reader on the
canonical EXIO basis.

For IMP it retains each external origin and records the exact current calculation

$$
N_a=\sum_o (\bar F_{oa}/\bar x_{oa})\bar u_{oa},\qquad
D_a=\sum_o\bar u_{oa},\qquad
G_a=\sum_o|\bar u_{oa}|,
$$

along with the conditioning diagnostic

$$
\kappa_a=|D_a|/G_a.
$$

The command reports, but does not adjudicate, very small $\kappa_a$, extreme
absolute intensities, source-reader disagreement, non-positive mapped production
output at an actively used origin, and mismatches between reconstructed and
persisted target sidecars. `--intensity-threshold` and
`--cancellation-ratio-threshold` are report-only thresholds. They are not
production tolerances, scientific validity cutoffs, or changes to signed-flow
semantics.

The explicit audit output directory receives
`production_magnitude_trace.csv`, `import_magnitude_trace.csv`,
`import_origin_trace.csv`, `source_reader_comparison.csv`, `source_errors.csv`, and
`magnitude_summary.json`. The origin trace is limited to target rows carrying at
least one diagnostic flag; the DOM/IMP magnitude tables enumerate every selected
sidecar stressor/sector combination.

## non-fail-fast and strict modes

Without `--strict`, the command writes the complete census and exits successfully
even when current-method blockers exist. This is the default discovery mode.

With `--strict`, the complete census is still written first, then the command exits
with status 2 if source-integrity errors or current-method support blockers remain.
Do not use strict mode until the state-space has been reviewed.

## Workflow rule

For a broad extension/geography/year generalization:

1. run the monetary source and no-environment gates first;
2. run this census before changing donor-support semantics;
3. inspect every current-method blocker and the exception lineage;
4. search repository methods/history/tests for the same invariant;
5. change implementation only when the scientific treatment is already settled;
6. add a cross-layer regression for any previously duplicated contract;
7. only then run the all-extension production path.

This workflow follows the cross-project contract-first debugging policy in
`../../ecolecon-agent-memory/coding/contract-first-scientific-debugging.md`.
