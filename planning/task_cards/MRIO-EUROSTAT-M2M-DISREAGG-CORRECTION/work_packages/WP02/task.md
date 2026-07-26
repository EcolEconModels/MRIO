# WP02 — Supplementary Blocks, Canonical Flows and Final Reaggregation

Work in:

`/home/agilra/EcolEconModels/MRIO`

## Prerequisite

Do not start from `05a9f17...`.

Start only from the independently accepted WP01 correction commit. Record that exact SHA as the WP02 baseline.

Verify:

```bash
git status --short
git rev-parse HEAD
```

The working tree must be clean.

Before runtime edits, save this entire instruction verbatim as:

```text
planning/task_cards/MRIO-EUROSTAT-M2M-DISREAGG-CORRECTION/
  work_packages/WP02/task.md
```

Do not replace it with a short summary.

## Planning amendment

Update the programme to version `1.1`.

Replace the coarse unimplemented WP2 placeholder rows with these exact auditable requirements:

```text
M-01       preserve_source_M_during_split
M-02       preserve_M_during_reaggregation
V-01       explicit_V_row_policy_inventory
V-02       conserve_primitive_V_parent_columns
V-03       derive_only_authorised_V_identity_rows
V-04       reject_unknown_or_unapproved_V_fallbacks
TOTAL-01   additive_TOTAL_Z_and_atomic_Y
TOTAL-02   construct_TOTAL_V_from_source_TOTAL_policy
TOTAL-03   preserve_published_TOTAL_M
XA-01      separate_flow_use_from_canonical_output
XA-02      compute_all_A_with_shared_domestic_output
XA-03      propagate_requested_pymrio_failures
REAGG-01   strict_explicit_child_mapping_coverage
REAGG-02   correct_block_axis_reaggregation
REAGG-03   conserve_primitive_totals_and_target_order
REAGG-04   validate_serialized_outputs
PROD-01    run_real_Austria_2021_pipeline
PROD-02    produce_complete_production_evidence
CLEAN-01   demote_duplicate_full_table_validation
CLEAN-02   retain_one_canonical_production_route
```

Assign every row to WP02.

Create `work_packages/WP02/owned_requirements.txt` with exactly those IDs.

Update `programme_status.md` to `WP02: implementing`.

## Non-negotiable transformation contract

The pipeline performs:

```text
Eurostat DOM and IMP
    → EXIO-informed split children
    → canonical TOTAL and x/A construction
    → explicit new-classification reaggregation
```

The final new classification is not required to reproduce the original Eurostat classification.

Do not impose:

```text
R_new(D(X)) = X
```

The local child-to-source-parent reconstruction checks remain the accepted WP01 contract.

---

## Phase 1 — Inventory before implementation

Before changing runtime code, inspect the real Austria 2021 source DOM, IMP and TOTAL V/M blocks.

Create:

```text
reports/agent_runs/MRIO-EUROSTAT-M2M-DISREAGG-CORRECTION/WP02/
  source_v_m_inventory.csv
```

For every supplementary row and flow, record:

```text
flow
row_code
source_label
present_in_V
present_in_M
source_role
policy_class
split_proxy_or_formula
TOTAL_construction
equation_authority
```

The expected source rows include, where present:

```text
CPA_TOTAL
P2_ADJ
P1
B1G
D1
D11
D21X31
D29X39
P51C
B2A3G
B2A3N
IMP
P7
P7_B0
P7_D0
P7_U2
P7_U3
TS_BP
```

Use source codes and existing reviewed source policy. Do not classify rows from broad string prefixes.

If the real source contains a nonzero row not covered by this instruction and existing policy evidence does not resolve it, stop before allocation and report the exact row.

---

## M-01 — Preserve source M during splitting

M has supplementary-account rows and final-demand columns. Neither axis is a product-sector axis.

Change the split path so that for each source flow:

```python
M_split = M_source.copy()
```

Requirements:

* same ordered row labels;
* same ordered column labels;
* same numeric values;
* same NaN/applicability mask;
* no invocation of `refresh_m_identity_rows()`;
* no blanking of P1, B1G, TS_BP or other rows;
* no recomputation from split Y or Z.

Delete or bypass `_refresh_final_demand_identity_rows()` in the product-splitting path.

Do not delete a generic helper still used by a genuinely different pipeline without tracing all callers.

Tests:

1. M containing finite values, zeros and NaNs survives splitting.
2. A mutation changing one value fails equality.
3. A mutation changing one NaN to zero fails the mask check.
4. Row or column reordering fails an ordered-equality check.

---

## M-02 — Preserve M during final reaggregation

Product reaggregation changes:

* both Z axes;
* Y rows;
* V columns;
* x index.

It does not change M.

In `aggregate_iot()` and the production aggregator:

```python
M_agg = M_split.copy()
```

Do not call `refresh_supplementary_rows()` on M.

Tests must verify exact ordered value/mask preservation through:

```text
source → split → TOTAL construction → final reaggregation → CSV/XLSX reopen
```

---

## V-01 — Replace prefix inference with an explicit row policy

Remove `_VALUE_ADDED_SUPPLEMENTARY_PREFIXES` and any equivalent `startswith()` policy from the production V split.

Create one authoritative explicit policy keyed by exact row code or reviewed role.

Use these classes:

### Derived rows

```text
CPA_TOTAL  → corresponding flow Z column sums
P1         → shared canonical domestic output x, where present
IMP        → IMP Z column sums for the DOM imported-use control, where authorised
P7         → transformed imported-product use/supply vector in TOTAL
TS_BP      → P1 + P7 in TOTAL, where the reviewed source contract authorises it
```

Do not infer P2_ADJ from Z column sums alone.

### Primitive rows split by declared proxy

At minimum:

```text
P2_ADJ
B1G
D1
D11
D21X31
D29X39
P51C
B2A3G
B2A3N
P7_B0
P7_D0
P7_U2
P7_U3
```

For production-account/value-added components, use the declared child output-share proxy unless the existing reviewed method provides a stronger row-specific source.

For P2_ADJ, use an explicitly documented column-input/output proxy selected after inspecting the source meaning. Do not silently treat it as plain intermediate use.

For P7 geographic components, use an explicit import-supply proxy and preserve every source parent value and the declared geographic partitions.

The policy object must record:

```text
row_code
role
primitive_or_derived
split_proxy
applicable_flows
TOTAL_construction
```

---

## V-02 — Conserve every primitive V parent-column value

For every primitive V row (r) and source parent product (j):

```text
source_V[r,j]
=
sum(split_V[r, children(j)])
```

Use the accepted WP01 child-to-parent map.

The check must be cellwise by row and parent column, not only a row total.

On failure report:

* flow;
* V row;
* parent product;
* source value;
* reconstructed value;
* signed gap;
* absolute gap;
* proxy used.

Add asymmetric mutation tests for at least:

```text
D1
D21X31
P51C
B2A3N
P7_B0
```

---

## V-03 — Derive only authorised identity rows

After primitive allocation and canonical x construction, derive the explicit identity rows listed under V-01.

Do not derive:

* B1G merely as `x - Z column sum` unless that exact valuation identity is authorised;
* P2_ADJ merely as a Z column sum;
* production-account components from one another merely because an algebraic identity appears plausible.

For every derived row, the implementation report must state:

```text
formula
flow
input blocks
valuation assumption
source-policy authority
test
```

---

## V-04 — Unknown rows and fallbacks

Unknown nonzero V rows must stop production.

The error must include:

* flow;
* row code and label;
* parent product;
* value;
* available policy classes.

Equal-share fallback must be disabled by default.

An optional fallback may exist only under an explicit CLI flag such as:

```text
--allow-v-equal-share-fallback
```

The canonical Austria production run must not use that flag.

When enabled, every fallback must record:

```text
flow
row
parent
children
source value
attempted proxy
proxy denominator
fallback shares
```

Report both fallback count and absolute monetary amount.

---

## TOTAL-01 — Construct only Z and primitive Y additively

After the accepted DOM and IMP split:

```text
Z_TOTAL = Z_DOM + Z_IMP
Y_TOTAL_atomic = Y_DOM_atomic + Y_IMP_atomic
```

Then reconstruct:

```text
P3
P5M
P5
TFU
TU
```

from generated components.

Do not add overlapping aggregate and detail columns independently.

Require identical structural child topology and order across DOM and IMP.

---

## TOTAL-02 — Build TOTAL V from published TOTAL V

Do not construct the complete TOTAL V block as:

```text
V_TOTAL = V_DOM + V_IMP
```

Load the published source TOTAL V schema and transform its product columns using the common child topology.

Then:

1. split primitive TOTAL V rows using their explicit policies;
2. derive only the authorised identity rows;
3. preserve source-parent values for every primitive row;
4. retain TOTAL-only rows such as P7 components and TS_BP.

DOM and IMP V schemas are allowed to differ from TOTAL.

Delete validators requiring identical V row schemas across all flows.

---

## TOTAL-03 — Preserve published TOTAL M

Write:

```python
M_TOTAL = source_TOTAL_M.copy()
```

Do not derive TOTAL M from DOM + IMP.

Replace `_validate_m_flow_contract()` with per-flow source equality:

```text
M_DOM output   = source M_DOM
M_IMP output   = source M_IMP
M_TOTAL output = source M_TOTAL
```

---

## XA-01 — Separate use totals from output

The split result must distinguish:

```text
use_total_flow = Z_flow row sums + atomic Y_flow row sums
x_domestic     = generated domestic product output
```

Do not write a flow-use vector to `x_split.csv`.

Preferred production sequence:

1. split DOM and IMP Z/Y and primitive V;
2. write `use_total_split.csv` for each primitive flow;
3. canonicalise all flows after DOM and IMP are available;
4. write one shared `x_split.csv` into DOM, IMP and TOTAL directories.

The shared x must be:

```text
x = DOM Z row sums + DOM atomic Y row sums
```

The published source P1 row is a rounded source control checked against this generated x.

Update `SplitResult` or introduce a clearly named pre-canonical result so that its `x` field cannot contain flow use.

Do not retain a public field named x with the old meaning.

---

## XA-02 — Recompute all A with shared domestic output

For every flow:

```text
A_flow = Z_flow · diag(x_domestic)^(-1)
```

Use the existing guarded `technical_coefficients()` function.

Tests must verify:

* DOM, IMP and TOTAL use the same denominator;
* a denominator from IMP use totals produces a failing mutation;
* zero output with nonzero column input fails;
* A is recomputed after final reaggregation;
* A is never summed from children.

---

## XA-03 — Propagate requested pymrio failures

When `create_io_system=False`, do no pymrio work.

When `create_io_system=True`:

* import or construction failure must propagate with context;
* `calc_system()` failure must propagate;
* do not return `io_system=None` after a broad exception.

Test with a controlled failing pymrio substitute or monkeypatch.

---

## REAGG-01 — Strict explicit mapping coverage

In `aggregate_iot_by_mapping.py`, perform strict child coverage against the explicit mapping CSV **before** `_augment_mapping_from_split_child_edges()`.

When strict mode is active:

* every emitted child must appear explicitly exactly once;
* missing edge children fail;
* auto-augmentation is disabled;
* conflicting duplicate targets fail.

Retain edge augmentation only under a clearly named non-strict compatibility option.

Add a mutation in which deleting one explicit edge-child row fails strict mode even though `split_child_edges.csv` could infer a target.

---

## REAGG-02 — Correct block-axis transformations

Final reaggregation must use:

```text
Z  → map and sum rows and columns
Y  → map and sum product rows
V  → map and sum product columns
M  → unchanged copy
x  → map and sum
A  → recompute from final Z and mapped x
```

Derived Y and authorised V identity rows are reconstructed after primitive aggregation.

Do not apply a product mapping to M.

---

## REAGG-03 — Conservation and exact target classification

Because the final target classification differs from the original Eurostat classification, validate:

```text
sum(Z_before) = sum(Z_after)
column total of each atomic Y category before = after
total of every primitive V row before = after
sum(x_before) = sum(x_after)
M_before = M_after
```

Also require:

* every emitted child maps exactly once;
* no selected child is dropped;
* exact expected target-sector set;
* exact expected target-sector order.

Do not use `R_new(D(X)) = X`.

---

## REAGG-04 — Serialization

After writing CSV and workbook outputs, reopen them and compare against the in-memory result.

Check separately:

```text
Z
Y
V
M
x
A
target labels
target order
NaN masks
```

Use strict labels/masks and tight floating tolerances.

A successful file write is not sufficient.

---

## PROD-01 — Real Austria 2021 production run

Run the canonical Austria 2021 pipeline from a clean output location using the repository’s official current mappings and locally available source data.

Do not delete source EXIOBASE zips or Eurostat source caches.

Run without:

```text
--allow-v-equal-share-fallback
```

The production run must exercise:

```text
DOM split
IMP split
TOTAL construction
new target reaggregation
CSV output
workbook output
focused validation
```

---

## PROD-02 — Required production evidence

Create:

```text
reports/agent_runs/MRIO-EUROSTAT-M2M-DISREAGG-CORRECTION/WP02/
  implementation_report.md
  implementation_coverage.tsv
  v_row_policy.csv
  production_validation_summary.csv
  final_target_sectors.csv
  fallback_summary.csv
```

`production_validation_summary.csv` must contain at least:

```text
check_id
object
flow
status
max_abs_gap
tolerance
affected_label
evidence_path
```

Report:

* source and mapping versions;
* exact production command;
* child count by parent;
* zero-weight structural edges;
* V policy for every source row;
* V fallback count and monetary amount;
* M equality for every flow;
* maximum Z/Y/V parent-reconstruction gaps;
* TOTAL additive Z/Y gaps;
* x equality across flows;
* maximum A formula gap;
* explicit final mapping coverage;
* final target-sector list;
* primitive-block conservation gaps;
* CSV/workbook reopen gaps;
* runtime and peak memory when available.

---

## CLEAN-01 — Demote the broad full-table validator

Do not expand `iot_full_table_validation.py`.

After focused production checks pass:

* ensure the canonical split/reaggregation pipeline does not depend on it;
* retain it only as an optional source-audit command if it has independent diagnostic value;
* clearly document that it is not the transformation correctness gate.

Delete it only when all callers and useful source-audit functions have been inspected and replaced. Prefer demotion over risky unrelated deletion.

---

## CLEAN-02 — One canonical route

Trace all active production scripts and remove or deprecate any weaker route that:

* splits TOTAL independently;
* constructs TOTAL V/M by blanket addition;
* writes flow-use totals as x;
* computes A with flow-specific use denominators;
* silently augments strict final mappings;
* refreshes M during product transformation.

Do not alter the separate Energy-Balance disaggregation pipeline unless shared code changes require a narrow compatibility update.

---

## Required focused test matrix

At minimum, tests must independently mutate and reject:

```text
M value changed
M NaN changed to zero
M row order changed
primitive V child mass moved between parents
unknown V row
unauthorised equal-share fallback
TOTAL V built by DOM+IMP
TOTAL M built by DOM+IMP
IMP use vector used as x
flow-specific denominator used for A
zero output with nonzero Z column
missing strict edge mapping
auto-augmentation hiding a missing strict mapping
A summed instead of recomputed
M product-mapped during reaggregation
wrong final target order
CSV value changed after write
workbook NaN mask changed
pymrio exception swallowed
```

Use asymmetric fixtures and independent direct-summation oracles.

Do not count one broad pipeline test as all of these mutation tests.

---

## Validation commands

Use:

```bash
PYTHON=/home/agilra/EcolEconModels/MRIO/.venv/bin/python
```

Run:

```bash
$PYTHON -m py_compile <all changed Python files>
$PYTHON -m pytest -q <all focused WP1 regression and WP2 tests>
$PYTHON -m ruff check <changed Python and test files>
$PYTHON -m pyright <changed core Python and tests>
$PYTHON scripts/split_eurostat_energy_by_exiobase.py --help
$PYTHON scripts/derive_total_iot_flow.py --help
$PYTHON scripts/aggregate_iot_by_mapping.py --help
$PYTHON scripts/validate_many_many_pipeline_outputs.py --help
git diff --check
```

Run Pyright at the accepted WP1 baseline and WP2 head on the same file list and report the diagnostic delta. Do not hide new diagnostics inside the inherited count.

---

## Mandatory return contract

The final implementer response must begin:

```text
WP02 IMPLEMENTATION RESULT
```

Then provide these sections exactly.

### Repository

* WP02 baseline SHA;
* final SHA;
* push status;
* working-tree status;
* complete changed-file list;
* `git diff --stat`;
* `git diff --numstat`.

### Requirement-by-requirement evidence

Provide a separate subsection for every ID:

```text
M-01
M-02
V-01
V-02
V-03
V-04
TOTAL-01
TOTAL-02
TOTAL-03
XA-01
XA-02
XA-03
REAGG-01
REAGG-02
REAGG-03
REAGG-04
PROD-01
PROD-02
CLEAN-01
CLEAN-02
```

Each subsection must contain all of:

```text
Status:
Files/functions:
Previous behaviour:
Implemented behaviour:
Economic/mathematical contract:
Named tests:
Actual test result:
Production evidence:
Known limitation:
```

No requirement may be covered only by “see tests” or a group summary.

### V row-policy table

Paste a complete table containing:

```text
flow
row_code
primitive_or_derived
split_proxy_or_formula
TOTAL_construction
source authority
fallback count
```

### M evidence

For DOM, IMP and TOTAL separately report:

```text
source shape
output shape
ordered labels equal
NaN masks equal
maximum absolute value gap
```

Report the same comparison before and after final reaggregation.

### TOTAL evidence

Report separately:

```text
Z DOM+IMP gap
atomic Y DOM+IMP gap
V construction method by row
M source-TOTAL equality gap
```

### x/A evidence

Report:

```text
x source/formula
DOM x vs IMP x gap
DOM x vs TOTAL x gap
A_DOM formula gap
A_IMP formula gap
A_TOTAL formula gap
final reaggregated A formula gap
zero-output guard test
```

### Reaggregation evidence

Report:

```text
number of emitted children
number explicitly mapped
number auto-augmented in strict mode
number dropped
expected target count
actual target count
exact target-order result
Z conservation gap
atomic Y conservation gaps by category
V row-total conservation gaps
x conservation gap
M equality result
```

`number auto-augmented in strict mode` must be zero.

### Fallback evidence

List every fallback type and:

```text
count
absolute monetary amount
largest affected parent
production permission
```

### Production command

Give the exact command and exit status.

### Serialization evidence

Give CSV and workbook reopen gaps separately for every block.

### Pyright delta

Give baseline, head, added, resolved and changed-line diagnostic counts.

### Validation commands

List every command and exit status individually.

### Deferred or unresolved matters

Write `none` or enumerate exact blockers. Do not claim completion while a blocking requirement remains.

## Commit

Suggested subject:

```text
Correct Eurostat supplementary flows and reaggregation
```

Commit only after every blocking requirement and validation succeeds. Attempt to push and report any credential error verbatim.
