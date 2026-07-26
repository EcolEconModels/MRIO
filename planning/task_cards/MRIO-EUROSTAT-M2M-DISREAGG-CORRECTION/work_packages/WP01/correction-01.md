# WP01 Correction 01 — Complete Split Validation and Donor Guards

Work in:

`/home/agilra/EcolEconModels/MRIO`

## Baseline

Required baseline commit:

`05a9f17f8d4a9790b47a6d4c3640e9f7003c2f3a`

Start with:

```bash
git status --short
git rev-parse HEAD
```

Do not proceed with a dirty working tree or another baseline.

This is a correction to WP01. Do not begin WP02 and do not alter V, M, TOTAL, canonical x/A, pymrio, final reaggregation or workbook semantics.

Before editing, save this entire instruction verbatim as:

```text
planning/task_cards/MRIO-EUROSTAT-M2M-DISREAGG-CORRECTION/
  work_packages/WP01/correction-01.md
```

Set WP01 to `correction_required` in `programme_status.md`.

## Required corrections

### C01 / SPLIT-02 — Remove x from split reconstruction

In `src/ecolecon_mrio/exio_share_split.py`:

* remove `x_split` from `_validate_split_reconstruction()`;
* remove aggregation and comparison of `x`;
* remove its call-site argument;
* do not replace it with another output/use-total comparison.

WP01 reconstruction must test only:

1. labelled source-parent Z cells;
2. non-overlapping source-parent Y values;
3. generated final-demand controls;
4. rounded published controls.

Canonical x semantics are owned by WP02.

Required test:

* a source P3 or P5 aggregate differing from its detail sum within the permitted source-rounding bound must pass WP01 reconstruction;
* the same mismatch beyond the bound must fail with the relevant source control named;
* the test must demonstrate that no strict x calculation runs first.

### C02 / SPLIT-02 — Validate TFU and TU explicitly

Add source and generated checks for:

```text
TFU = sum(non-overlapping atomic final demand)
TU  = row sum of Z + TFU
```

Requirements:

* published TFU and TU use the source-rounding envelope and correct component-term counts;
* generated TFU and TU use strict generated tolerances;
* TFU and TU are separate diagnostics;
* a corrupted generated TFU must fail even when atomic Y is correct;
* a corrupted generated TU must fail even when Z, atomic Y and TFU are correct.

Reuse or move the existing source-control derivation logic from `derive_total_iot_flow.py`; do not create another inconsistent implementation.

### C03 / SPLIT-02 — Remove the silent `0.01` source-quantum default

Change the public source-rounding quantum contract to:

```python
source_rounding_quantum: float | None = None
```

Rules:

1. Dataset mode uses the quantum inferred from parsed Eurostat metadata.
2. An explicit CLI/API override may replace the inferred quantum.
3. Local-file mode must not silently assume `0.01`.
4. When a published control equals its components under strict tolerance, no quantum is needed.
5. When a nonzero published-control gap requires a rounding envelope and no quantum is available, fail with an error requesting an explicit quantum.
6. Reject non-finite or non-positive explicit values.

Add an optional CLI argument:

```text
--source-rounding-quantum
```

Document its exact meaning.

Tests must cover:

* metadata-inferred quantum;
* explicit override;
* exact source controls without a quantum;
* rounded controls without a quantum failing;
* invalid zero, negative and non-finite quantum values.

### C04 / MAP-01 — Distinguish missing reference keys from zero values

In the flow-specific mapping resolver:

* require every structural target key used by an authorised calculated split to exist in the selected reference mapping;
* a present key with value `0.0` is a valid numerical zero;
* an absent key is a mapping/schema error.

The error must include:

* source code;
* missing target code or label;
* all structural targets;
* reference type used.

Tests:

1. missing target reference key fails;
2. present zero target succeeds when another target has positive mass;
3. all target keys present and ordinary positive values succeeds.

### C05 / MAP-04 — Validate every donor actually used by projection

Create one authoritative donor-domain check immediately before projection.

It must cover:

* raw EXIO members belonging to split children;
* nonzero EXIO members mapped to retained unsplit emitted Eurostat labels;
* members used in row or column donor blocks involving selected split parents.

Do not audit unrelated concordance targets absent from the emitted Eurostat table.

A missing member must fail before reindex/fill-zero and report:

* raw EXIO code;
* emitted label;
* original Eurostat parent or retained sector;
* flow;
* whether it was required on the split-child or retained-sector side.

Add a test in which:

* every selected child donor exists;
* one donor used only by a retained unsplit sector is absent;
* the split fails rather than silently replacing that donor with zero.

### C06 / MAP-02 — Make all-zero parent behaviour explicit

Before generic share normalisation, detect a selected parent whose complete donor profile has zero or non-finite mass.

Do not fall into a generic `Cannot normalize non-positive donor vector` message and do not invent equal shares silently.

The failure must identify:

* Eurostat parent;
* flow;
* emitted children;
* output, sales, input and final-demand donor masses.

It is acceptable for this correction to reject an all-zero parent explicitly. Do not implement an arbitrary economic allocation merely to support it.

Add one direct test.

### C07 / SPLIT-04 — Reject collisions with any other source parent

An emitted child owned by parent A must not equal:

* a retained unsplit source code;
* another emitted child;
* the original code of a different split parent B.

A child may equal its own parent only when it represents an explicitly unchanged one-child identity.

Add separate tests for:

* retained-sector collision;
* duplicate child ownership;
* child equal to another split parent’s code.

### C08 / API-01 — Finish donor-block API cleanup

Update the split CLI description and help text so it does not claim RAS/GRAS balancing.

Keep legacy filenames such as `*_balanced.csv`, `ras_info.json` or `balance_info.json` only where an existing compatibility contract requires them. They must remain clearly marked as deprecated aliases and must state:

```text
balancing_applied = false
```

Do not remove active RAS/GRAS functionality from the separate Energy-Balance pipeline.

Update tests so fake `balancing_stats` data are not supplied to helpers that no longer use it.

### C09 — Pyright delta evidence

Do not attempt to fix all inherited pandas-stub diagnostics.

Run Pyright on the same changed core/test file list at:

1. baseline `05a9f17...`;
2. the correction head.

Use JSON output where available and compare diagnostic fingerprints:

```text
file + line + rule + message
```

The correction report must classify diagnostics as:

* inherited unchanged;
* resolved;
* newly introduced;
* moved due only to formatting.

No newly introduced diagnostic on changed core lines may remain unexplained.

Avoid reformatting entire files. Do not edit unrelated lines merely to reduce diagnostic counts.

## Required tests

Run the existing focused WP1 suite plus direct tests for every correction above.

The final focused suite must contain independently named tests for:

```text
rounded P3/P5 accepted inside bound
rounded P3/P5 rejected outside bound
generated TFU corruption rejected
generated TU corruption rejected
missing source-rounding quantum rejected when needed
missing target reference key rejected
present zero target reference accepted
missing retained-sector donor rejected
all-zero parent gives contextual error
collision with another split parent rejected
CLI help contains no active RAS/GRAS claim
```

Do not claim that one broad end-to-end test covers these mutation cases.

## Validation

Use:

```bash
PYTHON=/home/agilra/EcolEconModels/MRIO/.venv/bin/python
```

Run:

```bash
$PYTHON -m py_compile <changed Python files>
$PYTHON -m pytest -q <all WP1 focused test files>
$PYTHON -m ruff check <changed Python and test files>
$PYTHON -m pyright <changed core and test files>
$PYTHON scripts/split_eurostat_energy_by_exiobase.py --help
git diff --check
```

No real Austria production run is required for this correction.

## Correction report files

Create:

```text
reports/agent_runs/MRIO-EUROSTAT-M2M-DISREAGG-CORRECTION/WP01/
  correction-01-implementation-report.md
  correction-01-coverage.tsv
```

`correction-01-coverage.tsv` must contain one row for each affected existing requirement:

```text
MAP-01
MAP-02
MAP-04
SPLIT-02
SPLIT-04
API-01
```

## Mandatory return contract

Your final response must use the following headings exactly.

### Repository

* baseline SHA;
* final SHA;
* push status;
* `git status --short`;
* complete changed-file list;
* `git diff --stat`;
* `git diff --numstat`.

### MAP-01

* exact defect corrected;
* exact functions and lines;
* before algorithm;
* after algorithm;
* named tests;
* actual test result;
* remaining limitation.

Repeat that complete structure separately for:

```text
MAP-02
MAP-04
SPLIT-02
SPLIT-04
API-01
```

### Source-rounding evidence

Report:

* inferred quantum test;
* explicit override test;
* exact-without-quantum test;
* rounded-without-quantum failure test;
* largest accepted and rejected example gaps.

### Donor-domain evidence

Report:

* selected-child donor set;
* retained-sector donor set;
* missing-donor mutation and exact error.

### Generated-control evidence

Report separately for:

```text
P3
P5M
P5
TFU
TU
```

For each: source treatment, generated formula and mutation test.

### Pyright delta

Report:

* baseline diagnostic count;
* correction diagnostic count;
* new diagnostic count;
* resolved diagnostic count;
* every new changed-line diagnostic, or explicitly `none`.

### Validation commands

List every command actually run and its exit status. Do not collapse them into “validation passed.”

### Deferred scope

Explicitly confirm that this correction did not change:

* V allocation;
* M;
* TOTAL construction;
* canonical x/A;
* pymrio;
* final reaggregation;
* workbook schemas;
* the broad full-table validator.

## Commit

Suggested subject:

```text
Complete WP1 split validation guards
```

Commit and attempt to push. A credential failure must be reported verbatim but does not justify starting WP02.
