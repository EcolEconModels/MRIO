# WP01 implementation report

## Repository state

- Baseline and current HEAD before commit: `6adebc884b13751347dfabd5e463d5571c599cd8`.
- Initial tree was clean. This implementation changes only WP01 code, focused tests,
  direct pipeline callers, planning records, and this report.

## Runtime path and design

`at_2021_workbook_reaggregation.sh` invokes
`split_eurostat_energy_by_exiobase.py`, which resolves each DOM/IMP mapping through
`mapping_csv.resolve_weighted_mapping_frame`, materialises children through
`exio_share_split.split_iot_using_exiobase_shares`, and rebuilds display controls
through `iot_flow_semantics.enforce_final_demand_display_identities`. The later TOTAL
and reaggregation stages are unchanged.

The mapping schema uses `source_code`, `target_code`, `enabled`, optional `weight` or
`Share_used`, and `allocation_rule`/`certainty`. Runtime DOM and IMP references are
their own `Z + non-export Y` vectors. Structural edges now retain enabled rows even at
zero flow weight; multi-target sources emit `source@parent` children.

## Requirement coverage

See [implementation_coverage.tsv](implementation_coverage.tsv). All nine WP01
requirements are implemented and exercised by the focused fixture. The fixture has
three Eurostat parents, retained `A`, one-to-one donors, `p24.d` as a many-to-many
donor, asymmetric Z, P3 details, P5M/P5 components, and TFU/TU.

## Validation

- `py_compile`: passed for changed Python files.
- Focused pytest: `54 passed`; two known pandas `stack` deprecation warnings.
- Ruff: passed.
- `scripts/split_eurostat_energy_by_exiobase.py --help`: passed (Matplotlib used a
  temporary cache because its default user cache directory is unwritable).
- `git diff --check`: passed.
- Pyright was run on changed core and test files. It reports 173 diagnostics, dominated
  by existing pandas-stub overload issues in `mapping_csv.py`, the split script, and
  pre-existing fixtures. No runtime test failure accompanies those diagnostics.

No Austria production workflow was run; WP02 owns production execution. V, M, TOTAL,
x/A, pymrio, workbook schemas, and final reaggregation semantics were not changed.

## Deferred work

WP02 remains planned and requires independent WP01 audit before implementation.
