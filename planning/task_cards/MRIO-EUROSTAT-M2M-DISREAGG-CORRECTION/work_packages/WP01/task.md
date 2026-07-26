# WP01 task

Implement MAP-01--04, SPLIT-01--04 and API-01 from the controlling programme
specification at baseline `6adebc884b13751347dfabd5e463d5571c599cd8`. Scope is
`mapping_csv.py`, `exio_share_split.py`, `iot_flow_semantics.py`, the split CLI and
its direct callers/tests. Do not alter V, M, TOTAL, x/A, final reaggregation,
workbook schemas, pymrio, or donor suitability assumptions.

Use the existing mapping resolver, donor-block splitter, final-demand helpers and
source-rounding helper. Validate the asymmetric DOM/IMP fixture, mapping success and
failure paths, required donors, child integrity, cellwise reconstruction, P5M, and
removed balancing arguments. Run compilation, focused tests, Ruff, Pyright, diff
checks, and the split CLI help command. Write the WP01 report and coverage table.
