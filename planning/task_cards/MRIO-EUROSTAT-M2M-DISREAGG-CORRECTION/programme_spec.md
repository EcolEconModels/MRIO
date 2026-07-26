# Eurostat many-to-many disaggregation correction

Version: 1.0. Baseline: `6adebc884b13751347dfabd5e463d5571c599cd8`.

The pipeline splits DOM and IMP Eurostat product IOT parents with EXIO donors,
materialises source-target children, derives TOTAL later, and then reaggregates to a
different target classification. It must not require final reaggregation to recover
the original Eurostat classification.

WP01 owns MAP-01--04, SPLIT-01--04, and API-01: authorised mappings, stable
structural edges, local source-parent reconstruction, P5M controls, child integrity,
and donor-block API honesty. WP02 will own M/V/TOTAL/x/A, final reaggregation,
production execution, and cleanup; it has no implementation task yet.

Cross-package invariants: local split reconstruction is source-parent based; DOM and
IMP retain the same structural edge identities; zero flow weights are numerical zero,
not absent topology; V, M, TOTAL, x/A semantics and final reaggregation remain outside
WP01.
