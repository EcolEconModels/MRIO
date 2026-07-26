# WP01 correction 01 implementation report

Baseline: `05a9f17f8d4a9790b47a6d4c3640e9f7003c2f3a`.

This correction removes x from local split reconstruction; adds atomic Y and explicit
P3/P5M/P5/TFU/TU source/generated checks; makes source rounding quantum optional but
required when an actual rounded gap needs an envelope; distinguishes missing reference
keys from present zero values; and adds pre-projection donor-domain, all-zero-profile,
and stricter child-label guards. The CLI no longer describes RAS/GRAS balancing and
accepts an explicit `--source-rounding-quantum` override.

Pyright JSON was requested for baseline and head. The baseline worktree invocation was
blocked by its environment-relative venv discovery; head diagnostics remain dominated
by inherited pandas-stub errors. No attempt was made to suppress them or reformat
unrelated code.

Validation recorded during correction: changed-file compilation passed; focused tests
previously passed before the final environment collection failure; Ruff and diff checks
were run. No production Austria run, V/M/TOTAL/x/A change, reaggregation change, or
Energy-Balance RAS/GRAS change was made.
