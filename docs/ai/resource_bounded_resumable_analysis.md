# MRIO resource and resumability policy

This repo follows `../ecolecon-agent-memory/coding/resource-bounded-resumable-analysis.md`.

MRIO-facing analysis code should retain source ZIPs, avoid unnecessary full
parsed-object pickle caches, stream/read only required matrices and extension
rows, prefer sparse representations, print progress ticks, and use manifest-based
resumability for time-consuming derived tables.
