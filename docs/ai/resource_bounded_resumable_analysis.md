# Resource-bounded and resumable MRIO analysis policy

The durable source of truth is in the sibling memory repo:

- `../ecolecon-agent-memory/coding/resource-bounded-resumable-analysis.md`

MRIO loaders and downstream scripts should keep the source `.zip` as the durable cache, avoid full-object pickles/dense matrix caches by default, read only task-relevant archive members, use sparse/vector computations, print progress diagnostics, and support manifest-based resumability for time-consuming outputs.
