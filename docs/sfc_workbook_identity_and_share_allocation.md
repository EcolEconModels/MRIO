# SFC Workbook Identity And Share Allocation Note

This note documents how the workbook exporter in `scripts/aggregate_iot_by_mapping.py` fills the cells outside the core intermediate block `Z` when sectors are split or re-aggregated.

## What is copied exactly from the current aggregated IOT

- The product-by-product intermediate block is written directly from the current `Z_after` matrix.
- The product rows in the final-demand block to the right are written directly from the current `Y_after` matrix.
- The product-row `TU` column is written as row sums of current intermediate use plus current `TFU`.
- The product-row `TFU` column is written as current `P3 + P5 + P6`.
- The top total row for the right-hand block is written as the column sums of the current `Y_after`, plus current `TU` and `TFU` totals.

These cells are accounting identities with respect to the current aggregated IOT, not estimates.

## What is identity-derived below the product rows

The exporter rebuilds several lower rows directly from current accounting identities:

- `Value added, gross = x - column_sum(Z)`
- `Output = x`
- `Total intermediate consumption /final use = column_sum(Z)`
- `Total supply at basic prices = Output + Imports of goods and services`

Here `x` is the current output vector implied by the aggregated IOT.

## What is share-allocated

Some supplementary workbook cells do not exist as separately observed data after a sector split. For those cells, the exporter uses proportional allocation based on current monetary values.

### Supplementary product columns below the main table

For lower supplementary rows that are not pure identities, parent-sector template values are distributed across child sectors using current monetary output shares within the parent basket.

Formally, for a template parent-sector value `v_p` and child sector `i` inside parent `p`, the exporter uses

$$
v_i = v_p \cdot \frac{x_i}{\sum_{j \in p} x_j}.
$$

This preserves the parent total when re-aggregated.

### Supplementary final-demand columns to the right

For lower supplementary rows in the right-hand block, the exporter uses current monetary final-demand shares to split detail categories under the workbook totals `P3`, `P5`, and `P6`.

If a template row contains a total such as `P3`, and the workbook also has detail columns such as `P3_S13`, `P3_S14`, and `P3_S15`, then the split is

$$
v_{k} = V_{P3} \cdot \frac{Y^{\text{current}}_{k}}{\sum_{h \in P3\text{-detail}} Y^{\text{current}}_{h}}.
$$

The same rule is applied for `P5` and `P6` detail columns. `TFU` is then rebuilt as `P3 + P5 + P6`, and `TU` is rebuilt as `product_total + TFU`.

## Validation rules enforced at export time

The exporter now raises an error if any of the following fail:

- Re-aggregated child-sector final-demand values do not match the original parent-sector IOT totals.
- Product-row right-hand final-demand cells in the workbook do not match the current aggregated IOT.
- The total row of the right-hand final-demand block does not match the current aggregated IOT column totals.
- Identity-derived lower rows do not match their current accounting identities.
- Share-allocated supplementary lower rows do not re-aggregate to the template parent totals when the template is authoritative for the exported flow, such as `TOTAL` or a matching flow-specific template.

The intent is to fail fast when the workbook ceases to be a faithful accounting representation of the current aggregated IOT.

## Economic interpretation

This procedure is economically justified as a bookkeeping closure rule, not as a structural behavioral model.

- For cells that are exact identities, the workbook is simply restating the current IOT.
- For supplementary split cells that are not separately observed, current-price monetary shares are the least committal proportional allocator available inside the same accounting system.
- The rule preserves adding-up constraints and avoids fabricating new totals.

## Academic defensibility and limits

This is defensible for workbook construction, reproducibility, and downstream accounting consistency, but it has clear limits.

- It is strong for preserving national-accounting identities.
- It is weak as evidence about the true sub-sector composition of household, investment, or export demand when no direct observations exist at that level.
- Results that depend on the detailed split of supplementary workbook rows or columns should therefore be described as share-allocated, not directly observed.

Where substantive conclusions depend on those split cells, a stronger presentation would report sensitivity checks using alternative allocators, for example:

- base-year monetary shares,
- physical energy shares where a physical counterpart exists,
- or no-split bounds for the unobserved supplementary detail.

That distinction is the key academic boundary: identities are hard information, while the share allocations are transparent auxiliary assumptions used only to close the workbook consistently.
