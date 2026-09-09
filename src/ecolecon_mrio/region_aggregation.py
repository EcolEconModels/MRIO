"""Compile unweighted MRIO region scopes and dispatch low-memory scope reads.

This module owns geography membership independently of sector mappings. It validates
many-to-one scope definitions against the source-region basis and exposes destination-
scope monetary use blocks whose production-output denominator remains a separate
producer-side account. Provider-specific ZIP parsing stays in the EXIOBASE module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ecolecon_mrio.iot_flow_semantics import _normalize_nonnegative_accounting_control


ScopeFlow = Literal["internal", "external_imports", "total"]
_REGION_MAPPING_COLUMNS = (
    "scope_code",
    "scope_label",
    "member_region",
    "member_order",
    "enabled",
    "notes",
)
_TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_VALUES = frozenset({"0", "false", "f", "no", "n", "off"})


@dataclass(frozen=True)
class RegionScope:
    """One unweighted many-source-regions-to-one-scope definition."""

    code: str
    members: tuple[str, ...]
    label: str | None = None


@dataclass(frozen=True)
class RegionAggregationPlan:
    """Validated disjoint region scopes compiled against one source-region basis."""

    source_regions: tuple[str, ...]
    scopes: tuple[RegionScope, ...]
    source_to_scope: pd.Series


@dataclass(frozen=True)
class MrioScopeSlice:
    """Low-memory destination-use slice on one consolidated region scope.

    ``Z`` and ``Y`` are extensive use flows into ``scope`` for the requested
    boundary. ``x_scope`` is production output of scope members and is therefore
    invariant across ``internal``, ``external_imports`` and ``total`` requests.
    A scope slice is not, in general, a balanced standalone IOT; in particular,
    callers must not derive technical coefficients from an external-import slice.
    """

    scope: RegionScope
    Z: pd.DataFrame
    Y: pd.DataFrame
    x_scope: pd.Series
    flow: ScopeFlow
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if self.flow not in {"internal", "external_imports", "total"}:
            raise ValueError(f"Unsupported scope flow {self.flow!r}.")
        z_rows = pd.Index(self.Z.index.astype(str), name="sector")
        z_cols = pd.Index(self.Z.columns.astype(str))
        y_rows = pd.Index(self.Y.index.astype(str), name="sector")
        x_rows = pd.Index(self.x_scope.index.astype(str), name="sector")
        if not z_rows.is_unique or not z_cols.is_unique or not y_rows.is_unique or not x_rows.is_unique:
            raise ValueError("MRIO scope-slice sector axes must be unique.")
        if not z_rows.equals(pd.Index(z_cols, name="sector")):
            raise ValueError("MRIO scope-slice Z must be square on one ordered sector basis.")
        if not y_rows.equals(z_rows):
            raise ValueError("MRIO scope-slice Y rows must exactly match the Z sector basis.")
        if not x_rows.equals(z_rows):
            raise ValueError("MRIO scope-slice production output must exactly match the Z sector basis.")


def _parse_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if cast(bool, pd.isna(value)):
        raise ValueError("Region mapping 'enabled' values must not be missing.")
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(f"Could not parse region mapping enabled value: {value!r}")


def _mapping_frame(mapping: Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(mapping, pd.DataFrame):
        frame = mapping.copy()
    else:
        path = Path(mapping)
        if not path.is_file():
            raise FileNotFoundError(f"Region-scope mapping not found: {path}")
        frame = pd.read_csv(path, keep_default_na=False)
    missing = set(_REGION_MAPPING_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(
            "Region-scope mapping is missing required columns: "
            f"{sorted(missing)!r}."
        )
    return frame.loc[:, list(_REGION_MAPPING_COLUMNS)].copy()


def _validated_source_regions(source_regions: Sequence[str]) -> tuple[str, ...]:
    regions = tuple(str(region).strip() for region in source_regions)
    if not regions or any(not region for region in regions):
        raise ValueError("source_regions must contain non-blank canonical region codes.")
    if len(set(regions)) != len(regions):
        raise ValueError("source_regions must be unique.")
    return regions


def _validate_scope(scope: RegionScope) -> tuple[str, ...]:
    code = str(scope.code).strip()
    members = tuple(str(member).strip() for member in scope.members)
    if not code:
        raise ValueError("RegionScope.code must not be blank.")
    if not members or any(not member for member in members):
        raise ValueError(f"RegionScope {code!r} must contain non-blank members.")
    if len(set(members)) != len(members):
        raise ValueError(f"RegionScope {code!r} contains duplicate members.")
    return members


def compile_region_aggregation(
    *,
    source_regions: Sequence[str],
    mapping: Path | pd.DataFrame,
    scopes: Sequence[str] | None = None,
    include_complement: bool = False,
    complement_code: str = "RoW",
) -> RegionAggregationPlan:
    """Validate many-to-one aggregation for one or more disjoint scopes."""

    # Freeze the actual provider-region basis first; complements are defined against
    # this basis rather than against a political list embedded in Python.
    source_region_order = _validated_source_regions(source_regions)
    source_region_set = set(source_region_order)
    frame = _mapping_frame(mapping)

    # Normalize only textual identifiers/metadata. Membership stays binary and no
    # numeric field is interpreted as an economic aggregation weight.
    for column in ("scope_code", "scope_label", "member_region", "notes"):
        frame[column] = frame[column].map(lambda value: str(value).strip())
    frame["enabled"] = frame["enabled"].map(_parse_enabled)
    active = frame.loc[frame["enabled"]].copy()
    if active.empty:
        raise ValueError("Region-scope mapping contains no enabled rows.")

    order_numeric = cast(pd.Series, pd.to_numeric(active["member_order"], errors="coerce"))
    if order_numeric.isna().any():
        raise ValueError("Enabled region mapping member_order values must be numeric.")
    order_int = order_numeric.astype("int64")
    if not order_numeric.eq(order_int).all() or bool((order_int <= 0).any()):
        raise ValueError("Enabled region mapping member_order values must be positive integers.")
    active["member_order"] = order_int

    # Preserve mapping-file scope order so compiled plans and diagnostics are
    # deterministic without assigning economic meaning to row order.
    active_scope_order = list(dict.fromkeys(active["scope_code"].astype(str)))
    if any(not code for code in active_scope_order):
        raise ValueError("Enabled region mapping scope_code values must not be blank.")
    if scopes is None:
        selected_codes = active_scope_order
    else:
        selected_codes = list(dict.fromkeys(str(code).strip() for code in scopes))
        if not selected_codes or any(not code for code in selected_codes):
            raise ValueError("scopes must contain non-blank scope codes when supplied.")
        missing_codes = [code for code in selected_codes if code not in set(active_scope_order)]
        if missing_codes:
            raise ValueError(f"Requested region scopes are absent or disabled: {missing_codes!r}")

    selected = active.loc[active["scope_code"].isin(selected_codes)].copy()
    if selected["member_region"].eq("").any():
        raise ValueError("Enabled region mapping member_region values must not be blank.")

    unknown = sorted(set(selected["member_region"]) - source_region_set)
    if unknown:
        raise ValueError(
            "Region mapping contains members absent from the selected source release: "
            f"{unknown!r}."
        )

    # Geography never disaggregates: one source region cannot feed two selected
    # aggregate scopes in the same plan.
    duplicate_members = selected.loc[
        selected["member_region"].duplicated(keep=False), "member_region"
    ].astype(str)
    if not duplicate_members.empty:
        raise ValueError(
            "Each source region may belong to at most one selected scope; duplicate members: "
            f"{sorted(set(duplicate_members))!r}."
        )

    duplicate_orders = selected.duplicated(
        subset=["scope_code", "member_order"], keep=False
    )
    if bool(duplicate_orders.any()):
        bad = selected.loc[duplicate_orders, ["scope_code", "member_order"]]
        raise ValueError(
            "member_order must be unique within each selected scope; duplicates: "
            f"{bad.to_dict(orient='records')!r}."
        )

    # Compile each selected scope in its declared member_order. The assignment map
    # is the single authoritative source->scope relation for later sparse aggregation.
    compiled: list[RegionScope] = []
    assignments: dict[str, str] = {}
    for code in selected_codes:
        group = selected.loc[selected["scope_code"] == code].sort_values(
            "member_order", kind="stable"
        )
        if group.empty:
            raise ValueError(f"Selected region scope {code!r} has no enabled members.")
        labels = [label for label in dict.fromkeys(group["scope_label"].astype(str)) if label]
        if len(labels) > 1:
            raise ValueError(
                f"Region scope {code!r} has inconsistent scope_label values: {labels!r}."
            )
        members = tuple(group["member_region"].astype(str))
        scope = RegionScope(code=code, members=members, label=labels[0] if labels else None)
        _validate_scope(scope)
        compiled.append(scope)
        assignments.update({member: code for member in members})

    # A complement is derived from the actual source release, never from a hard-coded
    # Rest-of-World country list.
    if include_complement:
        complement = str(complement_code).strip()
        if not complement:
            raise ValueError("complement_code must not be blank.")
        if complement in {scope.code for scope in compiled}:
            raise ValueError(f"complement_code {complement!r} collides with a selected scope.")
        complement_members = tuple(
            region for region in source_region_order if region not in assignments
        )
        if not complement_members:
            raise ValueError("Requested complement is empty for the selected source-region basis.")
        compiled.append(RegionScope(code=complement, members=complement_members, label=None))
        assignments.update({member: complement for member in complement_members})

    # Keep source_to_scope on the source provider's order. For a partial plan its
    # index contains only assigned regions; full-object aggregation can later require
    # complete coverage explicitly.
    assigned_order = [region for region in source_region_order if region in assignments]
    source_to_scope = pd.Series(
        [assignments[region] for region in assigned_order],
        index=pd.Index(assigned_order, name="source_region"),
        name="scope_code",
        dtype="object",
    )
    return RegionAggregationPlan(
        source_regions=source_region_order,
        scopes=tuple(compiled),
        source_to_scope=source_to_scope,
    )



def _validate_complete_region_plan(
    plan: RegionAggregationPlan,
    *,
    source_regions: tuple[str, ...],
) -> dict[str, str]:
    """Validate that one plan is a complete many-to-one partition of the IOT regions."""

    plan_regions = _validated_source_regions(plan.source_regions)
    if plan_regions != source_regions:
        raise ValueError(
            "RegionAggregationPlan source_regions do not exactly match the IOT "
            f"source-region order: plan={plan_regions!r}, iot={source_regions!r}."
        )
    if not plan.scopes:
        raise ValueError("Complete MRIO region aggregation requires at least one target scope.")

    scope_codes: list[str] = []
    expected: dict[str, str] = {}
    for scope in plan.scopes:
        code = str(scope.code).strip()
        members = _validate_scope(scope)
        if code in scope_codes:
            raise ValueError(f"RegionAggregationPlan contains duplicate scope code {code!r}.")
        scope_codes.append(code)
        for member in members:
            if member not in source_regions:
                raise ValueError(
                    f"RegionAggregationPlan member {member!r} is absent from the IOT "
                    "source-region basis."
                )
            if member in expected:
                raise ValueError(
                    f"RegionAggregationPlan assigns source region {member!r} more than once."
                )
            expected[member] = code

    missing = [region for region in source_regions if region not in expected]
    if missing:
        raise ValueError(
            "Complete MRIO region aggregation requires every source region to be assigned "
            f"exactly once; missing={missing!r}. Compile a complement when needed."
        )
    if not plan.source_to_scope.index.is_unique:
        raise ValueError("RegionAggregationPlan source_to_scope index must be unique.")
    actual_index = tuple(map(str, plan.source_to_scope.index))
    if actual_index != source_regions:
        raise ValueError(
            "RegionAggregationPlan source_to_scope must cover the complete source-region "
            "basis in source order."
        )
    actual = {
        str(region): str(scope_code)
        for region, scope_code in plan.source_to_scope.items()
    }
    if actual != expected:
        raise ValueError(
            "RegionAggregationPlan source_to_scope disagrees with its declared scope memberships."
        )
    return expected


def aggregate_mrio_regions(
    matrices: Mapping[str, csr_matrix | pd.Series | pd.DataFrame],
    *,
    basis_index: pd.MultiIndex,
    plan: RegionAggregationPlan,
    metadata: Mapping[str, object] | None = None,
) -> tuple[
    dict[str, csr_matrix | pd.Series | pd.DataFrame],
    pd.MultiIndex,
    dict[str, object],
]:
    """Aggregate one complete primitive MRIO IOT over regions, preserving sectors.

    The input and output contracts mirror ``load_mrio_iot_matrices``: the primitive
    source accounts are exactly ``Z``, ``x`` and ``Y``. Geography is many-to-one
    only. All source regions must be assigned exactly once so no monetary mass can
    disappear silently. ``A`` and summed ``y`` are deliberately not accepted or
    returned; derive coefficients only from the aggregated ``Z`` and ``x``.
    """

    required = {"Z", "x", "Y"}
    supplied = set(map(str, matrices))
    missing = sorted(required - supplied)
    extra = sorted(supplied - required)
    if missing or extra:
        raise ValueError(
            "aggregate_mrio_regions requires exactly primitive Z, x, and Y accounts; "
            f"missing={missing!r}, extra={extra!r}. Derived A/y inputs are not accepted."
        )

    Z = matrices["Z"]
    x = matrices["x"]
    Y = matrices["Y"]
    if not isinstance(Z, csr_matrix):
        raise TypeError("aggregate_mrio_regions requires Z as scipy.sparse.csr_matrix.")
    if not isinstance(x, pd.Series):
        raise TypeError("aggregate_mrio_regions requires x as pandas Series.")
    if not isinstance(Y, pd.DataFrame):
        raise TypeError("aggregate_mrio_regions requires Y as pandas DataFrame.")
    if not isinstance(basis_index, pd.MultiIndex) or basis_index.nlevels != 2:
        raise TypeError("Region aggregation requires a 2-level (region, sector) basis index.")
    if list(basis_index.names) != ["region", "sector"]:
        raise ValueError("Region aggregation basis names must be exactly ['region', 'sector'].")
    if not basis_index.is_unique:
        raise ValueError("Region aggregation basis must contain unique region-sector rows.")
    if Z.shape != (len(basis_index), len(basis_index)):
        raise ValueError("Z shape does not match the declared region-sector basis.")
    if not x.index.equals(basis_index):
        raise ValueError("x index must exactly match the region-sector basis.")
    if not Y.index.equals(basis_index):
        raise ValueError("Y row index must exactly match the region-sector basis.")
    if not isinstance(Y.columns, pd.MultiIndex) or Y.columns.nlevels != 2:
        raise TypeError("Y columns must use a 2-level (region, category) MultiIndex.")
    if list(Y.columns.names) != ["region", "category"]:
        raise ValueError("Y column names must be exactly ['region', 'category'].")
    if not Y.columns.is_unique:
        raise ValueError("Y final-demand columns must be unique.")
    if Z.nnz and not np.isfinite(Z.data).all():
        raise ValueError("Z contains non-finite monetary values.")
    x_numeric = cast(pd.Series, pd.to_numeric(x, errors="coerce"))
    x_values = x_numeric.to_numpy(dtype="float64")
    if not np.isfinite(x_values).all() or bool((x_values < 0.0).any()):
        raise ValueError("x contains missing, malformed, non-finite, or negative monetary values.")
    Y_values = Y.to_numpy(dtype="float64", copy=False)
    if not np.isfinite(Y_values).all():
        raise ValueError("Y contains non-finite monetary values.")

    region_values = basis_index.get_level_values("region").astype(str)
    sector_values = basis_index.get_level_values("sector").astype(str)
    source_regions = tuple(dict.fromkeys(map(str, region_values)))
    sector_order: tuple[str, ...] | None = None
    for region in source_regions:
        current = tuple(map(str, sector_values[region_values == region]))
        if not current or len(set(current)) != len(current) or any(not sector for sector in current):
            raise ValueError(
                f"Region {region!r} has an empty, blank, or duplicate sector basis."
            )
        if sector_order is None:
            sector_order = current
        elif current != sector_order:
            raise ValueError(
                "Region-only aggregation requires the same unique ordered sector basis "
                f"for every source region; region={region!r}."
            )
    if sector_order is None:
        raise ValueError("Region aggregation source basis is empty.")

    assignments = _validate_complete_region_plan(plan, source_regions=source_regions)
    target_regions = tuple(str(scope.code).strip() for scope in plan.scopes)
    target_basis = pd.MultiIndex.from_product(
        [target_regions, sector_order], names=["region", "sector"]
    )
    target_row_lookup = {key: pos for pos, key in enumerate(target_basis)}
    row_targets = np.asarray(
        [
            target_row_lookup[(assignments[str(region)], str(sector))]
            for region, sector in basis_index
        ],
        dtype=np.int64,
    )

    fd_region_values = Y.columns.get_level_values("region").astype(str)
    fd_category_values = Y.columns.get_level_values("category").astype(str)
    fd_regions = tuple(dict.fromkeys(map(str, fd_region_values)))
    if fd_regions != source_regions:
        raise ValueError(
            "Y final-demand region order must exactly match the IOT source-region order."
        )
    category_order: tuple[str, ...] | None = None
    for region in fd_regions:
        current = tuple(map(str, fd_category_values[fd_region_values == region]))
        if (
            not current
            or len(set(current)) != len(current)
            or any(not category for category in current)
        ):
            raise ValueError(
                f"Y region {region!r} has an empty, blank, or duplicate final-demand "
                "category basis."
            )
        if category_order is None:
            category_order = current
        elif current != category_order:
            raise ValueError(
                "Region-only aggregation requires the same unique ordered final-demand "
                f"category basis for every source region; region={region!r}."
            )
    if category_order is None:
        raise ValueError("Y final-demand column basis is empty.")

    target_Y_columns = pd.MultiIndex.from_product(
        [target_regions, category_order], names=["region", "category"]
    )
    target_col_lookup = {key: pos for pos, key in enumerate(target_Y_columns)}
    col_targets = np.asarray(
        [
            target_col_lookup[(assignments[str(region)], str(category))]
            for region, category in Y.columns
        ],
        dtype=np.int64,
    )

    # Z^R = R Z R^T. Remap sparse row/column coordinates directly so WP2 does not
    # densify the full EXIO transaction matrix merely to aggregate geography.
    source_coo = Z.tocoo(copy=True)
    Z_agg = csr_matrix(
        (
            source_coo.data,
            (row_targets[source_coo.row], row_targets[source_coo.col]),
        ),
        shape=(len(target_basis), len(target_basis)),
        dtype="float64",
    )
    Z_agg.sum_duplicates()
    Z_agg.eliminate_zeros()

    # x^R = R x.
    x_agg_values = np.zeros(len(target_basis), dtype="float64")
    np.add.at(x_agg_values, row_targets, x_values)
    x_agg = pd.Series(x_agg_values, index=target_basis, name="x", dtype="float64")

    # Y^R = R Y H^T. Aggregate rows first, then destination/category columns.
    Y_rows_agg = np.zeros((len(target_basis), Y_values.shape[1]), dtype="float64")
    np.add.at(Y_rows_agg, row_targets, Y_values)
    Y_agg_values = np.zeros(
        (len(target_basis), len(target_Y_columns)), dtype="float64"
    )
    np.add.at(Y_agg_values.T, col_targets, Y_rows_agg.T)
    Y_agg = pd.DataFrame(
        Y_agg_values,
        index=target_basis,
        columns=target_Y_columns,
        dtype="float64",
    )

    out_metadata = dict(metadata or {})
    if "region_aggregation" in out_metadata:
        raise ValueError(
            "aggregate_mrio_regions refuses to overwrite existing region_aggregation "
            "provenance; aggregate from primitive source accounts instead."
        )
    out_metadata["region_aggregation"] = {
        "method": "unweighted_many_to_one_extensive",
        "source_regions": list(source_regions),
        "target_regions": list(target_regions),
        "scope_members": {
            str(scope.code): list(map(str, scope.members)) for scope in plan.scopes
        },
        "complete_coverage": True,
        "sector_count_per_region": len(sector_order),
        "final_demand_categories": list(category_order),
    }
    return {"Z": Z_agg, "x": x_agg, "Y": Y_agg}, target_basis, out_metadata


_ENVIRONMENTAL_ACCOUNT_COLUMNS = (
    "axis_position",
    "axis_label",
    "boundary",
    "denominator_role",
    "extension",
    "stressor",
    "unit",
    "value",
    "monetary_denominator",
    "intensity",
)


def _environmental_text(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Environmental account {field} must not be blank.")
    return text


def _validated_region_sector_extension_inputs(
    production_values: pd.Series,
    output: pd.Series,
) -> tuple[pd.MultiIndex, tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    """Validate one complete production extension row against canonical monetary x."""
    if not isinstance(production_values, pd.Series) or not isinstance(output, pd.Series):
        raise TypeError("Production extension values and monetary output must be pandas Series.")
    index = production_values.index
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise TypeError("Production extension values require MultiIndex(region, sector).")
    if list(index.names) != ["region", "sector"]:
        raise ValueError("Production extension index names must be exactly ['region', 'sector'].")
    if not index.is_unique:
        raise ValueError("Production extension region-sector index must be unique.")
    if not output.index.equals(index):
        raise ValueError("Production extension values and monetary output must use the exact same basis.")

    production_numeric = cast(pd.Series, pd.to_numeric(production_values, errors="coerce"))
    output_numeric = cast(pd.Series, pd.to_numeric(output, errors="coerce"))
    f_values = production_numeric.to_numpy(dtype="float64")
    x_values = output_numeric.to_numpy(dtype="float64")
    if not np.isfinite(f_values).all():
        raise ValueError("Production extension values must be finite; missing evidence is not zero.")
    if not np.isfinite(x_values).all() or bool((x_values < 0.0).any()):
        raise ValueError("Production monetary output must be finite and non-negative.")

    region_values = index.get_level_values("region").astype(str)
    sector_values = index.get_level_values("sector").astype(str)
    source_regions = tuple(dict.fromkeys(map(str, region_values)))
    sector_order: tuple[str, ...] | None = None
    for region in source_regions:
        current = tuple(map(str, sector_values[region_values == region]))
        if not current or len(set(current)) != len(current) or any(not value for value in current):
            raise ValueError(f"Production extension region {region!r} has a malformed sector basis.")
        if sector_order is None:
            sector_order = current
        elif current != sector_order:
            raise ValueError(
                "Production extension requires the same unique ordered sector basis "
                f"for every source region; region={region!r}."
            )
    if not source_regions or sector_order is None:
        raise ValueError("Production extension basis is empty.")
    return index, source_regions, sector_order, f_values, x_values


def _validated_product_projection(
    product_projection: pd.DataFrame,
    *,
    source_sectors: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Validate one complete non-negative source-product partition P."""
    if not isinstance(product_projection, pd.DataFrame):
        raise TypeError("product_projection must be a pandas DataFrame.")
    if product_projection.empty:
        raise ValueError("product_projection must contain at least one target row.")
    target_labels = tuple(str(value).strip() for value in product_projection.index)
    source_labels = tuple(str(value).strip() for value in product_projection.columns)
    if any(not value for value in target_labels) or len(set(target_labels)) != len(target_labels):
        raise ValueError("product_projection target labels must be unique and non-blank.")
    if any(not value for value in source_labels) or len(set(source_labels)) != len(source_labels):
        raise ValueError("product_projection source labels must be unique and non-blank.")
    missing = [value for value in source_sectors if value not in set(source_labels)]
    extra = [value for value in source_labels if value not in set(source_sectors)]
    if missing or extra:
        raise ValueError(
            "product_projection must cover the complete source-sector basis exactly; "
            f"missing={missing!r}, extra={extra!r}."
        )
    source_to_position = {label: pos for pos, label in enumerate(source_labels)}
    order = [source_to_position[label] for label in source_sectors]
    try:
        values = product_projection.to_numpy(dtype="float64")[:, order]
    except (TypeError, ValueError) as exc:
        raise ValueError("product_projection weights must be numeric.") from exc
    if not np.isfinite(values).all() or bool((values < 0.0).any()):
        raise ValueError("product_projection weights must be finite and non-negative.")
    column_sums = values.sum(axis=0)
    if not np.allclose(column_sums, 1.0, rtol=0.0, atol=1e-9):
        bad = {
            source_sectors[pos]: float(total)
            for pos, total in enumerate(column_sums)
            if abs(float(total) - 1.0) > 1e-9
        }
        raise ValueError(
            "product_projection must preserve each source product with weights summing "
            f"to 1.0; mismatches={dict(list(bad.items())[:10])!r}."
        )
    frame = pd.DataFrame(
        values,
        index=pd.Index(target_labels, name="axis_label"),
        columns=pd.Index(source_sectors, name="source_sector"),
        dtype="float64",
    )
    return frame, target_labels


def _environmental_account_frame(
    *,
    axis_labels: Sequence[str],
    boundary: str,
    denominator_role: str,
    extension: str,
    stressor: str,
    unit: str,
    values: np.ndarray,
    monetary_denominator: np.ndarray,
    allow_unavailable_values: bool = False,
) -> pd.DataFrame:
    labels = tuple(str(value).strip() for value in axis_labels)
    if not labels or any(not value for value in labels) or len(set(labels)) != len(labels):
        raise ValueError("Environmental account axis labels must be unique and non-blank.")
    physical = np.asarray(values, dtype="float64")
    denominator = np.asarray(monetary_denominator, dtype="float64")
    if physical.ndim != 1 or denominator.ndim != 1 or len(physical) != len(labels) or len(denominator) != len(labels):
        raise ValueError("Environmental account values and denominators must match the one-dimensional axis.")
    if not np.isfinite(denominator).all():
        raise ValueError("Environmental account monetary denominators must be finite.")
    if allow_unavailable_values:
        if np.isinf(physical).any():
            raise ValueError("Unavailable environmental account values may be NaN but never +/-Inf.")
    elif not np.isfinite(physical).all():
        raise ValueError("Environmental account values must be finite.")
    intensity = np.full(len(labels), np.nan, dtype="float64")
    ratio_mask = (denominator != 0.0) & np.isfinite(physical)
    np.divide(physical, denominator, out=intensity, where=ratio_mask)
    # Pandas accepts the heterogeneous column mapping at runtime; annotate the
    # constructor boundary explicitly so Pyright does not widen numeric columns.
    frame_data: dict[str, Any] = {
        "axis_position": np.arange(1, len(labels) + 1, dtype=np.int64),
        "axis_label": labels,
        "boundary": _environmental_text(boundary, field="boundary"),
        "denominator_role": _environmental_text(denominator_role, field="denominator_role"),
        "extension": _environmental_text(extension, field="extension"),
        "stressor": _environmental_text(stressor, field="stressor"),
        "unit": _environmental_text(unit, field="unit"),
        "value": physical,
        "monetary_denominator": denominator,
        "intensity": intensity,
    }
    return pd.DataFrame(
        frame_data, columns=cast(Any, list(_ENVIRONMENTAL_ACCOUNT_COLUMNS))
    )


def aggregate_scope_production_extension(
    production_values: pd.Series,
    output: pd.Series,
    *,
    scope: RegionScope,
    extension: str,
    stressor: str,
    unit: str,
    product_projection: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate territorial production F and x over a scope, then divide.

    If ``product_projection`` is supplied, both extensive numerator and denominator
    are projected with the same complete source-product partition before intensity
    is formed. Raw or regional ``F/x`` intensities are never projected directly.
    """
    index, source_regions, sectors, f_values, x_values = _validated_region_sector_extension_inputs(
        production_values, output
    )
    members = _validate_scope(scope)
    unknown = [member for member in members if member not in set(source_regions)]
    if unknown:
        raise ValueError(f"Scope production members are absent from the extension basis: {unknown!r}.")
    regions = index.get_level_values("region").astype(str)
    f_scope = np.zeros(len(sectors), dtype="float64")
    x_scope = np.zeros(len(sectors), dtype="float64")
    for member in members:
        mask = np.asarray(regions == member, dtype=bool)
        f_scope += f_values[mask]
        x_scope += x_values[mask]

    if product_projection is None:
        labels = sectors
        physical = f_scope
        denominator = x_scope
    else:
        P, labels = _validated_product_projection(
            product_projection, source_sectors=sectors
        )
        p_values = P.to_numpy(dtype="float64")
        physical = p_values @ f_scope
        denominator = p_values @ x_scope
    return _environmental_account_frame(
        axis_labels=labels,
        boundary="scope_production",
        denominator_role="scope_production_output",
        extension=extension,
        stressor=stressor,
        unit=unit,
        values=physical,
        monetary_denominator=denominator,
    )


def aggregate_external_import_extension_direct(
    production_values: pd.Series,
    output: pd.Series,
    import_use: pd.Series,
    *,
    scope: RegionScope,
    product_projection: pd.DataFrame,
    extension: str,
    stressor: str,
    unit: str,
) -> pd.DataFrame:
    """Build the origin-retaining direct production account for scope imports.

    For every external origin ``o``, apply ``P`` independently to F, x, and actual
    destination-scope import use u. Only then form ``(P F_o)/(P x_o)`` and weight it
    by ``P u_o``. Origins are collapsed only after this origin-specific coefficient
    has been applied. Signed finite use is preserved; it is never clamped or replaced
    by a share proxy.
    """
    index, source_regions, sectors, f_values, x_values = _validated_region_sector_extension_inputs(
        production_values, output
    )
    members = _validate_scope(scope)
    unknown = [member for member in members if member not in set(source_regions)]
    if unknown:
        raise ValueError(f"Import scope members are absent from the production basis: {unknown!r}.")
    member_set = set(members)
    external_origins = tuple(region for region in source_regions if region not in member_set)
    P, target_labels = _validated_product_projection(
        product_projection, source_sectors=sectors
    )
    p_values = P.to_numpy(dtype="float64")

    expected_index = pd.MultiIndex.from_tuples(
        [
            (str(region), str(sector))
            for region, sector in index
            if str(region) in set(external_origins)
        ],
        names=["region", "sector"],
    )
    if not isinstance(import_use, pd.Series):
        raise TypeError("import_use must be a pandas Series indexed by external origin and sector.")
    if not isinstance(import_use.index, pd.MultiIndex) or list(import_use.index.names) != ["region", "sector"]:
        raise ValueError("import_use index must be MultiIndex(region, sector).")
    if not import_use.index.equals(expected_index):
        raise ValueError(
            "import_use must cover every external origin-sector row exactly in source order."
        )
    use_numeric = cast(pd.Series, pd.to_numeric(import_use, errors="coerce"))
    use_values = use_numeric.to_numpy(dtype="float64")
    if not np.isfinite(use_values).all():
        raise ValueError("import_use must be finite; missing monetary use is not zero.")

    physical_total = np.zeros(len(target_labels), dtype="float64")
    denominator_total = np.zeros(len(target_labels), dtype="float64")
    regions = index.get_level_values("region").astype(str)
    offset = 0
    for origin in external_origins:
        mask = np.asarray(regions == origin, dtype=bool)
        origin_count = int(mask.sum())
        if origin_count != len(sectors):
            raise RuntimeError(f"External origin {origin!r} lost the validated sector basis.")
        f_bar = p_values @ f_values[mask]
        x_bar = p_values @ x_values[mask]
        u_origin = use_values[offset : offset + origin_count]
        offset += origin_count
        u_bar = p_values @ u_origin
        undefined = (u_bar != 0.0) & (x_bar <= 0.0)
        if bool(undefined.any()):
            labels = [target_labels[pos] for pos in np.flatnonzero(undefined)[:10]]
            raise ValueError(
                "External-import direct extension has nonzero mapped import use with "
                f"non-positive mapped production output: origin={origin!r}, targets={labels!r}."
            )
        coefficient = np.full(len(target_labels), np.nan, dtype="float64")
        np.divide(f_bar, x_bar, out=coefficient, where=x_bar > 0.0)
        contribution = np.zeros(len(target_labels), dtype="float64")
        active = u_bar != 0.0
        contribution[active] = coefficient[active] * u_bar[active]
        physical_total += contribution
        denominator_total += u_bar
    if offset != len(use_values):
        raise RuntimeError("External-import use accounting did not consume its exact source basis.")
    return _environmental_account_frame(
        axis_labels=target_labels,
        boundary="external_imports_direct",
        denominator_role="scope_external_import_use",
        extension=extension,
        stressor=stressor,
        unit=unit,
        values=physical_total,
        monetary_denominator=denominator_total,
    )


def aggregate_scope_final_demand_extension(
    final_demand_values: pd.Series,
    monetary_final_demand: pd.Series,
    *,
    scope: RegionScope,
    extension: str,
    stressor: str,
    unit: str,
) -> pd.DataFrame:
    """Aggregate direct final-demand F_Y over destination members by category.

    ``F_Y`` stays on the destination final-demand category axis. There is
    deliberately no product-projection argument and no imported-RoW counterpart.
    """
    if not isinstance(final_demand_values, pd.Series):
        raise TypeError("final_demand_values must be a pandas Series.")
    index = final_demand_values.index
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise TypeError("F_Y values require MultiIndex(region, final_demand_category).")
    if list(index.names) != ["region", "final_demand_category"]:
        raise ValueError(
            "F_Y index names must be exactly ['region', 'final_demand_category']."
        )
    if not index.is_unique:
        raise ValueError("F_Y destination/category index must be unique.")
    f_numeric = cast(pd.Series, pd.to_numeric(final_demand_values, errors="coerce"))
    f_values = f_numeric.to_numpy(dtype="float64")
    if not np.isfinite(f_values).all():
        raise ValueError("F_Y values must be finite; missing direct emissions are not zero.")

    members = _validate_scope(scope)
    regions = index.get_level_values("region").astype(str)
    categories = index.get_level_values("final_demand_category").astype(str)
    available_regions = set(map(str, regions))
    unknown = [member for member in members if member not in available_regions]
    if unknown:
        raise ValueError(f"F_Y is missing scope destination members: {unknown!r}.")
    category_order: tuple[str, ...] | None = None
    physical: np.ndarray | None = None
    for member in members:
        mask = np.asarray(regions == member, dtype=bool)
        current = tuple(map(str, categories[mask]))
        if not current or len(set(current)) != len(current) or any(not value for value in current):
            raise ValueError(f"F_Y member {member!r} has a malformed category basis.")
        if category_order is None:
            category_order = current
            physical = np.zeros(len(current), dtype="float64")
        elif current != category_order:
            raise ValueError(
                "F_Y requires the same unique ordered final-demand category basis for "
                f"every scope member; member={member!r}."
            )
        assert physical is not None
        physical += f_values[mask]
    if category_order is None or physical is None:
        raise ValueError("F_Y scope aggregation produced an empty category basis.")

    if not isinstance(monetary_final_demand, pd.Series):
        raise TypeError("monetary_final_demand must be a pandas Series by category.")
    if not monetary_final_demand.index.is_unique:
        raise ValueError("monetary_final_demand category index must be unique.")
    denominator = monetary_final_demand.copy()
    denominator.index = pd.Index(
        [str(value).strip() for value in denominator.index],
        name="final_demand_category",
    )
    expected_index = pd.Index(category_order, name="final_demand_category")
    if not denominator.index.equals(expected_index):
        raise ValueError(
            "monetary_final_demand must exactly match the scope F_Y category basis in order."
        )
    denominator_numeric = cast(pd.Series, pd.to_numeric(denominator, errors="coerce"))
    denominator_values = denominator_numeric.to_numpy(dtype="float64")
    if not np.isfinite(denominator_values).all():
        raise ValueError("monetary_final_demand must be finite.")
    inconsistent_zero = (denominator_values == 0.0) & (physical != 0.0)
    if bool(inconsistent_zero.any()):
        labels = [category_order[pos] for pos in np.flatnonzero(inconsistent_zero)[:10]]
        raise ValueError(
            "Direct final-demand extension has nonzero F_Y with zero monetary final "
            f"demand for categories={labels!r}."
        )
    return _environmental_account_frame(
        axis_labels=category_order,
        boundary="scope_final_demand_direct",
        denominator_role="scope_final_demand",
        extension=extension,
        stressor=stressor,
        unit=unit,
        values=physical,
        monetary_denominator=denominator_values,
    )


def _environmental_donor_allows_zero_denominator_value(
    *, boundary: str, denominator_role: str
) -> bool:
    """Return whether finite donor F may coexist with a zero monetary denominator.

    Production F/x and signed external-import direct accounts are extensive donor
    evidence.  Their numerator remains valid when the coefficient is undefined at
    zero denominator; target retargeting decides whether that missing coefficient is
    harmless zero-mass evidence or a positive-mass support failure.
    """
    return (boundary, denominator_role) in {
        ("scope_production", "scope_production_output"),
        ("external_imports_direct", "scope_external_import_use"),
    }


def _validated_environmental_account_group(
    account: pd.DataFrame,
    *,
    expected_boundary: str,
    expected_denominator_role: str,
    allow_zero_denominator_nonzero_value: bool = False,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, tuple[str, str, str]]:
    if not isinstance(account, pd.DataFrame):
        raise TypeError("Environmental account must be a pandas DataFrame.")
    if list(account.columns) != list(_ENVIRONMENTAL_ACCOUNT_COLUMNS):
        raise ValueError(
            "Environmental account columns must exactly match the canonical long-form schema."
        )
    if account.empty:
        raise ValueError("Environmental account must not be empty.")
    if set(cast(pd.Series, account["boundary"]).astype(str)) != {expected_boundary}:
        raise ValueError(f"Expected only boundary={expected_boundary!r}.")
    if set(cast(pd.Series, account["denominator_role"]).astype(str)) != {expected_denominator_role}:
        raise ValueError(f"Expected denominator_role={expected_denominator_role!r}.")
    labels = tuple(cast(pd.Series, account["axis_label"]).astype(str).tolist())
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Environmental account axis labels must be unique and non-blank.")
    positions_numeric = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, account["axis_position"]), errors="coerce"),
    )
    positions = positions_numeric.to_numpy(dtype="float64")
    expected_positions = np.arange(1, len(labels) + 1, dtype="float64")
    if not np.array_equal(positions, expected_positions):
        raise ValueError("Environmental account axis_position must be one-based and contiguous.")
    values_numeric = cast(
        pd.Series, pd.to_numeric(cast(pd.Series, account["value"]), errors="coerce")
    )
    denominator_numeric = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, account["monetary_denominator"]), errors="coerce"),
    )
    values = values_numeric.to_numpy(dtype="float64")
    denominator = denominator_numeric.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or not np.isfinite(denominator).all():
        raise ValueError("Environmental account values and denominators must be finite.")
    inconsistent_zero = (denominator == 0.0) & (values != 0.0)
    if not allow_zero_denominator_nonzero_value and bool(inconsistent_zero.any()):
        bad = [labels[pos] for pos in np.flatnonzero(inconsistent_zero)[:10]]
        raise ValueError(
            "Environmental account has nonzero physical value at zero monetary denominator: "
            f"labels={bad!r}."
        )
    implied = np.full(len(labels), np.nan, dtype="float64")
    np.divide(values, denominator, out=implied, where=denominator != 0.0)
    raw_reported = cast(pd.Series, account["intensity"])
    reported_numeric = cast(
        pd.Series, pd.to_numeric(raw_reported, errors="coerce")
    )
    invalid_reported = raw_reported.notna() & reported_numeric.isna()
    if bool(invalid_reported.any()):
        raise ValueError("Environmental account intensity must be numeric or missing.")
    reported = reported_numeric.to_numpy(dtype="float64")
    nonzero_denominator = denominator != 0.0
    if not np.allclose(
        reported[nonzero_denominator],
        implied[nonzero_denominator],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Environmental account intensity disagrees with value/denominator.")
    if bool((~np.isnan(reported[~nonzero_denominator])).any()):
        raise ValueError("Environmental account intensity must be missing at zero denominator.")
    identity: list[str] = []
    for column in ("extension", "stressor", "unit"):
        unique = list(dict.fromkeys(cast(pd.Series, account[column]).astype(str)))
        if len(unique) != 1 or not unique[0].strip():
            raise ValueError(f"Environmental account {column} must identify exactly one non-blank group.")
        identity.append(unique[0].strip())
    return labels, values, denominator, (identity[0], identity[1], identity[2])


def _combine_total_supply_extension_arrays(
    *,
    axis_labels: tuple[str, ...],
    production_values: np.ndarray,
    production_denominator: np.ndarray,
    import_values: np.ndarray,
    import_denominator: np.ndarray,
    target_domestic_output: pd.Series,
    target_import_use: pd.Series,
    target_total_use: pd.Series,
    identity: tuple[str, str, str],
    rtol: float,
    atol: float,
) -> pd.DataFrame:
    """Apply direct DOM/IMP coefficients while preserving unavailable evidence."""
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("Total-supply identity tolerances must be non-negative.")
    n = len(axis_labels)
    prod_values = np.asarray(production_values, dtype="float64")
    prod_den = np.asarray(production_denominator, dtype="float64")
    imp_values = np.asarray(import_values, dtype="float64")
    imp_den = np.asarray(import_denominator, dtype="float64")
    for name, values in (
        ("production_values", prod_values),
        ("production_denominator", prod_den),
        ("import_values", imp_values),
        ("import_denominator", imp_den),
    ):
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite vector on the environmental target axis.")

    def control(series: pd.Series, *, name: str, allow_signed: bool) -> np.ndarray:
        if not isinstance(series, pd.Series) or not series.index.is_unique:
            raise ValueError(f"{name} must be a unique pandas Series.")
        normalized = series.copy()
        normalized.index = pd.Index([str(value).strip() for value in normalized.index])
        if tuple(normalized.index) != axis_labels:
            raise ValueError(
                f"{name} must exactly match the environmental target-product axis in order."
            )
        try:
            numeric = cast(pd.Series, pd.to_numeric(normalized, errors="raise")).astype("float64")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric.") from exc
        if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
            raise ValueError(f"{name} must be finite.")
        if not allow_signed:
            numeric = _normalize_nonnegative_accounting_control(numeric, context=name)
        return numeric.to_numpy(dtype="float64")

    domestic = control(target_domestic_output, name="target_domestic_output", allow_signed=False)
    import_use = control(target_import_use, name="target_import_use", allow_signed=True)
    total = control(target_total_use, name="target_total_use", allow_signed=False)
    if not np.allclose(domestic + import_use, total, rtol=float(rtol), atol=float(atol)):
        max_gap = float(np.max(np.abs((domestic + import_use) - total))) if n else 0.0
        raise ValueError(
            "Target total-use identity failed: domestic output + imported-product use must "
            f"equal total use; max_abs_gap={max_gap:.12g}."
        )

    prod_coeff = np.full(n, np.nan, dtype="float64")
    imp_coeff = np.full(n, np.nan, dtype="float64")
    np.divide(prod_values, prod_den, out=prod_coeff, where=prod_den != 0.0)
    np.divide(imp_values, imp_den, out=imp_coeff, where=imp_den != 0.0)
    prod_available = np.isfinite(prod_coeff)
    imp_available = np.isfinite(imp_coeff)
    unavailable = (
        ((domestic > 0.0) & ~prod_available)
        | ((import_use != 0.0) & ~imp_available)
        | ((total == 0.0) & ~prod_available & ~imp_available)
    )

    physical = np.full(n, np.nan, dtype="float64")
    computable = ~unavailable
    if bool(computable.any()):
        prod_contribution = np.where(domestic > 0.0, prod_coeff * domestic, 0.0)
        imp_contribution = np.where(import_use != 0.0, imp_coeff * import_use, 0.0)
        combined = prod_contribution + imp_contribution
        if not np.isfinite(combined[computable]).all():
            raise RuntimeError(
                "Supported total-use environmental combination produced non-finite values."
            )
        physical[computable] = combined[computable]

    extension, stressor, unit = identity
    return _environmental_account_frame(
        axis_labels=axis_labels,
        boundary="total_use_direct",
        denominator_role="target_total_use",
        extension=extension,
        stressor=stressor,
        unit=unit,
        values=physical,
        monetary_denominator=total,
        allow_unavailable_values=True,
    )


def combine_total_supply_extension_direct(
    *,
    production_account: pd.DataFrame,
    import_account: pd.DataFrame,
    target_domestic_output: pd.Series,
    target_import_supply: pd.Series,
    target_total_supply: pd.Series,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> pd.DataFrame:
    """Combine direct DOM and import coefficients on target transaction-use controls.

    The public function and keyword names are retained for API compatibility, but
    ``target_import_supply`` is the signed imported-product use vector and
    ``target_total_supply`` is the corresponding total transaction-use vector.
    """
    prod_labels, prod_values, prod_den, prod_identity = _validated_environmental_account_group(
        production_account,
        expected_boundary="scope_production",
        expected_denominator_role="scope_production_output",
        allow_zero_denominator_nonzero_value=_environmental_donor_allows_zero_denominator_value(
            boundary="scope_production",
            denominator_role="scope_production_output",
        ),
    )
    imp_labels, imp_values, imp_den, imp_identity = _validated_environmental_account_group(
        import_account,
        expected_boundary="external_imports_direct",
        expected_denominator_role="scope_external_import_use",
        allow_zero_denominator_nonzero_value=_environmental_donor_allows_zero_denominator_value(
            boundary="external_imports_direct",
            denominator_role="scope_external_import_use",
        ),
    )
    if imp_labels != prod_labels:
        raise ValueError("Production and import environmental accounts must use the same target-product axis.")
    if imp_identity != prod_identity:
        raise ValueError("Production and import environmental accounts must identify the same extension/stressor/unit.")
    return _combine_total_supply_extension_arrays(
        axis_labels=prod_labels,
        production_values=prod_values,
        production_denominator=prod_den,
        import_values=imp_values,
        import_denominator=imp_den,
        target_domestic_output=target_domestic_output,
        target_import_use=target_import_supply,
        target_total_use=target_total_supply,
        identity=prod_identity,
        rtol=float(rtol),
        atol=float(atol),
    )


def _combine_target_supply_extension_sidecars_direct(
    production_sidecar: pd.DataFrame,
    import_sidecar: pd.DataFrame,
    *,
    target_domestic_output: pd.Series,
    target_import_use: pd.Series,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> pd.DataFrame:
    """Combine model-facing DOM/IMP sidecars while preserving unavailable evidence.

    A positive-mass DOM or IMP component without a defined direct environmental
    coefficient makes the corresponding TOTAL estimate unavailable (NaN); it is
    never treated as physical zero. An unavailable zero-mass component does not
    poison a supported positive-mass counterpart.
    """
    required = {
        "sector", "boundary", "denominator_role", "extension", "stressor", "unit",
        "value", "monetary_output", "intensity",
    }
    for label, frame, boundary, role in (
        ("DOM", production_sidecar, "scope_production", "target_domestic_output"),
        ("IMP", import_sidecar, "external_imports_direct", "target_import_use"),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} environmental sidecar missing columns: {sorted(missing)!r}")
        if set(frame["boundary"].astype(str)) != {boundary}:
            raise ValueError(f"{label} environmental sidecar has the wrong accounting boundary.")
        if set(frame["denominator_role"].astype(str)) != {role}:
            raise ValueError(f"{label} environmental sidecar has the wrong denominator role.")
    labels = tuple(str(value).strip() for value in target_domestic_output.index)
    if tuple(str(value).strip() for value in target_import_use.index) != labels:
        raise ValueError("DOM/IMP environmental target denominators must use one ordered sector axis.")
    if len(set(labels)) != len(labels) or any(not label for label in labels):
        raise ValueError("Environmental target sector axis must be unique and non-blank.")

    dom_target = target_domestic_output.astype("float64").copy()
    imp_target = target_import_use.astype("float64").copy()
    dom_target.index = pd.Index(labels)
    imp_target.index = pd.Index(labels)
    dom_target = _normalize_nonnegative_accounting_control(
        dom_target, context="target_domestic_output"
    )
    if not np.isfinite(imp_target.to_numpy(dtype="float64")).all():
        raise ValueError("target_import_use must be finite.")
    target_total = (dom_target + imp_target).astype("float64")
    target_total.index = pd.Index(labels)
    target_total = _normalize_nonnegative_accounting_control(
        target_total, context="target_total_use"
    )

    keys = ["sector", "extension", "stressor", "unit"]
    if production_sidecar.duplicated(keys).any() or import_sidecar.duplicated(keys).any():
        raise ValueError("Environmental sidecar contains duplicate sector/stressor rows.")
    groups = pd.concat(
        [
            production_sidecar[["extension", "stressor", "unit"]],
            import_sidecar[["extension", "stressor", "unit"]],
        ],
        ignore_index=True,
    ).drop_duplicates(ignore_index=True)

    def arrays_for(
        frame: pd.DataFrame,
        *,
        group_key: tuple[str, str, str],
        target: pd.Series,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        ext, stressor, unit = group_key
        subset = frame.loc[
            frame["extension"].astype(str).eq(ext)
            & frame["stressor"].astype(str).eq(stressor)
            & frame["unit"].astype(str).eq(unit)
        ].copy()
        subset["sector"] = subset["sector"].astype(str)
        extra = sorted(set(subset["sector"]) - set(labels))
        if extra:
            raise ValueError(
                f"{side} environmental sidecar has sectors outside the target axis: {extra!r}"
            )
        by_sector = subset.set_index("sector")
        values = np.full(len(labels), np.nan, dtype="float64")
        available = np.zeros(len(labels), dtype=bool)
        target_values = target.to_numpy(dtype="float64")

        for pos, sector in enumerate(labels):
            if sector not in by_sector.index:
                requires_row = target_values[pos] != 0.0 if side == "IMP" else target_values[pos] > 0.0
                if requires_row:
                    raise ValueError(
                        f"{side} environmental sidecar is structurally missing a nonzero-mass row: "
                        f"{(sector, ext, stressor, unit)!r}"
                    )
                continue

            row = cast(pd.Series, by_sector.loc[sector])
            try:
                observed_den = float(row["monetary_output"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{side} environmental denominator must be numeric for "
                    f"{(sector, ext, stressor, unit)!r}"
                ) from exc
            if not np.isfinite(observed_den) or not np.isclose(
                observed_den, target_values[pos], rtol=float(rtol), atol=float(atol)
            ):
                raise ValueError(
                    f"{side} environmental denominator mismatch for "
                    f"{(sector, ext, stressor, unit)!r}"
                )

            raw_value_object = row["value"]
            raw_value = cast(
                pd.Series,
                pd.to_numeric(pd.Series([raw_value_object]), errors="coerce"),
            ).iloc[0]
            if cast(bool, pd.notna(raw_value_object)) and cast(bool, pd.isna(raw_value)):
                raise ValueError(
                    f"{side} environmental value must be numeric or NA for "
                    f"{(sector, ext, stressor, unit)!r}"
                )

            raw_intensity_object = row["intensity"]
            raw_intensity = cast(
                pd.Series,
                pd.to_numeric(pd.Series([raw_intensity_object]), errors="coerce"),
            ).iloc[0]
            if cast(bool, pd.notna(raw_intensity_object)) and cast(bool, pd.isna(raw_intensity)):
                raise ValueError(
                    f"{side} environmental intensity must be numeric or NA for "
                    f"{(sector, ext, stressor, unit)!r}"
                )

            if pd.isna(raw_value):
                if pd.notna(raw_intensity):
                    raise ValueError(
                        f"{side} unavailable environmental value requires unavailable intensity."
                    )
                continue

            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(
                    f"{side} environmental value must be finite when evidence is available."
                )
            if observed_den == 0.0:
                if value != 0.0:
                    raise ValueError(
                        f"{side} zero-denominator supported estimate must have value=0."
                    )
                if pd.notna(raw_intensity):
                    raise ValueError(
                        f"{side} environmental intensity must be missing at zero denominator."
                    )
            else:
                if (
                    pd.isna(raw_intensity)
                    or not np.isfinite(float(raw_intensity))
                    or not np.isclose(
                        float(raw_intensity),
                        value / observed_den,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ):
                    raise ValueError(
                        f"{side} environmental intensity disagrees with value/denominator."
                    )
            values[pos] = value
            available[pos] = True
        return values, available

    rows: list[pd.DataFrame] = []
    dom_mass = dom_target.to_numpy(dtype="float64")
    imp_mass = imp_target.to_numpy(dtype="float64")
    total_mass = target_total.to_numpy(dtype="float64")

    for group in groups.itertuples(index=False):
        group = cast(Any, group)
        identity = (str(group.extension), str(group.stressor), str(group.unit))
        prod_values, prod_available = arrays_for(
            production_sidecar, group_key=identity, target=dom_target, side="DOM"
        )
        imp_values, imp_available = arrays_for(
            import_sidecar, group_key=identity, target=imp_target, side="IMP"
        )

        required_unavailable = (
            ((dom_mass > 0.0) & ~prod_available)
            | ((imp_mass != 0.0) & ~imp_available)
        )
        unsupported_zero_mass = (
            (total_mass == 0.0) & ~prod_available & ~imp_available
        )
        unavailable = required_unavailable | unsupported_zero_mass

        physical = np.full(len(labels), np.nan, dtype="float64")
        computable = ~unavailable
        if bool(computable.any()):
            prod_contribution = np.where(dom_mass > 0.0, prod_values, 0.0)
            imp_contribution = np.where(imp_mass != 0.0, imp_values, 0.0)
            computed = prod_contribution + imp_contribution
            if not np.isfinite(computed[computable]).all():
                raise RuntimeError(
                    "Supported TOTAL environmental components produced non-finite physical values."
                )
            physical[computable] = computed[computable]

        intensity = np.full(len(labels), np.nan, dtype="float64")
        ratio_mask = computable & (total_mass > 0.0)
        np.divide(physical, total_mass, out=intensity, where=ratio_mask)

        extension, stressor, unit = identity
        sidecar = pd.DataFrame(
            {
                "sector": labels,
                "boundary": "total_use_direct",
                "denominator_role": "target_total_use",
                "extension": extension,
                "stressor": stressor,
                "unit": unit,
                "value": physical,
                "monetary_output": total_mass,
                "intensity": intensity,
            }
        )
        rows.append(sidecar)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _retarget_direct_environmental_account(
    account: pd.DataFrame,
    *,
    target_denominator: pd.Series,
    boundary: str,
    denominator_role: str,
    allow_signed_denominator: bool = False,
) -> pd.DataFrame:
    source_boundary = str(account["boundary"].iloc[0])
    source_denominator_role = str(account["denominator_role"].iloc[0])
    labels, _values, _denominator, identity = _validated_environmental_account_group(
        account,
        expected_boundary=source_boundary,
        expected_denominator_role=source_denominator_role,
        allow_zero_denominator_nonzero_value=_environmental_donor_allows_zero_denominator_value(
            boundary=source_boundary, denominator_role=source_denominator_role
        ),
    )
    target = cast(pd.Series, target_denominator.copy().astype("float64"))
    target.index = pd.Index([str(v) for v in target.index])
    if tuple(target.index) != labels:
        raise ValueError("Target environmental denominator must exactly match the projected sector axis.")
    if not np.isfinite(target.to_numpy(dtype="float64")).all():
        raise ValueError(f"Target environmental denominator ({boundary}) must be finite.")
    if allow_signed_denominator and denominator_role != "target_import_use":
        raise ValueError("Signed environmental retargeting is allowed only for target_import_use.")
    allow_signed = allow_signed_denominator or denominator_role == "target_import_use"
    if not allow_signed:
        target = _normalize_nonnegative_accounting_control(
            target,
            context=f"Target environmental denominator ({boundary})",
        )
    coeff = cast(
        pd.Series,
        pd.to_numeric(account["intensity"], errors="coerce"),
    ).to_numpy(dtype="float64")
    target_values = target.to_numpy(dtype="float64")
    coefficient_available = np.isfinite(coeff)

    # Scope-production donor F is extensive evidence even when mapped donor output
    # is zero.  The coefficient is then unavailable.  A positive domestic target
    # output genuinely requires that coefficient and remains fail-closed; a zero-mass
    # target keeps value/intensity unavailable rather than fabricating physical zero.
    if denominator_role == "target_domestic_output":
        unsupported_positive = (target_values > 0.0) & ~coefficient_available
        if bool(unsupported_positive.any()):
            bad = [labels[pos] for pos in np.flatnonzero(unsupported_positive)[:10]]
            raise ValueError(
                "Positive target domestic output requires a defined mapped production "
                f"donor coefficient: sectors={bad!r}."
            )

    # A supported coefficient applied to zero target mass is an actual physical zero.
    # An unavailable coefficient remains unavailable even when target mass is zero.
    # Signed import targets retain the same state distinction without turning a
    # missing coefficient into a domestic or zero proxy.
    physical = np.full(len(labels), np.nan, dtype="float64")
    physical[coefficient_available] = (
        coeff[coefficient_available] * target_values[coefficient_available]
    )
    extension, stressor, unit = identity
    return _environmental_account_frame(
        axis_labels=labels,
        boundary=boundary,
        denominator_role=denominator_role,
        extension=extension,
        stressor=stressor,
        unit=unit,
        values=physical,
        monetary_denominator=target_values,
        allow_unavailable_values=True,
    )


def _environmental_account_to_split_sidecar(account: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical WP3 account rows to the compatibility target-side schema."""
    required = set(_ENVIRONMENTAL_ACCOUNT_COLUMNS)
    if set(account.columns) != required:
        raise ValueError("Environmental account does not match the canonical WP3 schema.")
    out = account.rename(
        columns={"axis_label": "sector", "monetary_denominator": "monetary_output"}
    ).copy()
    return out.loc[
        :,
        [
            "sector",
            "boundary",
            "denominator_role",
            "extension",
            "stressor",
            "unit",
            "value",
            "monetary_output",
            "intensity",
        ],
    ]


def _environmental_account_to_raw_sidecar(
    account: pd.DataFrame, *, kind: str
) -> pd.DataFrame:
    out = account.rename(columns={"axis_label": "source_label"}).copy()
    out.insert(2, "kind", str(kind))
    return out.loc[
        :,
        [
            "source_label",
            "extension",
            "kind",
            "stressor",
            "unit",
            "value",
            "monetary_denominator",
            "intensity",
        ],
    ]


def build_scope_target_environmental_extensions(
    *,
    source_zip: str | Path,
    scope: RegionScope,
    flow: ScopeFlow,
    source_output: pd.Series,
    source_basis: pd.MultiIndex,
    product_projection: pd.DataFrame,
    target_denominator: pd.Series,
    extension_requests: Mapping[str, Sequence[str]],
    system: str = "pxp",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Build target DOM/IMP environmental sidecars and scope F_Y evidence.

    DOM uses scope-production coefficients; IMP uses origin-retaining direct import
    coefficients. The coefficient is formed on EXIO evidence first and only then
    applied to the authoritative Eurostat flow denominator. Direct F_Y is emitted
    only for the internal/DOM call because it has no import-origin decomposition;
    its final-demand categories are appended to the same target-side long table.
    """
    if flow not in {"internal", "external_imports"}:
        raise ValueError("Target environmental split sidecars support only internal/DOM or external-import/IMP flows.")
    if not extension_requests:
        return pd.DataFrame(), None
    from ecolecon_mrio.exiobase3 import (
        read_exiobase3_scope_external_import_use_by_origin,
        read_exiobase3_scope_final_demand_totals,
    )
    from ecolecon_mrio.exiobase3_extensions import load_exiobase3_extension_rows_with_units

    if not isinstance(source_output.index, pd.MultiIndex) or not source_output.index.equals(source_basis):
        raise ValueError("Environmental source output must exactly use the canonical EXIO region-sector basis.")
    # Product projections in the fuels pipeline use EXIO codes while the primitive
    # x/F basis may use official names. Translate the validated basis once, without
    # changing geography or ordering.
    from ecolecon_mrio.exiobase3 import _code_index, _exio_label_to_code_lookup
    regions = source_basis.get_level_values("region").astype(str)
    raw_sectors = pd.Index(source_basis.get_level_values("sector").astype(str))
    projection_columns = set(map(str, product_projection.columns))
    if set(raw_sectors).issubset(projection_columns):
        coded_sectors = raw_sectors
    else:
        coded_sectors = _code_index(
            raw_sectors, label_to_code=_exio_label_to_code_lookup(str(system))
        )
    coded_basis = pd.MultiIndex.from_arrays([regions, coded_sectors], names=["region", "sector"])
    coded_output = pd.Series(
        source_output.to_numpy(dtype="float64"), index=coded_basis, dtype="float64", name="x"
    )
    if list(map(str, product_projection.columns)) != list(dict.fromkeys(map(str, coded_sectors))):
        raise ValueError("Environmental product projection columns must exactly match the EXIO source-sector order.")

    import_use: pd.Series | None = None
    if flow == "external_imports":
        raw_import_use = read_exiobase3_scope_external_import_use_by_origin(
            source_zip=source_zip, scope=scope
        )
        raw_use_regions = raw_import_use.index.get_level_values("region").astype(str)
        raw_use_sectors = pd.Index(raw_import_use.index.get_level_values("sector").astype(str))
        use_codes = _code_index(raw_use_sectors, label_to_code=_exio_label_to_code_lookup(str(system)))
        import_use = pd.Series(
            raw_import_use.to_numpy(dtype="float64"),
            index=pd.MultiIndex.from_arrays([raw_use_regions, use_codes], names=["region", "sector"]),
            dtype="float64",
            name="scope_external_import_use",
        )

    target_rows: list[pd.DataFrame] = []
    raw_rows: list[pd.DataFrame] = []
    fy_denominator = (
        read_exiobase3_scope_final_demand_totals(source_zip=source_zip, scope=scope)
        if flow == "internal"
        else None
    )
    for extension, keywords in extension_requests.items():
        production_rows = load_exiobase3_extension_rows_with_units(
            source_zip,
            extension=str(extension),
            kind="F",
            keywords=tuple(keywords),
            region=None,
            system=str(system),
            basis_index=source_basis,
        )
        for row in production_rows:
            coded_values = pd.Series(
                row.values.to_numpy(dtype="float64"),
                index=coded_basis,
                dtype="float64",
                name=row.stressor,
            )
            scope_raw = aggregate_scope_production_extension(
                coded_values,
                coded_output,
                scope=scope,
                extension=row.extension,
                stressor=row.stressor,
                unit=row.unit,
            )
            if flow == "internal":
                donor = aggregate_scope_production_extension(
                    coded_values,
                    coded_output,
                    scope=scope,
                    product_projection=product_projection,
                    extension=row.extension,
                    stressor=row.stressor,
                    unit=row.unit,
                )
                target = _retarget_direct_environmental_account(
                    donor,
                    target_denominator=target_denominator,
                    boundary="scope_production",
                    denominator_role="target_domestic_output",
                )
                raw_rows.append(_environmental_account_to_raw_sidecar(scope_raw, kind="F"))
            else:
                assert import_use is not None
                donor = aggregate_external_import_extension_direct(
                    coded_values,
                    coded_output,
                    import_use,
                    scope=scope,
                    product_projection=product_projection,
                    extension=row.extension,
                    stressor=row.stressor,
                    unit=row.unit,
                )
                target = _retarget_direct_environmental_account(
                    donor,
                    target_denominator=target_denominator,
                    boundary="external_imports_direct",
                    denominator_role="target_import_use",
                    allow_signed_denominator=True,
                )
            target_rows.append(_environmental_account_to_split_sidecar(target))

        if flow == "internal":
            assert fy_denominator is not None
            try:
                final_rows = load_exiobase3_extension_rows_with_units(
                    source_zip,
                    extension=str(extension),
                    kind="F_Y",
                    keywords=tuple(keywords),
                    region=None,
                    system=str(system),
                )
            except (FileNotFoundError, KeyError) as exc:
                raise ValueError(
                    f"No stressors matched extension={extension!r}, kind='F_Y'."
                ) from exc
            for row in final_rows:
                fy_account = aggregate_scope_final_demand_extension(
                    row.values,
                    fy_denominator,
                    scope=scope,
                    extension=row.extension,
                    stressor=row.stressor,
                    unit=row.unit,
                )
                # F_Y already has the correct scope/category account. The compatibility
                # target sidecar overloads ``sector`` with this category axis so model
                # consumers can read one environmental_extensions.csv; it is never
                # product-projected or sector-reaggregated.
                target_rows.append(_environmental_account_to_split_sidecar(fy_account))
                raw_rows.append(_environmental_account_to_raw_sidecar(fy_account, kind="F_Y"))

    if not target_rows:
        raise ValueError("Requested environmental extensions produced no target sector rows.")
    target_table = pd.concat(target_rows, ignore_index=True)
    raw_table = pd.concat(raw_rows, ignore_index=True) if raw_rows else None
    return target_table, raw_table

def load_mrio_iot_scope_slice(
    dataset: str,
    year: int,
    *,
    scope: RegionScope,
    flow: ScopeFlow,
    data_root: str | Path | None = None,
    exio_system: str = "pxp",
    doi: str | None = None,
    force_download: bool = False,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    verbose: bool = False,
) -> MrioScopeSlice:
    """Stream one destination-scope IOT use block through the selected provider.

    This is the low-memory counterpart to ``load_mrio_iot_matrices``. It returns
    extensive ``Z``/``Y`` use accounts plus scope-member production ``x_scope``;
    it never derives ``A`` or environmental intensities.
    """

    _validate_scope(scope)
    dataset_norm = str(dataset).strip().lower()
    if flow not in {"internal", "external_imports", "total"}:
        raise ValueError("flow must be one of: internal, external_imports, total")
    if dataset_norm not in {"exiobase", "exiobase3"}:
        raise NotImplementedError(
            "load_mrio_iot_scope_slice currently supports only dataset='exiobase'."
        )

    from ecolecon_mrio.data import DEFAULT_EXIOBASE3_DOI
    from ecolecon_mrio.exiobase3 import _load_exiobase3_iot_scope_slice

    return _load_exiobase3_iot_scope_slice(
        year=int(year),
        scope=scope,
        system=str(exio_system),
        flow=flow,
        data_root=data_root,
        doi=str(doi or DEFAULT_EXIOBASE3_DOI),
        force_download=bool(force_download),
        allow_download_missing=bool(allow_download_missing),
        resume_downloads=bool(resume_downloads),
        show_download_progress=bool(show_download_progress),
        download_chunk_mb=int(download_chunk_mb),
        download_timeout_seconds=download_timeout_seconds,
        download_retries=int(download_retries),
        verbose=bool(verbose),
    )
