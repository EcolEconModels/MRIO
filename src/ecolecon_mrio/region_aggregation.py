"""Compile unweighted MRIO region scopes and dispatch low-memory scope reads.

This module owns geography membership independently of sector mappings. It validates
many-to-one scope definitions against the source-region basis and exposes destination-
scope monetary use blocks whose production-output denominator remains a separate
producer-side account. Provider-specific ZIP parsing stays in the EXIOBASE module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


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
    if not np.isfinite(x_values).all():
        raise ValueError("x contains missing, malformed, or non-finite monetary values.")
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
