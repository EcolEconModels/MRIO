"""Compile unweighted MRIO region scopes and dispatch low-memory scope reads.

This module owns geography membership independently of sector mappings. It validates
many-to-one scope definitions against the source-region basis and exposes destination-
scope monetary use blocks whose production-output denominator remains a separate
producer-side account. Provider-specific ZIP parsing stays in the EXIOBASE module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, cast

import pandas as pd


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
