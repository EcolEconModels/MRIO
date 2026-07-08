from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping as ABCMapping
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np
import pandas as pd

from ecolecon_mrio.balancing import BalanceStats
from ecolecon_mrio.disaggregation import SplitProfiles, calc_A, calc_x
from ecolecon_mrio.exiobase3 import (
    ExioRegionSlice,
    load_exiobase3_region_slice,
)
from ecolecon_mrio.exiobase3_extensions import (
    extract_quantity_and_price as extract_exio_quantity_and_price,
)


@dataclass(frozen=True)
class SplitResult:
    """Result of splitting one or more aggregate sectors in an IO table."""

    Z: pd.DataFrame
    Y: pd.DataFrame
    x: pd.Series
    A: pd.DataFrame
    split_profiles: dict[str, SplitProfiles]
    balancing_method: str
    balancing_stats: BalanceStats
    exio_selected_sectors: dict[str, list[str]]
    exio_quantity_and_price: pd.DataFrame | None
    V: pd.DataFrame | None = None
    M: pd.DataFrame | None = None
    supplementary_diagnostics: pd.DataFrame | None = None
    fallback_diagnostics: pd.DataFrame | None = None
    disaggregated_sector_value_per_physical_input: pd.DataFrame | None = None
    io_system: object | None = None  # pymrio.IOSystem when available

    @property
    def fallback_count(self) -> int:
        if self.fallback_diagnostics is None:
            return 0
        return int(len(self.fallback_diagnostics))


def _aggregate_rows_to_original(
    df: pd.DataFrame,
    *,
    split_profiles: Mapping[str, SplitProfiles],
) -> pd.DataFrame:
    mapping = {
        str(subsector): str(aggregate_sector)
        for aggregate_sector, prof in split_profiles.items()
        for subsector in prof.subsectors
    }
    out = df.copy()
    out.index = pd.Index([mapping.get(str(idx), str(idx)) for idx in out.index])
    grouped = out.groupby(level=0).sum()
    return cast(pd.DataFrame, grouped)

def _aggregate_square_to_original(
    df: pd.DataFrame,
    *,
    split_profiles: Mapping[str, SplitProfiles],
) -> pd.DataFrame:
    mapping = {
        str(subsector): str(aggregate_sector)
        for aggregate_sector, prof in split_profiles.items()
        for subsector in prof.subsectors
    }
    out = df.copy()
    out.index = pd.Index([mapping.get(str(idx), str(idx)) for idx in out.index])
    out.columns = pd.Index([mapping.get(str(col), str(col)) for col in out.columns])
    grouped_rows = cast(pd.DataFrame, out.groupby(level=0).sum())
    grouped_cols = cast(pd.DataFrame, grouped_rows.T.groupby(level=0).sum().T)
    return grouped_cols

def _largest_abs_gap(diff: pd.DataFrame | pd.Series) -> tuple[str, float]:
    if isinstance(diff, pd.Series):
        if diff.empty:
            return "", 0.0
        label = str(diff.abs().idxmax())
        return label, float(diff.abs().max())
    if diff.empty:
        return "", 0.0
    stack = diff.abs().stack(dropna=False)
    if stack.empty:
        return "", 0.0
    where = stack.idxmax()
    if isinstance(where, tuple) and len(where) >= 2:
        label = f"{where[0]} -> {where[1]}"
    else:
        label = str(where)
    return label, float(stack.max())

def _expanded_sector_order(
    *,
    sectors_old: Sequence[str],
    split_profiles: Mapping[str, SplitProfiles],
) -> list[str]:
    sectors_new: list[str] = []
    for sector in map(str, sectors_old):
        if sector in split_profiles:
            sectors_new.extend(list(map(str, split_profiles[sector].subsectors)))
        else:
            sectors_new.append(sector)
    return sectors_new


def _normalize_vector(values: pd.Series) -> pd.Series:
    numeric_obj = pd.to_numeric(values, errors="coerce")
    if not isinstance(numeric_obj, pd.Series):
        raise TypeError("Expected normalized donor vector input to remain a pandas Series.")
    numeric = numeric_obj.fillna(0.0).astype("float64")
    total = float(numeric.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Cannot normalize non-positive donor vector.")
    return numeric / total

def _edge_child_source_target(subsector: object) -> tuple[str, str] | None:
    """Return (EXIO source, Eurostat target) for edge-child codes like p24.d@CPA_C20."""
    text = str(subsector).strip()
    if "@" not in text:
        return None
    source, target = text.split("@", 1)
    source = source.strip()
    target = target.strip()
    if not source or not target:
        return None
    return source, target


def _edge_member_weight(
    *,
    subsector: object,
    member: object,
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None,
) -> float:
    """Weight an EXIO member when a split child represents a source-target edge."""
    edge = _edge_child_source_target(subsector)
    if edge is None:
        return 1.0
    edge_source, edge_target = edge
    member_s = str(member).strip()
    if member_s != edge_source:
        return 0.0
    if eurostat_sector_mapping is None:
        return 1.0
    return float(eurostat_sector_mapping.get(edge_target, {}).get(edge_source, 0.0))


def _weighted_exio_member_sum(
    values: pd.Series,
    *,
    subsector: object,
    members: Sequence[str],
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None,
) -> float:
    total = 0.0
    for member in map(str, members):
        if member not in values.index:
            continue
        total += float(values.loc[member]) * _edge_member_weight(
            subsector=subsector,
            member=member,
            eurostat_sector_mapping=eurostat_sector_mapping,
        )
    return float(total)


def _raw_exio_members_from_profile(prof: SplitProfiles | None) -> list[str]:
    if prof is None or not prof.subsector_to_exio_sectors:
        return []
    out: list[str] = []
    for members in prof.subsector_to_exio_sectors.values():
        out.extend(str(member) for member in members if str(member).strip())
    return list(dict.fromkeys(out))


def _raw_exio_members_from_profiles(profiles: Mapping[str, SplitProfiles]) -> list[str]:
    out: list[str] = []
    for prof in profiles.values():
        out.extend(_raw_exio_members_from_profile(prof))
    return list(dict.fromkeys(out))


def raw_exio_members_from_split_profiles(profiles: Mapping[str, SplitProfiles]) -> list[str]:
    """Return raw EXIO member codes used by split profiles, ignoring edge-child labels."""
    return _raw_exio_members_from_profiles(profiles)


def aggregation_mapping_from_split_profiles(
    profiles: Mapping[str, SplitProfiles],
) -> dict[str, str]:
    """Return emitted split child -> aggregate parent mapping for reaggregation.

    This is the explicit ``H`` mapping used to reaggregate split outputs. It is
    edge-label aware: materialized children such as ``p24.d@CPA_C20`` map through
    the target-side aggregate that produced them, not through the raw EXIO source
    alone.
    """
    mapping: dict[str, str] = {}
    for aggregate_sector, prof in profiles.items():
        aggregate = str(aggregate_sector)
        for child in prof.subsectors:
            child_s = str(child).strip()
            if not child_s:
                continue
            if child_s in mapping and mapping[child_s] != aggregate:
                raise ValueError(
                    f"Split child {child_s!r} maps to multiple aggregate parents: "
                    f"{mapping[child_s]!r} and {aggregate!r}."
                )
            mapping[child_s] = aggregate
    return mapping


def emitted_sector_to_exio_members_from_split_profiles(
    profiles: Mapping[str, SplitProfiles],
) -> dict[str, dict[str, list[str]]]:
    """Return aggregate -> emitted split child -> raw EXIO members.

    This is the post-Phase-4 mapping for downstream outputs. It preserves
    materialized edge-child labels such as ``p24.d@CPA_C20`` while keeping
    the raw EXIO member codes needed for extension extraction and weighted
    ratio-of-sums calculations.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for aggregate_sector, prof in profiles.items():
        child_map: dict[str, list[str]] = {}
        if prof.subsector_to_exio_sectors:
            for child, members in prof.subsector_to_exio_sectors.items():
                child_s = str(child).strip()
                raw_members = [str(member).strip() for member in members if str(member).strip()]
                if child_s and raw_members:
                    child_map[child_s] = list(dict.fromkeys(raw_members))
        else:
            for child in prof.subsectors:
                child_s = str(child).strip()
                if child_s:
                    child_map[child_s] = [child_s]
        if child_map:
            out[str(aggregate_sector)] = child_map
    return out


def build_projection_matrix_for_split_profiles(
    split_profiles: Mapping[str, SplitProfiles],
    *,
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None = None,
    raw_exio_sectors: Sequence[str] | None = None,
    emitted_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the emitted-label by raw-EXIO projection matrix ``P``.

    Rows are emitted output labels and columns are raw EXIO sector codes. Split
    children are projected through their raw EXIO members, edge children through
    source-target concordance weights, and unsplit Eurostat sectors through the
    resolved EXIO-to-Eurostat mapping. When ``emitted_labels`` is omitted, rows
    are limited to the split children in ``split_profiles`` for diagnostics.
    """
    emitted_mapping = emitted_sector_to_exio_members_from_split_profiles(split_profiles)
    inferred_raw: list[str] = []
    inferred_children: list[str] = []
    child_members: dict[str, list[str]] = {}
    for child_map in emitted_mapping.values():
        for child, members in child_map.items():
            child_s = str(child).strip()
            if child_s:
                inferred_children.append(child_s)
                child_members[child_s] = list(dict.fromkeys(str(member).strip() for member in members if str(member).strip()))
            for member in members:
                member_s = str(member).strip()
                if member_s and member_s not in inferred_raw:
                    inferred_raw.append(member_s)

    if emitted_labels is None:
        row_labels = list(dict.fromkeys(inferred_children))
    else:
        row_labels = list(dict.fromkeys(map(str, emitted_labels)))

    if raw_exio_sectors is None:
        raw_order = inferred_raw
    else:
        raw_order = list(dict.fromkeys(str(sector).strip() for sector in raw_exio_sectors if str(sector).strip()))

    P = pd.DataFrame(0.0, index=row_labels, columns=raw_order, dtype="float64")
    if P.empty:
        return P

    available_raw = set(raw_order)
    for label in row_labels:
        if label in child_members:
            for member in child_members[label]:
                member_s = str(member).strip()
                if member_s not in available_raw:
                    continue
                weight = _edge_member_weight(
                    subsector=label,
                    member=member_s,
                    eurostat_sector_mapping=eurostat_sector_mapping,
                )
                if np.isfinite(weight) and weight != 0.0:
                    P.loc[label, member_s] = float(P.loc[label, member_s]) + float(weight)
            continue

        if eurostat_sector_mapping is not None and label in eurostat_sector_mapping:
            for member, weight in eurostat_sector_mapping[label].items():
                member_s = str(member).strip()
                weight_f = float(weight)
                if member_s in available_raw and np.isfinite(weight_f) and weight_f != 0.0:
                    P.loc[label, member_s] = float(P.loc[label, member_s]) + weight_f
            continue

        if label in available_raw:
            P.loc[label, label] = 1.0

    return P


def project_exio_donor_blocks_for_split_profiles(
    *,
    exio_Z: pd.DataFrame,
    exio_Y: pd.DataFrame,
    split_profiles: Mapping[str, SplitProfiles],
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None = None,
    raw_exio_sectors: Sequence[str] | None = None,
    emitted_labels: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Project raw EXIO donor blocks using ``P Z P.T`` and ``P Y``."""
    P = build_projection_matrix_for_split_profiles(
        split_profiles=split_profiles,
        eurostat_sector_mapping=eurostat_sector_mapping,
        raw_exio_sectors=raw_exio_sectors,
        emitted_labels=emitted_labels,
    )
    raw_order = list(map(str, P.columns))

    Z_raw = exio_Z.copy().astype("float64")
    Z_raw.index = Z_raw.index.astype(str)
    Z_raw.columns = Z_raw.columns.astype(str)
    Y_raw = exio_Y.copy().astype("float64")
    Y_raw.index = Y_raw.index.astype(str)
    Y_raw.columns = Y_raw.columns.astype(str)

    Z_block = Z_raw.reindex(index=raw_order, columns=raw_order).fillna(0.0)
    Y_block = Y_raw.reindex(index=raw_order).fillna(0.0)

    p_arr = P.to_numpy(dtype="float64")
    Z_projected = pd.DataFrame(
        p_arr @ Z_block.to_numpy(dtype="float64") @ p_arr.T,
        index=P.index.copy(),
        columns=P.index.copy(),
        dtype="float64",
    )
    Y_projected = pd.DataFrame(
        p_arr @ Y_block.to_numpy(dtype="float64"),
        index=P.index.copy(),
        columns=Y_block.columns.copy(),
        dtype="float64",
    )
    return P, Z_projected, Y_projected

def _source_targets_from_eurostat_mapping(
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, float]]:
    source_targets: dict[str, dict[str, float]] = {}
    if eurostat_sector_mapping is None:
        return source_targets
    for target_code, source_weights in eurostat_sector_mapping.items():
        target = str(target_code)
        for source_code, weight in source_weights.items():
            source = str(source_code)
            weight_float = float(weight)
            if weight_float == 0.0:
                continue
            source_targets.setdefault(source, {})[target] = weight_float
    return source_targets


def _edge_aware_sector_to_exio_sectors(
    sector_to_exio_sectors: Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None,
    *,
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None,
) -> Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None:
    """Derive edge-child output labels for selected multi-target EXIO sources.

    The public split manifest stays parent -> raw EXIO source. When a raw source
    maps to multiple Eurostat targets and the selected parent is one of them, the
    emitted split child is source@parent while retaining the raw source as its
    EXIO donor member.
    """
    if sector_to_exio_sectors is None or eurostat_sector_mapping is None:
        return sector_to_exio_sectors

    source_targets = _source_targets_from_eurostat_mapping(eurostat_sector_mapping)
    if not source_targets:
        return sector_to_exio_sectors

    def _child_label(parent: str, source: str) -> str:
        targets = source_targets.get(source, {})
        if len(targets) > 1 and parent in targets:
            return f"{source}@{parent}"
        return source

    expanded: dict[str, list[str] | dict[str, list[str]]] = {}
    for parent_code, value in sector_to_exio_sectors.items():
        parent = str(parent_code)
        if isinstance(value, ABCMapping):
            grouped: dict[str, list[str]] = {}
            for group_code, members in value.items():
                for source_code in members:
                    source = str(source_code)
                    child = _child_label(parent, source)
                    grouped.setdefault(child, [])
                    if source not in grouped[child]:
                        grouped[child].append(source)
            expanded[parent] = grouped
        else:
            children: dict[str, list[str]] = {}
            for source_code in value:
                source = str(source_code)
                child = _child_label(parent, source)
                children.setdefault(child, [])
                if source not in children[child]:
                    children[child].append(source)
            expanded[parent] = children

    return expanded  # pyright: ignore[reportReturnType]

def _validate_split_reconstruction(
    *,
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    split_profiles: Mapping[str, SplitProfiles],
    Z_split: pd.DataFrame,
    Y_split: pd.DataFrame,
    x_split: pd.Series,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> None:
    """Fail fast if split sectors do not aggregate back to the original IO totals."""

    Z_expected = eurostat_Z.copy().astype("float64")
    Y_expected = eurostat_Y.copy().astype("float64")
    Z_expected.index = Z_expected.index.astype(str)
    Z_expected.columns = Z_expected.columns.astype(str)
    Y_expected.index = Y_expected.index.astype(str)
    Y_expected.columns = Y_expected.columns.astype(str)

    Z_back = _aggregate_square_to_original(Z_split.astype("float64"), split_profiles=split_profiles)
    Y_back = _aggregate_rows_to_original(Y_split.astype("float64"), split_profiles=split_profiles)
    x_back = _aggregate_rows_to_original(x_split.astype("float64").to_frame("x"), split_profiles=split_profiles)["x"]

    Z_back = Z_back.reindex(index=Z_expected.index, columns=Z_expected.columns).fillna(0.0)
    Y_back = Y_back.reindex(index=Y_expected.index, columns=Y_expected.columns).fillna(0.0)
    x_expected = (Z_expected.sum(axis=1) + Y_expected.sum(axis=1)).astype("float64")
    x_back = x_back.reindex(index=x_expected.index).fillna(0.0)
    z_row_expected = Z_expected.sum(axis=1).astype("float64")
    z_col_expected = Z_expected.sum(axis=0).astype("float64")
    z_row_back = Z_back.sum(axis=1).astype("float64")
    z_col_back = Z_back.sum(axis=0).astype("float64")

    checks = [
        ("Y", Y_back, Y_expected),
        ("Z row totals", z_row_back, z_row_expected),
        ("Z column totals", z_col_back, z_col_expected),
        ("x", x_back, x_expected),
    ]
    errors: list[str] = []
    for label, actual, expected in checks:
        if isinstance(actual, pd.DataFrame):
            ok = np.allclose(actual.to_numpy(), expected.to_numpy(), rtol=rtol, atol=atol, equal_nan=True)
            if not ok:
                diff = actual - expected
                where, gap = _largest_abs_gap(diff)
                errors.append(
                    f"{label} reconstruction failed: largest_abs_gap={gap:.10f} at {where}."
                )
        else:
            ok = np.allclose(actual.to_numpy(), expected.to_numpy(), rtol=rtol, atol=atol, equal_nan=True)
            if not ok:
                diff = actual - expected
                where, gap = _largest_abs_gap(diff)
                errors.append(
                    f"{label} reconstruction failed: largest_abs_gap={gap:.10f} at {where}."
                )

    if errors:
        raise ValueError(
            "Disaggregation validation failed: split result does not aggregate back to the original IOT.\n"
            + "\n".join(errors)
        )


_VALUE_ADDED_SUPPLEMENTARY_PREFIXES: tuple[str, ...] = (
    "B",
    "D1",
    "D2",
    "D3",
    "D29",
    "D39",
    "P1",
    "P51C",
    "CPA_TOTAL",
)


def _supplementary_row_proxy_share(
    row_code: object,
    prof: SplitProfiles,
) -> tuple[pd.Series, str]:
    """Choose the Phase-5 proxy vector for one supplementary/lower V row."""
    code = str(row_code).strip().upper()
    if code.startswith(_VALUE_ADDED_SUPPLEMENTARY_PREFIXES):
        return prof.output_share, "output_share_proxy_for_value_added_row"
    return prof.intermediate_input_share, "projected_exio_column_use_proxy"


def _normalized_supplementary_weights(
    share: pd.Series,
    *,
    children: Sequence[str],
) -> tuple[pd.Series, float, bool]:
    """Return non-negative normalized weights and the pre-normalization denominator."""
    child_list = list(map(str, children))
    numeric_obj = pd.to_numeric(share.reindex(child_list), errors="coerce")
    if not isinstance(numeric_obj, pd.Series):
        raise TypeError("Expected supplementary proxy share to remain a pandas Series.")
    weights = numeric_obj.fillna(0.0).astype("float64").clip(lower=0.0)
    denom = float(weights.sum())  # pyright: ignore[reportOptionalMemberAccess]  # pandas Series after runtime guard; stub may infer Optional
    if not np.isfinite(denom) or denom <= 0.0:
        return pd.Series(1.0 / len(child_list), index=child_list, dtype="float64"), denom, True
    return weights / denom, denom, False  # pyright: ignore[reportOptionalOperand]  # pandas Series after nonzero finite denominator guard


def _normalise_supplementary_row_label(row_code: object) -> str:
    return " ".join(str(row_code).strip().casefold().replace("\\", "/").split())


def _supplementary_accounting_identity_kind(row_code: object) -> str | None:
    code = str(row_code).strip().upper()
    label = _normalise_supplementary_row_label(row_code)
    if code in {"P1", "OUTPUT"} or label == "output":
        return "output"
    if code in {"P2_ADJ", "P2ADJ"} or label == "total intermediate consumption /final use":
        return "intermediate_use"
    if code in {"B1G", "B1GQ", "VALUE_ADDED_GROSS"} or label == "value added, gross":
        return "value_added"
    if code in {"TS_BP", "TOTAL_SUPPLY_AT_BASIC_PRICES"} or label == "total supply at basic prices":
        return "total_supply"
    return None


def _refresh_final_demand_identity_rows(
    M_out: pd.DataFrame | None,
    *,
    Y_split: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, list[dict[str, object]]]:
    if M_out is None:
        return None, []

    M_out = M_out.copy().astype("float64")
    M_out.index = M_out.index.astype(str)
    M_out.columns = M_out.columns.astype(str)
    if Y_split is None:
        return M_out, []

    fd_totals = Y_split.reindex(columns=M_out.columns).fillna(0.0).astype("float64").sum(axis=0)
    diagnostics: list[dict[str, object]] = []
    for row_code in list(map(str, M_out.index)):
        if _supplementary_accounting_identity_kind(row_code) != "intermediate_use":
            continue
        mass_before = float(M_out.loc[row_code].sum())
        M_out.loc[row_code, M_out.columns] = fd_totals
        diagnostics.append(
            {
                "row_code": row_code,
                "parent_code": "__accounting_identity__",
                "child_codes": "|".join(map(str, M_out.columns)),
                "allocation_method": "accounting_identity_final_demand_column_sum",
                "denominator": 1.0,
                "fallback_reason": "",
                "mass_before": mass_before,
                "mass_after": float(M_out.loc[row_code].sum()),
                "weights": "derived_from_split_Y_column_sums",
            }
        )
    return M_out, diagnostics


def _split_supplementary_rows_columnwise(
    *,
    eurostat_V: pd.DataFrame | None,
    eurostat_M: pd.DataFrame | None,
    sectors_old: Sequence[str],
    sectors_new: Sequence[str],
    split_profiles: Mapping[str, SplitProfiles],
    Z_split: pd.DataFrame | None = None,
    Y_split: pd.DataFrame | None = None,
    x_split: pd.Series | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    if eurostat_V is None:
        M_out, diagnostics = _refresh_final_demand_identity_rows(eurostat_M, Y_split=Y_split)
        diag = pd.DataFrame(diagnostics) if diagnostics else None
        return None, M_out, diag

    V = eurostat_V.copy().astype("float64")
    V.index = V.index.astype(str)
    V.columns = V.columns.astype(str)
    V_split = pd.DataFrame(0.0, index=V.index, columns=list(map(str, sectors_new)), dtype="float64")
    diagnostics: list[dict[str, object]] = []

    for parent in map(str, sectors_old):
        if parent not in V.columns:
            continue
        if parent in split_profiles:
            prof = split_profiles[parent]
            children = list(map(str, prof.subsectors))
        else:
            prof = None
            children = [parent]

        for row_code in map(str, V.index):
            row_total = float(V.loc[row_code, parent])
            if prof is None:
                weights = pd.Series(1.0, index=children, dtype="float64")
                denominator = 1.0
                method = "unsplit_identity"
                fallback_reason = ""
            else:
                share, method = _supplementary_row_proxy_share(row_code, prof)
                weights, denominator, used_equal_fallback = _normalized_supplementary_weights(
                    share,
                    children=children,
                )
                fallback_reason = "non_positive_or_non_finite_proxy_denominator" if used_equal_fallback else ""
                if used_equal_fallback:
                    method = "equal_share_fallback"

            for child in children:
                V_split.loc[row_code, child] = row_total * float(weights.loc[child])

            diagnostics.append(
                {
                    "row_code": row_code,
                    "parent_code": parent,
                    "child_codes": "|".join(children),
                    "allocation_method": method,
                    "denominator": denominator,
                    "fallback_reason": fallback_reason,
                    "mass_before": row_total,
                    "mass_after": float(V_split.loc[row_code, children].sum()),
                    "weights": "|".join(f"{child}={float(weights.loc[child]):.12g}" for child in children),
                }
            )

    identity_child_codes = "|".join(map(str, V_split.columns))
    x_identity: pd.Series | None = None
    if x_split is not None:
        x_obj = pd.to_numeric(x_split.reindex(V_split.columns), errors="coerce")
        if not isinstance(x_obj, pd.Series):
            raise TypeError("Expected x_split identity vector to remain a pandas Series.")
        x_identity = x_obj.fillna(0.0).astype("float64")
        for row_code in list(map(str, V_split.index)):
            if _supplementary_accounting_identity_kind(row_code) == "output":
                mass_before = float(V_split.loc[row_code].sum())
                V_split.loc[row_code, V_split.columns] = x_identity
                diagnostics.append(
                    {
                        "row_code": row_code,
                        "parent_code": "__accounting_identity__",
                        "child_codes": identity_child_codes,
                        "allocation_method": "accounting_identity_x_split",
                        "denominator": 1.0,
                        "fallback_reason": "",
                        "mass_before": mass_before,
                        "mass_after": float(V_split.loc[row_code].sum()),
                        "weights": "derived_from_split_x",
                    }
                )

    z_col_use: pd.Series | None = None
    if Z_split is not None:
        z_use = Z_split.reindex(index=V_split.columns, columns=V_split.columns).fillna(0.0).astype("float64")
        z_col_use = z_use.sum(axis=0).astype("float64")
        for row_code in list(map(str, V_split.index)):
            if _supplementary_accounting_identity_kind(row_code) == "intermediate_use":
                mass_before = float(V_split.loc[row_code].sum())
                V_split.loc[row_code, V_split.columns] = z_col_use
                diagnostics.append(
                    {
                        "row_code": row_code,
                        "parent_code": "__accounting_identity__",
                        "child_codes": identity_child_codes,
                        "allocation_method": "accounting_identity_z_column_sum",
                        "denominator": 1.0,
                        "fallback_reason": "",
                        "mass_before": mass_before,
                        "mass_after": float(V_split.loc[row_code].sum()),
                        "weights": "derived_from_split_Z_column_sums",
                    }
                )

    if x_identity is not None and z_col_use is not None:
        value_added = (x_identity - z_col_use).astype("float64")
        for row_code in list(map(str, V_split.index)):
            if _supplementary_accounting_identity_kind(row_code) == "value_added":
                mass_before = float(V_split.loc[row_code].sum())
                V_split.loc[row_code, V_split.columns] = value_added
                diagnostics.append(
                    {
                        "row_code": row_code,
                        "parent_code": "__accounting_identity__",
                        "child_codes": identity_child_codes,
                        "allocation_method": "accounting_identity_value_added_residual",
                        "denominator": 1.0,
                        "fallback_reason": "",
                        "mass_before": mass_before,
                        "mass_after": float(V_split.loc[row_code].sum()),
                        "weights": "derived_from_split_x_minus_Z_column_sums",
                    }
                )

    if x_identity is not None:
        imports_vector = None
        imports_row_code = ""
        for candidate in ("P7", "P7_CIF", "P7_FOB"):
            if candidate in V_split.index:
                imports_row_code = candidate
                imports_vector = V_split.loc[candidate].reindex(V_split.columns).fillna(0.0).astype("float64")
                break
        if imports_vector is not None:
            for row_code in list(map(str, V_split.index)):
                if _supplementary_accounting_identity_kind(row_code) == "total_supply":
                    mass_before = float(V_split.loc[row_code].sum())
                    V_split.loc[row_code, V_split.columns] = x_identity + imports_vector
                    diagnostics.append(
                        {
                            "row_code": row_code,
                            "parent_code": "__accounting_identity__",
                            "child_codes": identity_child_codes,
                            "allocation_method": "accounting_identity_x_plus_imports",
                            "denominator": 1.0,
                            "fallback_reason": "",
                            "mass_before": mass_before,
                            "mass_after": float(V_split.loc[row_code].sum()),
                            "weights": f"derived_from_split_x_plus_{imports_row_code}",
                        }
                    )

    M_out, M_diagnostics = _refresh_final_demand_identity_rows(eurostat_M, Y_split=Y_split)
    diagnostics.extend(M_diagnostics)
    diag = pd.DataFrame(diagnostics) if diagnostics else None
    return V_split, M_out, diag

def _fd_category_keys_for_eurostat_code(fd_code: str) -> list[str] | None:
    code = str(fd_code)
    if code == "P3":
        return [
            "households",
            "non-profit organisations serving households",
            "npish",
            "government",
        ]
    if code == "P3_S14":
        return ["households"]
    if code == "P3_S15":
        return ["non-profit organisations serving households", "npish"]
    if code == "P3_S13":
        return ["government"]
    if code == "P5":
        return ["gross fixed capital formation", "inventories", "valuables"]
    if code == "P51G":
        return ["gross fixed capital formation"]
    if code == "P52":
        return ["inventories"]
    if code == "P53":
        return ["valuables"]
    if code in {"P6", "P6_B0", "P6_D0", "P6_U2", "P6_U3"}:
        return ["exports"]
    return None


def _split_iot_using_donor_blocks(
    *,
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    eurostat_V: pd.DataFrame | None = None,
    eurostat_M: pd.DataFrame | None = None,
    exio_slice: ExioRegionSlice,
    split_profiles: Mapping[str, SplitProfiles],
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None,
    create_io_system: bool,
) -> SplitResult:
    Z = eurostat_Z.copy().astype("float64")
    Y = eurostat_Y.copy().astype("float64")
    Z.index = Z.index.astype(str)
    Z.columns = Z.columns.astype(str)
    Y.index = Y.index.astype(str)
    Y.columns = Y.columns.astype(str)

    sectors_old = list(map(str, Z.index))
    sectors_new = _expanded_sector_order(sectors_old=sectors_old, split_profiles=split_profiles)
    Z_split = pd.DataFrame(0.0, index=sectors_new, columns=sectors_new, dtype="float64")
    Y_split = pd.DataFrame(0.0, index=sectors_new, columns=Y.columns.astype(str), dtype="float64")

    _, donor_Z, donor_Y = project_exio_donor_blocks_for_split_profiles(
        exio_Z=exio_slice.Z_domestic,
        exio_Y=exio_slice.Y_region,
        split_profiles=split_profiles,
        eurostat_sector_mapping=eurostat_sector_mapping,
        raw_exio_sectors=list(map(str, exio_slice.sectors)),
        emitted_labels=sectors_new,
    )

    fallback_records: list[dict[str, object]] = []

    for row_sector in sectors_old:
        row_labels = list(split_profiles[row_sector].subsectors) if row_sector in split_profiles else [row_sector]
        for col_sector in sectors_old:
            col_labels = list(split_profiles[col_sector].subsectors) if col_sector in split_profiles else [col_sector]
            cell_total = float(Z.loc[row_sector, col_sector])
            if cell_total == 0.0:
                continue

            if len(row_labels) == 1 and len(col_labels) == 1:
                Z_split.loc[row_labels[0], col_labels[0]] = cell_total
                continue

            donor_values = donor_Z.reindex(index=row_labels, columns=col_labels).fillna(0.0).astype("float64")

            donor_total = float(donor_values.to_numpy(dtype="float64").sum())
            if np.isfinite(donor_total) and donor_total > 0.0:
                alloc = donor_values * (cell_total / donor_total)
            elif row_sector in split_profiles and col_sector in split_profiles:
                row_share = _normalize_vector(
                    split_profiles[row_sector].intermediate_sales_share.reindex(row_labels).fillna(0.0)
                )
                col_share = _normalize_vector(
                    split_profiles[col_sector].intermediate_input_share.reindex(col_labels).fillna(0.0)
                )
                alloc = pd.DataFrame(
                    np.outer(row_share.to_numpy(), col_share.to_numpy()) * cell_total,
                    index=row_labels,
                    columns=col_labels,
                )
                fallback_records.append(
                    {
                        "row_sector": str(row_sector),
                        "col_sector": str(col_sector),
                        "cell_total": float(cell_total),
                        "row_labels": "|".join(map(str, row_labels)),
                        "col_labels": "|".join(map(str, col_labels)),
                        "donor_total": float(donor_total),
                        "fallback_type": "both_split_outer_product",
                    }
                )
            elif row_sector in split_profiles:
                row_share = _normalize_vector(
                    split_profiles[row_sector].intermediate_sales_share.reindex(row_labels).fillna(0.0)
                )
                alloc = pd.DataFrame(
                    row_share.to_numpy().reshape(-1, 1) * cell_total,
                    index=row_labels,
                    columns=col_labels,
                )
                fallback_records.append(
                    {
                        "row_sector": str(row_sector),
                        "col_sector": str(col_sector),
                        "cell_total": float(cell_total),
                        "row_labels": "|".join(map(str, row_labels)),
                        "col_labels": "|".join(map(str, col_labels)),
                        "donor_total": float(donor_total),
                        "fallback_type": "row_split_sales_share",
                    }
                )
            elif col_sector in split_profiles:
                col_share = _normalize_vector(
                    split_profiles[col_sector].intermediate_input_share.reindex(col_labels).fillna(0.0)
                )
                alloc = pd.DataFrame(
                    col_share.to_numpy().reshape(1, -1) * cell_total,
                    index=row_labels,
                    columns=col_labels,
                )
                fallback_records.append(
                    {
                        "row_sector": str(row_sector),
                        "col_sector": str(col_sector),
                        "cell_total": float(cell_total),
                        "row_labels": "|".join(map(str, row_labels)),
                        "col_labels": "|".join(map(str, col_labels)),
                        "donor_total": float(donor_total),
                        "fallback_type": "col_split_input_share",
                    }
                )
            else:
                alloc = pd.DataFrame([[cell_total]], index=row_labels, columns=col_labels)
                fallback_records.append(
                    {
                        "row_sector": str(row_sector),
                        "col_sector": str(col_sector),
                        "cell_total": float(cell_total),
                        "row_labels": "|".join(map(str, row_labels)),
                        "col_labels": "|".join(map(str, col_labels)),
                        "donor_total": float(donor_total),
                        "fallback_type": "both_split_marginal_share",
                    }
                )

            Z_split.loc[row_labels, col_labels] = alloc.to_numpy(dtype="float64")

    for row_sector in sectors_old:
        row_labels = list(split_profiles[row_sector].subsectors) if row_sector in split_profiles else [row_sector]
        if len(row_labels) == 1:
            Y_split.loc[row_labels[0], :] = Y.loc[row_sector, :].to_numpy(dtype="float64")
            continue

        for fd_code in Y.columns.astype(str):
            fd_total = float(Y.loc[row_sector, fd_code])
            if fd_total == 0.0:
                continue
            keys = _fd_category_keys_for_eurostat_code(str(fd_code))
            donor_share = None
            if keys:
                cols = [col for col in donor_Y.columns if any(key in str(col).lower() for key in keys)]
                if cols:
                    donor_values = donor_Y.reindex(index=row_labels, columns=cols).fillna(0.0).sum(axis=1)
                    donor_total = float(donor_values.sum())
                    if np.isfinite(donor_total) and donor_total > 0.0:
                        donor_share = donor_values / donor_total
            if donor_share is None:
                donor_share = split_profiles[row_sector].final_demand_share.reindex(row_labels).fillna(0.0)
                donor_share = _normalize_vector(donor_share)
                fallback_records.append(
                    {
                        "row_sector": str(row_sector),
                        "col_sector": str(fd_code),
                        "cell_total": float(fd_total),
                        "row_labels": "|".join(map(str, row_labels)),
                        "col_labels": str(fd_code),
                        "donor_total": float("nan"),
                        "fallback_type": "final_demand_share",
                    }
                )
            Y_split.loc[row_labels, fd_code] = donor_share.reindex(row_labels).fillna(0.0).to_numpy(dtype="float64") * fd_total

    x_split = calc_x(Z_split, Y_split)
    A_split = calc_A(Z_split, x_split)
    V_split, M_split, supplementary_diagnostics = _split_supplementary_rows_columnwise(
        eurostat_V=eurostat_V,
        eurostat_M=eurostat_M,
        sectors_old=sectors_old,
        sectors_new=sectors_new,
        split_profiles=split_profiles,
        Z_split=Z_split,
        Y_split=Y_split,
        x_split=x_split,
    )

    io_system = None
    if create_io_system:
        try:
            import pymrio  # pyright: ignore[reportMissingImports]

            io_system = pymrio.IOSystem()
            io_system.Z = Z_split
            io_system.Y = Y_split
            io_system.x = x_split
            io_system.A = A_split
        except ImportError:
            pass

    fallback_count = len(fallback_records)
    method = "DONOR_BLOCK" if fallback_count == 0 else "DONOR_BLOCK_WITH_SHARE_FALLBACK"
    return SplitResult(
        Z=Z_split,
        Y=Y_split,
        x=x_split,
        A=A_split,
        split_profiles=dict(split_profiles),
        balancing_method=method,
        balancing_stats=BalanceStats(0, 0.0, 0.0),
        exio_selected_sectors={},
        exio_quantity_and_price=None,
        V=V_split,
        M=M_split,
        supplementary_diagnostics=supplementary_diagnostics,
        fallback_diagnostics=pd.DataFrame(fallback_records) if fallback_records else None,
        io_system=io_system,
    )


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, index_col=0)
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        obj = pd.read_pickle(path)
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"Expected DataFrame in table pickle: {path}")
        return obj
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, index_col=0)
    raise ValueError(f"Unsupported table format: {path}")

def load_local_iot(
    z_path: str | Path,
    y_path: str | Path,
    *,
    dtype: str | np.dtype = "float64",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load an IO table provided as two tables (Z and Y).

    This loader is file-format agnostic (csv/parquet/pickle/xlsx).
    """
    Z = _read_table(Path(z_path)).astype(dtype)
    Y = _read_table(Path(y_path)).astype(dtype)

    Z.index = Z.index.astype(str)
    Z.columns = Z.columns.astype(str)
    Y.index = Y.index.astype(str)
    Y.columns = Y.columns.astype(str)

    if not Z.index.equals(Z.columns):
        raise ValueError("Z must be square with matching index/columns.")
    if not Y.index.equals(Z.index):
        raise ValueError("Y must have the same index as Z.")

    return Z, Y

def build_split_profiles_from_exiobase_domestic_block(
    exio_slice: ExioRegionSlice,
    *,
    sector_to_exio_sectors: Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    sector_patterns: Mapping[str, Sequence[str]] | None = None,
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[dict[str, SplitProfiles], dict[str, list[str]]]:
    """
    Build split profiles for any set of aggregate sectors.

    Exactly one of:
    - `sector_to_exio_sectors`: mapping aggregate -> explicit EXIO sector labels (recommended).
    - `sector_patterns`: mapping aggregate -> substring patterns (template/quickstart; less robust).
    """
    if (sector_to_exio_sectors is None) == (sector_patterns is None):
        raise ValueError("Provide exactly one of sector_to_exio_sectors or sector_patterns.")

    profiles: dict[str, SplitProfiles] = {}
    selected: dict[str, list[str]] = {}
    exio_sector_index = pd.Index(exio_slice.sectors.astype(str))
    available_codes = set(map(str, exio_sector_index))

    if sector_to_exio_sectors is not None:
        from ecolecon_mrio.disaggregation import SplitProfiles, normalize_share

        Z_rr = exio_slice.Z_domestic
        Y_r = exio_slice.Y_region
        x_use = (Z_rr.sum(axis=1) + Y_r.sum(axis=1)).astype("float64")

        # Pre-compute sums to avoid repetitive matrix summing in loop
        # These are series indexed by exio sector
        Z_sales_sum = Z_rr.sum(axis=1)
        Z_inputs_sum = Z_rr.sum(axis=0)
        Y_fd_sum = Y_r.sum(axis=1)

        for aggregate, exio_sectors in sector_to_exio_sectors.items():
            if isinstance(exio_sectors, ABCMapping):
                subsector_to_exio: dict[str, list[str]] = {}
                for sub, members in exio_sectors.items():
                    sub_code = str(sub).strip()
                    requested_codes = [str(member).strip() for member in members if str(member).strip() != ""]
                    missing = [code for code in requested_codes if code not in available_codes]
                    if missing:
                        raise ValueError(
                            f"aggregate_sector={aggregate!r}, subsector={sub_code!r}: unknown EXIO sector code(s): {missing!r}"
                        )
                    subsector_to_exio[sub_code] = requested_codes

                # Aggregate EXIO vectors into subsector totals, then normalize to shares.
                # Optimized: slices from pre-computed sums instead of summing matrix block every time.
                x_sub = pd.Series(
                    {
                        sub: _weighted_exio_member_sum(
                            x_use,
                            subsector=sub,
                            members=list(map(str, members)),
                            eurostat_sector_mapping=eurostat_sector_mapping,
                        )
                        for sub, members in subsector_to_exio.items()
                    },
                    dtype="float64",
                )
                sales_sub = pd.Series(
                    {
                        sub: _weighted_exio_member_sum(
                            Z_sales_sum,
                            subsector=sub,
                            members=list(map(str, members)),
                            eurostat_sector_mapping=eurostat_sector_mapping,
                        )
                        for sub, members in subsector_to_exio.items()
                    },
                    dtype="float64",
                )
                inputs_sub = pd.Series(
                    {
                        sub: _weighted_exio_member_sum(
                            Z_inputs_sum,
                            subsector=sub,
                            members=list(map(str, members)),
                            eurostat_sector_mapping=eurostat_sector_mapping,
                        )
                        for sub, members in subsector_to_exio.items()
                    },
                    dtype="float64",
                )
                fd_sub = pd.Series(
                    {
                        sub: _weighted_exio_member_sum(
                            Y_fd_sum,
                            subsector=sub,
                            members=list(map(str, members)),
                            eurostat_sector_mapping=eurostat_sector_mapping,
                        )
                        for sub, members in subsector_to_exio.items()
                    },
                    dtype="float64",
                )

                output_share = normalize_share(x_sub, name=f"{aggregate}.output_share")
                sales_share = normalize_share(sales_sub, name=f"{aggregate}.intermediate_sales_share")
                input_share = normalize_share(inputs_sub, name=f"{aggregate}.intermediate_input_share")
                try:
                    final_demand_share = normalize_share(fd_sub, name=f"{aggregate}.final_demand_share")
                except ValueError:
                    final_demand_share = sales_share.copy()

                subsectors = [str(s) for s in x_sub.index]
                rename = dict(zip(x_sub.index.astype(str), subsectors))

                profiles[str(aggregate)] = SplitProfiles(
                    aggregate_sector=str(aggregate),
                    subsectors=tuple(subsectors),
                    output_share=output_share.rename(index=rename),
                    intermediate_sales_share=sales_share.rename(index=rename),
                    intermediate_input_share=input_share.rename(index=rename),
                    final_demand_share=final_demand_share.rename(index=rename),
                    subsector_to_exio_sectors=subsector_to_exio,
                )

                selected[str(aggregate)] = list(dict.fromkeys(map(str, subsector_to_exio.keys())))
            else:
                requested_codes = [str(member).strip() for member in exio_sectors if str(member).strip() != ""]
                missing = [code for code in requested_codes if code not in available_codes]
                if missing:
                    raise ValueError(f"aggregate_sector={aggregate!r}: unknown EXIO sector code(s): {missing!r}")
                subsector_to_exio = {code: [code] for code in requested_codes}

                x_sub = pd.Series(
                    {
                        sub: _weighted_exio_member_sum(
                            x_use,
                            subsector=sub,
                            members=list(map(str, members)),
                            eurostat_sector_mapping=eurostat_sector_mapping,
                        )
                        for sub, members in subsector_to_exio.items()
                    },
                    dtype="float64",
                )
                sales_sub = pd.Series(
                    {
                        sub: _weighted_exio_member_sum(
                            Z_sales_sum,
                            subsector=sub,
                            members=list(map(str, members)),
                            eurostat_sector_mapping=eurostat_sector_mapping,
                        )
                        for sub, members in subsector_to_exio.items()
                    },
                    dtype="float64",
                )
                inputs_sub = pd.Series(
                    {
                        sub: _weighted_exio_member_sum(
                            Z_inputs_sum,
                            subsector=sub,
                            members=list(map(str, members)),
                            eurostat_sector_mapping=eurostat_sector_mapping,
                        )
                        for sub, members in subsector_to_exio.items()
                    },
                    dtype="float64",
                )
                fd_sub = pd.Series(
                    {
                        sub: _weighted_exio_member_sum(
                            Y_fd_sum,
                            subsector=sub,
                            members=list(map(str, members)),
                            eurostat_sector_mapping=eurostat_sector_mapping,
                        )
                        for sub, members in subsector_to_exio.items()
                    },
                    dtype="float64",
                )

                output_share = normalize_share(x_sub, name=f"{aggregate}.output_share")
                sales_share = normalize_share(sales_sub, name=f"{aggregate}.intermediate_sales_share")
                input_share = normalize_share(inputs_sub, name=f"{aggregate}.intermediate_input_share")
                try:
                    final_demand_share = normalize_share(fd_sub, name=f"{aggregate}.final_demand_share")
                except ValueError:
                    final_demand_share = sales_share.copy()

                subsectors = [str(s) for s in x_sub.index]
                rename = dict(zip(x_sub.index.astype(str), subsectors))

                profiles[str(aggregate)] = SplitProfiles(
                    aggregate_sector=str(aggregate),
                    subsectors=tuple(subsectors),
                    output_share=output_share.rename(index=rename),
                    intermediate_sales_share=sales_share.rename(index=rename),
                    intermediate_input_share=input_share.rename(index=rename),
                    final_demand_share=final_demand_share.rename(index=rename),
                    subsector_to_exio_sectors=subsector_to_exio,
                )
                selected[str(aggregate)] = list(dict.fromkeys(requested_codes))
        return profiles, selected

    assert sector_patterns is not None
    Z_rr = exio_slice.Z_domestic
    Y_r = exio_slice.Y_region
    x_use = (Z_rr.sum(axis=1) + Y_r.sum(axis=1)).astype("float64")
    Z_sales_sum = Z_rr.sum(axis=1)
    Z_inputs_sum = Z_rr.sum(axis=0)
    Y_fd_sum = Y_r.sum(axis=1)

    from ecolecon_mrio.disaggregation import SplitProfiles, normalize_share

    for aggregate, patterns in sector_patterns.items():
        matched = [code for code in available_codes if any(str(pattern).casefold() in code.casefold() for pattern in patterns)]
        if not matched:
            raise ValueError(f"aggregate_sector={aggregate!r}: no EXIO sector codes matched patterns {tuple(patterns)!r}")

        subsector_to_exio = {code: [code] for code in matched}
        x_sub = pd.Series({sub: float(x_use.loc[pd.Index(members)].sum()) for sub, members in subsector_to_exio.items()}, dtype="float64")
        sales_sub = pd.Series({sub: float(Z_sales_sum.loc[pd.Index(members)].sum()) for sub, members in subsector_to_exio.items()}, dtype="float64")
        inputs_sub = pd.Series({sub: float(Z_inputs_sum.loc[pd.Index(members)].sum()) for sub, members in subsector_to_exio.items()}, dtype="float64")
        fd_sub = pd.Series({sub: float(Y_fd_sum.loc[pd.Index(members)].sum()) for sub, members in subsector_to_exio.items()}, dtype="float64")

        output_share = normalize_share(x_sub, name=f"{aggregate}.output_share")
        sales_share = normalize_share(sales_sub, name=f"{aggregate}.intermediate_sales_share")
        input_share = normalize_share(inputs_sub, name=f"{aggregate}.intermediate_input_share")
        try:
            final_demand_share = normalize_share(fd_sub, name=f"{aggregate}.final_demand_share")
        except ValueError:
            final_demand_share = sales_share.copy()

        profiles[str(aggregate)] = SplitProfiles(
            aggregate_sector=str(aggregate),
            subsectors=tuple(matched),
            output_share=output_share,
            intermediate_sales_share=sales_share,
            intermediate_input_share=input_share,
            final_demand_share=final_demand_share,
            subsector_to_exio_sectors=subsector_to_exio,
        )
        selected[str(aggregate)] = list(dict.fromkeys(matched))
    return profiles, selected


def _resolve_subsector_to_exio_sectors(
    *,
    agg_sector: str,
    prof: SplitProfiles,
    sector_to_exio_sectors: Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None,
) -> Mapping[str, Sequence[str]] | None:
    subsector_map = prof.subsector_to_exio_sectors
    if subsector_map:
        return subsector_map

    if sector_to_exio_sectors is None:
        return None

    agg_map = sector_to_exio_sectors.get(str(agg_sector))
    if isinstance(agg_map, ABCMapping):
        resolved: dict[str, list[str]] = {}
        for sub, members in agg_map.items():
            sub_label = str(sub)
            if sub_label not in prof.subsectors:
                continue
            resolved[sub_label] = list(map(str, members))
        return resolved or None

    if agg_map is not None:
        exio_list = list(map(str, agg_map))
        if len(exio_list) == len(prof.subsectors):
            return {
                subsector: [exio_sector]
                for subsector, exio_sector in zip(prof.subsectors, exio_list)
            }

    return None


def _map_exio_ratio_to_disaggregated_sectors(
    *,
    exio_df: pd.DataFrame | None,
    split_profiles: dict[str, SplitProfiles],
    sector_to_exio_sectors: Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None,
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None,
    numerator_col: str,
    denominator_col: str,
    out_ratio_col: str,
) -> pd.DataFrame | None:
    if exio_df is None or not split_profiles:
        return None

    df = exio_df.copy()
    required_cols = {"exio_sector", "extension", "stressor", "unit", numerator_col, denominator_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"exio_df missing required columns for mapping: {sorted(missing_cols)!r}"
        )

    df["exio_sector"] = df["exio_sector"].astype(str)
    df[numerator_col] = pd.to_numeric(df[numerator_col], errors="coerce")
    df[denominator_col] = pd.to_numeric(df[denominator_col], errors="coerce")

    rows: list[dict[str, object]] = []
    group_cols = ["extension", "stressor", "unit"]
    for agg_sector, prof in split_profiles.items():
        subsector_map = _resolve_subsector_to_exio_sectors(
            agg_sector=str(agg_sector),
            prof=prof,
            sector_to_exio_sectors=sector_to_exio_sectors,
        )
        if not subsector_map:
            continue

        for subsector, exio_sectors in subsector_map.items():
            weight_map: dict[str, float] = {}
            for exio_sector in map(str, exio_sectors):
                weight = _edge_member_weight(
                    subsector=subsector,
                    member=exio_sector,
                    eurostat_sector_mapping=eurostat_sector_mapping,
                )
                if np.isfinite(weight) and weight != 0.0:
                    weight_map[exio_sector] = float(weight)
            if not weight_map:
                continue

            exio_subset = df[df["exio_sector"].isin(list(weight_map.keys()))].copy()
            if exio_subset.empty:
                continue

            grouped = exio_subset.groupby(group_cols, dropna=False)
            for keys, group in grouped:
                key_tuple = keys if isinstance(keys, tuple) else (keys,)
                if len(key_tuple) != len(group_cols):
                    continue
                exio_sector_series = cast(pd.Series, group["exio_sector"])
                weights = exio_sector_series.map(weight_map).fillna(0.0).astype("float64")
                num_values = cast(pd.Series, pd.to_numeric(group[numerator_col], errors="coerce")).astype("float64")
                den_values = cast(pd.Series, pd.to_numeric(group[denominator_col], errors="coerce")).astype("float64")

                total_num = float((num_values * weights).sum())
                if not np.isfinite(total_num):
                    continue
                total_den = float((den_values * weights).sum())
                if not np.isfinite(total_den) or total_den == 0:
                    continue

                rows.append(
                    {
                        "disaggregated_sector": subsector,
                        "extension": key_tuple[0],
                        "stressor": key_tuple[1],  # pyright: ignore[reportGeneralTypeIssues]
                        "unit": key_tuple[2],  # pyright: ignore[reportGeneralTypeIssues]
                        numerator_col: total_num,
                        denominator_col: total_den,
                        out_ratio_col: total_num / total_den,
                    }
                )

    if not rows:
        return None
    return pd.DataFrame(rows)

def map_exio_value_per_physical_input(
    exio_quantity_and_price: pd.DataFrame | None,
    split_profiles: dict[str, SplitProfiles],
    sector_to_exio_sectors: Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.DataFrame | None:
    """Map EXIOBASE unit-value proxies to emitted disaggregated sectors."""
    if exio_quantity_and_price is None or not split_profiles:
        return None

    exio_qp = exio_quantity_and_price.copy()
    if "value_per_physical_input" not in exio_qp.columns and "implied_price" in exio_qp.columns:
        exio_qp = exio_qp.rename(columns={"implied_price": "value_per_physical_input"})

    return _map_exio_ratio_to_disaggregated_sectors(
        exio_df=exio_qp,
        split_profiles=split_profiles,
        sector_to_exio_sectors=sector_to_exio_sectors,
        eurostat_sector_mapping=eurostat_sector_mapping,
        numerator_col="monetary_output",
        denominator_col="quantity",
        out_ratio_col="value_per_physical_input",
    )


def map_exio_emission_intensity(
    exio_emissions_and_intensity: pd.DataFrame | None,
    split_profiles: dict[str, SplitProfiles],
    sector_to_exio_sectors: Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.DataFrame | None:
    """Map EXIOBASE emissions intensity to emitted disaggregated sectors."""
    return _map_exio_ratio_to_disaggregated_sectors(
        exio_df=exio_emissions_and_intensity,
        split_profiles=split_profiles,
        sector_to_exio_sectors=sector_to_exio_sectors,
        eurostat_sector_mapping=eurostat_sector_mapping,
        numerator_col="emission",
        denominator_col="monetary_output",
        out_ratio_col="emission_intensity",
    )


def split_iot_using_exiobase_split_profiles(
    *,
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    eurostat_V: pd.DataFrame | None = None,
    eurostat_M: pd.DataFrame | None = None,
    exio_slice: ExioRegionSlice,
    sector_to_exio_sectors: (
        Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None
    ) = None,
    sector_patterns: Mapping[str, Sequence[str]] | None = None,
    ras_max_iter: int = 50_000,
    ras_tol: float = 1e-10,
    balancing_scope: str = "local",
    coverage_expected_exio_sectors: Mapping[str, Sequence[str]] | None = None,
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None = None,
    coverage_tol: float = 0.0,
    extract_quantity_and_price: bool = False,
    quantity_price_extensions: Sequence[str] | None = None,
    quantity_price_stressor_keywords: Sequence[str] | None = None,
    quantity_price_cache_dir: str | Path | None = None,
    create_io_system: bool = True,
) -> SplitResult:
    """
    Split one or more aggregate sectors in (Z,Y) using EXIOBASE donor structure where available.

    The allocation uses full donor blocks for Z and donor final-demand-category shares for Y.
    Where donor information is unavailable, it falls back to the existing marginal-share logic.
    """
    effective_sector_to_exio_sectors = _edge_aware_sector_to_exio_sectors(
        sector_to_exio_sectors,
        eurostat_sector_mapping=eurostat_sector_mapping,
    )
    profiles, selected = build_split_profiles_from_exiobase_domestic_block(
        exio_slice,
        sector_to_exio_sectors=effective_sector_to_exio_sectors,
        sector_patterns=sector_patterns,
        eurostat_sector_mapping=eurostat_sector_mapping,
    )

    if coverage_expected_exio_sectors is not None:
        if coverage_tol < 0:
            raise ValueError("coverage_tol must be >= 0")

        use_total = (exio_slice.Z_domestic.sum(axis=1) + exio_slice.Y_region.sum(axis=1)).astype("float64")
        use_total.index = use_total.index.astype(str)

        errors: list[str] = []
        for agg, expected in coverage_expected_exio_sectors.items():
            agg_s = str(agg)
            expected_list = list(map(str, expected))
            profile = profiles.get(agg_s)
            raw_selected = _raw_exio_members_from_profile(profile)
            selected_list = raw_selected or list(map(str, selected.get(agg_s, [])))

            missing_expected = [s for s in expected_list if s not in use_total.index]
            if missing_expected:
                errors.append(
                    f"Coverage check failed: aggregate_sector={agg_s!r} has expected EXIO sectors that are not present "
                    f"in the EXIO slice (flow={exio_slice.flow!r}): {missing_expected!r}"
                )
                continue

            missing_selected = [s for s in selected_list if s not in use_total.index]
            if missing_selected:
                errors.append(
                    f"Coverage check failed: aggregate_sector={agg_s!r} has selected EXIO sectors that are not present "
                    f"in the EXIO slice (flow={exio_slice.flow!r}): {missing_selected!r}"
                )
                continue

            expected_sum = float(use_total.loc[expected_list].sum())
            selected_sum = float(use_total.loc[selected_list].sum())
            if expected_sum == 0:
                coverage = 1.0 if selected_sum == 0 else 0.0
            else:
                coverage = selected_sum / expected_sum

            if coverage < 1.0 - float(coverage_tol):
                missing = sorted(set(expected_list) - set(selected_list))
                missing_contrib = (
                    use_total.loc[missing].sort_values(ascending=False).head(10).to_dict() if missing else {}
                )
                errors.append(
                    "Coverage check failed for EXIO constituent set.\n"
                    f"  aggregate_sector: {agg_s}\n"
                    f"  exio_flow: {exio_slice.flow}\n"
                    f"  expected_count: {len(expected_list)}\n"
                    f"  selected_count: {len(selected_list)}\n"
                    f"  selected/expected (by EXIO Z+Y row totals): {coverage:.6g} (tol={coverage_tol})\n"
                    f"  missing_sectors: {missing!r}\n"
                    f"  top_missing_contrib: {missing_contrib!r}"
                )

        if errors:
            raise ValueError("\n\n".join(errors))

    res = _split_iot_using_donor_blocks(
        eurostat_Z=eurostat_Z,
        eurostat_Y=eurostat_Y,
        eurostat_V=eurostat_V,
        eurostat_M=eurostat_M,
        exio_slice=exio_slice,
        split_profiles=profiles,
        eurostat_sector_mapping=eurostat_sector_mapping,
        create_io_system=create_io_system,
    )

    _validate_split_reconstruction(
        eurostat_Z=eurostat_Z,
        eurostat_Y=eurostat_Y,
        split_profiles=profiles,
        Z_split=res.Z,
        Y_split=res.Y,
        x_split=res.x,
    )

    exio_qp = None
    if extract_quantity_and_price:
        if quantity_price_extensions is None or quantity_price_stressor_keywords is None:
            raise ValueError(
                "When extract_quantity_and_price=True, also provide "
                "quantity_price_extensions and quantity_price_stressor_keywords."
            )
        exio_sectors = _raw_exio_members_from_profiles(profiles)

        monetary_output = (exio_slice.Z_domestic.sum(axis=1) + exio_slice.Y_region.sum(axis=1)).astype(
            "float64"
        )
        exio_qp = extract_exio_quantity_and_price(
            container=exio_slice.source_zip,
            region=exio_slice.region,
            sectors=exio_sectors,
            monetary_output=monetary_output,
            extensions=tuple(quantity_price_extensions),
            stressor_keywords=tuple(quantity_price_stressor_keywords),
            cache_dir=quantity_price_cache_dir,
        )

    return SplitResult(
        Z=res.Z,
        Y=res.Y,
        x=res.x,
        A=res.A,
        split_profiles=res.split_profiles,
        balancing_method=res.balancing_method,
        balancing_stats=res.balancing_stats,
        exio_selected_sectors=selected,
        exio_quantity_and_price=exio_qp,
        V=res.V,
        M=res.M,
        supplementary_diagnostics=res.supplementary_diagnostics,
        fallback_diagnostics=res.fallback_diagnostics,
        disaggregated_sector_value_per_physical_input=res.disaggregated_sector_value_per_physical_input,
        io_system=res.io_system,
    )


def split_iot_using_exiobase_shares(
    *,
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    eurostat_V: pd.DataFrame | None = None,
    eurostat_M: pd.DataFrame | None = None,
    exio_year: int,
    exio_region: str,
    exio_slice: ExioRegionSlice | None = None,
    exio_system: str = "pxp",
    exio_cache_root: str | Path | None = None,
    sector_to_exio_sectors: (
        Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None
    ) = None,
    sector_patterns: Mapping[str, Sequence[str]] | None = None,
    ras_max_iter: int = 50_000,
    ras_tol: float = 1e-10,
    balancing_scope: str = "local",
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None = None,
    coverage_expected_exio_sectors: Mapping[str, Sequence[str]] | None = None,
    coverage_tol: float = 0.0,
    extract_quantity_and_price: bool = False,
    quantity_price_extensions: Sequence[str] | None = None,
    quantity_price_stressor_keywords: Sequence[str] | None = None,
    quantity_price_cache_dir: str | Path | None = None,
    create_io_system: bool = True,
) -> SplitResult:
    """Convenience wrapper: load an EXIOBASE region slice if needed, then split+balance."""
    if exio_slice is None:
        exio_slice = load_exiobase3_region_slice(
            year=exio_year,
            region=exio_region,
            system=exio_system,
            cache_root=exio_cache_root,
        )
    return split_iot_using_exiobase_split_profiles(
        eurostat_Z=eurostat_Z,
        eurostat_Y=eurostat_Y,
        eurostat_V=eurostat_V,
        eurostat_M=eurostat_M,
        exio_slice=exio_slice,
        sector_to_exio_sectors=sector_to_exio_sectors,
        sector_patterns=sector_patterns,
        ras_max_iter=ras_max_iter,
        ras_tol=ras_tol,
        balancing_scope=balancing_scope,
        eurostat_sector_mapping=eurostat_sector_mapping,
        coverage_expected_exio_sectors=coverage_expected_exio_sectors,
        coverage_tol=coverage_tol,
        extract_quantity_and_price=extract_quantity_and_price,
        quantity_price_extensions=quantity_price_extensions,
        quantity_price_stressor_keywords=quantity_price_stressor_keywords,
        quantity_price_cache_dir=quantity_price_cache_dir,
        create_io_system=create_io_system,
    )


def write_split_report_xlsx(
    out_path: str | Path,
    *,
    result: SplitResult,
    aggregates: Sequence[str] | None = None,
) -> Path:
    """
    Write an XLSX summary report for a split result.

    Note: mapping/concordance workflows are CSV-driven, but reports can still be XLSX.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if aggregates is None:
        aggregates = tuple(result.split_profiles.keys())

    subsectors: list[str] = []
    for agg in aggregates:
        prof = result.split_profiles.get(str(agg))
        if prof is None:
            continue
        subsectors.extend(list(map(str, prof.subsectors)))
    subsectors = list(dict.fromkeys(subsectors))

    def _safe_sheet(name: str) -> str:
        return str(name)[:31]

    rows: list[pd.DataFrame] = []
    for agg in aggregates:
        prof = result.split_profiles.get(str(agg))
        if prof is None:
            continue
        df = pd.DataFrame(
            {
                "subsector": list(map(str, prof.subsectors)),
                "output_share": prof.output_share.reindex(prof.subsectors).values,
                "intermediate_sales_share": prof.intermediate_sales_share.reindex(prof.subsectors).values,
                "intermediate_input_share": prof.intermediate_input_share.reindex(prof.subsectors).values,
                "final_demand_share": prof.final_demand_share.reindex(prof.subsectors).values,
            }
        )
        df.insert(0, "aggregate_sector", str(agg))
        rows.append(df)
    profiles_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    with pd.ExcelWriter(out_path) as xw:  # pyright: ignore[reportAbstractUsage]
        profiles_df.to_excel(xw, sheet_name=_safe_sheet("split_profiles"), index=False)
        result.Z.loc[subsectors, subsectors].to_excel(xw, sheet_name=_safe_sheet("Z_split_submatrix"))
        result.Y.loc[subsectors, :].to_excel(xw, sheet_name=_safe_sheet("Y_split_rows"))

    return out_path


def write_energy_split_report_xlsx(
    out_path: str | Path,
    *,
    result: SplitResult,
    aggregates: Sequence[str],
) -> Path:
    # Backwards-compatible alias
    return write_split_report_xlsx(out_path, result=result, aggregates=aggregates)
