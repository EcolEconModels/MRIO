#!/usr/bin/env python3
"""Construct canonical additive flows and validate the single IO contract."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping, cast

import numpy as np
import pandas as pd

from scipy import sparse

from ecolecon_mrio.balancing import kl_balance_linear_equalities
from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.diagnostics import calc_spectral_radius
from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT, load_eurostat_iot
from ecolecon_mrio.eurostat_iot_source_policy import NAIO_10_CP1700_ACCOUNTING_POLICY
from ecolecon_mrio.iot_full_table_validation import validate_eurostat_iot_full_table
from ecolecon_mrio.region_aggregation import _combine_target_supply_extension_sidecars_direct
from ecolecon_mrio.iot_flow_semantics import (
    enforce_final_demand_display_identities,
    flow_use_total,
    rounded_source_control_diagnostic,
    source_atomic_final_demand,
    technical_coefficients,
)

_FRAME_FILES = ("Z", "Y", "V", "M")
_SPLIT_NAMES = {name: f"{name}_split.csv" for name in _FRAME_FILES}
_AGG_NAMES = {name: f"{name}_agg.csv" for name in _FRAME_FILES}
_GENERATED_IDENTITY_RTOL = 1e-12
_GENERATED_IDENTITY_ATOL = 1e-10
_TRANSFORMATION_COHERENCE_REPORT = "transformation_coherence_validation.csv"


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame.astype("float64")


def _read_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one data column in {path}, found {frame.shape[1]}")
    series = frame.iloc[:, 0].astype("float64")
    series.index = series.index.astype(str)
    return series


def _write_series(series: pd.Series, path: Path) -> None:
    out = series.astype("float64").copy()
    out.name = out.name or "value"
    out.to_csv(path, header=True)


def _write_frame(frame: pd.DataFrame, directory: Path, filename: str) -> None:
    frame.to_csv(directory / filename)


def _max_gap(left: pd.DataFrame | pd.Series, right: pd.DataFrame | pd.Series) -> float:
    if type(left) is not type(right):
        raise TypeError("Both objects must be DataFrames or both must be Series")
    if isinstance(left, pd.DataFrame):
        if not left.index.equals(right.index) or not left.columns.equals(right.columns):
            return float("inf")
        if not left.isna().equals(right.isna()):
            return float("inf")
    else:
        if not left.index.equals(right.index):
            return float("inf")
        if not left.isna().equals(right.isna()):
            return float("inf")
    left_values = left.to_numpy(dtype="float64")
    right_values = right.to_numpy(dtype="float64")
    if np.isinf(left_values).any() or np.isinf(right_values).any():
        return float("inf")
    values = left_values - right_values
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else 0.0


def _assert_close(label: str, left: pd.DataFrame | pd.Series, right: pd.DataFrame | pd.Series, *, rtol: float, atol: float) -> float:
    """Require elementwise generated equality under ``atol + rtol * scale``.

    A single global scale can hide a bad small cell beside a very large cell.  The
    generated-table contract is strict but numerical: matching labels/NaN masks are
    required, while ordinary floating reduction noise is compared element by element.
    """
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be finite and non-negative")
    gap = _max_gap(left, right)
    if not np.isfinite(gap):
        raise ValueError(
            f"{label} failed: labels/NaN masks differ or non-finite values are present"
        )
    left_values = left.to_numpy(dtype="float64")
    right_values = right.to_numpy(dtype="float64")
    jointly_missing = np.isnan(left_values) & np.isnan(right_values)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if np.any(~finite & ~jointly_missing):
        raise ValueError(f"{label} failed: non-finite compared values are present")
    abs_gap = np.where(jointly_missing, 0.0, np.abs(left_values - right_values))
    tolerance = atol + rtol * np.maximum(np.abs(left_values), np.abs(right_values))
    tolerance = np.where(jointly_missing, np.inf, tolerance)
    bad = abs_gap > tolerance
    if bool(np.any(bad)):
        normalized = np.zeros_like(abs_gap)
        positive = tolerance > 0.0
        np.divide(abs_gap, tolerance, out=normalized, where=positive)
        normalized[~positive & (abs_gap > 0.0)] = np.inf
        worst = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
        raise ValueError(
            f"{label} failed: max_abs_gap={float(np.max(abs_gap)):.12g}, "
            f"worst_abs_gap={float(abs_gap[worst]):.12g}, "
            f"worst_tolerance={float(tolerance[worst]):.12g}"
        )
    return float(np.max(abs_gap)) if abs_gap.size else 0.0


_ROUNDED_Y_CONTROLS = ("P3", "P5M", "P5", "P6_D0", "P6_U3", "TFU", "TU")


def _resolve_source_rounding_quantum(
    references: Mapping[str, ParsedEurostatIOT],
    override: float | None,
) -> float:
    if override is not None:
        if not np.isfinite(override) or override <= 0.0:
            raise ValueError("--source-rounding-quantum must be finite and positive")
        return float(override)
    inferred = {
        float(reference.meta["source_rounding_quantum"])
        for reference in references.values()
        if reference.meta.get("source_rounding_quantum") is not None
    }
    if len(inferred) != 1:
        raise ValueError(
            "Could not establish one source rounding quantum across DOM/IMP/TOTAL; "
            f"inferred={sorted(inferred)!r}. Pass --source-rounding-quantum explicitly."
        )
    quantum = inferred.pop()
    if not np.isfinite(quantum) or quantum <= 0.0:
        raise ValueError(f"Invalid inferred source rounding quantum: {quantum!r}")
    return quantum


def _derived_y_control(
    reference: ParsedEurostatIOT,
    identity: str,
) -> tuple[pd.Series, pd.Series]:
    Y = reference.Y.astype("float64")
    if identity == "P3":
        detail = ["P3_S13", "P3_S14", "P3_S15"]
        if not set(detail).issubset(Y.columns):
            raise ValueError("P3 published-control validation requires its complete detail partition")
        values = Y.loc[:, detail].sum(axis=1).astype("float64")
        counts = Y.loc[:, detail].notna().sum(axis=1).astype("int64")
        return values, counts
    if identity == "P5":
        if {"P51G", "P52", "P53"}.issubset(Y.columns):
            detail = ["P51G", "P52", "P53"]
        elif {"P51G", "P5M"}.issubset(Y.columns) and not {"P52", "P53"}.intersection(Y.columns):
            detail = ["P51G", "P5M"]
        elif "P5" in source_atomic_final_demand(Y).columns:
            detail = ["P5"]
        else:
            raise ValueError("P5 published-control validation requires one complete recognised investment basis")
        values = Y.loc[:, detail].sum(axis=1).astype("float64")
        counts = Y.loc[:, detail].notna().sum(axis=1).astype("int64")
        return values, counts
    if identity == "P5M":
        if {"P52", "P53"}.issubset(Y.columns):
            detail = ["P52", "P53"]
        elif "P5M" in source_atomic_final_demand(Y).columns:
            detail = ["P5M"]
        else:
            raise ValueError("P5M published-control validation requires P52/P53 detail or atomic nested P5M")
        values = Y.loc[:, detail].sum(axis=1).astype("float64")
        counts = Y.loc[:, detail].notna().sum(axis=1).astype("int64")
        return values, counts
    if identity == "P6_D0":
        values = cast(pd.Series, Y.loc[:, "P6"]).sub(
            cast(pd.Series, Y.loc[:, "P6_B0"])
        ).astype("float64")
        return values, Y.loc[:, ["P6", "P6_B0"]].notna().sum(axis=1).astype("int64")
    if identity == "P6_U3":
        values = cast(pd.Series, Y.loc[:, "P6_B0"]).sub(
            cast(pd.Series, Y.loc[:, "P6_U2"])
        ).astype("float64")
        return values, Y.loc[:, ["P6_B0", "P6_U2"]].notna().sum(axis=1).astype("int64")

    atomic = source_atomic_final_demand(Y)
    values = atomic.sum(axis=1).astype("float64")
    counts = atomic.notna().sum(axis=1).astype("int64")
    if identity == "TFU":
        return values, counts
    if identity == "TU":
        values = reference.Z.sum(axis=1).add(values, fill_value=0.0).astype("float64")
        counts = reference.Z.notna().sum(axis=1).add(counts, fill_value=0).astype("int64")
        return values, counts
    raise ValueError(f"Unsupported rounded Y control: {identity}")


def _derived_source_output(reference: ParsedEurostatIOT) -> tuple[pd.Series, pd.Series]:
    """Derive source output and term counts without exact control validation."""
    atomic = source_atomic_final_demand(reference.Y)
    values = reference.Z.sum(axis=1).add(atomic.sum(axis=1), fill_value=0.0).astype("float64")
    counts = reference.Z.notna().sum(axis=1).add(
        atomic.notna().sum(axis=1), fill_value=0
    ).astype("int64")
    return values, counts


def _validate_rounded_series(
    *,
    flow: str,
    identity: str,
    published: pd.Series,
    derived: pd.Series,
    component_counts: pd.Series,
    source_rounding_quantum: float,
    rtol: float,
    atol: float,
) -> list[dict[str, object]]:
    labels = published.index
    if not derived.index.equals(labels) or not component_counts.index.equals(labels):
        raise ValueError(f"{flow} {identity}: rounded-control index mismatch")
    diagnostics: list[dict[str, object]] = []
    for code in labels.astype(str):
        diagnostic = rounded_source_control_diagnostic(
            identity=identity,
            published=float(published.loc[code]),
            derived=float(derived.loc[code]),
            component_term_count=int(component_counts.loc[code]),
            source_rounding_quantum=source_rounding_quantum,
            rtol=rtol,
            atol=atol,
        )
        diagnostic.update({"stage": "source_control", "flow": flow, "product_code": code})
        diagnostics.append(diagnostic)
    return diagnostics


def _validate_source_ts_bp_controls(
    *, references: Mapping[str, ParsedEurostatIOT], source_rounding_quantum: float,
    rtol: float, atol: float,
) -> list[dict[str, object]]:
    """Validate published TS_BP under its flow-specific source control."""
    diagnostics: list[dict[str, object]] = []
    controls = (
        ("DOM", "DOM_TS_BP_equals_P1", ("TS_BP", "P1"), 1),
        ("TOTAL", "TOTAL_TS_BP_equals_P1_plus_P7", ("TS_BP", "P1", "P7"), 2),
    )
    for flow, identity, rows, component_term_count in controls:
        published, *components = (_source_row(references[flow], row) for row in rows)
        if published is None or any(component is None for component in components):
            continue
        first_component = components[0]
        assert first_component is not None
        derived = first_component.copy()
        for component in components[1:]:
            assert component is not None
            derived = derived.add(component)
        counts = pd.Series(component_term_count, index=published.index, dtype="int64")
        for diagnostic in _validate_rounded_series(
            flow=flow,
            identity=identity,
            published=published,
            derived=derived,
            component_counts=counts,
            source_rounding_quantum=source_rounding_quantum,
            rtol=rtol,
            atol=atol,
        ):
            diagnostic.update({"stage": "source_supply_control", "source_identity": identity})
            diagnostics.append(diagnostic)
    return diagnostics


def _derived_final_demand_residual_rows(
    reference: ParsedEurostatIOT, *, identity: str
) -> set[str]:
    """Return source rows whose display residual was reconstructed during parsing."""
    raw = reference.meta.get("derived_final_demand_residual_cells", [])
    if raw is None:
        return set()
    if not isinstance(raw, list):
        raise ValueError("derived_final_demand_residual_cells metadata must be a list.")
    rows: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("derived final-demand residual metadata entries must be mappings.")
        if str(item.get("column", "")) != str(identity):
            continue
        row = str(item.get("row", "")).strip()
        if not row:
            raise ValueError("derived final-demand residual metadata contains a blank row.")
        rows.add(row)
    return rows

def _validate_y_source_reconstruction(
    *,
    flow: str,
    reconstructed: pd.DataFrame,
    reference: ParsedEurostatIOT,
    source_rounding_quantum: float,
    rtol: float,
    atol: float,
) -> list[dict[str, object]]:
    common_controls = [
        code for code in _ROUNDED_Y_CONTROLS
        if code in reference.Y.columns and code in reconstructed.columns
    ]
    ordinary_columns = [
        code for code in reference.Y.columns if str(code) not in set(common_controls)
    ]
    _assert_close(
        f"{flow} ordinary Y reconstruction",
        reconstructed.loc[:, ordinary_columns],
        reference.Y.loc[:, ordinary_columns],
        rtol=rtol,
        atol=atol,
    )
    diagnostics: list[dict[str, object]] = []
    for identity in common_controls:
        derived, counts = _derived_y_control(reference, identity)
        _assert_close(
            f"{flow} generated {identity} reconstruction",
            reconstructed.loc[:, identity],
            derived,
            rtol=rtol,
            atol=atol,
        )
        parser_derived_rows = _derived_final_demand_residual_rows(
            reference, identity=identity
        )
        published_rows = [
            str(code)
            for code in reference.Y.index.astype(str)
            if str(code) not in parser_derived_rows
        ]
        if published_rows:
            diagnostics.extend(
                _validate_rounded_series(
                    flow=flow,
                    identity=identity,
                    published=reference.Y.loc[published_rows, identity].astype("float64"),
                    derived=derived.loc[published_rows],
                    component_counts=counts.loc[published_rows],
                    source_rounding_quantum=source_rounding_quantum,
                    rtol=rtol,
                    atol=atol,
                )
            )
    return diagnostics


def _manifest_children(value: object) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    raise TypeError(f"Unexpected exio_selected_sectors value: {type(value).__name__}")


def _load_equal_manifests(dom_dir: Path, imp_dir: Path) -> dict[str, object]:
    paths = [dom_dir / "exio_selected_sectors.json", imp_dir / "exio_selected_sectors.json"]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("DOM and IMP must both contain exio_selected_sectors.json")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    topology = {str(parent): set(_manifest_children(children)) for parent, children in manifests[0].items()}
    other = {str(parent): set(_manifest_children(children)) for parent, children in manifests[1].items()}
    if topology != other:
        raise ValueError("DOM and IMP split-child topology differs; additive TOTAL is not well-defined")
    return manifests[0]


def _child_to_parent(manifest: Mapping[str, object], source_codes: Iterable[str]) -> dict[str, str]:
    source_set = set(map(str, source_codes))
    mapping = {code: code for code in source_set}
    for parent, children in manifest.items():
        for child in _manifest_children(children):
            previous = mapping.get(child)
            if previous is not None and previous != str(parent) and child in source_set:
                raise ValueError(f"Split child {child} collides with unsplit source code {previous}")
            mapping[child] = str(parent)
    return mapping


def _aggregate_square(frame: pd.DataFrame, child_parent: Mapping[str, str], source_codes: pd.Index) -> pd.DataFrame:
    rows = pd.Index([child_parent.get(str(code), str(code)) for code in frame.index])
    cols = pd.Index([child_parent.get(str(code), str(code)) for code in frame.columns])
    row_agg = cast(pd.DataFrame, frame.groupby(rows, sort=False).sum())
    col_agg = cast(pd.DataFrame, row_agg.T.groupby(cols, sort=False).sum().T)
    expected = set(source_codes.astype(str))
    if set(col_agg.index.astype(str)) != expected or set(col_agg.columns.astype(str)) != expected:
        raise ValueError(
            "Split reconstruction produced missing/orphan parent sectors: "
            f"rows_extra={sorted(set(col_agg.index.astype(str)) - expected)!r}, "
            f"rows_missing={sorted(expected - set(col_agg.index.astype(str)))!r}, "
            f"cols_extra={sorted(set(col_agg.columns.astype(str)) - expected)!r}, "
            f"cols_missing={sorted(expected - set(col_agg.columns.astype(str)))!r}."
        )
    return cast(pd.DataFrame, col_agg.reindex(index=source_codes, columns=source_codes))


def _aggregate_rows(frame: pd.DataFrame, child_parent: Mapping[str, str], source_codes: pd.Index) -> pd.DataFrame:
    groups = pd.Index([child_parent.get(str(code), str(code)) for code in frame.index])
    grouped = cast(pd.DataFrame, frame.groupby(groups, sort=False).sum())
    expected = set(source_codes.astype(str))
    if set(grouped.index.astype(str)) != expected:
        raise ValueError(
            "Row reconstruction produced missing/orphan parent sectors: "
            f"extra={sorted(set(grouped.index.astype(str)) - expected)!r}, "
            f"missing={sorted(expected - set(grouped.index.astype(str)))!r}."
        )
    return cast(pd.DataFrame, grouped.reindex(index=source_codes))


def _aggregate_columns(frame: pd.DataFrame, child_parent: Mapping[str, str], source_codes: pd.Index) -> pd.DataFrame:
    groups = pd.Index([child_parent.get(str(code), str(code)) for code in frame.columns])
    grouped = cast(
        pd.DataFrame,
        frame.T.groupby(groups, sort=False).sum(min_count=1).T,
    )
    expected = set(source_codes.astype(str))
    if set(grouped.columns.astype(str)) != expected:
        raise ValueError(
            "Column reconstruction produced missing/orphan parent sectors: "
            f"extra={sorted(set(grouped.columns.astype(str)) - expected)!r}, "
            f"missing={sorted(expected - set(grouped.columns.astype(str)))!r}."
        )
    return cast(pd.DataFrame, grouped.reindex(columns=source_codes))



def _load_dom_split_output_shares(
    dom_dir: Path, *, child_parent: Mapping[str, str], children: pd.Index
) -> pd.Series:
    """Return DOM EXIO producer-output shares on the canonical child axis."""
    frame = pd.read_csv(dom_dir / "split_output_shares.csv")
    required = {"parent", "child", "output_share"}
    if set(frame.columns) != required or frame["child"].astype(str).duplicated().any():
        raise ValueError("Invalid DOM split_output_shares.csv schema or duplicate child.")
    frame = frame.assign(
        parent=frame["parent"].astype(str),
        child=frame["child"].astype(str),
        output_share=pd.to_numeric(frame["output_share"], errors="coerce"),
    )
    children = pd.Index(children.astype(str))
    shares = pd.Series(1.0, index=children, dtype="float64")
    split_parents = {
        parent
        for parent in dict.fromkeys(child_parent.get(str(c), str(c)) for c in children)
        if sum(child_parent.get(str(c), str(c)) == parent for c in children) > 1
    }
    for parent in split_parents:
        group = [str(c) for c in children if child_parent.get(str(c), str(c)) == parent]
        rows = frame.loc[frame["parent"].eq(parent)].set_index("child")
        if set(rows.index.astype(str)) != set(group):
            raise ValueError(f"Incomplete DOM output shares for split parent {parent!r}.")
        values = rows["output_share"].reindex(group).astype("float64")
        if (
            not np.isfinite(values.to_numpy()).all()
            or bool((values < 0.0).any())
            or not np.isclose(float(values.sum()), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(f"Invalid DOM output shares for split parent {parent!r}.")
        shares.loc[group] = values.to_numpy()
    extra = set(frame["child"]) - set(children)
    if extra:
        raise ValueError(f"DOM output shares contain non-canonical children: {sorted(extra)!r}.")
    return shares



# PRODUCER_TECH_RECONCILIATION_V2
def _ordered_parent_groups(
    children: pd.Index, child_parent: Mapping[str, str]
) -> tuple[pd.Index, dict[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for child in children.astype(str):
        parent = child_parent.get(str(child), str(child))
        groups.setdefault(parent, []).append(str(child))
    return pd.Index(list(groups), dtype="object"), groups


def _output_conditioned_technology_prior(
    Z: pd.DataFrame,
    *,
    child_parent: Mapping[str, str],
    dom_output_share: pd.Series,
    x_domestic: pd.Series,
    flow: str,
) -> pd.DataFrame:
    """Build a prior proportional within each parent block to donor A times target x."""
    children = pd.Index(Z.index.astype(str))
    if not Z.index.equals(Z.columns):
        raise ValueError(f"{flow} technology prior requires square aligned Z.")
    _, groups = _ordered_parent_groups(children, child_parent)
    scale = pd.Series(1.0, index=children, dtype="float64")
    # Remove donor production scale column by column: Z^E_ab * x^AT_b / s^x,E_b
    # is proportional within each parent to A^E_ab * x^AT_b.
    for parent, group in groups.items():
        if len(group) <= 1:
            continue
        output = dom_output_share.reindex(group).astype("float64")
        x = x_domestic.reindex(group).astype("float64").mask(
            lambda values: values.abs() <= _GENERATED_IDENTITY_ATOL, 0.0
        )
        bad = x.gt(_GENERATED_IDENTITY_ATOL) & output.le(_GENERATED_IDENTITY_ATOL)
        if bool(bad.any()):
            bad_children = list(map(str, bad.index[bad]))
            raise ValueError(
                "Positive target output has zero DOM EXIO output support while constructing "
                f"the {flow} technology prior: parent={parent!r}, children={bad_children!r}."
            )
        scale.loc[group] = x.div(
            output.where(output.gt(_GENERATED_IDENTITY_ATOL))
        ).fillna(0.0)
    prior = Z.astype("float64").clip(lower=0.0).mul(scale, axis=1)
    prior = prior.mask(prior.abs() <= _GENERATED_IDENTITY_ATOL, 0.0)
    if not np.isfinite(prior.to_numpy(dtype="float64")).all():
        raise ValueError(f"{flow} technology prior contains non-finite values.")
    return prior


# WHOLE_IOT_PRODUCTION_V_RECONCILIATION_V2
def _signed_parent_magnitude_prior(
    *,
    published: pd.Series,
    proxy: pd.Series,
    child_parent: Mapping[str, str],
    children: pd.Index,
    row_code: str,
) -> tuple[pd.Series, pd.Series]:
    """Return a sign-preserving non-negative child magnitude prior.

    The source parent sign is evidence; absent child-specific evidence the split
    must not invent offsetting tax/subsidy or positive/negative account entries
    inside one parent merely to make the monetary table balance.  The non-negative
    magnitude is therefore the KL variable and the parent sign enters the linear
    accounting constraint separately.
    """
    prior = pd.Series(0.0, index=children, dtype="float64")
    signs = pd.Series(0.0, index=children, dtype="float64")
    _, groups = _ordered_parent_groups(children, child_parent)
    for parent, group in groups.items():
        if parent not in published.index:
            raise ValueError(f"Source {row_code} is missing producer parent {parent!r}.")
        value = float(published.loc[parent])
        if not np.isfinite(value):
            raise ValueError(f"Source {row_code} is non-finite for parent {parent!r}.")
        if abs(value) <= _GENERATED_IDENTITY_ATOL:
            continue
        weights = proxy.reindex(group).astype("float64")
        if not np.isfinite(weights.to_numpy(dtype="float64")).all() or bool((weights < 0.0).any()):
            raise ValueError(
                f"Invalid {row_code} child-allocation prior for parent {parent!r}: "
                f"proxy={weights.to_dict()!r}."
            )
        denominator = float(weights.sum())
        if denominator <= _GENERATED_IDENTITY_ATOL:
            raise ValueError(
                f"No structural support for nonzero {row_code} parent {parent!r}: "
                f"published={value:.12g}."
            )
        prior.loc[group] = weights.to_numpy(dtype="float64") * (abs(value) / denominator)
        signs.loc[group] = 1.0 if value > 0.0 else -1.0
    return prior, signs


def _append_linear_constraint(
    rows: list[int],
    columns: list[int],
    values: list[float],
    targets: list[float],
    entries: Iterable[tuple[int, float]],
    target: float,
) -> None:
    """Append one sparse equality row, omitting explicit zero coefficients."""
    row = len(targets)
    for column, value in entries:
        coefficient = float(value)
        if coefficient == 0.0:
            continue
        rows.append(row)
        columns.append(int(column))
        values.append(coefficient)
    targets.append(float(target))


def _reconcile_iot_intermediate_transactions(
    Z_dom: pd.DataFrame,
    Z_imp: pd.DataFrame,
    *,
    child_parent: Mapping[str, str],
    dom_output_share: pd.Series,
    x_domestic: pd.Series,
    reference_total: ParsedEurostatIOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame, list[dict[str, object]]]:
    """Jointly reconcile DOM/IMP Z and the production-account V closure.

    Atomic final demand remains fixed in Step 1, so DOM row use fixes child output.
    EXIO supplies only the non-negative transaction prior.  The hard source
    equalities are flow-specific child intermediate-sales rows, every Eurostat
    parent Z cell (one algebraically redundant block per supplier parent is omitted
    from the numerical system but checked afterwards), and the parent D21X31
    control.  At child level the exact production account is

        u_b + D21X31_b + B1G_b = x_b.

    D21X31 and B1G enter as sign-preserving non-negative magnitudes.  This removes
    the former proportional-D21 capacity formula: child GVA is now allocated in
    the same information projection instead of appearing only as a non-negativity
    boundary.  Published parent B1G is independently rounded source evidence; its
    generated aggregate is implied by P1, Z and D21X31 and is validated later by
    the source-rounding contract rather than redundantly forced as another exact
    equality.
    """
    children = pd.Index(Z_dom.index.astype(str))
    if not Z_dom.index.equals(Z_dom.columns):
        raise ValueError("DOM Z must be square before joint IOT reconciliation.")
    if not Z_imp.index.equals(children) or not Z_imp.columns.equals(children):
        raise ValueError("DOM and IMP child topology/order differs before joint IOT reconciliation.")
    n = len(children)
    parent_axis, groups = _ordered_parent_groups(children, child_parent)
    p = len(parent_axis)
    x = x_domestic.reindex(children).astype("float64")
    if not np.isfinite(x.to_numpy(dtype="float64")).all() or bool((x < -_GENERATED_IDENTITY_ATOL).any()):
        raise ValueError("Target domestic output must be finite and non-negative.")
    x = x.mask(x.abs() <= _GENERATED_IDENTITY_ATOL, 0.0)

    block_targets = {
        "DOM": _aggregate_square(Z_dom, child_parent, parent_axis),
        "IMP": _aggregate_square(Z_imp, child_parent, parent_axis),
    }
    row_targets = {
        "DOM": Z_dom.sum(axis=1).astype("float64"),
        "IMP": Z_imp.sum(axis=1).astype("float64"),
    }
    priors = {
        "DOM": _output_conditioned_technology_prior(
            Z_dom,
            child_parent=child_parent,
            dom_output_share=dom_output_share,
            x_domestic=x,
            flow="DOM",
        ),
        "IMP": _output_conditioned_technology_prior(
            Z_imp,
            child_parent=child_parent,
            dom_output_share=dom_output_share,
            x_domestic=x,
            flow="IMP",
        ),
    }

    source_d21 = _source_row(reference_total, "D21X31")
    source_b1g = _source_row(reference_total, "B1G")
    if source_d21 is None or source_b1g is None:
        # Non-production-account fixtures/datasets still receive the source-cell
        # Z reconciliation, but must not claim the production-V method downstream.
        production_v: pd.DataFrame | None = None
        d21_prior = pd.Series(0.0, index=children, dtype="float64")
        d21_sign = pd.Series(0.0, index=children, dtype="float64")
        b1g_prior = pd.Series(0.0, index=children, dtype="float64")
        b1g_sign = pd.Series(0.0, index=children, dtype="float64")
        production_account_available = False
    else:
        production_account_available = True
        preliminary_columns = Z_dom.add(Z_imp).sum(axis=0).reindex(children).astype("float64")
        # Product-tax/subsidy evidence follows the existing intermediate-input
        # proxy, but is now a soft child prior rather than a hard proportional rule.
        d21_prior, d21_sign = _signed_parent_magnitude_prior(
            published=source_d21,
            proxy=preliminary_columns.where(x.gt(0.0), 0.0),
            child_parent=child_parent,
            children=children,
            row_code="D21X31",
        )
        # The exact generated parent GVA is implied by P1 - Z - D21X31.  Use that
        # implied total for support/scale and the published B1G only as independent
        # rounded source validation.  This avoids fabricating epsilon support when
        # a published 0.00 is merely a rounded small balance.
        implied_parent_b1g = pd.Series(0.0, index=parent_axis, dtype="float64")
        for parent, group in groups.items():
            implied_parent_b1g.loc[parent] = (
                float(x.reindex(group).sum())
                - float(Z_dom.add(Z_imp).loc[:, group].to_numpy(dtype="float64").sum())
                - float(source_d21.loc[parent])
            )
        b1g_prior, b1g_sign = _signed_parent_magnitude_prior(
            published=implied_parent_b1g,
            proxy=x,
            child_parent=child_parent,
            children=children,
            row_code="implied_B1G",
        )
        production_v = pd.DataFrame(index=pd.Index(["D21X31", "B1G"]), columns=children, dtype="float64")

    parent_id = {str(parent): i for i, parent in enumerate(parent_axis.astype(str))}
    child_parent_ids = np.array(
        [parent_id[child_parent.get(str(child), str(child))] for child in children],
        dtype=np.int64,
    )
    parent_children = {
        i: np.flatnonzero(child_parent_ids == i).astype(np.int64)
        for i in range(p)
    }

    dom_offset = 0
    imp_offset = n * n
    d21_offset = 2 * n * n
    b1g_offset = d21_offset + n
    variable_count = b1g_offset + n
    prior = np.concatenate(
        [
            priors["DOM"].to_numpy(dtype="float64").reshape(-1),
            priors["IMP"].to_numpy(dtype="float64").reshape(-1),
            d21_prior.to_numpy(dtype="float64"),
            b1g_prior.to_numpy(dtype="float64"),
        ]
    )

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    targets: list[float] = []

    def z_index(offset: int, row: int, column: int) -> int:
        return offset + row * n + column

    # Exact child intermediate-sales rows for each flow.
    for flow, offset in (("DOM", dom_offset), ("IMP", imp_offset)):
        for row in range(n):
            _append_linear_constraint(
                matrix_rows,
                matrix_columns,
                matrix_values,
                targets,
                ((z_index(offset, row, column), 1.0) for column in range(n)),
                float(row_targets[flow].iloc[row]),
            )

    # Parent Z cells are also exact.  Within each supplier parent, the final buyer
    # block is implied by the child-row totals plus all preceding buyer blocks, so
    # omit that one numerical row to avoid a redundant dual equality.  Post-solve
    # validation still checks every source parent cell.
    if p > 1:
        for flow, offset in (("DOM", dom_offset), ("IMP", imp_offset)):
            for supplier_parent in range(p):
                supplier_rows = parent_children[supplier_parent]
                for buyer_parent in range(p - 1):
                    buyer_columns = parent_children[buyer_parent]
                    _append_linear_constraint(
                        matrix_rows,
                        matrix_columns,
                        matrix_values,
                        targets,
                        (
                            (z_index(offset, int(row), int(column)), 1.0)
                            for row in supplier_rows
                            for column in buyer_columns
                        ),
                        float(block_targets[flow].iloc[supplier_parent, buyer_parent]),
                    )

    if production_account_available:
        assert source_d21 is not None
        # D21X31 is a primitive parent source control.  Its child magnitudes are
        # inferred jointly but aggregate exactly to the published parent amount.
        for parent_position, parent in enumerate(parent_axis.astype(str)):
            child_positions = parent_children[parent_position]
            _append_linear_constraint(
                matrix_rows,
                matrix_columns,
                matrix_values,
                targets,
                ((d21_offset + int(column), 1.0) for column in child_positions),
                abs(float(source_d21.loc[parent])),
            )

        # Child production account: raw DOM+IMP intermediate inputs, signed
        # D21X31 and signed B1G must close exactly to fixed Step-1 output.
        for column in range(n):
            entries: list[tuple[int, float]] = [
                (z_index(dom_offset, row, column), 1.0) for row in range(n)
            ]
            entries.extend(
                (z_index(imp_offset, row, column), 1.0) for row in range(n)
            )
            entries.extend(
                [
                    (d21_offset + column, float(d21_sign.iloc[column])),
                    (b1g_offset + column, float(b1g_sign.iloc[column])),
                ]
            )
            _append_linear_constraint(
                matrix_rows,
                matrix_columns,
                matrix_values,
                targets,
                entries,
                float(x.iloc[column]),
            )

    constraint_matrix = sparse.coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(targets), variable_count),
        dtype="float64",
    ).tocsr()
    solution, stats = kl_balance_linear_equalities(
        prior,
        constraint_matrix,
        np.asarray(targets, dtype="float64"),
        rtol=_GENERATED_IDENTITY_RTOL,
        atol=_GENERATED_IDENTITY_ATOL,
    )
    flat = solution.reshape(-1)
    reconciled = {
        "DOM": pd.DataFrame(
            flat[dom_offset:imp_offset].reshape(n, n),
            index=children,
            columns=children,
        ),
        "IMP": pd.DataFrame(
            flat[imp_offset:d21_offset].reshape(n, n),
            index=children,
            columns=children,
        ),
    }

    checks: list[dict[str, object]] = []
    for flow in ("DOM", "IMP"):
        row_gap = _assert_close(
            f"{flow} reconciled child-row margin",
            reconciled[flow].sum(axis=1),
            row_targets[flow],
            rtol=_GENERATED_IDENTITY_RTOL,
            atol=_GENERATED_IDENTITY_ATOL,
        )
        parent_gap = _assert_close(
            f"{flow} reconciled Eurostat parent Z cells",
            _aggregate_square(reconciled[flow], child_parent, parent_axis),
            block_targets[flow],
            rtol=_GENERATED_IDENTITY_RTOL,
            atol=_GENERATED_IDENTITY_ATOL,
        )
        checks.extend(
            [
                {
                    "validation_class": "transformation_coherence",
                    "check_id": "tc.child_row_margin",
                    "flow": flow,
                    "status": "pass",
                    "max_abs_gap": row_gap,
                    "rtol": _GENERATED_IDENTITY_RTOL,
                    "atol": _GENERATED_IDENTITY_ATOL,
                },
                {
                    "validation_class": "transformation_coherence",
                    "check_id": "tc.parent_z_cell",
                    "flow": flow,
                    "status": "pass",
                    "max_abs_gap": parent_gap,
                    "rtol": _GENERATED_IDENTITY_RTOL,
                    "atol": _GENERATED_IDENTITY_ATOL,
                },
            ]
        )

    total_columns = reconciled["DOM"].sum(axis=0).add(reconciled["IMP"].sum(axis=0))
    zero_children = [str(child) for child in children if float(x.loc[child]) == 0.0]
    zero_gap = 0.0
    if zero_children:
        zero_inputs = total_columns.reindex(zero_children).astype("float64")
        zero_gap = _assert_close(
            "zero-output producer columns",
            zero_inputs,
            pd.Series(0.0, index=zero_inputs.index, dtype="float64"),
            rtol=0.0,
            atol=_GENERATED_IDENTITY_ATOL,
        )

    production_gap = 0.0
    d21_parent_gap = 0.0
    b1g_violation = 0.0
    generated_b1g = pd.Series(np.nan, index=children, dtype="float64")
    if production_account_available:
        assert source_d21 is not None and source_b1g is not None and production_v is not None
        d21_magnitude = pd.Series(flat[d21_offset:b1g_offset], index=children, dtype="float64")
        b1g_magnitude = pd.Series(flat[b1g_offset:], index=children, dtype="float64")
        generated_d21 = d21_magnitude.mul(d21_sign)
        auxiliary_b1g = b1g_magnitude.mul(b1g_sign)
        generated_b1g = x - total_columns - generated_d21
        production_gap = _assert_close(
            "joint production-account B1G auxiliary",
            generated_b1g,
            auxiliary_b1g,
            rtol=_GENERATED_IDENTITY_RTOL,
            atol=_GENERATED_IDENTITY_ATOL,
        )
        production_v.loc["D21X31"] = generated_d21
        production_v.loc["B1G"] = generated_b1g
        d21_parent_gap = _assert_close(
            "joint D21X31 parent reconstruction",
            _aggregate_columns(
                production_v.loc[["D21X31"]], child_parent, source_d21.index.astype(str)
            ).loc["D21X31"],
            source_d21,
            rtol=_GENERATED_IDENTITY_RTOL,
            atol=_GENERATED_IDENTITY_ATOL,
        )
        constrained_children = [
            child
            for parent, group in groups.items()
            if float(source_b1g.loc[parent]) >= -_GENERATED_IDENTITY_ATOL
            for child in group
        ]
        if constrained_children:
            negatives = (-generated_b1g.reindex(constrained_children)).clip(lower=0.0)
            b1g_violation = float(negatives.max())
            if b1g_violation > _GENERATED_IDENTITY_ATOL:
                child = str(negatives.idxmax())
                raise ValueError(
                    "Joint production-account reconciliation generated negative child B1G "
                    f"under a non-negative source parent: child={child!r}, "
                    f"B1G={float(generated_b1g.loc[child]):.12g}."
                )
        checks.extend(
            [
                {
                    "validation_class": "transformation_coherence",
                    "check_id": "tc.production_account_identity",
                    "flow": "TOTAL",
                    "status": "pass",
                    "max_abs_gap": production_gap,
                    "rtol": _GENERATED_IDENTITY_RTOL,
                    "atol": _GENERATED_IDENTITY_ATOL,
                },
                {
                    "validation_class": "transformation_coherence",
                    "check_id": "tc.parent_d21x31",
                    "flow": "TOTAL",
                    "status": "pass",
                    "max_abs_gap": d21_parent_gap,
                    "rtol": _GENERATED_IDENTITY_RTOL,
                    "atol": _GENERATED_IDENTITY_ATOL,
                },
                {
                    "validation_class": "transformation_coherence",
                    "check_id": "tc.nonnegative_child_b1g",
                    "flow": "TOTAL",
                    "status": "pass",
                    "max_abs_gap": b1g_violation,
                    "rtol": 0.0,
                    "atol": _GENERATED_IDENTITY_ATOL,
                },
                {
                    "validation_class": "transformation_coherence",
                    "check_id": "tc.zero_target_output_column",
                    "flow": "TOTAL",
                    "status": "pass",
                    "max_abs_gap": zero_gap,
                    "rtol": 0.0,
                    "atol": _GENERATED_IDENTITY_ATOL,
                },
            ]
        )

    checks.append(
        {
            "validation_class": "transformation_coherence",
            "check_id": "tc.kl_convergence",
            "flow": "TOTAL",
            "status": "pass",
            "max_abs_gap": stats.max_abs_gap,
            "rtol": _GENERATED_IDENTITY_RTOL,
            "atol": _GENERATED_IDENTITY_ATOL,
        }
    )

    diagnostic_records: list[dict[str, object]] = []
    before_dom = Z_dom.sum(axis=0)
    before_imp = Z_imp.sum(axis=0)
    for parent, group in groups.items():
        published_d21 = float("nan") if source_d21 is None else float(source_d21.loc[parent])
        published_b1g = float("nan") if source_b1g is None else float(source_b1g.loc[parent])
        generated_parent_b1g = float(generated_b1g.reindex(group).sum()) if production_account_available else float("nan")
        for child in group:
            diagnostic_records.append(
                {
                    "buyer_parent": parent,
                    "buyer_child": child,
                    "domestic_output": float(x.loc[child]),
                    "published_parent_D21X31": published_d21,
                    "published_parent_B1G": published_b1g,
                    "generated_parent_B1G": generated_parent_b1g,
                    "before_DOM_column_input": float(before_dom.loc[child]),
                    "before_IMP_column_input": float(before_imp.loc[child]),
                    "after_DOM_column_input": float(reconciled["DOM"].sum(axis=0).loc[child]),
                    "after_IMP_column_input": float(reconciled["IMP"].sum(axis=0).loc[child]),
                    "after_TOTAL_column_input": float(total_columns.loc[child]),
                    "joint_D21X31": (
                        float(production_v.loc["D21X31", child])
                        if production_v is not None else float("nan")
                    ),
                    "generated_B1G": float(generated_b1g.loc[child]),
                    "kl_iterations": stats.iterations,
                    "kl_max_normalized_gap": stats.max_normalized_gap,
                }
            )
    return (
        reconciled["DOM"],
        reconciled["IMP"],
        production_v,
        pd.DataFrame(diagnostic_records),
        checks,
    )

_V_OUTPUT_PRIMITIVES = {"D1", "D11", "D29X39", "P51C", "B3G"}
_SIGNED_INVESTMENT_ATOMS = frozenset({"P5", "P5M", "P52", "P53"})


def _allocate_source_parent_vector(
    *, published: pd.Series, child_parent: Mapping[str, str], children: pd.Index,
    proxy: pd.Series, flow: str, row_code: str,
) -> pd.Series:
    """Allocate one signed primitive parent vector with non-negative child weights."""
    out = pd.Series(0.0, index=children, dtype="float64")
    for parent in published.index.astype(str):
        child_list = [str(child) for child in children if child_parent.get(str(child), str(child)) == parent]
        value = float(published.loc[parent])
        if not child_list:
            raise ValueError(f"No emitted children for V parent {parent!r}")
        if np.isnan(value):
            out.loc[child_list] = np.nan
            continue
        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite primitive V value: flow={flow!r}, "
                f"row_code={row_code!r}, parent={parent!r}, value={value!r}."
            )
        if len(child_list) == 1:
            out.loc[child_list[0]] = value
            continue
        if value == 0.0:
            continue
        weights = proxy.reindex(child_list).astype("float64")
        if not np.isfinite(weights.to_numpy(dtype="float64")).all() or bool((weights < 0.0).any()):
            raise ValueError(
                f"Invalid V allocation proxy: flow={flow!r}, row_code={row_code!r}, "
                f"parent={parent!r}, proxy={weights.to_dict()!r}."
            )
        denominator = float(weights.sum())
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                f"No usable V allocation proxy: flow={flow!r}, row_code={row_code!r}, "
                f"parent={parent!r}, children={child_list!r}, published_value={value!r}, "
                f"proxy_denominator={denominator!r}."
            )
        out.loc[child_list] = weights.to_numpy() * (value / denominator)
    return out


def _source_row(reference: ParsedEurostatIOT, row: str) -> pd.Series | None:
    """Return one finite SDMX primitive row, or None when the row is unavailable."""
    if reference.V is None or row not in reference.V.index:
        return None
    values = reference.V.loc[row].astype("float64")
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError(f"Source V row {row!r} contains non-finite product values")
    return values


def _allocate_import_partition(
    *, aggregate: pd.Series, published_part: pd.Series, published_total: pd.Series,
    child_parent: Mapping[str, str], children: pd.Index, row_code: str,
) -> pd.Series:
    """Apply source geographic proportions to transaction-derived imported supply."""
    out = pd.Series(0.0, index=children, dtype="float64")
    for parent in published_total.index.astype(str):
        child_list = [str(c) for c in children if child_parent.get(str(c), str(c)) == parent]
        total = float(published_total.loc[parent])
        part = float(published_part.loc[parent])
        if not np.isfinite(total) or not np.isfinite(part) or part < 0.0 or part > total:
            raise ValueError(
                f"{row_code} must be a non-negative partition of P7 for {parent}: "
                f"part={part!r}, total={total!r}."
            )
        derived = aggregate.reindex(child_list).astype("float64")
        if total == 0.0:
            if abs(part) > 0.0:
                raise ValueError(f"{row_code} has a nonzero component under zero source P7 for {parent}")
            continue
        out.loc[child_list] = derived.to_numpy() * (part / total)
    return out


def _construct_transaction_v(
    *, flow: str, reference: ParsedEurostatIOT, child_parent: Mapping[str, str],
    Z: pd.DataFrame, Z_total: pd.DataFrame, Z_imp: pd.DataFrame,
    x_domestic: pd.Series, import_supply_proxy: pd.Series,
    primitive_reference: ParsedEurostatIOT | None = None,
    production_account_v: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Construct V from primitive observations and exact transaction identities."""
    if reference.V is None:
        return None, pd.DataFrame()
    source = reference.V.copy().astype("float64")
    source.index = source.index.astype(str)
    source.columns = source.columns.astype(str)
    primitive = reference if primitive_reference is None else primitive_reference
    primitive_v = None if primitive.V is None else primitive.V.astype("float64")
    children = pd.Index(Z.columns.astype(str))
    V = pd.DataFrame(np.nan, index=source.index, columns=children, dtype="float64")
    diagnostics: list[dict[str, object]] = []
    output_proxy = x_domestic.reindex(children).astype("float64")
    input_proxy = Z_total.sum(axis=0).reindex(children).astype("float64")

    # Primitive production rows retain their SDMX parent totals while using the
    # transaction-derived domestic output distribution inside each parent.
    for row in sorted(_V_OUTPUT_PRIMITIVES.intersection(source.index)):
        if primitive_v is None or row not in primitive_v.index:
            raise ValueError(f"Authoritative primitive V row {row!r} is unavailable")
        allocated = _allocate_source_parent_vector(
            published=primitive_v.loc[row], child_parent=child_parent, children=children,
            proxy=output_proxy, flow=flow, row_code=row,
        )
        V.loc[row] = allocated
        reconstructed = _aggregate_columns(
            allocated.rename(row).to_frame().T, child_parent, primitive_v.columns
        ).loc[row]
        gap = _assert_close(
            f"{flow} primitive V reconstruction {row}",
            reconstructed,
            primitive_v.loc[row],
            rtol=0.0,
            atol=1e-9,
        )
        diagnostics.append({
            "flow": flow, "row": row, "role": "primitive_output_weighted",
            "max_abs_gap": gap,
        })

    # D21X31 is a production-account primitive.  On the canonical path its child
    # allocation comes from the same KL reconciliation as Z; standalone fixtures
    # without that transient result retain the source-conserving proxy fallback.
    if "D21X31" in source.index:
        if primitive_v is None or "D21X31" not in primitive_v.index:
            raise ValueError("Authoritative primitive V row 'D21X31' is unavailable")
        if production_account_v is not None:
            if "D21X31" not in production_account_v.index:
                raise ValueError("Joint production-account result is missing D21X31")
            allocated_d21 = production_account_v.loc["D21X31"].reindex(children).astype("float64")
            role = "joint_production_account"
        else:
            allocated_d21 = _allocate_source_parent_vector(
                published=primitive_v.loc["D21X31"], child_parent=child_parent, children=children,
                proxy=input_proxy, flow=flow, row_code="D21X31",
            )
            role = "primitive_input_weighted_fallback"
        V.loc["D21X31"] = allocated_d21
        reconstructed_d21 = _aggregate_columns(
            allocated_d21.rename("D21X31").to_frame().T,
            child_parent,
            primitive_v.columns,
        ).loc["D21X31"]
        d21_gap = _assert_close(
            f"{flow} primitive V reconstruction D21X31",
            reconstructed_d21,
            primitive_v.loc["D21X31"],
            rtol=_GENERATED_IDENTITY_RTOL if production_account_v is not None else 0.0,
            atol=_GENERATED_IDENTITY_ATOL if production_account_v is not None else 1e-9,
        )
        diagnostics.append({
            "flow": flow, "row": "D21X31", "role": role,
            "max_abs_gap": d21_gap,
        })

    # Exact transaction identities are populated only where the source flow
    # exposes the corresponding row; IMP never receives a production account.
    if "CPA_TOTAL" in source.index:
        V.loc["CPA_TOTAL"] = Z.sum(axis=0).reindex(children)
    if "P1" in source.index:
        V.loc["P1"] = x_domestic.reindex(children) if flow in {"DOM", "TOTAL"} else 0.0
    if "IMP" in source.index:
        V.loc["IMP"] = Z_imp.sum(axis=0).reindex(children)

    # P7 is imports-CIF supply by product, not the IMP transaction-row-use vector.
    # Preserve each published parent P7 and use a non-negative transaction proxy
    # only to distribute that supply control among newly emitted child products.
    source_p7 = _source_row(reference, "P7")
    allocated_p7: pd.Series | None = None
    if source_p7 is not None and "P7" in source.index:
        allocated_p7 = _allocate_source_parent_vector(
            published=source_p7,
            child_parent=child_parent,
            children=children,
            proxy=import_supply_proxy,
            flow=flow,
            row_code="P7",
        )
        V.loc["P7"] = allocated_p7

    # Geographic P7 rows remain supply-side partitions of the allocated P7 vector.
    if (
        source_p7 is not None
        and allocated_p7 is not None
        and "P7_B0" in source.index
        and "P7_D0" in source.index
    ):
        b0 = _allocate_import_partition(
            aggregate=allocated_p7, published_part=source.loc["P7_B0"],
            published_total=source_p7, child_parent=child_parent, children=children,
            row_code="P7_B0",
        )
        V.loc["P7_B0"] = b0
        V.loc["P7_D0"] = allocated_p7 - b0
        if "P7_U2" in source.index and "P7_U3" in source.index:
            u2 = _allocate_import_partition(
                aggregate=b0, published_part=source.loc["P7_U2"],
                published_total=source.loc["P7_B0"], child_parent=child_parent,
                children=children, row_code="P7_U2",
            )
            V.loc["P7_U2"] = u2
            V.loc["P7_U3"] = b0 - u2

    # Consolidated tables can publish standalone extra-EU P7_D0 without an
    # intra-EU P7_B0 counterpart. Preserve that published source row independently.
    if (
        source_p7 is not None
        and allocated_p7 is not None
        and "P7_D0" in source.index
        and "P7_B0" not in source.index
    ):
        V.loc["P7_D0"] = _allocate_source_parent_vector(
            published=source.loc["P7_D0"],
            child_parent=child_parent,
            children=children,
            proxy=import_supply_proxy,
            flow=flow,
            row_code="P7_D0",
        )

    # The production column closes after taxes on products. Eurostat may expose
    # either gross operating surplus directly or the finer net-surplus + CFC basis;
    # do not require the finer basis when only the gross row is transmitted.
    if {"P2_ADJ", "B1G", "B2A3N", "B2A3G"}.intersection(source.index):
        if flow not in {"DOM", "TOTAL"}:
            raise ValueError(f"{flow} exposes production-account rows unexpectedly")
        if "D21X31" not in V.index or V.loc["D21X31"].isna().any():
            raise ValueError(f"{flow} production account requires D21X31")
        p2 = input_proxy + V.loc["D21X31"]
        b1g = output_proxy - p2
        if production_account_v is not None:
            if "B1G" not in production_account_v.index:
                raise ValueError("Joint production-account result is missing B1G")
            _assert_close(
                f"{flow} joint production-account B1G",
                b1g,
                production_account_v.loc["B1G"].reindex(children).astype("float64"),
                rtol=_GENERATED_IDENTITY_RTOL,
                atol=_GENERATED_IDENTITY_ATOL,
            )
        if primitive_v is not None and "B1G" in primitive_v.index:
            published_b1g = primitive_v.loc["B1G"].astype("float64")
            for parent in published_b1g.index.astype(str):
                if float(published_b1g.loc[parent]) < -_GENERATED_IDENTITY_ATOL:
                    continue
                child_list = [
                    str(child) for child in children
                    if child_parent.get(str(child), str(child)) == parent
                ]
                negative = b1g.reindex(child_list) < -_GENERATED_IDENTITY_ATOL
                if bool(negative.any()):
                    bad = list(map(str, negative.index[negative]))
                    raise ValueError(
                        "Generated child B1G is negative under a non-negative published parent B1G: "
                        f"flow={flow!r}, parent={parent!r}, children={bad!r}."
                    )
        if "P2_ADJ" in source.index:
            V.loc["P2_ADJ"] = p2
        if "B1G" in source.index:
            V.loc["B1G"] = b1g
        if "B2A3N" in source.index:
            required_net = {"D1", "D29X39", "P51C"}
            if any(row not in V.index or V.loc[row].isna().any() for row in required_net):
                raise ValueError(f"{flow} net-surplus closure requires {sorted(required_net)!r}")
            b2n = b1g - V.loc["D1"] - V.loc["D29X39"] - V.loc["P51C"]
            V.loc["B2A3N"] = b2n
            if "B2A3G" in source.index:
                V.loc["B2A3G"] = b2n + V.loc["P51C"]
        elif "B2A3G" in source.index:
            required_gross = {"D1", "D29X39"}
            if any(row not in V.index or V.loc[row].isna().any() for row in required_gross):
                raise ValueError(f"{flow} gross-surplus closure requires {sorted(required_gross)!r}")
            V.loc["B2A3G"] = b1g - V.loc["D1"] - V.loc["D29X39"]
    if "TS_BP" in source.index:
        # TOTAL is a supply table (P1 + P7); DOM is a domestic-output table (P1).
        if flow == "TOTAL":
            if allocated_p7 is None:
                raise ValueError("TOTAL TS_BP requires P7")
            V.loc["TS_BP"] = output_proxy + allocated_p7
        elif flow == "DOM":
            if "P1" not in V.index or V.loc["P1"].isna().any():
                raise ValueError("DOM TS_BP requires generated P1")
            V.loc["TS_BP"] = V.loc["P1"]
        else:
            raise ValueError("IMP TS_BP has no production/supply role")

    # A numeric source row without a declared primitive or identity role is a
    # genuine accounting gap; all-blank template-only rows are handled at export.
    for row in source.index.astype(str):
        if V.loc[row].isna().all():
            values = source.loc[row].to_numpy(dtype="float64")
            if np.any(np.abs(np.nan_to_num(values, nan=0.0)) > 0.0):
                raise ValueError(f"Unknown nonzero V row: flow={flow!r}, row={row!r}")
    return V, pd.DataFrame(diagnostics)


def _construct_transaction_m(
    *, flow: str, reference: ParsedEurostatIOT, Y: pd.DataFrame,
    Z: pd.DataFrame, Y_imp: pd.DataFrame, Z_imp: pd.DataFrame,
    V: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Refresh M identities while preserving primitive SDMX values and masks."""
    if reference.M is None:
        return None
    M = reference.M.copy().astype("float64")
    M.index = M.index.astype(str)
    M.columns = M.columns.astype(str)
    if V is None:
        return M

    # P7/P7_* and TS_BP are supply-by-product rows. Their cells on the final-use
    # M block are structurally non-applicable and must never be populated from Y_imp.
    supply_only_rows = cast(pd.Index, M.index).intersection(
        pd.Index(["P7", "P7_B0", "P7_D0", "P7_U2", "P7_U3", "TS_BP"])
    )
    if len(supply_only_rows):
        M.loc[supply_only_rows, :] = np.nan

    # The TOTAL display column under applicable supplementary use/production rows
    # is the matching V grand total. Supply-only rows remain non-applicable in M.
    if "TOTAL" in M.columns:
        shared_rows = (
            cast(pd.Index, M.index)
            .intersection(cast(pd.Index, V.index))
            .difference(supply_only_rows)
        )
        for row in shared_rows:
            M.loc[row, "TOTAL"] = float(V.loc[row].sum(min_count=1))
    if "CPA_TOTAL" in M.index:
        for column in M.columns:
            if column == "TOTAL":
                continue
            if column in Y.columns:
                M.loc["CPA_TOTAL", column] = float(Y[column].sum())
    if "IMP" in M.index and flow == "DOM":
        for column in M.columns:
            if column == "TOTAL":
                M.loc["IMP", column] = float(Z_imp.to_numpy(dtype="float64").sum())
            elif column in Y_imp.columns:
                M.loc["IMP", column] = float(Y_imp[column].sum())
    # Rebuild non-overlapping display controls only where the source schema says
    # the supplementary observation is applicable.
    m_rows = cast(pd.Index, M.index).astype(str)
    for row in m_rows:
        if {"P3_S13", "P3_S14", "P3_S15"}.issubset(M.columns) and not pd.isna(M.loc[row, "P3"]):
            M.loc[row, "P3"] = float(M.loc[row, ["P3_S13", "P3_S14", "P3_S15"]].sum())
        if {"P52", "P53"}.issubset(M.columns) and not pd.isna(M.loc[row, "P5M"]):
            M.loc[row, "P5M"] = float(M.loc[row, ["P52", "P53"]].sum())
        if {"P51G", "P52", "P53"}.issubset(M.columns) and not pd.isna(M.loc[row, "P5"]):
            M.loc[row, "P5"] = float(M.loc[row, ["P51G", "P52", "P53"]].sum())
        elif {"P51G", "P5M"}.issubset(M.columns) and not pd.isna(M.loc[row, "P5"]):
            M.loc[row, "P5"] = float(M.loc[row, ["P51G", "P5M"]].sum())
        atomic_cols = [c for c in source_atomic_final_demand(Y).columns if c in M.columns]
        if "TFU" in M.columns and not pd.isna(M.loc[row, "TFU"]):
            M.loc[row, "TFU"] = float(M.loc[row, atomic_cols].sum())
        if "TU" in M.columns and not pd.isna(M.loc[row, "TU"]):
            intermediate = float(M.loc[row, "TOTAL"]) if "TOTAL" in M.columns else 0.0
            final = float(M.loc[row, "TFU"]) if "TFU" in M.columns else float(M.loc[row, atomic_cols].sum())
            M.loc[row, "TU"] = intermediate + final

    # P2_ADJ is a display-column identity, so construct it only after D21X31 and
    # every Y/M display aggregate have reached their canonical generated values.
    if "P2_ADJ" in M.index and "D21X31" in M.index:
        m_columns = cast(pd.Index, M.columns).astype(str)
        for column in m_columns:
            if column == "TOTAL":
                continue
            if column in Y.columns and not pd.isna(M.loc["D21X31", column]):
                M.loc["P2_ADJ", column] = float(Y[column].sum()) + float(M.loc["D21X31", column])
    return M


def _validate_coefficient_sidecars(
    dirs: Mapping[str, Path], *, suffix: str, atol: float, rtol: float
) -> None:
    z_name = "Z_split.csv" if suffix == "split" else "Z_agg.csv"
    y_name = "Y_split.csv" if suffix == "split" else "Y_agg.csv"
    x_name = "x_split.csv" if suffix == "split" else "x_agg.csv"
    a_name = "A_split.csv" if suffix == "split" else "A_agg.csv"
    use_name = "use_total_split.csv" if suffix == "split" else "use_total_agg.csv"
    Z = {flow: _read_frame(path / z_name) for flow, path in dirs.items()}
    Y = {flow: _read_frame(path / y_name) for flow, path in dirs.items()}
    use = {flow: flow_use_total(Z[flow], Y[flow]) for flow in dirs}
    output = _read_series(dirs["DOM"] / x_name)
    _assert_close(
        f"{suffix} DOM x=row-use",
        output,
        use["DOM"],
        rtol=rtol,
        atol=atol,
    )
    _assert_close(
        f"{suffix} TOTAL use=DOM+IMP",
        use["TOTAL"],
        use["DOM"].add(use["IMP"]),
        rtol=rtol,
        atol=atol,
    )
    for flow, directory in dirs.items():
        expected_A = technical_coefficients(Z[flow], output)
        _assert_close(
            f"{suffix} {flow} use_total",
            _read_series(directory / use_name), use[flow], rtol=rtol, atol=atol,
        )
        _assert_close(
            f"{suffix} {flow} output",
            _read_series(directory / x_name), output, rtol=rtol, atol=atol,
        )
        _assert_close(
            f"{suffix} {flow} A",
            _read_frame(directory / a_name), expected_A, rtol=rtol, atol=atol,
        )


def _validate_generated_full_table(
    *, variants: Mapping[str, ParsedEurostatIOT], report_path: Path
) -> None:
    """Run the single generated-table accounting gate and persist its evidence."""
    report = validate_eurostat_iot_full_table(
        variants,
        policy=NAIO_10_CP1700_ACCOUNTING_POLICY,
        mode="generated",
    )
    report.to_frame().to_csv(report_path, index=False)
    report.require_ok()


def _read_generated_variants(
    dirs: Mapping[str, Path], *, suffix: str
) -> dict[str, ParsedEurostatIOT]:
    names = _SPLIT_NAMES if suffix == "split" else _AGG_NAMES
    return {
        flow: ParsedEurostatIOT(
            Z=_read_frame(directory / names["Z"]),
            Y=_read_frame(directory / names["Y"]),
            V=_read_frame(directory / names["V"]),
            M=_read_frame(directory / names["M"]),
            meta={},
        )
        for flow, directory in dirs.items()
    }



def _derive_total_environmental_extensions(dirs: Mapping[str, Path]) -> None:
    """Build TOTAL product extensions and carry destination-scope F_Y unchanged."""
    dom_path = dirs["DOM"] / "environmental_extensions.csv"
    imp_path = dirs["IMP"] / "environmental_extensions.csv"
    if not dom_path.exists() and not imp_path.exists():
        return
    if not dom_path.exists() or not imp_path.exists():
        raise FileNotFoundError(
            "Boundary-aware TOTAL environmental output requires both DOM and IMP environmental sidecars."
        )
    dom = pd.read_csv(dom_path)
    imp = pd.read_csv(imp_path)
    dom_final_mask = dom["boundary"].astype(str).eq("scope_final_demand_direct")
    dom_final_demand = dom.loc[dom_final_mask].copy()
    dom_product = dom.loc[~dom_final_mask].copy()
    if not dom_final_demand.empty and set(
        dom_final_demand["denominator_role"].astype(str)
    ) != {"scope_final_demand"}:
        raise ValueError(
            "DOM scope_final_demand_direct rows must use denominator_role='scope_final_demand'."
        )
    if bool(imp["boundary"].astype(str).eq("scope_final_demand_direct").any()):
        raise ValueError(
            "IMP must not contain scope_final_demand_direct rows; F_Y has no import-origin counterpart."
        )

    dom_use = _read_series(dirs["DOM"] / "use_total_split.csv")
    imp_use = _read_series(dirs["IMP"] / "use_total_split.csv")
    total_product = _combine_target_supply_extension_sidecars_direct(
        dom_product,
        imp,
        target_domestic_output=dom_use,
        target_import_use=imp_use,
    )
    total = (
        pd.concat([total_product, dom_final_demand], ignore_index=True)
        if not dom_final_demand.empty
        else total_product
    )
    total.to_csv(dirs["TOTAL"] / "environmental_extensions.csv", index=False)

    # Carry source F/F_Y identities forward as provenance only. The TOTAL target
    # sidecar owns total_use_direct / target_total_use; downstream
    # reaggregation must never infer a target boundary from the raw source file.
    raw_dom = dirs["DOM"] / "exio_environmental_extensions.csv"
    if raw_dom.exists():
        shutil.copy2(raw_dom, dirs["TOTAL"] / raw_dom.name)

def derive_total_flow(outdir: Path, *, eurostat_dataset: str, eurostat_filters: Mapping[str, object], eurostat_row_dim: str, eurostat_col_dim: str, eurostat_stk_flow_dim: str = "stk_flow", source_rounding_quantum: float | None = None, atol: float = 1e-6, rtol: float = 1e-9) -> None:
    """Canonicalise DOM/IMP transactions into a cleared production account.

    Domestic product output is the complete domestic-origin row use. The IMP
    transaction table is imported-product use and may be signed; P7 remains a
    separate imports-CIF supply control. Primitive supplementary rows retain source
    parent totals; generated rows are recomputed only for declared identities.
    """
    dirs = {flow: outdir / f"{eurostat_stk_flow_dim}={flow}" for flow in ("DOM", "IMP", "TOTAL")}
    for flow in ("DOM", "IMP"):
        if not dirs[flow].is_dir():
            raise FileNotFoundError(f"Required primitive flow directory not found: {dirs[flow]}")
    manifest = _load_equal_manifests(dirs["DOM"], dirs["IMP"])
    Z_dom, Z_imp = (_read_frame(dirs[flow] / "Z_split.csv") for flow in ("DOM", "IMP"))
    Y_dom, Y_imp = (_read_frame(dirs[flow] / "Y_split.csv") for flow in ("DOM", "IMP"))
    if not Z_dom.index.equals(Z_imp.index) or not Z_dom.columns.equals(Z_imp.columns):
        raise ValueError("DOM and IMP child topology/order differs for Z; cannot construct TOTAL.")
    if not Y_dom.index.equals(Y_imp.index) or not Y_dom.columns.equals(Y_imp.columns):
        raise ValueError("DOM and IMP child topology/order differs for Y; cannot construct TOTAL.")
    Y_dom = enforce_final_demand_display_identities(Y_dom, Z=Z_dom)
    Y_imp = enforce_final_demand_display_identities(Y_imp, Z=Z_imp)
    _write_frame(Y_dom, dirs["DOM"], "Y_split.csv")
    _write_frame(Y_imp, dirs["IMP"], "Y_split.csv")
    if dirs["TOTAL"].exists():
        shutil.rmtree(dirs["TOTAL"])
    dirs["TOTAL"].mkdir(parents=True)
    Z_total = Z_dom + Z_imp
    Y_total = Y_dom + Y_imp
    Y_total = enforce_final_demand_display_identities(Y_total, Z=Z_total)
    _write_frame(Z_total, dirs["TOTAL"], "Z_split.csv")
    _write_frame(Y_total, dirs["TOTAL"], "Y_split.csv")
    _derive_total_environmental_extensions(dirs)
    Z = {"DOM": Z_dom, "IMP": Z_imp, "TOTAL": Z_total}
    Y = {"DOM": Y_dom, "IMP": Y_imp, "TOTAL": Y_total}
    use = {flow: flow_use_total(Z[flow], Y[flow]) for flow in Z}
    references: dict[str, ParsedEurostatIOT] = {}
    diagnostics: list[dict[str, object]] = []
    for flow in ("DOM", "IMP", "TOTAL"):
        filters = dict(eurostat_filters)
        filters[eurostat_stk_flow_dim] = flow
        references[flow] = load_eurostat_iot(dataset=eurostat_dataset, filters=filters, row_dim=eurostat_row_dim, col_dim=eurostat_col_dim, storage_folder=default_data_root() / "eurostat")
    quantum = _resolve_source_rounding_quantum(references, source_rounding_quantum)
    diagnostics.extend(_validate_source_ts_bp_controls(
        references=references,
        source_rounding_quantum=quantum,
        rtol=rtol,
        atol=atol,
    ))
    source_codes = references["TOTAL"].Z.index.astype(str)
    child_parent = _child_to_parent(manifest, source_codes)
    # WHOLE_IOT_KL_RECONCILIATION_V1: Y is fixed and source-validated first;
    # DOM row use then fixes child output.  DOM and IMP intermediate transactions
    # are reconciled jointly so source Z cells, child row margins, and the TOTAL
    # production account are one simultaneous feasible set. EXIO is only the KL prior.
    x_domestic = use["DOM"].rename("x")
    dom_output_share = _load_dom_split_output_shares(
        dirs["DOM"], child_parent=child_parent, children=Z_dom.index
    )
    for flow in ("DOM", "IMP"):
        diagnostics.extend(
            _validate_y_source_reconstruction(
                flow=flow,
                reconstructed=_aggregate_rows(
                    Y[flow], child_parent, references[flow].Z.index.astype(str)
                ),
                reference=references[flow],
                source_rounding_quantum=quantum,
                rtol=rtol,
                atol=atol,
            )
        )
    pre_reconciliation_use = dict(use)
    Z_dom, Z_imp, production_account_v, reconciliation, transformation_checks = (
        _reconcile_iot_intermediate_transactions(
            Z_dom,
            Z_imp,
            child_parent=child_parent,
            dom_output_share=dom_output_share,
            x_domestic=x_domestic,
            reference_total=references["TOTAL"],
        )
    )

    Z_total = Z_dom + Z_imp
    Z = {"DOM": Z_dom, "IMP": Z_imp, "TOTAL": Z_total}
    reconciled_use = {flow: flow_use_total(Z[flow], Y[flow]) for flow in Z}
    dom_output_gap = _assert_close(
        "DOM reconciliation preserves canonical output",
        reconciled_use["DOM"],
        x_domestic,
        rtol=_GENERATED_IDENTITY_RTOL,
        atol=_GENERATED_IDENTITY_ATOL,
    )
    imp_use_gap = _assert_close(
        "IMP reconciliation preserves imported-product row use",
        reconciled_use["IMP"],
        pre_reconciliation_use["IMP"],
        rtol=_GENERATED_IDENTITY_RTOL,
        atol=_GENERATED_IDENTITY_ATOL,
    )
    total_use_gap = _assert_close(
        "TOTAL use remains DOM plus IMP after reconciliation",
        reconciled_use["TOTAL"],
        reconciled_use["DOM"].add(reconciled_use["IMP"]),
        rtol=_GENERATED_IDENTITY_RTOL,
        atol=_GENERATED_IDENTITY_ATOL,
    )
    A_dom = technical_coefficients(Z_dom, x_domestic)
    rho_dom = float(calc_spectral_radius(A_dom))
    if not np.isfinite(rho_dom) or rho_dom >= 1.0:
        raise ValueError(
            "Generated domestic Leontief system is not productive: "
            f"spectral_radius={rho_dom!r}; require rho(A_DOM) < 1."
        )
    transformation_checks.extend(
        [
            {
                "validation_class": "transformation_coherence",
                "check_id": "tc.dom_output_margin",
                "flow": "DOM",
                "status": "pass",
                "max_abs_gap": dom_output_gap,
                "rtol": _GENERATED_IDENTITY_RTOL,
                "atol": _GENERATED_IDENTITY_ATOL,
            },
            {
                "validation_class": "transformation_coherence",
                "check_id": "tc.imp_row_use_margin",
                "flow": "IMP",
                "status": "pass",
                "max_abs_gap": imp_use_gap,
                "rtol": _GENERATED_IDENTITY_RTOL,
                "atol": _GENERATED_IDENTITY_ATOL,
            },
            {
                "validation_class": "transformation_coherence",
                "check_id": "tc.total_use_additivity",
                "flow": "TOTAL",
                "status": "pass",
                "max_abs_gap": total_use_gap,
                "rtol": _GENERATED_IDENTITY_RTOL,
                "atol": _GENERATED_IDENTITY_ATOL,
            },
            {
                "validation_class": "transformation_coherence",
                "check_id": "tc.dom_leontief_productivity",
                "flow": "DOM",
                "status": "pass",
                "max_abs_gap": max(rho_dom - 1.0, 0.0),
                "rtol": 0.0,
                "atol": 0.0,
                "spectral_radius": rho_dom,
            },
        ]
    )
    use = reconciled_use
    for flow in ("DOM", "IMP", "TOTAL"):
        _write_frame(Z[flow], dirs[flow], "Z_split.csv")
    (outdir / "producer_column_conditioning.csv").unlink(missing_ok=True)
    reconciliation.to_csv(
        outdir / "producer_technology_reconciliation.csv", index=False
    )
    import_atomic = source_atomic_final_demand(Y_imp)
    nonsigned_import_columns = [
        code for code in import_atomic.columns if code not in _SIGNED_INVESTMENT_ATOMS
    ]
    import_supply_proxy = Z_imp.sum(axis=1).add(
        import_atomic.loc[:, nonsigned_import_columns].sum(axis=1), fill_value=0.0
    ).astype("float64")
    if not np.isfinite(import_supply_proxy.to_numpy(dtype="float64")).all() or bool(
        (import_supply_proxy < -1e-9).any()
    ):
        raise ValueError("P7 child-allocation proxy must be finite and non-negative.")
    import_supply_proxy = import_supply_proxy.clip(lower=0.0)
    published_p1 = _source_row(references["DOM"], "P1")
    if published_p1 is not None:
        derived_p1 = cast(
            pd.Series,
            _aggregate_rows(
                x_domestic.to_frame("P1"), child_parent, published_p1.index
            )["P1"],
        )
        counts = references["DOM"].Z.notna().sum(axis=1).add(
            source_atomic_final_demand(references["DOM"].Y).notna().sum(axis=1),
            fill_value=0,
        ).astype("int64")
        diagnostics.extend(_validate_rounded_series(
            flow="DOM", identity="P1", published=published_p1,
            derived=derived_p1, component_counts=counts,
            source_rounding_quantum=quantum, rtol=rtol, atol=atol,
        ))
    for flow in Z:
        _write_series(use[flow], dirs[flow] / "use_total_split.csv")
        _write_series(x_domestic, dirs[flow] / "x_split.csv")
        _write_frame(technical_coefficients(Z[flow], x_domestic), dirs[flow], "A_split.csv")
    diagnostics.append({
        "stage": "analytical_contract", "flow": "DOM", "object": "P1",
        "contract": "transaction_derived_DOM_row_use",
        "max_abs_gap": dom_output_gap,
        "ok": True,
    })
    for flow in ("DOM", "IMP"):
        reference = references[flow]
        reconstructed_z = _aggregate_square(Z[flow], child_parent, reference.Z.index.astype(str))
        diagnostics.append({"stage": "source_reconstruction", "flow": flow, "object": "Z", "max_abs_gap": _assert_close(f"{flow} Z reconstruction", reconstructed_z, reference.Z, rtol=rtol, atol=atol), "ok": True})
    # Published TOTAL can differ from DOM+IMP by independently rounded source cells.
    total_z = _aggregate_square(Z_total, child_parent, references["TOTAL"].Z.index.astype(str))
    total_atomic = _aggregate_rows(source_atomic_final_demand(Y_total), child_parent, references["TOTAL"].Z.index.astype(str))
    source_atomic = source_atomic_final_demand(references["TOTAL"].Y).reindex(index=total_atomic.index, columns=total_atomic.columns)
    for object_name, generated, published in (("Z", total_z, references["TOTAL"].Z), ("atomic_Y", total_atomic, source_atomic)):
        for row in generated.index.astype(str):
            for col in generated.columns.astype(str):
                diag = rounded_source_control_diagnostic(identity=f"TOTAL:{object_name}", published=float(published.loc[row, col]), derived=float(generated.loc[row, col]), component_term_count=2, source_rounding_quantum=quantum, rtol=rtol, atol=atol)
                diag.update({"stage": "TOTAL_source_reconstruction", "flow": "TOTAL", "object": object_name, "row": row, "column": col})
                diagnostics.append(diag)
    total_reconstructed_y = _aggregate_rows(Y_total, child_parent, references["TOTAL"].Z.index.astype(str))
    for identity in _ROUNDED_Y_CONTROLS:
        if identity in total_reconstructed_y.columns:
            expected = _derived_y_control(ParsedEurostatIOT(Z=total_z, Y=total_reconstructed_y, V=None, M=None, meta={}), identity)[0]
            _assert_close(f"TOTAL generated {identity}", total_reconstructed_y[identity], expected, rtol=rtol, atol=atol)
    canonical_v: dict[str, pd.DataFrame | None] = {}
    canonical_m: dict[str, pd.DataFrame | None] = {}
    for flow in ("DOM", "IMP", "TOTAL"):
        V, v_diagnostics = _construct_transaction_v(
            flow=flow, reference=references[flow], child_parent=child_parent,
            Z=Z[flow], Z_total=Z_total, Z_imp=Z_imp,
            x_domestic=x_domestic, import_supply_proxy=import_supply_proxy,
            primitive_reference=references["TOTAL"] if flow in {"DOM", "TOTAL"} else None,
            production_account_v=(
                production_account_v if flow in {"DOM", "TOTAL"} else None
            ),
        )
        canonical_v[flow] = V
        if V is not None:
            _write_frame(V, dirs[flow], "V_split.csv")
            v_diagnostics.to_csv(
                dirs[flow] / "supplementary_row_split_diagnostics.csv",
                index=False,
            )
        M = _construct_transaction_m(
            flow=flow, reference=references[flow], Y=Y[flow], Z=Z[flow],
            Y_imp=Y_imp, Z_imp=Z_imp, V=V,
        )
        canonical_m[flow] = M
        if M is not None:
            _write_frame(M, dirs[flow], "M_split.csv")
    for flow in ("DOM", "TOTAL"):
        source_b1g = _source_row(references[flow], "B1G")
        generated_v = canonical_v[flow]
        if source_b1g is None or generated_v is None or "B1G" not in generated_v.index:
            continue
        generated_parent_b1g = _aggregate_columns(
            generated_v.loc[["B1G"]], child_parent, source_b1g.index.astype(str)
        ).loc["B1G"]
        p1_terms = references["DOM"].Z.notna().sum(axis=1).add(
            source_atomic_final_demand(references["DOM"].Y).notna().sum(axis=1),
            fill_value=0,
        )
        p2_terms = references["TOTAL"].Z.notna().sum(axis=0).add(1)
        b1g_terms = p1_terms.reindex(source_b1g.index).add(
            p2_terms.reindex(source_b1g.index), fill_value=0
        ).astype("int64")
        diagnostics.extend(
            _validate_rounded_series(
                flow=flow,
                identity="B1G",
                published=source_b1g,
                derived=generated_parent_b1g,
                component_counts=b1g_terms,
                source_rounding_quantum=quantum,
                rtol=rtol,
                atol=atol,
            )
        )
    if eurostat_dataset == NAIO_10_CP1700_ACCOUNTING_POLICY.dataset:
        _validate_generated_full_table(
            variants={
                flow: ParsedEurostatIOT(
                    Z=Z[flow],
                    Y=Y[flow],
                    V=canonical_v[flow],
                    M=canonical_m[flow],
                    meta={},
                )
                for flow in ("DOM", "IMP", "TOTAL")
            },
            report_path=outdir / "split_full_table_validation.csv",
        )
    pd.DataFrame(transformation_checks).to_csv(
        outdir / _TRANSFORMATION_COHERENCE_REPORT, index=False
    )
    (dirs["TOTAL"] / "exio_selected_sectors.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (dirs["TOTAL"] / "split_info.json").write_text(json.dumps({
        "flow_construction": "DOM_plus_IMP",
        "x_semantics": "transaction_derived_DOM_P1",
        "A_semantics": {
            "DOM": "cleared_domestic_leontief_coefficients",
            "IMP": "imported_inputs_per_domestic_output",
            "TOTAL": "total_inputs_per_domestic_output",
        },
        "V_status": "transaction_identities_plus_primitive_rows",
        "producer_column_reconciliation": "joint_dom_imp_kl_production_account_v",
        "producer_column_reconciliation_diagnostic": "../producer_technology_reconciliation.csv",
        "producer_column_constraint_semantics": "EXIO_is_KL_prior; Eurostat_parent_Z_and_child_rows_are_equalities; D21X31_is_joint_parent_control; child_production_account_closes_with_B1G_auxiliary",
        "transformation_coherence_validation": f"../{_TRANSFORMATION_COHERENCE_REPORT}",
        "full_table_validation": "../split_full_table_validation.csv",
        "validation_classes": [
            "accounting_identity_and_source_conservation",
            "transformation_coherence",
        ],
        "balancing_applied": True,
        "balancing_method": "KL_LINEAR_EQUALITY_PRODUCTION_ACCOUNT_V",
    }, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(diagnostics).to_csv(outdir / "derived_total_flow_identity.csv", index=False)


def validate_additive_outputs(outdir: Path, *, suffix: str, atol: float, rtol: float) -> None:
    (outdir / f"{suffix}_flow_identity.csv").unlink(missing_ok=True)
    dirs = {flow: outdir / f"stk_flow={flow}" for flow in ("DOM", "IMP", "TOTAL")}
    for directory in dirs.values():
        if not directory.is_dir():
            raise FileNotFoundError(f"Required flow directory not found: {directory}")
    _validate_coefficient_sidecars(
        dirs, suffix=suffix, atol=atol, rtol=rtol
    )
    _validate_generated_full_table(
        variants=_read_generated_variants(dirs, suffix=suffix),
        report_path=outdir / f"{suffix}_full_table_validation.csv",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--validate-only", choices=("split", "agg"))
    parser.add_argument("--eurostat-dataset")
    parser.add_argument("--eurostat-filters")
    parser.add_argument("--eurostat-row-dim")
    parser.add_argument("--eurostat-col-dim")
    parser.add_argument("--eurostat-stk-flow-dim", default="stk_flow")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument(
        "--source-rounding-quantum",
        type=float,
        help="Override the inferred source reporting quantum for rounded published controls.",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.validate_only:
        validate_additive_outputs(args.outdir, suffix=args.validate_only, rtol=args.rtol, atol=args.atol)
        return 0
    required = (args.eurostat_dataset, args.eurostat_filters, args.eurostat_row_dim, args.eurostat_col_dim)
    if not all(required):
        parser.error("derivation requires Eurostat dataset, filters, row dimension, and column dimension")
    derive_total_flow(args.outdir, eurostat_dataset=str(args.eurostat_dataset), eurostat_filters=json.loads(str(args.eurostat_filters)), eurostat_row_dim=str(args.eurostat_row_dim), eurostat_col_dim=str(args.eurostat_col_dim), eurostat_stk_flow_dim=str(args.eurostat_stk_flow_dim), source_rounding_quantum=args.source_rounding_quantum, rtol=args.rtol, atol=args.atol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
