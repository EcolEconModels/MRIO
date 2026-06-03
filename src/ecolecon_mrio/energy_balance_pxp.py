from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ecolecon_mrio.disaggregation import SplitProfiles, disaggregate_and_balance
from ecolecon_mrio.eurostat_energy_balances import load_eurostat_energy_balance_slice

ALLOWED_PARENTS: frozenset[str] = frozenset({"CPA_C19", "CPA_C20", "CPA_D"})
ALLOWED_COMPONENT_KINDS: frozenset[str] = frozenset(
    {"energy_quantity", "advanced_external_quantity", "selector_residual", "parent_residual"}
)
ALLOWED_QUANTITY_RULES: frozenset[str] = frozenset(
    {
        "energy_balance_selector",
        "external_quantity_cap",
        "selector_total_minus_group_external",
        "parent_output_minus_declared_components",
    }
)
SUPPORTED_PRICE_UNITS: frozenset[str] = frozenset({"EUR/KWH", "EUR/T", "EUR/MWH_LHV"})
SUPPORTED_EB_DATASETS: frozenset[str] = frozenset({"nrg_bal_c", "nrg_bal_peh"})
MWH_PER_TOE = 11.63
GWH_PER_MTOE = 11_630.0


@dataclass(frozen=True)
class EnergyBalancePxpSplitPlan:
    split_profiles: dict[str, SplitProfiles]
    column_allocator: dict[str, dict[str, pd.Series]]
    component_values: pd.DataFrame
    split_shares: pd.DataFrame
    selector_residuals: pd.DataFrame
    parent_residuals: pd.DataFrame
    price_resolution: pd.DataFrame
    diagnostics: dict[str, pd.DataFrame]
    reaggregation_mapping: dict[str, str]


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return " ".join(str(value).strip().split())


def _parse_boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _normalize_text(value).casefold()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Could not parse boolean value: {value!r}")


def _split_tokens(value: object) -> tuple[str, ...]:
    text = _normalize_text(value)
    if not text:
        return tuple()
    return tuple(token.strip() for token in text.split("|") if token.strip())


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def _as_float(value: object, *, default: float | None = None) -> float:
    text = _normalize_text(value)
    if text == "":
        if default is None:
            raise ValueError("Missing numeric value.")
        return float(default)
    return float(text)


def _active_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "enabled" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["enabled"].map(_parse_boolish)].copy()


def load_energy_balance_pxp_split_frame(path: str | Path) -> pd.DataFrame:
    frame = _read_csv(path)
    required = {
        "order",
        "enabled",
        "parent_code",
        "parent_label",
        "child_code",
        "child_label",
        "component_kind",
        "quantity_rule",
        "subsplit_group",
        "eb_selector_system",
        "eb_product_selector",
        "eb_flow_selector",
        "quantity_unit",
        "external_quantity_mtoe",
        "external_quantity_source",
        "price_code",
        "residual_rule",
        "share_roles",
        "allow_zero",
        "notes",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Split CSV is missing required columns: {missing!r}")
    frame["order"] = pd.to_numeric(frame["order"], errors="raise")
    frame["enabled"] = frame["enabled"].map(_parse_boolish)
    frame["allow_zero"] = frame["allow_zero"].map(_parse_boolish)
    for column in required - {"order", "enabled", "allow_zero"}:
        frame[column] = frame[column].map(_normalize_text)
    return frame.sort_values("order", kind="stable").reset_index(drop=True)


def load_energy_balance_pxp_reaggregation_frame(path: str | Path) -> pd.DataFrame:
    frame = _read_csv(path)
    required = {"source_code", "source_label", "target_code", "target_label", "enabled", "notes"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Reaggregation CSV is missing required columns: {missing!r}")
    frame["enabled"] = frame["enabled"].map(_parse_boolish)
    for column in required - {"enabled"}:
        frame[column] = frame[column].map(_normalize_text)
    return frame


def load_energy_balance_pxp_input_technology_frame(path: str | Path) -> pd.DataFrame:
    frame = _read_csv(path)
    required = {
        "order",
        "enabled",
        "parent_code",
        "parent_label",
        "buyer_split_code",
        "buyer_split_label",
        "supplier_selector_system",
        "supplier_selector",
        "allocation_weight",
        "apply_to",
        "rule",
        "fallback_method",
        "notes",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Technology CSV is missing required columns: {missing!r}")
    frame["order"] = pd.to_numeric(frame["order"], errors="raise")
    frame["enabled"] = frame["enabled"].map(_parse_boolish)
    frame["allocation_weight"] = pd.to_numeric(frame["allocation_weight"], errors="coerce")
    for column in required - {"order", "enabled", "allocation_weight"}:
        frame[column] = frame[column].map(_normalize_text)
    return frame.sort_values("order", kind="stable").reset_index(drop=True)


def load_energy_balance_pxp_prices_frame(path: str | Path) -> pd.DataFrame:
    frame = _read_csv(path)
    required = {
        "price_code",
        "price_label",
        "price_value",
        "price_unit",
        "toe_per_t",
        "eur_per_mwh_lhv",
        "status",
        "applies_to",
        "source_invariant",
        "notes",
        "conversion_note",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prices CSV is missing required columns: {missing!r}")
    for column in required - {"price_value", "toe_per_t", "eur_per_mwh_lhv", "source_invariant"}:
        frame[column] = frame[column].map(_normalize_text)
    frame["price_value"] = pd.to_numeric(frame["price_value"], errors="coerce")
    frame["toe_per_t"] = pd.to_numeric(frame["toe_per_t"], errors="coerce")
    frame["eur_per_mwh_lhv"] = pd.to_numeric(frame["eur_per_mwh_lhv"], errors="coerce")
    frame["source_invariant"] = frame["source_invariant"].map(_parse_boolish)
    return frame


def load_energy_balance_dictionary_frame(path: str | Path) -> pd.DataFrame:
    frame = _read_csv(path)
    required = {"dataset", "dimension", "code", "label", "description", "provenance_source"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Energy Balance dictionary is missing required columns: {missing!r}")
    for column in required:
        frame[column] = frame[column].map(_normalize_text)
    blanks = {
        column: int(frame[column].astype(str).str.strip().eq("").sum())
        for column in required
    }
    bad = {key: value for key, value in blanks.items() if value}
    if bad:
        raise ValueError(f"Energy Balance dictionary has blank required values: {bad!r}")
    dupes = frame.duplicated(subset=["dataset", "dimension", "code"], keep=False)
    if dupes.any():
        sample = frame.loc[dupes, ["dataset", "dimension", "code"]].head(20).to_dict("records")
        raise ValueError(f"Energy Balance dictionary has duplicate dataset/dimension/code rows: {sample!r}")
    return frame


def _parse_eb_flow_selector(flow_selector: object) -> tuple[str, tuple[str, ...]]:
    text = _normalize_text(flow_selector)
    if not text or ":" not in text:
        raise ValueError(
            f"Unsupported eb_flow_selector {text!r}. "
            "Use explicit forms like 'nrg_bal_c:NRG_PR_E' or 'nrg_bal_peh:GEP'."
        )
    dataset, rhs = text.split(":", 1)
    dataset = dataset.strip()
    rhs = rhs.strip()
    if dataset not in SUPPORTED_EB_DATASETS:
        raise ValueError(f"Unsupported eb_flow_selector dataset {dataset!r} in {text!r}.")
    if rhs.startswith("SUM(") and rhs.endswith(")"):
        rhs = rhs[4:-1].strip()
    codes = _split_tokens(rhs)
    if not codes:
        raise ValueError(f"eb_flow_selector {text!r} did not contain nrg_bal code(s).")
    return dataset, codes


def _dictionary_codes(dictionary_frame: pd.DataFrame, *, dataset: str, dimension: str) -> set[str]:
    mask = dictionary_frame["dataset"].eq(dataset) & dictionary_frame["dimension"].eq(dimension)
    return set(dictionary_frame.loc[mask, "code"].astype(str).str.strip())


def validate_split_against_energy_balance_dictionary(split_frame: pd.DataFrame, dictionary_frame: pd.DataFrame) -> None:
    enabled = split_frame.loc[split_frame["enabled"] & (split_frame["component_kind"] != "parent_residual")].copy()
    errors: list[str] = []
    for row in enabled.itertuples(index=False):
        dataset, nrg_bal_codes = _parse_eb_flow_selector(row.eb_flow_selector)
        official_nrg_bal = _dictionary_codes(dictionary_frame, dataset=dataset, dimension="nrg_bal")
        official_siec = _dictionary_codes(dictionary_frame, dataset=dataset, dimension="siec")
        missing_flows = [code for code in nrg_bal_codes if code not in official_nrg_bal]
        missing_products = [code for code in _split_tokens(row.eb_product_selector) if code not in official_siec]
        if missing_flows:
            errors.append(
                f"{row.child_code}: eb_flow_selector {row.eb_flow_selector!r} has non-official nrg_bal code(s) "
                f"{missing_flows!r} for dataset={dataset!r}"
            )
        if missing_products:
            errors.append(
                f"{row.child_code}: eb_product_selector {row.eb_product_selector!r} has non-official SIEC code(s) "
                f"{missing_products!r} for dataset={dataset!r}"
            )
    if errors:
        raise ValueError("Energy Balance split CSV uses non-official EB codes:\n" + "\n".join(errors[:50]))


def resolve_price_row(row: pd.Series) -> float:
    unit = _normalize_text(row["price_unit"]).upper()
    value = float(row["price_value"])
    if unit == "EUR/KWH":
        return value * 1000.0
    if unit == "EUR/T":
        toe_per_t = float(row.get("toe_per_t", np.nan))
        if not np.isfinite(toe_per_t) or toe_per_t <= 0:
            raise ValueError(f"Cannot convert EUR/t price without positive toe_per_t for {row['price_code']!r}.")
        return value / (toe_per_t * MWH_PER_TOE)
    if unit == "EUR/MWH_LHV":
        return value
    raise ValueError(f"Unsupported price unit {row['price_unit']!r} for price_code={row['price_code']!r}.")


def build_price_resolution(prices_frame: pd.DataFrame) -> pd.DataFrame:
    out = prices_frame.copy()
    out["resolved_eur_per_mwh_lhv"] = out.apply(resolve_price_row, axis=1)
    return out


def _validate_split_frame(split_frame: pd.DataFrame) -> None:
    invalid_parent_codes = sorted(set(split_frame["parent_code"].astype(str)) - set(ALLOWED_PARENTS))
    if invalid_parent_codes:
        raise ValueError(
            "EB split path only allows CPA_C19, CPA_C20, and CPA_D parents; "
            f"found {invalid_parent_codes!r}."
        )
    if (split_frame["parent_code"] == "CPA_C10-12").any() or (split_frame["child_code"] == "CPA_C10-12").any():
        raise ValueError("CPA_C10-12 is not allowed in the EB pxp split path.")
    invalid_kinds = sorted(set(split_frame["component_kind"]) - set(ALLOWED_COMPONENT_KINDS))
    if invalid_kinds:
        raise ValueError(f"Unsupported component_kind values: {invalid_kinds!r}")
    invalid_rules = sorted(set(split_frame["quantity_rule"]) - set(ALLOWED_QUANTITY_RULES))
    if invalid_rules:
        raise ValueError(f"Unsupported quantity_rule values: {invalid_rules!r}")

    active = split_frame.loc[split_frame["enabled"]].copy()
    for parent_code, group in active.groupby("parent_code", sort=False):
        if not (group["component_kind"] == "parent_residual").any():
            raise ValueError(f"Parent {parent_code!r} is missing an active parent_residual row.")
    active_d = active.loc[active["parent_code"].eq("CPA_D")]
    if not active_d.empty and not (active_d["child_code"].eq("EB_D_ELECTRICITY_BY_NUCLEAR")).any():
        raise ValueError("EB_D_ELECTRICITY_BY_NUCLEAR must be active when CPA_D is split.")

    for row in active.loc[active["component_kind"] != "parent_residual"].itertuples(index=False):
        _parse_eb_flow_selector(row.eb_flow_selector)
        if not _split_tokens(row.eb_product_selector):
            raise ValueError(f"Enabled split row {row.child_code!r} has no eb_product_selector.")
        if not row.price_code:
            raise ValueError(f"Enabled split row {row.child_code!r} has no price_code.")


def _validate_reaggregation_frame(split_frame: pd.DataFrame, reaggregation_frame: pd.DataFrame) -> dict[str, str]:
    enabled_children = set(split_frame.loc[split_frame["enabled"], "child_code"].astype(str))
    enabled_reagg = reaggregation_frame.loc[reaggregation_frame["enabled"]].copy()
    enabled_sources = set(enabled_reagg["source_code"].astype(str))
    missing_from_reagg = sorted(enabled_children - enabled_sources)
    extra_sources = sorted(enabled_sources - enabled_children)
    if missing_from_reagg:
        raise ValueError(f"Enabled split children missing from reaggregation: {missing_from_reagg!r}")
    if extra_sources:
        raise ValueError(f"Enabled reaggregation source codes missing from split CSV: {extra_sources!r}")
    dupes = enabled_reagg.duplicated(subset=["source_code"], keep=False)
    if dupes.any():
        sample = sorted(enabled_reagg.loc[dupes, "source_code"].unique().tolist())
        raise ValueError(f"Reaggregation maps a source child to more than one target: {sample!r}")
    return {str(row.source_code): str(row.target_code) for row in enabled_reagg.itertuples(index=False)}


def _validate_prices_frame(split_frame: pd.DataFrame, prices_frame: pd.DataFrame) -> pd.DataFrame:
    resolved = build_price_resolution(prices_frame)
    price_lookup = set(resolved["price_code"].astype(str))
    active_non_parent = split_frame.loc[split_frame["enabled"] & (split_frame["component_kind"] != "parent_residual")]
    missing = sorted({code for code in active_non_parent["price_code"].astype(str) if not code or code not in price_lookup})
    if missing:
        raise ValueError(f"Enabled split rows reference missing price_code(s): {missing!r}")
    placeholder = resolved.loc[resolved["status"].astype(str).eq("placeholder_to_replace")]
    if not placeholder.empty:
        warnings.warn(
            "Placeholder Energy Balance prices are still in use: "
            + ", ".join(placeholder["price_code"].astype(str).tolist()),
            RuntimeWarning,
            stacklevel=2,
        )
    return resolved


def _validate_technology_frame(split_frame: pd.DataFrame, technology_frame: pd.DataFrame) -> None:
    if technology_frame.empty:
        raise ValueError("Technology CSV is required even though Phase 2 baseline does not yet apply it.")
    enabled_tech = technology_frame.loc[technology_frame["enabled"]].copy()
    if enabled_tech.empty:
        raise ValueError("Technology CSV has no enabled rows.")
    active_children = set(split_frame.loc[split_frame["enabled"], "child_code"].astype(str))
    active_parents = set(split_frame.loc[split_frame["enabled"], "parent_code"].astype(str))
    bad_parents = sorted(set(enabled_tech["parent_code"].astype(str)) - active_parents)
    if bad_parents:
        raise ValueError(f"Technology CSV contains invalid parent_code values: {bad_parents!r}")
    bad_children: set[str] = set()
    for value in enabled_tech["buyer_split_code"]:
        for token in _split_tokens(value):
            if token not in active_children:
                bad_children.add(token)
    if bad_children:
        raise ValueError(f"Technology CSV references unknown active child code(s): {sorted(bad_children)!r}")
    bad_supplier_tokens: list[str] = []
    for value in enabled_tech["supplier_selector"]:
        for token in _split_tokens(value):
            if token.startswith("EB_"):
                bad_supplier_tokens.append(token)
            if token in active_children:
                bad_supplier_tokens.append(token)
    if bad_supplier_tokens:
        raise ValueError(
            "Technology supplier selectors must refer to original Eurostat rows or explicit label:/prefix:/regex: selectors, "
            f"not EB child codes: {sorted(set(bad_supplier_tokens))!r}"
        )


def validate_energy_balance_pxp_inputs(
    *,
    split_frame: pd.DataFrame,
    reaggregation_frame: pd.DataFrame,
    technology_frame: pd.DataFrame,
    prices_frame: pd.DataFrame,
    energy_balance_dictionary_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _validate_split_frame(split_frame)
    if energy_balance_dictionary_frame is not None:
        validate_split_against_energy_balance_dictionary(split_frame, energy_balance_dictionary_frame)
    reaggregation_mapping = _validate_reaggregation_frame(split_frame, reaggregation_frame)
    resolved_prices = _validate_prices_frame(split_frame, prices_frame)
    _validate_technology_frame(split_frame, technology_frame)
    # Keep one cheap check here so callers get the map shape before compile.
    if not reaggregation_mapping:
        raise ValueError("Reaggregation mapping is empty.")
    return resolved_prices


def _coerce_quantity_table(quantity_table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(quantity_table, pd.DataFrame):
        raise TypeError("quantity_table_override must be a pandas DataFrame; legacy tuple mappings are not supported.")
    required = {
        "child_code",
        "parent_code",
        "subsplit_group",
        "eb_product_selector",
        "eb_flow_selector",
        "quantity_GWh",
        "selector_total_GWh",
    }
    missing = sorted(required - set(quantity_table.columns))
    if missing:
        raise ValueError(f"Quantity table is missing required columns: {missing!r}")
    out = quantity_table.copy()
    for column in ("child_code", "parent_code", "subsplit_group", "eb_product_selector", "eb_flow_selector", "quantity_source", "notes"):
        if column in out.columns:
            out[column] = out[column].map(_normalize_text)
    out["quantity_GWh"] = pd.to_numeric(out["quantity_GWh"], errors="coerce")
    out["selector_total_GWh"] = pd.to_numeric(out["selector_total_GWh"], errors="coerce")
    if out["quantity_GWh"].isna().any() or out["selector_total_GWh"].isna().any():
        raise ValueError("Quantity table contains non-numeric quantity_GWh or selector_total_GWh values.")
    if "quantity_source" not in out.columns:
        out["quantity_source"] = "override"
    if "notes" not in out.columns:
        out["notes"] = ""
    return out


def _collect_required_flow_codes(split_frame: pd.DataFrame) -> dict[str, set[str]]:
    required: dict[str, set[str]] = {"nrg_bal_c": set(), "nrg_bal_peh": set()}
    active = split_frame.loc[split_frame["enabled"] & (split_frame["component_kind"] != "parent_residual")]
    for row in active.itertuples(index=False):
        dataset, codes = _parse_eb_flow_selector(row.eb_flow_selector)
        required[dataset].update(codes)
    return required


def _load_required_energy_balance_tables(
    *,
    split_frame: pd.DataFrame,
    geo: str,
    year: int,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    force_reparse: bool = False,
) -> dict[str, pd.DataFrame]:
    required = _collect_required_flow_codes(split_frame)
    out: dict[str, pd.DataFrame] = {}
    for dataset, codes in required.items():
        if not codes:
            out[dataset] = pd.DataFrame(dtype="float64")
            continue
        frame, _meta = load_eurostat_energy_balance_slice(
            geo=geo,
            time=year,
            nrg_bal=sorted(codes),
            siec=None,
            unit="GWH",
            freq="A",
            dataset=dataset,
            storage_folder=storage_folder,
            force_download=force_download,
            force_reparse=force_reparse,
        )
        out[dataset] = frame.astype("float64").fillna(0.0)
    return out


def _sum_quantity_for_row(row: object, tables: Mapping[str, pd.DataFrame]) -> tuple[float, str]:
    dataset, nrg_bal_codes = _parse_eb_flow_selector(getattr(row, "eb_flow_selector"))
    table = tables.get(dataset, pd.DataFrame(dtype="float64"))
    if table.empty:
        raise ValueError(f"No table loaded for dataset {dataset!r}; cannot resolve child {row.child_code!r}.")
    missing_rows = [code for code in nrg_bal_codes if code not in table.index.astype(str)]
    if missing_rows:
        raise ValueError(f"Energy Balance table {dataset!r} is missing nrg_bal row(s) {missing_rows!r} for {row.child_code!r}.")
    selected = table.loc[list(nrg_bal_codes), :]
    if isinstance(selected, pd.Series):
        selected = selected.to_frame().T
    product_tokens = _split_tokens(getattr(row, "eb_product_selector"))
    missing_products = [token for token in product_tokens if token not in selected.columns.astype(str)]
    if missing_products:
        if bool(getattr(row, "allow_zero")):
            return 0.0, f"{dataset}:{'|'.join(nrg_bal_codes)}; missing allowed products={missing_products!r}"
        raise ValueError(f"Energy Balance table {dataset!r} is missing SIEC code(s) {missing_products!r} for {row.child_code!r}.")
    quantity = float(pd.to_numeric(selected.loc[:, list(product_tokens)].to_numpy().ravel(), errors="coerce").sum())
    return quantity, f"{dataset}:{'|'.join(nrg_bal_codes)}"


def build_energy_balance_pxp_quantity_lookup(
    *,
    split_frame: pd.DataFrame,
    geo: str,
    year: int,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    force_reparse: bool = False,
) -> pd.DataFrame:
    active = split_frame.loc[split_frame["enabled"] & (split_frame["component_kind"] != "parent_residual")].copy()
    columns = [
        "child_code",
        "parent_code",
        "subsplit_group",
        "eb_product_selector",
        "eb_flow_selector",
        "quantity_GWh",
        "selector_total_GWh",
        "quantity_source",
        "notes",
    ]
    if active.empty:
        return pd.DataFrame(columns=columns)
    tables = _load_required_energy_balance_tables(
        split_frame=split_frame,
        geo=geo,
        year=year,
        storage_folder=storage_folder,
        force_download=force_download,
        force_reparse=force_reparse,
    )
    records: list[dict[str, object]] = []
    for row in active.sort_values(["parent_code", "order"], kind="stable").itertuples(index=False):
        quantity, source_note = _sum_quantity_for_row(row, tables)
        # Zero selector totals are legitimate for derived rows such as
        # advanced_external_quantity and selector_residual: the downstream
        # component builder caps advanced quantities at the selector total and
        # computes conventional residuals as zero.  Only direct measured
        # energy_quantity rows use allow_zero as a hard zero guard.
        if quantity == 0.0 and not bool(row.allow_zero) and str(row.component_kind) == "energy_quantity":
            raise ValueError(
                f"Zero or missing quantity for {row.child_code!r} is not allowed. "
                f"selector={row.eb_product_selector!r}, flow={row.eb_flow_selector!r}."
            )
        records.append(
            {
                "child_code": str(row.child_code),
                "parent_code": str(row.parent_code),
                "subsplit_group": str(row.subsplit_group),
                "eb_product_selector": str(row.eb_product_selector),
                "eb_flow_selector": str(row.eb_flow_selector),
                "quantity_GWh": float(quantity),
                "selector_total_GWh": float(quantity),
                "quantity_source": source_note,
                "notes": str(row.notes),
            }
        )
    return pd.DataFrame(records, columns=columns).reset_index(drop=True)


def build_energy_balance_component_values(
    split_frame: pd.DataFrame,
    *,
    resolved_prices: pd.DataFrame,
    quantity_lookup: pd.DataFrame,
    parent_output_by_code: Mapping[str, float],
    tol: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quantity_table = _coerce_quantity_table(quantity_lookup)
    price_lookup = resolved_prices.set_index("price_code")["resolved_eur_per_mwh_lhv"].to_dict()
    price_status = resolved_prices.set_index("price_code")["status"].to_dict()

    active = split_frame.loc[split_frame["enabled"]].sort_values(["parent_code", "order"], kind="stable").copy()
    non_parent = active.loc[active["component_kind"] != "parent_residual"].copy()
    parent_rows = active.loc[active["component_kind"] == "parent_residual"].copy()

    quantity_by_child = quantity_table.set_index("child_code").to_dict("index") if not quantity_table.empty else {}
    selector_total_by_group = (
        quantity_table.groupby(["parent_code", "subsplit_group", "eb_flow_selector"], sort=False)["selector_total_GWh"].max().to_dict()
        if not quantity_table.empty else {}
    )
    advanced_used_by_group: dict[tuple[str, str], float] = {}

    rows: list[dict[str, object]] = []
    selector_residual_rows: list[dict[str, object]] = []
    parent_residual_rows: list[dict[str, object]] = []
    missing_or_zero_rows: list[dict[str, object]] = []

    for row in non_parent.itertuples(index=False):
        q_record = quantity_by_child.get(str(row.child_code))
        if q_record is None:
            raise ValueError(f"Quantity lookup is missing child_code {row.child_code!r}.")
        selector_total = float(
            selector_total_by_group.get(
                (str(row.parent_code), str(row.subsplit_group), str(row.eb_flow_selector)),
                q_record.get("selector_total_GWh", np.nan),
            )
        )
        price = float(price_lookup[str(row.price_code)])
        quantity_gwh: float

        if row.component_kind == "energy_quantity":
            quantity_gwh = float(q_record["quantity_GWh"])
            if quantity_gwh == 0.0:
                missing_or_zero_rows.append(
                    {
                        "parent_code": row.parent_code,
                        "child_code": row.child_code,
                        "subsplit_group": row.subsplit_group,
                        "selector": row.eb_product_selector,
                        "flow_selector": row.eb_flow_selector,
                        "reason": "zero_selector",
                    }
                )
        elif row.component_kind == "advanced_external_quantity":
            text = _normalize_text(row.external_quantity_mtoe)
            if text == "" or text.upper().startswith("TODO"):
                if not row.allow_zero:
                    raise ValueError(f"Missing advanced quantity for {row.child_code!r}.")
                quantity_gwh = 0.0
                missing_or_zero_rows.append(
                    {
                        "parent_code": row.parent_code,
                        "child_code": row.child_code,
                        "subsplit_group": row.subsplit_group,
                        "selector": row.eb_product_selector,
                        "flow_selector": row.eb_flow_selector,
                        "reason": "pending_advanced_quantity",
                    }
                )
            else:
                requested = _as_float(text) * GWH_PER_MTOE
                used_before = float(advanced_used_by_group.get((row.parent_code, row.subsplit_group), 0.0))
                remaining = max(float(selector_total) - used_before, 0.0)
                quantity_gwh = min(requested, remaining)
                advanced_used_by_group[(row.parent_code, row.subsplit_group)] = used_before + quantity_gwh
        elif row.component_kind == "selector_residual":
            used = float(advanced_used_by_group.get((row.parent_code, row.subsplit_group), 0.0))
            quantity_gwh = float(selector_total) - used
            if quantity_gwh < -tol:
                raise ValueError(f"Selector residual negative for {row.child_code!r}: {quantity_gwh:.12g} GWh.")
            if quantity_gwh < 0.0:
                warnings.warn(
                    f"Selector residual for {row.child_code!r} is within tolerance and has been clamped to zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                quantity_gwh = 0.0
            selector_residual_rows.append(
                {
                    "parent_code": row.parent_code,
                    "child_code": row.child_code,
                    "subsplit_group": row.subsplit_group,
                    "selector_total_GWh": float(selector_total),
                    "advanced_used_GWh": used,
                    "residual_GWh": quantity_gwh,
                }
            )
        else:
            raise ValueError(f"Unexpected component_kind for non-parent row: {row.component_kind!r}")

        value_meur = quantity_gwh * price / 1000.0
        rows.append(
            {
                "order": int(row.order),
                "enabled": bool(row.enabled),
                "parent_code": str(row.parent_code),
                "parent_label": str(row.parent_label),
                "child_code": str(row.child_code),
                "child_label": str(row.child_label),
                "component_kind": str(row.component_kind),
                "quantity_rule": str(row.quantity_rule),
                "subsplit_group": str(row.subsplit_group),
                "quantity_GWh": float(quantity_gwh),
                "price_code": str(row.price_code),
                "price_EUR_per_MWh_LHV": price,
                "value_MEUR": float(value_meur),
                "price_status": str(price_status.get(str(row.price_code), "")),
                "allow_zero": bool(row.allow_zero),
                "notes": str(row.notes),
            }
        )

    rows_df = pd.DataFrame(rows)
    for parent_code, group in parent_rows.groupby("parent_code", sort=False):
        parent_output = float(parent_output_by_code.get(str(parent_code), np.nan))
        if not np.isfinite(parent_output):
            raise ValueError(f"Missing parent output for {parent_code!r}.")
        declared = float(rows_df.loc[rows_df["parent_code"].eq(parent_code), "value_MEUR"].sum()) if not rows_df.empty else 0.0
        residual = parent_output - declared
        if residual < -tol:
            raise ValueError(f"Parent residual negative for {parent_code!r}: {residual:.12g} MEUR.")
        if residual < 0.0:
            warnings.warn(
                f"Parent residual for {parent_code!r} is within tolerance and has been clamped to zero.",
                RuntimeWarning,
                stacklevel=2,
            )
            residual = 0.0
        for row in group.itertuples(index=False):
            parent_residual_rows.append(
                {
                    "order": int(row.order),
                    "enabled": bool(row.enabled),
                    "parent_code": str(row.parent_code),
                    "parent_label": str(row.parent_label),
                    "child_code": str(row.child_code),
                    "child_label": str(row.child_label),
                    "component_kind": str(row.component_kind),
                    "quantity_rule": str(row.quantity_rule),
                    "subsplit_group": str(row.subsplit_group),
                    "quantity_GWh": np.nan,
                    "price_code": "",
                    "price_EUR_per_MWh_LHV": np.nan,
                    "value_MEUR": float(residual),
                    "parent_output_MEUR": float(parent_output),
                    "declared_components_MEUR": declared,
                    "residual_MEUR": float(residual),
                    "price_status": "",
                    "allow_zero": bool(row.allow_zero),
                    "notes": str(row.notes),
                }
            )
    if parent_residual_rows:
        rows_df = pd.concat([rows_df, pd.DataFrame(parent_residual_rows)], ignore_index=True, sort=False)
    rows_df = rows_df.sort_values(["parent_code", "order"], kind="stable").reset_index(drop=True)

    shares: list[dict[str, object]] = []
    for parent_code, group in rows_df.groupby("parent_code", sort=False):
        total = float(pd.to_numeric(group["value_MEUR"], errors="coerce").fillna(0.0).sum())
        if total <= 0:
            raise ValueError(f"Parent {parent_code!r} has non-positive split value total.")
        for row in group.itertuples(index=False):
            shares.append(
                {
                    "parent_code": str(parent_code),
                    "child_code": str(row.child_code),
                    "child_label": str(row.child_label),
                    "value_MEUR": float(row.value_MEUR),
                    "child_share": float(row.value_MEUR) / total,
                }
            )
    return (
        rows_df,
        pd.DataFrame(shares).sort_values(["parent_code", "child_code"], kind="stable").reset_index(drop=True),
        pd.DataFrame(selector_residual_rows),
        pd.DataFrame(parent_residual_rows),
        pd.DataFrame(missing_or_zero_rows),
    )


def build_split_profiles_from_component_values(component_values: pd.DataFrame) -> dict[str, SplitProfiles]:
    out: dict[str, SplitProfiles] = {}
    for parent_code, group in component_values.groupby("parent_code", sort=False):
        children = list(group["child_code"].astype(str))
        values = pd.to_numeric(group["value_MEUR"], errors="coerce").fillna(0.0).astype("float64")
        total = float(values.sum())
        if total <= 0:
            raise ValueError(f"Parent {parent_code!r} has non-positive value total.")
        shares = pd.Series((values / total).to_numpy(dtype="float64"), index=children, dtype="float64")
        out[str(parent_code)] = SplitProfiles(
            aggregate_sector=str(parent_code),
            subsectors=tuple(children),
            output_share=shares.copy(),
            intermediate_sales_share=shares.copy(),
            intermediate_input_share=shares.copy(),
            final_demand_share=shares.copy(),
        )
    return out


def _selector_token_matches(row_code: str, row_label: str, token: str) -> bool:
    """
    Explicit selector syntax:
      CPA_C19          exact original sector-code match
      label:<text>     exact normalized label match
      prefix:<text>    explicit prefix match
      regex:<pattern>  regex match against row code or row label

    There is deliberately no implicit prefix matching.
    """
    token = _normalize_text(token)
    if not token:
        return False

    row_code_n = _normalize_text(row_code).casefold()
    row_label_n = _normalize_text(row_label).casefold()

    if token.startswith("label:"):
        wanted = _normalize_text(token[len("label:") :]).casefold()
        return row_label_n == wanted
    if token.startswith("prefix:"):
        wanted = _normalize_text(token[len("prefix:") :]).casefold()
        return row_code_n.startswith(wanted) or row_label_n.startswith(wanted)
    if token.startswith("regex:"):
        pattern = token[len("regex:") :]
        return re.search(pattern, row_code, flags=re.IGNORECASE) is not None or re.search(
            pattern, row_label, flags=re.IGNORECASE
        ) is not None

    return row_code_n == token.casefold()


def _selector_matches(row_code: str, row_label: str, selector: str) -> bool:
    return any(_selector_token_matches(row_code, row_label, token) for token in _split_tokens(selector))


def _validate_supplier_selectors_against_original_rows(
    technology_frame: pd.DataFrame,
    *,
    original_row_codes: set[str],
) -> None:
    invalid: list[str] = []
    active = technology_frame.loc[technology_frame["enabled"]].copy()
    for row in active.itertuples(index=False):
        for token in _split_tokens(row.supplier_selector):
            if token.startswith(("label:", "prefix:", "regex:")):
                continue
            if token not in original_row_codes:
                invalid.append(f"{row.parent_code}:{row.buyer_split_code}:{token}")

    if invalid:
        raise ValueError(
            "Technology supplier_selector contains bare tokens that are not original Eurostat sector rows. "
            "Use only original row codes, or explicit label:/prefix:/regex: selectors. "
            f"Examples: {invalid[:20]!r}"
        )


def build_column_allocator(
    Z: pd.DataFrame,
    technology_frame: pd.DataFrame,
    *,
    split_profiles: Mapping[str, SplitProfiles],
    row_code_to_label: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, pd.DataFrame]]:
    """Build supplier-specific child-column shares from the technology CSV."""
    allocator: dict[str, dict[str, pd.Series]] = {}
    matches_rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    enabled = technology_frame.loc[
        technology_frame["enabled"] & technology_frame["apply_to"].astype(str).str.upper().eq("Z_COLUMNS")
    ].copy()

    empty = {
        "column_allocator_matches": pd.DataFrame(),
        "column_allocator_fallbacks": pd.DataFrame(),
        "column_allocator_summary": pd.DataFrame(),
    }
    if enabled.empty:
        return allocator, empty

    Z = Z.copy()
    Z.index = Z.index.astype(str)
    Z.columns = Z.columns.astype(str)
    original_row_codes = set(map(str, Z.index))
    _validate_supplier_selectors_against_original_rows(enabled, original_row_codes=original_row_codes)

    row_code_to_label = {str(k): str(v) for k, v in (row_code_to_label or {}).items()}

    for parent_code, parent_group in enabled.groupby("parent_code", sort=False):
        parent_code = str(parent_code)
        prof = split_profiles.get(parent_code)
        if prof is None:
            continue

        child_labels = list(map(str, prof.subsectors))
        fallback_share = prof.intermediate_input_share.reindex(child_labels).fillna(0.0).astype("float64")
        fallback_total = float(fallback_share.sum())
        if fallback_total <= 0:
            raise ValueError(f"Non-positive intermediate_input_share for parent {parent_code!r}.")
        fallback_share = fallback_share / fallback_total

        parent_matches = 0
        parent_fallbacks = 0
        matched_value = 0.0
        fallback_value = 0.0
        parent_total = 0.0

        for supplier_row in Z.index.astype(str):
            supplier_label = row_code_to_label.get(str(supplier_row), str(supplier_row))
            original_cell_value = float(Z.loc[supplier_row, parent_code]) if parent_code in Z.columns else 0.0
            row_contrib = pd.Series(0.0, index=child_labels, dtype="float64")
            matched_rules: list[str] = []
            matched_selectors: list[str] = []

            for tech_row in parent_group.itertuples(index=False):
                if not _selector_matches(str(supplier_row), supplier_label, tech_row.supplier_selector):
                    continue
                buyer_tokens = [token for token in _split_tokens(tech_row.buyer_split_code) if token in child_labels]
                if not buyer_tokens:
                    continue
                weight = float(tech_row.allocation_weight)
                if not np.isfinite(weight) or weight <= 0:
                    weight = 1.0

                if str(tech_row.rule) == "equal_split":
                    base = pd.Series(1.0, index=buyer_tokens, dtype="float64")
                else:
                    base = fallback_share.reindex(buyer_tokens).fillna(0.0).astype("float64")
                    if float(base.sum()) <= 0:
                        base = pd.Series(1.0, index=buyer_tokens, dtype="float64")
                base = base / float(base.sum())
                contrib = pd.Series(0.0, index=child_labels, dtype="float64")
                contrib.loc[buyer_tokens] = weight * base
                row_contrib = row_contrib.add(contrib, fill_value=0.0)
                matched_rules.append(str(tech_row.rule))
                matched_selectors.append(str(tech_row.supplier_selector))

            if matched_rules:
                total = float(row_contrib.sum())
                if total <= 0:
                    row_contrib = fallback_share.copy()
                    allocation_method = "matched_but_zero_weight_child_share_fallback"
                else:
                    row_contrib = row_contrib / total
                    allocation_method = "technology_rule"
                allocator.setdefault(parent_code, {})[str(supplier_row)] = row_contrib.astype("float64")
                parent_matches += 1
                matched_value += original_cell_value
                matches_rows.append(
                    {
                        "parent_code": parent_code,
                        "supplier_row": str(supplier_row),
                        "supplier_label": supplier_label,
                        "supplier_selector": "|".join(matched_selectors),
                        "buyer_split_code": "|".join(map(str, row_contrib.index[row_contrib > 0])),
                        "original_cell_value": original_cell_value,
                        "allocation_method": allocation_method,
                        "matched_rule": "|".join(matched_rules),
                        "allocation_vector_json": row_contrib.to_json(),
                    }
                )
            else:
                allocator.setdefault(parent_code, {})[str(supplier_row)] = fallback_share.copy()
                parent_fallbacks += 1
                fallback_value += original_cell_value
                if original_cell_value != 0.0:
                    fallback_rows.append(
                        {
                            "parent_code": parent_code,
                            "supplier_row": str(supplier_row),
                            "supplier_label": supplier_label,
                            "supplier_selector": "",
                            "buyer_split_code": "|".join(child_labels),
                            "original_cell_value": original_cell_value,
                            "allocation_method": "child_share_fallback",
                            "matched_rule": "fallback",
                            "allocation_vector_json": fallback_share.to_json(),
                        }
                    )
            parent_total += original_cell_value

        summary_rows.append(
            {
                "parent_code": parent_code,
                "n_supplier_rows": int(len(Z.index)),
                "n_matched_supplier_rows": int(parent_matches),
                "n_fallback_supplier_rows": int(parent_fallbacks),
                "original_cell_value_sum": float(parent_total),
                "matched_cell_value_sum": float(matched_value),
                "fallback_cell_value_sum": float(fallback_value),
            }
        )

    diagnostics = {
        "column_allocator_matches": pd.DataFrame(matches_rows),
        "column_allocator_fallbacks": pd.DataFrame(fallback_rows),
        "column_allocator_summary": pd.DataFrame(summary_rows),
    }
    return allocator, diagnostics


def compile_energy_balance_pxp_split_plan(
    *,
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    split_frame: pd.DataFrame,
    reaggregation_frame: pd.DataFrame,
    technology_frame: pd.DataFrame,
    prices_frame: pd.DataFrame,
    energy_balance_dictionary_frame: pd.DataFrame | None = None,
    geo: str,
    year: int,
    quantity_table_override: pd.DataFrame | None = None,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    force_reparse: bool = False,
    tol: float = 1e-10,
) -> EnergyBalancePxpSplitPlan:
    resolved_prices = validate_energy_balance_pxp_inputs(
        split_frame=split_frame,
        reaggregation_frame=reaggregation_frame,
        technology_frame=technology_frame,
        prices_frame=prices_frame,
        energy_balance_dictionary_frame=energy_balance_dictionary_frame,
    )
    if quantity_table_override is None:
        quantity_table = build_energy_balance_pxp_quantity_lookup(
            split_frame=split_frame,
            geo=geo,
            year=year,
            storage_folder=storage_folder,
            force_download=force_download,
            force_reparse=force_reparse,
        )
    else:
        quantity_table = _coerce_quantity_table(quantity_table_override)
    parent_output_by_code = (Z.sum(axis=1) + Y.sum(axis=1)).astype("float64").to_dict()
    component_values, split_shares, selector_residuals, parent_residuals, missing_or_zero = build_energy_balance_component_values(
        split_frame,
        resolved_prices=resolved_prices,
        quantity_lookup=quantity_table,
        parent_output_by_code=parent_output_by_code,
        tol=tol,
    )
    split_profiles = build_split_profiles_from_component_values(component_values)
    column_allocator, column_allocator_diagnostics = build_column_allocator(
        Z,
        technology_frame,
        split_profiles=split_profiles,
    )
    reaggregation_mapping = _validate_reaggregation_frame(split_frame, reaggregation_frame)
    diagnostics = {
        "missing_or_zero_selectors": missing_or_zero,
        "quantity_lookup": quantity_table,
        **column_allocator_diagnostics,
        "validation_summary": pd.DataFrame(
            [
                {
                    "parents": ",".join(sorted(split_profiles.keys())),
                    "n_components": int(len(component_values)),
                    "n_zero_or_missing_selectors": int(len(missing_or_zero)),
                    "placeholder_prices": ",".join(
                        resolved_prices.loc[resolved_prices["status"].astype(str).eq("placeholder_to_replace"), "price_code"].astype(str).tolist()
                    ),
                    "technology_allocator_status": "applied_with_child_share_fallback",
                }
            ]
        ),
    }
    return EnergyBalancePxpSplitPlan(
        split_profiles=split_profiles,
        column_allocator=column_allocator,
        component_values=component_values,
        split_shares=split_shares,
        selector_residuals=selector_residuals,
        parent_residuals=parent_residuals,
        price_resolution=resolved_prices,
        diagnostics=diagnostics,
        reaggregation_mapping=reaggregation_mapping,
    )


def run_energy_balance_pxp_split(
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    plan: EnergyBalancePxpSplitPlan,
    *,
    balancing_scope: str = "local",
    max_iter: int = 50_000,
    tol: float = 1e-10,
    create_io_system: bool = False,
):
    return disaggregate_and_balance(
        Z,
        Y,
        split_profiles=plan.split_profiles,
        column_allocator=plan.column_allocator,
        balancing_scope=balancing_scope,
        max_iter=max_iter,
        tol=tol,
        create_io_system=create_io_system,
    )


def write_energy_balance_pxp_diagnostics(
    outdir: str | Path,
    *,
    result: Mapping[str, object],
    missing_or_zero_selectors: pd.DataFrame | None = None,
) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for key in ("component_values", "split_shares", "selector_residuals", "parent_residuals", "price_resolution"):
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            value.to_csv(out / f"{key}.csv", index=False)
    diagnostics = result.get("diagnostics", {})
    if isinstance(diagnostics, Mapping):
        for key, value in diagnostics.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(out / f"{key}.csv", index=False)
    if missing_or_zero_selectors is not None:
        missing_or_zero_selectors.to_csv(out / "missing_or_zero_selectors.csv", index=False)
    meta = {"diagnostic_keys": sorted([str(k) for k, v in (diagnostics or {}).items() if isinstance(v, pd.DataFrame)])}
    (out / "diagnostics_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
