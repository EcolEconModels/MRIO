from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ecolecon_mrio.balancing import BalanceStats
from ecolecon_mrio.disaggregation import SplitProfiles, disaggregate_and_balance
from ecolecon_mrio.exiobase3 import (
    ExioRegionSlice,
    build_split_profiles_from_exio_domestic_block,
    load_exiobase3_region_slice,
)
from ecolecon_mrio.exiobase3_extensions import (
    extract_quantity_and_price as extract_exio_quantity_and_price,
)


@dataclass(frozen=True)
class SplitResult:
    Z: pd.DataFrame
    Y: pd.DataFrame
    x: pd.Series
    A: pd.DataFrame
    split_profiles: dict[str, SplitProfiles]
    balancing_method: str
    balancing_stats: BalanceStats
    exio_selected_sectors: dict[str, list[str]]
    exio_quantity_and_price: pd.DataFrame | None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, index_col=0)
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, index_col=0)
    raise ValueError(f"Unsupported table format: {path}")


def load_local_eurostat_iot(
    z_path: str | Path,
    y_path: str | Path,
    *,
    dtype: str | np.dtype = "float64",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a Eurostat national IOT provided as two tables (Z and Y).

    This loader is file-format agnostic (csv/parquet/pickle/xlsx).
    """
    Z = _read_table(Path(z_path)).astype(dtype)
    Y = _read_table(Path(y_path)).astype(dtype)

    Z.index = Z.index.astype(str)
    Z.columns = Z.columns.astype(str)
    Y.index = Y.index.astype(str)
    Y.columns = Y.columns.astype(str)

    if not Z.index.equals(Z.columns):
        raise ValueError("Eurostat Z must be square with matching index/columns.")
    if not Y.index.equals(Z.index):
        raise ValueError("Eurostat Y must have the same index as Z.")

    return Z, Y


def build_split_profiles_from_exiobase_domestic_block(
    exio_slice: ExioRegionSlice,
    *,
    sector_patterns: Mapping[str, Sequence[str]],
    prefix_subsectors: bool = True,
    prefix_template: str = "{aggregate}::",
) -> tuple[dict[str, SplitProfiles], dict[str, list[str]]]:
    """
    Build split profiles (and record matched EXIO sectors) for any set of aggregate sectors.

    Parameters
    ----------
    sector_patterns:
        Mapping of aggregate sector code -> list of substring patterns used to match EXIO sectors.
    prefix_subsectors:
        If True, prefix each generated subsector label with `prefix_template.format(aggregate=...)`.
    prefix_template:
        Formatting template for subsector labels when prefixing is enabled.
    """
    profiles: dict[str, SplitProfiles] = {}
    selected: dict[str, list[str]] = {}
    for aggregate, patterns in sector_patterns.items():
        prefix = prefix_template.format(aggregate=str(aggregate)) if prefix_subsectors else None
        prof, matched = build_split_profiles_from_exio_domestic_block(
            exio_slice,
            aggregate_sector=str(aggregate),
            exio_sector_patterns=tuple(patterns),
            prefix_labels_with=prefix,
        )
        profiles[str(aggregate)] = prof
        selected[str(aggregate)] = list(matched)
    return profiles, selected


def split_iot_using_exiobase_split_profiles(
    *,
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    exio_slice: ExioRegionSlice,
    sector_patterns: Mapping[str, Sequence[str]],
    prefix_subsectors: bool = True,
    prefix_template: str = "{aggregate}::",
    ras_max_iter: int = 50_000,
    ras_tol: float = 1e-10,
    extract_quantity_and_price: bool = False,
    quantity_price_extensions: Sequence[str] | None = None,
    quantity_price_stressor_keywords: Sequence[str] | None = None,
    quantity_price_cache_dir: str | Path | None = None,
) -> SplitResult:
    """
    Split one or more aggregate sectors in (Z,Y) using EXIOBASE domestic-block shares and balance via RAS/GRAS.

    This is a generic utility: callers specify which aggregates to split via `sector_patterns`.
    """
    profiles, selected = build_split_profiles_from_exiobase_domestic_block(
        exio_slice,
        sector_patterns=sector_patterns,
        prefix_subsectors=prefix_subsectors,
        prefix_template=prefix_template,
    )

    res = disaggregate_and_balance(
        eurostat_Z, eurostat_Y, split_profiles=profiles, max_iter=ras_max_iter, tol=ras_tol
    )

    exio_qp = None
    if extract_quantity_and_price:
        if quantity_price_extensions is None or quantity_price_stressor_keywords is None:
            raise ValueError(
                "When extract_quantity_and_price=True, also provide "
                "quantity_price_extensions and quantity_price_stressor_keywords."
            )
        exio_sectors: list[str] = []
        for agg in profiles.keys():
            exio_sectors.extend(selected.get(str(agg), []))
        exio_sectors = list(dict.fromkeys(exio_sectors))

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
    )


def split_iot_using_exiobase_shares(
    *,
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    exio_year: int,
    exio_region: str,
    exio_slice: ExioRegionSlice | None = None,
    exio_system: str = "pxp",
    exio_cache_root: str | Path | None = None,
    sector_patterns: Mapping[str, Sequence[str]],
    prefix_subsectors: bool = True,
    prefix_template: str = "{aggregate}::",
    ras_max_iter: int = 50_000,
    ras_tol: float = 1e-10,
    extract_quantity_and_price: bool = False,
    quantity_price_extensions: Sequence[str] | None = None,
    quantity_price_stressor_keywords: Sequence[str] | None = None,
    quantity_price_cache_dir: str | Path | None = None,
) -> SplitResult:
    """
    Convenience wrapper: load an EXIOBASE region slice if needed, then split+balance.
    """
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
        exio_slice=exio_slice,
        sector_patterns=sector_patterns,
        prefix_subsectors=prefix_subsectors,
        prefix_template=prefix_template,
        ras_max_iter=ras_max_iter,
        ras_tol=ras_tol,
        extract_quantity_and_price=extract_quantity_and_price,
        quantity_price_extensions=quantity_price_extensions,
        quantity_price_stressor_keywords=quantity_price_stressor_keywords,
        quantity_price_cache_dir=quantity_price_cache_dir,
    )


def write_energy_split_report_xlsx(
    out_path: str | Path,
    *,
    result: SplitResult,
    aggregates: Sequence[str],
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    energy_subsectors: list[str] = []
    for agg in aggregates:
        prof = result.split_profiles.get(str(agg))
        if prof is None:
            continue
        energy_subsectors.extend(list(map(str, prof.subsectors)))
    energy_subsectors = list(dict.fromkeys(energy_subsectors))

    def _safe_sheet(name: str) -> str:
        return str(name)[:31]

    rows: list[pd.DataFrame] = []
    for agg in aggregates:
        prof = result.split_profiles.get(str(agg))
        if prof is None:
            continue
        subs = pd.Index(prof.subsectors, dtype=str)
        df = pd.DataFrame(
            {
                "aggregate_sector": str(agg),
                "subsector": subs,
                "output_share": prof.output_share.reindex(subs).to_numpy(),
                "intermediate_sales_share": prof.intermediate_sales_share.reindex(subs).to_numpy(),
                "intermediate_input_share": prof.intermediate_input_share.reindex(subs).to_numpy(),
                "final_demand_share": prof.final_demand_share.reindex(subs).to_numpy(),
            }
        )
        rows.append(df)
    weights = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    for agg in aggregates:
        prof = result.split_profiles.get(str(agg))
        if prof is None:
            continue
        subs = list(map(str, prof.subsectors))
        row_total = float(result.Z.loc[subs, :].sum(axis=1).sum() + result.Y.loc[subs, :].sum(axis=1).sum())
        col_total = float(result.Z.loc[:, subs].sum(axis=0).sum())
        summary_rows.append(
            {
                "aggregate_sector": str(agg),
                "num_subsectors": len(subs),
                "sum_output_share": float(prof.output_share.sum()),
                "sum_intermediate_sales_share": float(prof.intermediate_sales_share.sum()),
                "sum_intermediate_input_share": float(prof.intermediate_input_share.sum()),
                "sum_final_demand_share": float(prof.final_demand_share.sum()),
                "total_row_output_in_split_tables": row_total,
                "total_col_intermediate_input_in_split_tables": col_total,
            }
        )
    summary = pd.DataFrame(summary_rows)

    Z_rows = result.Z.loc[energy_subsectors, :] if energy_subsectors else pd.DataFrame()
    Z_cols = result.Z.loc[:, energy_subsectors] if energy_subsectors else pd.DataFrame()
    Z_block = result.Z.loc[energy_subsectors, energy_subsectors] if energy_subsectors else pd.DataFrame()
    Y_rows = result.Y.loc[energy_subsectors, :] if energy_subsectors else pd.DataFrame()

    exio_sel = (
        pd.DataFrame(
            [
                {"aggregate_sector": agg, "exio_sector": s}
                for agg, sectors in result.exio_selected_sectors.items()
                for s in sectors
            ]
        )
        if result.exio_selected_sectors
        else pd.DataFrame()
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name=_safe_sheet("summary"), index=False)
        weights.to_excel(xw, sheet_name=_safe_sheet("split_weights"), index=False)
        exio_sel.to_excel(xw, sheet_name=_safe_sheet("exio_selected"), index=False)
        Z_rows.to_excel(xw, sheet_name=_safe_sheet("Z_rows_energy"))
        Z_cols.to_excel(xw, sheet_name=_safe_sheet("Z_cols_energy"))
        Z_block.to_excel(xw, sheet_name=_safe_sheet("Z_energy_block"))
        Y_rows.to_excel(xw, sheet_name=_safe_sheet("Y_rows_energy"))
        if result.exio_quantity_and_price is not None:
            result.exio_quantity_and_price.to_excel(xw, sheet_name=_safe_sheet("exio_qty_price"), index=False)

    return out_path
