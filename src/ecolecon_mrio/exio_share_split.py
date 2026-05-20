from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping as ABCMapping
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ecolecon_mrio.balancing import BalanceStats
from ecolecon_mrio.disaggregation import SplitProfiles, calc_A, calc_x
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
    disaggregated_sector_value_per_physical_input: pd.DataFrame | None = None
    io_system: object | None = None  # pymrio.IOSystem when available


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
    return out.groupby(level=0).sum()


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
    out = out.groupby(level=0).sum()
    return out.T.groupby(level=0).sum().T


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
    return f"{where[0]} -> {where[1]}", float(stack.max())


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
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype("float64")
    total = float(numeric.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Cannot normalize non-positive donor vector.")
    return numeric / total


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


def _weighted_block_total(
    donor_block: pd.DataFrame,
    *,
    row_weights: Mapping[str, float],
    col_weights: Mapping[str, float],
) -> float:
    row = pd.Series({str(k): float(v) for k, v in row_weights.items() if float(v) != 0.0}, dtype="float64")
    col = pd.Series({str(k): float(v) for k, v in col_weights.items() if float(v) != 0.0}, dtype="float64")
    if row.empty or col.empty:
        return 0.0

    sub = donor_block.reindex(index=row.index.astype(str), columns=col.index.astype(str)).fillna(0.0)
    return float(row.to_numpy(dtype="float64") @ sub.to_numpy(dtype="float64") @ col.to_numpy(dtype="float64"))


def _build_label_to_exio_weights(
    *,
    eurostat_sectors: Sequence[str],
    exio_sectors: Sequence[str],
    split_profiles: Mapping[str, SplitProfiles],
    sector_to_exio_sectors: Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None,
    eurostat_sector_mapping: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}

    if eurostat_sector_mapping is not None:
        for sector, weights in eurostat_sector_mapping.items():
            clean_weights = {
                str(exio_sector): float(weight)
                for exio_sector, weight in weights.items()
                if float(weight) != 0.0
            }
            if clean_weights:
                out[str(sector)] = clean_weights

    available_exio = set(map(str, exio_sectors))
    for sector in map(str, eurostat_sectors):
        if sector not in out and sector in available_exio:
            out[sector] = {sector: 1.0}

    for agg_sector, prof in split_profiles.items():
        subsector_map = _resolve_subsector_to_exio_sectors(
            agg_sector=str(agg_sector),
            prof=prof,
            sector_to_exio_sectors=sector_to_exio_sectors,
        )
        if not subsector_map:
            continue
        for subsector, exio_members in subsector_map.items():
            members = [str(member) for member in exio_members if str(member) in available_exio]
            if members:
                out[str(subsector)] = {member: 1.0 for member in members}

    return out


def _build_fd_category_donor_shares(
    *,
    exio_slice: ExioRegionSlice,
    split_profiles: Mapping[str, SplitProfiles],
    sector_to_exio_sectors: Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None,
    eurostat_fd_codes: Sequence[str],
) -> dict[str, dict[str, pd.Series]]:
    Y_region = exio_slice.Y_region.copy().astype("float64")
    Y_region.index = Y_region.index.astype(str)
    Y_region.columns = Y_region.columns.astype(str)
    donor_shares: dict[str, dict[str, pd.Series]] = {}

    for agg_sector, prof in split_profiles.items():
        subsector_map = _resolve_subsector_to_exio_sectors(
            agg_sector=str(agg_sector),
            prof=prof,
            sector_to_exio_sectors=sector_to_exio_sectors,
        )
        if not subsector_map:
            continue

        shares_by_code: dict[str, pd.Series] = {}
        for fd_code in map(str, eurostat_fd_codes):
            keys = _fd_category_keys_for_eurostat_code(fd_code)
            if not keys:
                continue
            cols = [
                col
                for col in Y_region.columns
                if any(key in str(col).lower() for key in keys)
            ]
            if not cols:
                continue

            donor_values = pd.Series(0.0, index=list(map(str, prof.subsectors)), dtype="float64")
            for subsector, exio_members in subsector_map.items():
                donor_values.loc[str(subsector)] = float(
                    Y_region.reindex(index=list(map(str, exio_members)), columns=cols).fillna(0.0).to_numpy(dtype="float64").sum()
                )

            total = float(donor_values.sum())
            if np.isfinite(total) and total > 0.0:
                shares_by_code[fd_code] = donor_values / total

        if shares_by_code:
            donor_shares[str(agg_sector)] = shares_by_code

    return donor_shares


def _split_iot_using_donor_blocks(
    *,
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    exio_slice: ExioRegionSlice,
    split_profiles: Mapping[str, SplitProfiles],
    sector_to_exio_sectors: Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None,
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

    label_to_exio_weights = _build_label_to_exio_weights(
        eurostat_sectors=sectors_old,
        exio_sectors=list(map(str, exio_slice.sectors)),
        split_profiles=split_profiles,
        sector_to_exio_sectors=sector_to_exio_sectors,
        eurostat_sector_mapping=eurostat_sector_mapping,
    )
    fd_donor_shares = _build_fd_category_donor_shares(
        exio_slice=exio_slice,
        split_profiles=split_profiles,
        sector_to_exio_sectors=sector_to_exio_sectors,
        eurostat_fd_codes=Y.columns.astype(str),
    )

    donor_Z = exio_slice.Z_domestic.copy().astype("float64")
    donor_Z.index = donor_Z.index.astype(str)
    donor_Z.columns = donor_Z.columns.astype(str)

    donor_fallback_count = 0

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

            donor_values = pd.DataFrame(0.0, index=row_labels, columns=col_labels, dtype="float64")
            for row_label in row_labels:
                row_weights = label_to_exio_weights.get(str(row_label))
                if not row_weights:
                    continue
                for col_label in col_labels:
                    col_weights = label_to_exio_weights.get(str(col_label))
                    if not col_weights:
                        continue
                    donor_values.loc[row_label, col_label] = _weighted_block_total(
                        donor_Z,
                        row_weights=row_weights,
                        col_weights=col_weights,
                    )

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
            elif row_sector in split_profiles:
                row_share = _normalize_vector(
                    split_profiles[row_sector].intermediate_sales_share.reindex(row_labels).fillna(0.0)
                )
                alloc = pd.DataFrame(
                    row_share.to_numpy().reshape(-1, 1) * cell_total,
                    index=row_labels,
                    columns=col_labels,
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
            else:
                alloc = pd.DataFrame([[cell_total]], index=row_labels, columns=col_labels)
                donor_fallback_count += 1

            Z_split.loc[row_labels, col_labels] = alloc.to_numpy(dtype="float64")

    for row_sector in sectors_old:
        row_labels = list(split_profiles[row_sector].subsectors) if row_sector in split_profiles else [row_sector]
        if len(row_labels) == 1:
            Y_split.loc[row_labels[0], :] = Y.loc[row_sector, :].to_numpy(dtype="float64")
            continue

        donor_shares_by_code = fd_donor_shares.get(str(row_sector), {})
        for fd_code in Y.columns.astype(str):
            fd_total = float(Y.loc[row_sector, fd_code])
            if fd_total == 0.0:
                continue
            donor_share = donor_shares_by_code.get(str(fd_code))
            if donor_share is None:
                donor_share = split_profiles[row_sector].final_demand_share.reindex(row_labels).fillna(0.0)
                donor_share = _normalize_vector(donor_share)
            Y_split.loc[row_labels, fd_code] = donor_share.reindex(row_labels).fillna(0.0).to_numpy(dtype="float64") * fd_total

    x_split = calc_x(Z_split, Y_split)
    A_split = calc_A(Z_split, x_split)

    io_system = None
    if create_io_system:
        try:
            import pymrio

            io_system = pymrio.IOSystem()
            io_system.Z = Z_split
            io_system.Y = Y_split
            io_system.x = x_split
            io_system.A = A_split
        except ImportError:
            pass

    method = "DONOR_BLOCK" if donor_fallback_count == 0 else "DONOR_BLOCK_WITH_SHARE_FALLBACK"
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
        io_system=io_system,
    )


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
                    {sub: float(x_use.loc[pd.Index(members)].sum()) for sub, members in subsector_to_exio.items()},
                    dtype="float64",
                )
                sales_sub = pd.Series(
                    {
                        sub: float(Z_sales_sum.loc[pd.Index(members)].sum())
                        for sub, members in subsector_to_exio.items()
                    },
                    dtype="float64",
                )
                inputs_sub = pd.Series(
                    {
                        sub: float(Z_inputs_sum.loc[pd.Index(members)].sum())
                        for sub, members in subsector_to_exio.items()
                    },
                    dtype="float64",
                )
                fd_sub = pd.Series(
                    {
                        sub: float(Y_fd_sum.loc[pd.Index(members)].sum())
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

                flat_members: list[str] = []
                for members in subsector_to_exio.values():
                    flat_members.extend(members)
                selected[str(aggregate)] = list(dict.fromkeys(flat_members))
            else:
                requested_codes = [str(member).strip() for member in exio_sectors if str(member).strip() != ""]
                missing = [code for code in requested_codes if code not in available_codes]
                if missing:
                    raise ValueError(f"aggregate_sector={aggregate!r}: unknown EXIO sector code(s): {missing!r}")
                subsector_to_exio = {code: [code] for code in requested_codes}

                x_sub = pd.Series(
                    {sub: float(x_use.loc[pd.Index(members)].sum()) for sub, members in subsector_to_exio.items()},
                    dtype="float64",
                )
                sales_sub = pd.Series(
                    {
                        sub: float(Z_sales_sum.loc[pd.Index(members)].sum())
                        for sub, members in subsector_to_exio.items()
                    },
                    dtype="float64",
                )
                inputs_sub = pd.Series(
                    {
                        sub: float(Z_inputs_sum.loc[pd.Index(members)].sum())
                        for sub, members in subsector_to_exio.items()
                    },
                    dtype="float64",
                )
                fd_sub = pd.Series(
                    {
                        sub: float(Y_fd_sum.loc[pd.Index(members)].sum())
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

    df[numerator_col] = pd.to_numeric(df[numerator_col], errors="coerce")
    df[denominator_col] = pd.to_numeric(df[denominator_col], errors="coerce")

    rows: list[dict] = []
    for agg_sector, prof in split_profiles.items():
        subsector_map = _resolve_subsector_to_exio_sectors(
            agg_sector=str(agg_sector),
            prof=prof,
            sector_to_exio_sectors=sector_to_exio_sectors,
        )
        if not subsector_map:
            continue

        for subsector, exio_sectors in subsector_map.items():
            exio_subset = df[df["exio_sector"].isin(list(map(str, exio_sectors)))]
            if exio_subset.empty:
                continue

            grouped = exio_subset.groupby(["extension", "stressor", "unit"], dropna=False)
            for (ext, stressor, unit), group in grouped:
                total_num = float(group[numerator_col].sum())
                if not np.isfinite(total_num):
                    continue
                total_den = float(group[denominator_col].sum())
                if not np.isfinite(total_den) or total_den == 0:
                    continue
                rows.append(
                    {
                        "disaggregated_sector": subsector,
                        "extension": ext,
                        "stressor": stressor,
                        "unit": unit,
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
) -> pd.DataFrame | None:
    """Map EXIOBASE unit-value proxies to disaggregated sectors."""
    if exio_quantity_and_price is None or not split_profiles:
        return None

    exio_qp = exio_quantity_and_price.copy()
    if "value_per_physical_input" not in exio_qp.columns and "implied_price" in exio_qp.columns:
        exio_qp = exio_qp.rename(columns={"implied_price": "value_per_physical_input"})

    return _map_exio_ratio_to_disaggregated_sectors(
        exio_df=exio_qp,
        split_profiles=split_profiles,
        sector_to_exio_sectors=sector_to_exio_sectors,
        numerator_col="monetary_output",
        denominator_col="quantity",
        out_ratio_col="value_per_physical_input",
    )


def map_exio_emission_intensity(
    exio_emissions_and_intensity: pd.DataFrame | None,
    split_profiles: dict[str, SplitProfiles],
    sector_to_exio_sectors: Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> pd.DataFrame | None:
    """Map EXIOBASE emissions intensity (ratio-of-sums) to disaggregated sectors."""
    return _map_exio_ratio_to_disaggregated_sectors(
        exio_df=exio_emissions_and_intensity,
        split_profiles=split_profiles,
        sector_to_exio_sectors=sector_to_exio_sectors,
        numerator_col="emission",
        denominator_col="monetary_output",
        out_ratio_col="emission_intensity",
    )


def split_iot_using_exiobase_split_profiles(
    *,
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
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
    profiles, selected = build_split_profiles_from_exiobase_domestic_block(
        exio_slice,
        sector_to_exio_sectors=sector_to_exio_sectors,
        sector_patterns=sector_patterns,
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
            selected_list = list(map(str, selected.get(agg_s, [])))

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
        exio_slice=exio_slice,
        split_profiles=profiles,
        sector_to_exio_sectors=sector_to_exio_sectors,
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
        disaggregated_sector_value_per_physical_input=res.disaggregated_sector_value_per_physical_input,
        io_system=res.io_system,
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

    with pd.ExcelWriter(out_path) as xw:
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
