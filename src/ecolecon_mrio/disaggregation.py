from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from ecolecon_mrio.balancing import BalanceStats, balance_biproportional

if TYPE_CHECKING:
    try:
        import pymrio
    except ImportError:
        pymrio = None  # type: ignore


@dataclass(frozen=True)
class SplitProfiles:
    """
    Split shares for one aggregate sector into multiple subsectors.

    Shares are 1D and should sum to 1 within each aggregate.
    """

    aggregate_sector: str
    subsectors: tuple[str, ...]
    output_share: pd.Series
    intermediate_sales_share: pd.Series
    intermediate_input_share: pd.Series
    final_demand_share: pd.Series


@dataclass(frozen=True)
class DisaggregationResult:
    """
    Result of disaggregation operation.
    
    Attributes:
        Z: Balanced intermediate transactions matrix
        Y: Balanced final demand matrix
        x: Total output vector
        A: Technical coefficients matrix
        split_profiles: Share profiles used for disaggregation
        balancing_method: Method used ("RAS" or "GRAS")
        balancing_stats: Convergence statistics
        io_system: Optional pymrio IOSystem for advanced analysis
    """
    Z: pd.DataFrame
    Y: pd.DataFrame
    x: pd.Series
    A: pd.DataFrame
    split_profiles: dict[str, SplitProfiles]
    balancing_method: str
    balancing_stats: BalanceStats
    io_system: object | None = None  # pymrio.IOSystem when available


def normalize_share(series: pd.Series | pd.DataFrame, *, name: str) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError(f"Cannot normalize {name}: expected 1D share vector, got shape={series.shape}.")
        series = series.iloc[:, 0]

    series = series.astype("float64").clip(lower=0)
    total = float(series.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Cannot normalize {name}: sum is {total}.")
    return series / total


def _outer_share_matrix(row_share: pd.Series, col_share: pd.Series) -> pd.DataFrame:
    """
    Build outer product matrix from normalized share vectors.
    
    Since both input shares sum to 1, their outer product also sums to 1.
    No additional normalization is needed.
    """
    M = np.outer(row_share.to_numpy(), col_share.to_numpy())
    # Validate assumption: input shares must be normalized.
    # Use a proper exception (not `assert`) so this cannot be skipped with `python -O`.
    tol = 1e-10
    row_sum = float(row_share.sum())
    col_sum = float(col_share.sum())
    if not np.isfinite(row_sum) or not np.isfinite(col_sum):
        raise ValueError(
            "Cannot build outer share matrix: share sums are non-finite. "
            f"row_share.sum()={row_sum}, col_share.sum()={col_sum}."
        )
    if abs(row_sum - 1.0) > tol or abs(col_sum - 1.0) > tol:
        row_nan = int(row_share.isna().sum())
        col_nan = int(col_share.isna().sum())
        raise ValueError(
            "Cannot build outer share matrix: expected normalized 1D shares (sum≈1). "
            f"row_share.sum()={row_sum} (nan={row_nan}), col_share.sum()={col_sum} (nan={col_nan}). "
            "This usually indicates missing values after reindexing or an un-normalized share vector."
        )
    return pd.DataFrame(M, index=row_share.index, columns=col_share.index)


def expand_io_tables(
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    split_profiles: Mapping[str, SplitProfiles],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expand an IO table (Z,Y) by replacing each aggregate sector with its subsectors.

    The expansion is "clean": aggregating the expanded tables back reproduces the
    original totals (before balancing).
    """
    Z = Z.copy()
    Y = Y.copy()
    Z.index = Z.index.astype(str)
    Z.columns = Z.columns.astype(str)
    Y.index = Y.index.astype(str)
    Y.columns = Y.columns.astype(str)

    if not Z.index.equals(Z.columns):
        raise ValueError("Z must be square with matching index/columns.")
    if not Y.index.equals(Z.index):
        raise ValueError("Y must have the same index as Z.")

    sectors_old = list(map(str, Z.index))
    split_sectors = list(split_profiles.keys())

    missing = [s for s in split_sectors if s not in sectors_old]
    if missing:
        raise ValueError(f"Aggregate sectors not found in Z/Y index: {missing}")

    sectors_new: list[str] = []
    for s in sectors_old:
        if s in split_profiles:
            sectors_new.extend(list(split_profiles[s].subsectors))
        else:
            sectors_new.append(s)

    Z0 = pd.DataFrame(0.0, index=sectors_new, columns=sectors_new)
    Y0 = pd.DataFrame(0.0, index=sectors_new, columns=list(map(str, Y.columns)))

    unchanged = [s for s in sectors_old if s not in split_profiles]
    Z0.loc[unchanged, unchanged] = Z.loc[unchanged, unchanged].to_numpy()
    Y0.loc[unchanged, :] = Y.loc[unchanged, :].to_numpy()

    for agg, prof in split_profiles.items():
        subs = list(prof.subsectors)
        # Pre-compute reindexed shares to avoid repeated lookups
        sales_share = prof.intermediate_sales_share.reindex(subs)
        input_share = prof.intermediate_input_share.reindex(subs)
        fd_share = prof.final_demand_share.reindex(subs)

        buyers_unchanged = [s for s in sectors_old if s not in split_profiles]
        suppliers_unchanged = buyers_unchanged

        # Split sales to unchanged buyers
        Z_row = Z.loc[agg, buyers_unchanged].astype("float64")
        if Z_row.sum() != 0:
            Z0.loc[subs, buyers_unchanged] = np.outer(sales_share.to_numpy(), Z_row.to_numpy())

        # Split purchases from unchanged suppliers
        Z_col = Z.loc[suppliers_unchanged, agg].astype("float64")
        if Z_col.sum() != 0:
            Z0.loc[suppliers_unchanged, subs] = np.outer(Z_col.to_numpy(), input_share.to_numpy())

        # Split within the aggregate block
        Z_agg_agg = float(Z.loc[agg, agg])
        if Z_agg_agg != 0:
            share_mat = _outer_share_matrix(sales_share, input_share)
            Z0.loc[subs, subs] = Z_agg_agg * share_mat

        # Split final demand row
        Y_row = Y.loc[agg, :].astype("float64")
        if Y_row.sum() != 0:
            Y0.loc[subs, :] = np.outer(fd_share.to_numpy(), Y_row.to_numpy())

    # Split interactions between two different aggregates (if any are being split)
    for agg_r, prof_r in split_profiles.items():
        # Pre-compute reindexed shares for row aggregate
        subs_r = list(prof_r.subsectors)
        sales_share_r = prof_r.intermediate_sales_share.reindex(subs_r)
        
        for agg_c, prof_c in split_profiles.items():
            if agg_r == agg_c:
                continue
            cell = float(Z.loc[agg_r, agg_c])
            if cell == 0:
                continue
            # Pre-compute reindexed shares for column aggregate
            subs_c = list(prof_c.subsectors)
            input_share_c = prof_c.intermediate_input_share.reindex(subs_c)
            
            share_mat = _outer_share_matrix(sales_share_r, input_share_c)
            Z0.loc[subs_r, subs_c] = cell * share_mat

    return Z0, Y0


def build_targets(
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    split_profiles: Mapping[str, SplitProfiles],
    sectors_new: Sequence[str],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Build row output targets x and intermediate input targets u for expanded sectors.

    Returns (x_targets, u_targets, fd_col_targets) where fd_col_targets are the
    final-demand column totals (kept at the original Y columns).
    """
    x_old = (Z.sum(axis=1) + Y.sum(axis=1)).astype("float64")
    u_old = Z.sum(axis=0).astype("float64")

    x_new = pd.Series(0.0, index=sectors_new)
    u_new = pd.Series(0.0, index=sectors_new)

    for s in Z.index.astype(str):
        if s not in split_profiles:
            x_new.loc[s] = float(x_old.loc[s])
            u_new.loc[s] = float(u_old.loc[s])
            continue

        prof = split_profiles[s]
        for sub in prof.subsectors:
            x_new.loc[sub] = float(x_old.loc[s]) * float(prof.output_share.loc[sub])
            u_new.loc[sub] = float(u_old.loc[s]) * float(prof.intermediate_input_share.loc[sub])

    fd_targets = Y.sum(axis=0).astype("float64")

    if not np.isfinite(x_new.to_numpy()).all() or not np.isfinite(u_new.to_numpy()).all():
        raise ValueError("Non-finite targets produced; check source tables and shares.")

    return x_new, u_new, fd_targets


def calc_x(Z: pd.DataFrame, Y: pd.DataFrame) -> pd.Series:
    return (Z.sum(axis=1) + Y.sum(axis=1)).astype("float64")


def calc_A(Z: pd.DataFrame, x: pd.Series) -> pd.DataFrame:
    x_safe = x.replace(0.0, np.nan)
    return Z.div(x_safe, axis=1).fillna(0.0)


def disaggregate_and_balance(
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    split_profiles: Mapping[str, SplitProfiles],
    max_iter: int = 50_000,
    tol: float = 1e-10,
    create_io_system: bool = True,
) -> DisaggregationResult:
    """
    Expand Z/Y using `split_profiles`, then biproportionally balance [Z|Y] to targets.

    Balancing uses RAS if feasible; otherwise it falls back to GRAS.
    
    Parameters:
        Z: Square intermediate transactions matrix
        Y: Final demand matrix (same row index as Z)
        split_profiles: Disaggregation share profiles
        max_iter: Maximum iterations for balancing
        tol: Convergence tolerance
        create_io_system: If True and pymrio available, create IOSystem object
        
    Returns:
        DisaggregationResult with balanced tables and optional IOSystem
    """
    Z0, Y0 = expand_io_tables(Z, Y, split_profiles=split_profiles)
    x_target, u_target, fd_targets = build_targets(Z, Y, split_profiles=split_profiles, sectors_new=Z0.index)

    M0 = pd.concat([Z0, Y0], axis=1)
    row_targets = x_target.to_numpy()
    col_targets = np.concatenate([u_target.to_numpy(), fd_targets.to_numpy()])

    M_bal, stats, method = balance_biproportional(
        M0.to_numpy(), row_targets=row_targets, col_targets=col_targets, max_iter=max_iter, tol=tol
    )

    M_bal_df = pd.DataFrame(M_bal, index=M0.index, columns=M0.columns)
    Z_bal = M_bal_df.loc[:, Z0.columns]
    Y_bal = M_bal_df.loc[:, Y0.columns]
    x_bal = calc_x(Z_bal, Y_bal)
    A_bal = calc_A(Z_bal, x_bal)

    # Optional: Create pymrio IOSystem for advanced analysis
    io_system = None
    if create_io_system:
        try:
            import pymrio
            io_system = pymrio.IOSystem()
            io_system.Z = Z_bal
            io_system.Y = Y_bal
            io_system.x = x_bal
            io_system.A = A_bal
            # Note: calc_system() will be called by user when needed
            # to avoid overhead if not using pymrio features
        except ImportError:
            # pymrio not available, skip IOSystem creation
            pass

    return DisaggregationResult(
        Z=Z_bal,
        Y=Y_bal,
        x=x_bal,
        A=A_bal,
        split_profiles=dict(split_profiles),
        balancing_method=method,
        balancing_stats=stats,
        io_system=io_system,
    )
