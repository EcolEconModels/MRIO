from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from ecolecon_mrio.balancing import BalanceStats, balance_biproportional, balance_biproportional_masked

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
    Optional subsector_to_exio_sectors captures the EXIO sectors assigned to each subsector.
    """

    aggregate_sector: str
    subsectors: tuple[str, ...]
    output_share: pd.Series
    intermediate_sales_share: pd.Series
    intermediate_input_share: pd.Series
    final_demand_share: pd.Series
    subsector_to_exio_sectors: dict[str, list[str]] | None = None


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


def _normalized_reindexed_share(series: pd.Series, labels: Sequence[str], *, name: str) -> pd.Series:
    share = series.reindex(list(map(str, labels))).fillna(0.0).astype("float64")
    total = float(share.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Cannot use {name}: non-positive/non-finite share sum {total}.")
    return share / total


def _column_share_for(
    *,
    row_sector: str,
    col_sector: str,
    col_labels: Sequence[str],
    split_profiles: Mapping[str, SplitProfiles],
    column_allocator: Mapping[str, Mapping[str, pd.Series]] | None,
) -> pd.Series:
    """Return column-child shares for one original supplier row and one buyer sector."""
    col_labels = list(map(str, col_labels))
    if col_sector not in split_profiles:
        return pd.Series([1.0], index=col_labels, dtype="float64")

    prof = split_profiles[col_sector]
    fallback = _normalized_reindexed_share(
        prof.intermediate_input_share,
        col_labels,
        name=f"intermediate_input_share[{col_sector}]",
    )

    if column_allocator is None:
        return fallback

    override = column_allocator.get(str(col_sector), {}).get(str(row_sector))
    if override is None:
        return fallback

    override_s = pd.Series(override, dtype="float64").reindex(col_labels).fillna(0.0).astype("float64")
    total = float(override_s.sum())
    if not np.isfinite(total) or total <= 0:
        return fallback
    return override_s / total


def expand_io_tables(
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    split_profiles: Mapping[str, SplitProfiles],
    column_allocator: Mapping[str, Mapping[str, pd.Series]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expand an IO table (Z,Y) by replacing each aggregate sector with its subsectors.

    There is deliberately only one expansion path. ``column_allocator`` optionally
    overrides column-child shares for purchases made by split buyer sectors from
    selected original supplier rows; row splits and final-demand splits still come
    from ``SplitProfiles``.
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
    missing = [s for s in split_profiles.keys() if str(s) not in sectors_old]
    if missing:
        raise ValueError(f"Aggregate sectors not found in Z/Y index: {missing}")

    sectors_new: list[str] = []
    for s in sectors_old:
        if s in split_profiles:
            sectors_new.extend(list(map(str, split_profiles[s].subsectors)))
        else:
            sectors_new.append(s)

    Z0 = pd.DataFrame(0.0, index=sectors_new, columns=sectors_new, dtype="float64")
    Y0 = pd.DataFrame(0.0, index=sectors_new, columns=list(map(str, Y.columns)), dtype="float64")

    for row_sector in sectors_old:
        if row_sector in split_profiles:
            row_labels = list(map(str, split_profiles[row_sector].subsectors))
            row_share = _normalized_reindexed_share(
                split_profiles[row_sector].intermediate_sales_share,
                row_labels,
                name=f"intermediate_sales_share[{row_sector}]",
            )
            fd_share = _normalized_reindexed_share(
                split_profiles[row_sector].final_demand_share,
                row_labels,
                name=f"final_demand_share[{row_sector}]",
            )
        else:
            row_labels = [row_sector]
            row_share = pd.Series([1.0], index=row_labels, dtype="float64")
            fd_share = pd.Series([1.0], index=row_labels, dtype="float64")

        for col_sector in sectors_old:
            cell_total = float(Z.loc[row_sector, col_sector])
            if cell_total == 0.0:
                continue

            if col_sector in split_profiles:
                col_labels = list(map(str, split_profiles[col_sector].subsectors))
            else:
                col_labels = [col_sector]

            col_share = _column_share_for(
                row_sector=row_sector,
                col_sector=col_sector,
                col_labels=col_labels,
                split_profiles=split_profiles,
                column_allocator=column_allocator,
            )
            Z0.loc[row_labels, col_labels] += (
                np.outer(row_share.to_numpy(dtype="float64"), col_share.to_numpy(dtype="float64")) * cell_total
            )

        Y_row = Y.loc[row_sector, :].astype("float64")
        if float(Y_row.sum()) != 0.0:
            Y0.loc[row_labels, :] += np.outer(fd_share.to_numpy(dtype="float64"), Y_row.to_numpy(dtype="float64"))

    return Z0, Y0


def build_targets(
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    split_profiles: Mapping[str, SplitProfiles],
    sectors_new: Sequence[str],
    Z0: pd.DataFrame | None = None,
    preserve_expanded_column_targets: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Build row output targets x and intermediate input targets u for expanded sectors.

    If ``preserve_expanded_column_targets`` is true, u targets come from
    ``Z0.sum(axis=0)``. This is needed when a technology allocator changes
    child-column totals relative to generic intermediate_input_share.
    """
    x_old = (Z.sum(axis=1) + Y.sum(axis=1)).astype("float64")
    u_old = Z.sum(axis=0).astype("float64")

    sectors_new = list(map(str, sectors_new))
    x_new = pd.Series(0.0, index=sectors_new, dtype="float64")
    u_new = pd.Series(0.0, index=sectors_new, dtype="float64")

    for s in Z.index.astype(str):
        if s not in split_profiles:
            x_new.loc[s] = float(x_old.loc[s])
            u_new.loc[s] = float(u_old.loc[s])
            continue

        prof = split_profiles[s]
        for sub in map(str, prof.subsectors):
            x_new.loc[sub] = float(x_old.loc[s]) * float(prof.output_share.loc[sub])
            u_new.loc[sub] = float(u_old.loc[s]) * float(prof.intermediate_input_share.loc[sub])

    if preserve_expanded_column_targets:
        if Z0 is None:
            raise ValueError("Z0 is required when preserve_expanded_column_targets=True.")
        u_new = Z0.sum(axis=0).astype("float64").reindex(sectors_new).fillna(0.0).astype("float64")

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
    column_allocator: Mapping[str, Mapping[str, pd.Series]] | None = None,
    balancing_scope: str = "local",
    max_iter: int = 50_000,
    tol: float = 1e-10,
    create_io_system: bool = True,
) -> DisaggregationResult:
    """
    Expand Z/Y using `split_profiles`, then biproportionally balance [Z|Y] to targets.

    Balancing uses RAS if feasible; otherwise it falls back to GRAS.

    `balancing_scope` controls which cells are allowed to change during balancing:
    - "global": all cells in [Z|Y] may change (standard RAS/GRAS).
    - "local": keep the untouched blocks fixed and only balance cells that touch the
      newly-created subsectors (recommended when you want the original IO table to
      remain identical outside the split sectors).
    
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
    Z0, Y0 = expand_io_tables(Z, Y, split_profiles=split_profiles, column_allocator=column_allocator)

    balancing_scope_n = str(balancing_scope).strip().lower()
    if balancing_scope_n not in {"global", "local"}:
        raise ValueError(f"balancing_scope must be 'local' or 'global', got {balancing_scope!r}.")

    preserve_allocator_column_targets = column_allocator is not None and balancing_scope_n == "local"
    x_target, u_target, fd_targets = build_targets(
        Z,
        Y,
        split_profiles=split_profiles,
        sectors_new=Z0.index,
        Z0=Z0,
        preserve_expanded_column_targets=preserve_allocator_column_targets,
    )

    M0 = pd.concat([Z0, Y0], axis=1)
    row_targets = x_target.to_numpy()
    col_targets = np.concatenate([u_target.to_numpy(), fd_targets.to_numpy()])

    if balancing_scope_n == "global":
        M_bal, stats, method = balance_biproportional(
            M0.to_numpy(), row_targets=row_targets, col_targets=col_targets, max_iter=max_iter, tol=tol
        )
    else:
        # Constrained/local balancing: keep the untouched (un-split) blocks fixed.
        subsectors: set[str] = set()
        for prof in split_profiles.values():
            subsectors.update(map(str, prof.subsectors))

        row_is_split = np.array([str(s) in subsectors for s in M0.index], dtype=bool)

        # Columns: first block is Z columns (sector labels), second block is final demand.
        z_cols = list(map(str, Z0.columns))
        col_is_split_Z = np.array([c in subsectors for c in z_cols], dtype=bool)
        n_z = len(z_cols)
        n_y = Y0.shape[1]

        # Adjustable mask:
        # - Z block: adjust cells where row or column touches a split subsector
        # - Y block: keep final-demand rows fixed. The clean expansion already
        #   preserves the original sector-by-final-demand totals exactly. Once
        #   those Y rows are fixed, the feasible row targets are the clean
        #   expansion row sums implied by [Z0|Y0], not the generic output-share
        #   targets used by the global balancer.
        mask_Z = row_is_split[:, None] | col_is_split_Z[None, :]
        mask_Y = np.zeros((len(M0.index), n_y), dtype=bool)
        adjustable = np.concatenate([mask_Z, mask_Y], axis=1)
        row_targets_local = M0.sum(axis=1).to_numpy(dtype="float64")

        M_bal, stats, method = balance_biproportional_masked(
            M0.to_numpy(),
            row_targets=row_targets_local,
            col_targets=col_targets,
            adjustable=adjustable,
            row_labels=list(map(str, M0.index)),
            col_labels=list(map(str, M0.columns)),
            max_iter=max_iter,
            tol=tol,
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
