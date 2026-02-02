from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping as ABCMapping
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


def _parse_bool(x: object) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return True
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s == "":
        return True
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value: {x!r}")


def read_exio_sector_mapping_csv(
    path: str | Path,
    *,
    aggregate_col: str = "aggregate_sector",
    subsector_col: str = "subsector",
    exio_col: str = "exio_sector",
    enabled_col: str = "enabled",
) -> dict[str, list[str]] | dict[str, dict[str, list[str]]]:
    """
    Read a mapping CSV for splitting: aggregate sector -> explicit EXIO sector labels.

    Required columns
    - `aggregate_sector`: the sector code/name in the target IO table to split.
    - `exio_sector`: exact EXIOBASE sector label (matches `ExioRegionSlice.sectors`).

    Optional columns
    - `subsector`: target subsector label for the split output. If provided, multiple EXIO
      sectors can be grouped into one output subsector by using the same `subsector` value.
    - `enabled`: boolean-ish (default True). Disabled rows are ignored.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in (aggregate_col, exio_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Mapping CSV {Path(path)} missing required columns: {missing}")

    if enabled_col in df.columns:
        mask = df[enabled_col].map(_parse_bool).astype(bool)
        df = df[mask].copy()

    # Strip whitespace from relevant columns
    cols_to_strip = [aggregate_col, exio_col]
    if subsector_col in df.columns:
        cols_to_strip.append(subsector_col)

    for c in cols_to_strip:
        df[c] = df[c].str.strip()

    # Filter empty values
    mask_empty = (df[aggregate_col] == "") | (df[exio_col] == "")
    if mask_empty.any():
        bad_rows = mask_empty[mask_empty].index.tolist()
        raise ValueError(
            f"Mapping CSV {Path(path)} has empty {aggregate_col} or {exio_col} value in rows: {bad_rows}"
        )

    if subsector_col not in df.columns:
        out_simple: dict[str, list[str]] = {}
        for agg, group in df.groupby(aggregate_col):
            out_simple[str(agg)] = list(group[exio_col])
        return out_simple

    # Fix subsectors globally: empty -> exio sector label
    mask_no_sub = df[subsector_col] == ""
    df.loc[mask_no_sub, subsector_col] = df.loc[mask_no_sub, exio_col]

    out_grouped: dict[str, dict[str, list[str]]] = {}
    for agg, agg_group in df.groupby(aggregate_col):
        sub_dict: dict[str, list[str]] = {}
        for sub, sub_group in agg_group.groupby(subsector_col):
            sub_dict[str(sub)] = list(sub_group[exio_col])
        out_grouped[str(agg)] = sub_dict

    return out_grouped


def _normalize_label(s: str) -> str:
    return " ".join(str(s).strip().casefold().split())


def resolve_exio_sector_labels(
    *,
    requested: Sequence[str],
    available: Sequence[str],
    allow_normalized_match: bool = False,
    context: str | None = None,
) -> list[str]:
    """
    Resolve a list of EXIO sector labels against an available sector index.

    - Default: strict exact matching only (robust; avoids accidental matches like 'gas' -> 'gasoline').
    - If allow_normalized_match=True: tries a whitespace/case normalized match when exact match fails.
    """
    available_list = list(map(str, available))
    available_set = set(available_list)
    norm_to_available: dict[str, list[str]] = {}
    for a in available_list:
        norm_to_available.setdefault(_normalize_label(a), []).append(a)

    resolved: list[str] = []
    missing: list[str] = []
    ambiguous: dict[str, list[str]] = {}

    for r in requested:
        r_s = str(r)
        if r_s in available_set:
            resolved.append(r_s)
            continue
        if allow_normalized_match:
            hits = norm_to_available.get(_normalize_label(r_s), [])
            if len(hits) == 1:
                resolved.append(hits[0])
                continue
            if len(hits) > 1:
                ambiguous[r_s] = hits
                continue
        missing.append(r_s)

    if ambiguous or missing:
        import difflib

        lines: list[str] = []
        if context:
            lines.append(str(context))
        if ambiguous:
            lines.append("Ambiguous normalized EXIO sector matches:")
            for k, hits in ambiguous.items():
                lines.append(f"  - {k!r} -> {hits!r}")
        if missing:
            lines.append("Missing EXIO sector labels (no exact match):")
            for m in missing:
                lines.append(f"  - {m!r}")
                sugg = difflib.get_close_matches(m, available_list, n=5, cutoff=0.65)
                if sugg:
                    lines.append(f"    suggestions: {sugg!r}")
        raise ValueError("\n".join(lines))

    return resolved


def build_split_profiles_from_exiobase_domestic_block(
    exio_slice: ExioRegionSlice,
    *,
    sector_to_exio_sectors: Mapping[str, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    sector_patterns: Mapping[str, Sequence[str]] | None = None,
    allow_normalized_exio_match: bool = False,
    prefix_subsectors: bool = True,
    prefix_template: str = "{aggregate}::",
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
            prefix = prefix_template.format(aggregate=str(aggregate)) if prefix_subsectors else None

            if isinstance(exio_sectors, ABCMapping):
                # Grouped mapping: subsector label -> list of EXIO labels
                subsector_to_exio: dict[str, list[str]] = {}
                for sub, members in exio_sectors.items():
                    resolved_members = resolve_exio_sector_labels(
                        requested=list(map(str, members)),
                        available=list(map(str, exio_sector_index)),
                        allow_normalized_match=allow_normalized_exio_match,
                        context=f"aggregate_sector={aggregate!r}, subsector={sub!r}",
                    )
                    subsector_to_exio[str(sub)] = resolved_members

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
                if prefix is not None:
                    subsectors = [f"{prefix}{s}" for s in subsectors]
                rename = dict(zip(x_sub.index.astype(str), subsectors))
                subsector_to_exio_prefixed = {
                    rename[str(sub)]: list(map(str, members)) for sub, members in subsector_to_exio.items()
                }

                profiles[str(aggregate)] = SplitProfiles(
                    aggregate_sector=str(aggregate),
                    subsectors=tuple(subsectors),
                    output_share=output_share.rename(index=rename),
                    intermediate_sales_share=sales_share.rename(index=rename),
                    intermediate_input_share=input_share.rename(index=rename),
                    final_demand_share=final_demand_share.rename(index=rename),
                    subsector_to_exio_sectors=subsector_to_exio_prefixed,
                )

                flat_members: list[str] = []
                for members in subsector_to_exio.values():
                    flat_members.extend(members)
                selected[str(aggregate)] = list(dict.fromkeys(flat_members))
            else:
                # Explicit mapping: each EXIO label becomes its own output subsector.
                resolved = resolve_exio_sector_labels(
                    requested=list(map(str, exio_sectors)),
                    available=list(map(str, exio_sector_index)),
                    allow_normalized_match=allow_normalized_exio_match,
                    context=f"aggregate_sector={aggregate!r}",
                )
                prof, matched = build_split_profiles_from_exio_domestic_block(
                    exio_slice,
                    aggregate_sector=str(aggregate),
                    exio_sector_patterns=tuple(resolved),
                    prefix_labels_with=prefix,
                    selector="exact",
                )
                profiles[str(aggregate)] = prof
                selected[str(aggregate)] = list(matched)
        return profiles, selected

    assert sector_patterns is not None
    for aggregate, patterns in sector_patterns.items():
        prefix = prefix_template.format(aggregate=str(aggregate)) if prefix_subsectors else None
        prof, matched = build_split_profiles_from_exio_domestic_block(
            exio_slice,
            aggregate_sector=str(aggregate),
            exio_sector_patterns=tuple(patterns),
            prefix_labels_with=prefix,
            selector="patterns",
        )
        profiles[str(aggregate)] = prof
        selected[str(aggregate)] = list(matched)
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
                matches = [s for s in prof.subsectors if s.endswith(f"::{sub_label}")]
                if len(matches) == 1:
                    sub_label = matches[0]
                else:
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
    allow_normalized_exio_match: bool = False,
    prefix_subsectors: bool = True,
    prefix_template: str = "{aggregate}::",
    ras_max_iter: int = 50_000,
    ras_tol: float = 1e-10,
    balancing_scope: str = "local",
    coverage_expected_exio_sectors: Mapping[str, Sequence[str]] | None = None,
    coverage_tol: float = 0.0,
    extract_quantity_and_price: bool = False,
    quantity_price_extensions: Sequence[str] | None = None,
    quantity_price_stressor_keywords: Sequence[str] | None = None,
    quantity_price_cache_dir: str | Path | None = None,
    create_io_system: bool = True,
) -> SplitResult:
    """
    Split one or more aggregate sectors in (Z,Y) using EXIOBASE domestic-block shares and balance via RAS/GRAS.
    """
    profiles, selected = build_split_profiles_from_exiobase_domestic_block(
        exio_slice,
        sector_to_exio_sectors=sector_to_exio_sectors,
        sector_patterns=sector_patterns,
        allow_normalized_exio_match=allow_normalized_exio_match,
        prefix_subsectors=prefix_subsectors,
        prefix_template=prefix_template,
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

    res = disaggregate_and_balance(
        eurostat_Z,
        eurostat_Y,
        split_profiles=profiles,
        balancing_scope=balancing_scope,
        max_iter=ras_max_iter,
        tol=ras_tol,
        create_io_system=create_io_system,
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
    allow_normalized_exio_match: bool = False,
    prefix_subsectors: bool = True,
    prefix_template: str = "{aggregate}::",
    ras_max_iter: int = 50_000,
    ras_tol: float = 1e-10,
    balancing_scope: str = "local",
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
        allow_normalized_exio_match=allow_normalized_exio_match,
        prefix_subsectors=prefix_subsectors,
        prefix_template=prefix_template,
        ras_max_iter=ras_max_iter,
        ras_tol=ras_tol,
        balancing_scope=balancing_scope,
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
