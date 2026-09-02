from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Sequence

import pandas as pd
import pymrio


WIOD16_WIOTS_STATA_DATAVERSE_URL = "https://dataverse.nl/api/access/datafile/199103"


def _write_download_log(path: Path, *, note: str, files: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": int(time.time()),
        "note": note,
        "files": [str(p) for p in files],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ensure_wiod16_wiots_stata_zip(
    data_dir: Path,
    *,
    force_download: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Ensure the WIOD 2016 WIOTs (STATA) zip is present locally.

    This is a large download (~600MB). We only download when `force_download=True`.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "WIOTS_in_STATA.zip"
    if zip_path.exists() and not force_download:
        return zip_path

    # Lazy import to keep base install light.
    import urllib.request

    tmp = zip_path.with_suffix(".zip.tmp")
    if verbose:
        action = "Re-downloading" if zip_path.exists() else "Downloading"
        print(f"{action} WIOD16 WIOTs STATA zip to {zip_path} ...")
    try:
        urllib.request.urlretrieve(WIOD16_WIOTS_STATA_DATAVERSE_URL, tmp)  # nosec B310
    except Exception as e:
        raise RuntimeError(
            "Failed to download WIOD 2016 WIOTs (STATA). "
            f"URL: {WIOD16_WIOTS_STATA_DATAVERSE_URL}. "
            f"Either download it manually to {zip_path} or retry with network access."
        ) from e
    # Basic integrity check.
    try:
        with zipfile.ZipFile(tmp, "r") as zf:
            zf.namelist()
    except Exception as e:
        raise RuntimeError(f"Downloaded WIOD16 WIOT zip is not a valid zip: {tmp}") from e

    tmp.replace(zip_path)
    _write_download_log(
        data_dir / "download_log_wiod16_wiots_stata.json",
        note="Downloaded WIOD 2016 WIOTs (STATA) from Dataverse." if not force_download else "Re-downloaded WIOD 2016 WIOTs (STATA) from Dataverse.",
        files=[zip_path],
    )
    return zip_path


def _iter_wiod16_members_for_year(zf: zipfile.ZipFile, year: int) -> list[str]:
    y = str(int(year))
    wanted_prefix = f"WIOT{y}_"
    members = [n for n in zf.namelist() if n.startswith(wanted_prefix) and n.lower().endswith(".dta")]
    return sorted(members)


def _wiod16_fd_category_labels(n: int) -> list[str]:
    # WIOD 2016 release typically uses 5 final demand categories.
    # We keep names stable but do not rely on them for computations.
    if n == 5:
        return ["HFCE", "NPISH", "GGFC", "GFCF", "INV"]
    return [f"FD{i+1}" for i in range(n)]


def parse_wiod16_wiot_frame(df: pd.DataFrame) -> pymrio.IOSystem:
    """
    Parse one WIOD 2016 WIOT year (as a wide DataFrame) into a pymrio IOSystem.

    Expected columns:
    - IndustryCode, IndustryDescription, Country, RNr, Year
    - v{CCC}{N} columns where CCC is country code (incl. ROW) and N is a number:
      * N=1..n_industries are intermediate-demand industry columns
      * N>n_industries are final-demand columns (same country)
    - Optional TOT column (ignored except for sanity checks)
    """
    required = {"IndustryCode", "Country", "RNr", "Year"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"WIOD16 WIOT frame missing required columns: {sorted(missing)}")

    # Identify industry rows (country-specific), and infer n_industries.
    industry_rows = df[df["Country"] != "TOT"].copy()
    if industry_rows.empty:
        raise ValueError("WIOD16 WIOT frame has no country-industry rows (Country != 'TOT').")

    n_industries = int(industry_rows["RNr"].max())
    if n_industries <= 0:
        raise ValueError(f"Invalid industry count inferred from RNr: {n_industries}")

    # Identify countries from rows to define ordering and column parsing.
    countries = list(dict.fromkeys(industry_rows["Country"].astype(str).tolist()))

    # Build sector ordering per country from rows.
    sector_by_country: dict[str, list[str]] = {}
    for c in countries:
        sub = industry_rows[industry_rows["Country"] == c].sort_values("RNr")
        # Some files may include non-industry rows inside countries; keep RNr 1..n_industries.
        sub = sub[sub["RNr"].between(1, n_industries)]
        codes = sub["IndustryCode"].astype(str).tolist()
        if len(codes) != n_industries:
            raise ValueError(
                f"Country {c}: expected {n_industries} industry codes, found {len(codes)}. "
                "This loader expects one row per industry with sequential RNr."
            )
        sector_by_country[c] = codes

    # Identify v{country}{number} columns.
    v_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("v") and len(c) >= 5]
    if not v_cols:
        raise ValueError("No WIOD16 value columns found (expected columns like 'vAUS1').")

    # Determine available numeric suffixes per country.
    import re

    re_v = re.compile(r"^v([A-Z]{3})(\d+)$")
    nums_by_country: dict[str, set[int]] = {c: set() for c in countries}
    for col in v_cols:
        m = re_v.match(col)
        if not m:
            continue
        c, n = m.group(1), int(m.group(2))
        if c in nums_by_country:
            nums_by_country[c].add(n)

    # Determine final demand column numbers as those beyond the industry range.
    fd_nums: list[int] = []
    for c in countries:
        fd_nums = sorted([n for n in nums_by_country.get(c, set()) if n > n_industries])
        if fd_nums:
            break
    if not fd_nums:
        raise ValueError("Could not infer final demand columns (no v{CCC}{N} found with N > n_industries).")

    fd_labels = _wiod16_fd_category_labels(len(fd_nums))

    # Build industry columns and FD columns in deterministic order.
    industry_col_names: list[str] = []
    fd_col_names: list[str] = []
    for c in countries:
        for i in range(1, n_industries + 1):
            industry_col_names.append(f"v{c}{i}")
        for n in fd_nums:
            fd_col_names.append(f"v{c}{n}")

    missing_ind_cols = [c for c in industry_col_names if c not in df.columns]
    if missing_ind_cols:
        raise ValueError(f"Missing expected industry columns in WIOD16 file (sample): {missing_ind_cols[:10]}")
    missing_fd_cols = [c for c in fd_col_names if c not in df.columns]
    if missing_fd_cols:
        raise ValueError(f"Missing expected final-demand columns in WIOD16 file (sample): {missing_fd_cols[:10]}")

    # Select industry rows strictly by RNr (1..n_industries) and preserve order.
    ind = industry_rows[industry_rows["RNr"].between(1, n_industries)].copy()
    ind = ind.sort_values(["Country", "RNr"])
    ind_index = pd.MultiIndex.from_arrays(
        [ind["Country"].astype(str).to_list(), ind["IndustryCode"].astype(str).to_list()],
        names=["region", "sector"],
    )

    # Build columns index for industries.
    col_regions: list[str] = []
    col_sectors: list[str] = []
    for c in countries:
        col_regions.extend([c] * n_industries)
        col_sectors.extend(sector_by_country[c])
    Z_columns = pd.MultiIndex.from_arrays([col_regions, col_sectors], names=["region", "sector"])

    # Z matrix.
    Z = pd.DataFrame(ind[industry_col_names].to_numpy(), index=ind_index, columns=Z_columns)

    # Y matrix.
    y_regions: list[str] = []
    y_cats: list[str] = []
    for c in countries:
        y_regions.extend([c] * len(fd_nums))
        y_cats.extend(fd_labels)
    Y_columns = pd.MultiIndex.from_arrays([y_regions, y_cats], names=["region", "category"])
    Y = pd.DataFrame(ind[fd_col_names].to_numpy(), index=ind_index, columns=Y_columns)

    # Output vector x: use the 'GO' row (Output at basic prices) if present.
    go = df[(df["Country"] == "TOT") & (df["IndustryCode"].astype(str) == "GO")]
    if len(go) != 1:
        raise ValueError(f"Expected exactly one GO row with Country='TOT', found {len(go)}.")
    go = go.iloc[0]
    x = pd.Series(go[industry_col_names].to_numpy(), index=Z_columns, name="x")
    x_df = pd.DataFrame(x)

    mrio = pymrio.IOSystem(Z=Z, Y=Y, x=x_df, year=int(go["Year"]), name="WIOD16")
    return mrio


def load_wiod16(
    *,
    year: int,
    data_dir: Path,
    force_download: bool = False,
    calc_all: bool = True,
    include_ghosh: bool = True,
    verbose: bool = False,
) -> pymrio.IOSystem:
    """
    Load WIOD 2016 release WIOT for a specific year (2000–2014) from the STATA zip.
    """
    if year < 2000 or year > 2014:
        raise ValueError(f"WIOD 2016 release covers years 2000–2014. Requested year={year}.")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Pymrio can save/load as a directory. Avoid using a path that looks like a
    # file but is (or becomes) a directory (see similar EXIOBASE cache issue).
    mrio_cache_dir = data_dir / f"mrio_system_wiod16_{year}"
    legacy_cache_dir = data_dir / f"mrio_system_wiod16_{year}.pkl"

    if not mrio_cache_dir.exists() and legacy_cache_dir.is_dir():
        try:
            legacy_cache_dir.rename(mrio_cache_dir)
        except Exception:
            mrio_cache_dir = legacy_cache_dir

    cache_para = mrio_cache_dir / "file_parameters.json"
    if mrio_cache_dir.exists() and not cache_para.exists():
        ts = int(time.time())
        mrio_cache_dir.rename(data_dir / f"mrio_system_wiod16_{year}.invalid.{ts}")

    if mrio_cache_dir.exists() and not force_download:
        if verbose:
            print(f"Loading cached WIOD16 IOSystem from {mrio_cache_dir} ...")
        try:
            mrio = pymrio.load(mrio_cache_dir)
        except FileNotFoundError:
            ts = int(time.time())
            mrio_cache_dir.rename(data_dir / f"mrio_system_wiod16_{year}.invalid.{ts}")
        else:
            if calc_all and getattr(mrio, "G", None) is None:
                mrio.calc_all(include_ghosh=include_ghosh)
            return mrio

    zip_path = ensure_wiod16_wiots_stata_zip(data_dir, force_download=force_download, verbose=verbose)

    # Read the year-specific .dta directly from the zip.
    try:
        import pyreadstat  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Loading WIOD 2016 from the STATA distribution requires `pyreadstat`.\n"
            "Install it via:\n"
            "  .venv/bin/python -m pip install -r requirements-wiod16-stata.txt\n"
            "or (editable install):\n"
            "  .venv/bin/python -m pip install -e \".[wiod16-stata]\" --no-build-isolation --no-deps"
        ) from e

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = _iter_wiod16_members_for_year(zf, year)
        if len(members) != 1:
            raise FileNotFoundError(
                f"WIOD16 STATA zip does not contain a unique .dta for year={year}. "
                f"Candidates: {members}"
            )
        member = members[0]
        if verbose:
            print(f"Parsing WIOD16 WIOT from {zip_path}::{member} ...")
        with zf.open(member) as f:
            # Requires `pyreadstat` installed for performance/reliability.
            df = pd.read_stata(f, convert_categoricals=False)

    mrio = parse_wiod16_wiot_frame(df)
    # If a partial previous run left a directory, move it aside.
    if mrio_cache_dir.exists():
        ts = int(time.time())
        mrio_cache_dir.rename(data_dir / f"mrio_system_wiod16_{year}.stale.{ts}")
    mrio.save(mrio_cache_dir)
    if calc_all and getattr(mrio, "G", None) is None:
        if verbose:
            print("Computing MRIO derived matrices (calc_all)...")
        mrio.calc_all(include_ghosh=include_ghosh)
    return mrio
