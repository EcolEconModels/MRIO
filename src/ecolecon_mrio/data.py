from __future__ import annotations

import os
import pickle
import time
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd
import pymrio
from pymrio.tools.iodownloader import EXIOBASE3_CONFIG, HEADERS, MRIOMetaData, _get_url_datafiles

from ecolecon_mrio.download_utils import DownloadSpec, download_file_resumable, zip_contains_member
from ecolecon_mrio.wiod16 import load_wiod16


def default_data_root() -> Path:
    """
    Return the default data root directory for caches (Eurostat/EXIOBASE/etc.).

    Prefers the env var `MRIO_DATA_ROOT` if set; otherwise anchors to the repo root
    (so running from `src/` doesn't accidentally create `src/data/...`).
    """
    env = os.environ.get("MRIO_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # src/ecolecon_mrio/data.py -> repo_root is two parents up from this file.
    return Path(__file__).resolve().parents[2] / "data"


def _resolve_exiobase3_url(year: int, system: str = "pxp", doi: str = "10.5281/zenodo.3583070") -> str:
    cfg = dict(EXIOBASE3_CONFIG)
    cfg["url_db_view"] = "https://doi.org/" + str(doi)
    exio_web_content = _get_url_datafiles(**cfg)
    year_token = f"IOT_{int(year)}_{system}.zip"
    for url in exio_web_content.data_urls:
        if year_token in url:
            return url
    raise FileNotFoundError(f"Could not resolve EXIOBASE source URL for year={year}, system={system}, doi={doi}")


def ensure_exiobase_year_zip(
    year: int,
    *,
    data_dir: str | Path,
    system: str = "pxp",
    force_download: bool = False,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    doi: str = "10.5281/zenodo.3583070",
    verbose: bool = False,
) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"IOT_{int(year)}_{system}.zip"

    if force_download and zip_path.exists():
        ts = int(time.time())
        zip_path.rename(data_dir / f"IOT_{int(year)}_{system}.zip.corrupt_or_forced.{ts}")

    if zip_path.exists() and not zip_contains_member(zip_path, "file_parameters.json"):
        ts = int(time.time())
        zip_path.rename(data_dir / f"IOT_{int(year)}_{system}.zip.corrupt.{ts}")

    if zip_path.exists():
        return zip_path

    if not allow_download_missing:
        raise FileNotFoundError(f"EXIOBASE zip not found in local cache and downloads are disabled: {zip_path}")

    url = _resolve_exiobase3_url(int(year), system=system, doi=doi)
    if verbose:
        print(f"EXIOBASE zip not found at {zip_path}; downloading {url} ...")

    downlog = MRIOMetaData._make_download_log(
        location=data_dir,
        description="Download log of EXIOBASE3",
        name="EXIO3",
        system=str(system),
        version=str(doi),
    )

    spec = DownloadSpec(
        url=url,
        dest_path=zip_path,
        label=f"EXIOBASE {int(year)} {system}",
        headers=HEADERS,
        chunk_size=max(1, int(download_chunk_mb)) * 1024 * 1024,
        integrity_check=lambda p: zip_contains_member(p, "file_parameters.json"),
    )
    res = download_file_resumable(
        spec,
        overwrite_existing=False,
        resume=bool(resume_downloads),
        progress=bool(show_download_progress and verbose),
        timeout_seconds=download_timeout_seconds,
        retries=max(1, int(download_retries)),
    )
    if res.status == "failed":
        raise RuntimeError(f"EXIOBASE download failed for year={year}: {res.reason}")
    if res.status == "skipped_existing":
        downlog._add_fileio(f"Skip download existing file {zip_path.name}")
    else:
        downlog._add_fileio(f"Downloaded {url} to {zip_path.name}")
    downlog.save()
    return zip_path


def _is_complete_exio_parsed_cache(cache_dir: Path) -> bool:
    """Best-effort completeness check for parsed EXIOBASE cache dirs."""
    if not cache_dir.exists() or not cache_dir.is_dir():
        return False
    return (cache_dir / "file_parameters.json").exists()


def _load_fast_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _save_fast_pickle(path: Path, obj, *, verbose: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    if verbose:
        print(f"Saved EXIOBASE fast cache to {path}")


def load_mrio(
    dataset: str,
    year: int,
    *,
    force_download: bool = False,
    data_root: str | Path | None = None,
    wiod_wiot_file: str | Path | None = None,
    calc_all: bool = True,
    include_ghosh: bool = True,
    verbose: bool = False,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    delete_source_zip_after_parse: bool = False,
) -> Tuple[pymrio.IOSystem, pd.Series]:
    """
    Load MRIO data using native pymrio download and parsing capabilities.
    Returns the mrio system and the calculated final demand vector y.
    
    Parameters
    ----------
    dataset : str
        One of "test", "oecd", "wiod", "exiobase".
    year : int
        Year of analysis.
        
    Returns
    -------
    mrio : pymrio.IOSystem
        The loaded input-output system with calculated inverses.
    y_full : pd.Series
        The total final demand vector (summed across categories).
    """
    if dataset == "test":
        if verbose:
            print("Loading internal pymrio test dataset (no download)...")
        mrio = pymrio.load_test()
        y_full = mrio.Y.sum(axis=1)

    elif dataset == "oecd":
        # OECD 2018 edition covers 2005-2015.
        # OECD 2016 edition covers 1995-2011.
        # We target the 2018 edition for year 2010 as it's a common replication point.
        # If year is outside 2005-2015, we might need a different version.
        
        # Pymrio defaults: version='v2021' (which uses 5-year bundles).
        # We explicitly request v2018 for individual year handling flexibility or simplicity if applicable.
        # Or we use v2021 if that's what the user has.
        # Based on file listing (ICIO2023), it seems user has v2021 (published 2023 maybe).
        
        # Let's try standard behavior:
        # If we want 2010, we can try detecting what's available or just ask pymrio to handle it.
        
        root = Path(data_root) if data_root is not None else default_data_root()
        data_dir = root / f"oecd_icio_{year}"
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            
        if verbose:
            print(f"Ensuring OECD data for {year} in {data_dir}...")

        # Avoid unnecessary network calls: only invoke pymrio's downloader when data
        # is missing (or the caller explicitly forces refresh).
        existing_archives = sorted(list(data_dir.glob("*.zip")))
        
        # Note: download_oecd verifies existence internally.
        # We specify version='v2018' to ensure we get the yearly files expected for 2010 replication if possible,
        # or stick to the simple API.
        # However, 2020 paper used 2010 data from Eurostat. OECD is the fallback.
        # Let's use the 'v2018' version logic which is stable for 2010.
        
        if force_download or not existing_archives:
            pymrio.download_oecd(storage_folder=data_dir, years=[year], version="v2018")

        # parser looks for files in the specific path
        mrio = pymrio.parse_oecd(path=data_dir, year=year)
        
        # Filter for Y columns that are not exports (if needed, or just sum)
        # OECD columns usually: "AUS_HFCE", etc.
        # Exports might be "AUS_EXP".
        y_cols = [c for c in mrio.Y.columns if "EXP" not in c]
        y_full = mrio.Y[y_cols].sum(axis=1)

    elif dataset == "wiod16":
        root = Path(data_root) if data_root is not None else default_data_root()
        data_dir = root / "wiod16"
        if not allow_download_missing:
            cache_dir = data_dir / f"mrio_system_wiod16_{year}"
            zip_path = data_dir / "WIOTS_in_STATA.zip"
            if not cache_dir.exists() and not zip_path.exists():
                raise FileNotFoundError(f"WIOD16 cache missing for year={year} and downloads are disabled: {data_dir}")
        parsed = load_wiod16(
            year=year,
            data_dir=data_dir,
            force_download=force_download,
            calc_all=calc_all,
            include_ghosh=include_ghosh,
            verbose=verbose,
        )
        return parsed.mrio, parsed.y_full

    elif dataset == "wiod":
        if year < 1995 or year > 2011:
            raise ValueError(
                "WIOD via `pymrio.download_wiod2013/parse_wiod` only supports the WIOD 2013 release "
                "(years 1995–2011). "
                f"Requested year={year}. Use --dataset oecd, or provide a compatible WIOD source/year."
            )
        root = Path(data_root) if data_root is not None else default_data_root()
        data_dir = root / f"wiod13_{year}"
        data_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Ensuring WIOD 2013 data for {year} in {data_dir}...")

        cache_path = data_dir / f"mrio_system_{year}.pkl"
        if cache_path.exists():
            mrio = pymrio.load(cache_path)
            y_cols = [c for c in mrio.Y.columns if "EXP" not in c]
            y_full = mrio.Y[y_cols].sum(axis=1)
            if calc_all and getattr(mrio, "G", None) is None:
                if verbose:
                    print("Computing MRIO derived matrices (calc_all)...")
                mrio.calc_all(include_ghosh=include_ghosh)
            return mrio, y_full

        yy = str(year)[-2:]

        def _wiot_candidates() -> list[Path]:
            return sorted(list(data_dir.glob(f"wiot{yy}*.xlsx")))

        wiot_path: Path | None = None

        if wiod_wiot_file is not None:
            candidate = Path(wiod_wiot_file)
            if not candidate.is_absolute():
                candidate = data_dir / candidate
            if not candidate.exists():
                raise FileNotFoundError(f"WIOD WIOT file not found: {candidate}")
            wiot_path = candidate
        else:
            candidates = _wiot_candidates()
            if force_download:
                # Force refresh even if WIOT exists (e.g., corrupted/partial local files).
                if verbose:
                    print(f"Forcing WIOD re-download for {year} into {data_dir}...")
                pymrio.download_wiod2013(storage_folder=data_dir, years=[year])
                candidates = _wiot_candidates()
            elif not candidates:
                if not allow_download_missing:
                    raise FileNotFoundError(f"WIOD WIOT file not found in local cache and downloads are disabled: {data_dir}")
                if verbose:
                    print(f"WIOD WIOT file not found in {data_dir}; downloading via pymrio...")
                pymrio.download_wiod2013(storage_folder=data_dir, years=[year])
                candidates = _wiot_candidates()

            if not candidates:
                raise FileNotFoundError(
                    f"WIOD WIOT file not found in {data_dir}. "
                    f"Expected a file matching wiot{yy}*.xlsx. "
                    "If you already downloaded WIOD, place the WIOT xlsx for this year in that folder, "
                    "or re-run with force_download=True (requires network), "
                    "or select a supported year (1995–2011)."
                )

            if len(candidates) == 1:
                wiot_path = candidates[0]
            else:
                exact = data_dir / f"wiot{yy}.xlsx"
                if exact in candidates:
                    wiot_path = exact
                else:
                    raise FileNotFoundError(
                        f"WIOD WIOT file is ambiguous in {data_dir}. "
                        f"Files matching wiot{yy}*.xlsx: {[p.name for p in candidates]}. "
                        "Remove/rename duplicates, or pass `wiod_wiot_file=...` to select the file explicitly."
                    )

        # Pass the specific file path to avoid pymrio's ambiguous directory scan.
        mrio = pymrio.parse_wiod(path=wiot_path, year=year)
        mrio.save(cache_path)
        
        y_cols = [c for c in mrio.Y.columns if "EXP" not in c]
        y_full = mrio.Y[y_cols].sum(axis=1)

    else:
        # Exiobase
        root = Path(data_root) if data_root is not None else default_data_root()
        data_dir = root / f"exiobase3_{year}"
        data_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Ensuring EXIOBASE 3 data for {year} in {data_dir}...")
        
        # Pymrio saves to a *directory*; avoid giving it a directory name with an
        # extension (e.g., "mrio_system.pkl") because `MRIOMetaData.save()` will
        # treat that as a filename and fail with IsADirectoryError.
        mrio_cache_dir = data_dir / "mrio_system"
        fast_cache_path = data_dir / "mrio_system.fast.pkl"
        legacy_cache_dir = data_dir / "mrio_system.pkl"
        mrio = None

        if not mrio_cache_dir.exists() and legacy_cache_dir.is_dir():
            # Backward-compat: previous runs may have created a directory named
            # "mrio_system.pkl". Prefer migrating it to the extension-less name.
            if verbose:
                print(f"Found legacy EXIOBASE cache dir at {legacy_cache_dir}")
            try:
                legacy_cache_dir.rename(mrio_cache_dir)
                if verbose:
                    print(f"Migrated legacy cache dir to {mrio_cache_dir}")
            except Exception:
                # If migration fails (e.g., permissions), just use the legacy dir
                # for reading; avoid writing to it again.
                mrio_cache_dir = legacy_cache_dir

        # Optional: force re-download/re-parse
        if force_download and mrio_cache_dir.exists():
            ts = int(time.time())
            mrio_cache_dir.rename(data_dir / f"mrio_system.forced.{ts}")
        if force_download and fast_cache_path.exists():
            ts = int(time.time())
            fast_cache_path.rename(data_dir / f"mrio_system.fast.forced.{ts}.pkl")

        # Check if cached system exists to avoid reparsing
        cache_para = mrio_cache_dir / "file_parameters.json"
        if mrio_cache_dir.exists() and not cache_para.exists():
            ts = int(time.time())
            mrio_cache_dir.rename(data_dir / f"mrio_system.invalid.{ts}")

        if mrio_cache_dir.exists() and fast_cache_path.exists():
            if verbose:
                print(f"Loading EXIOBASE fast cache from {fast_cache_path} ...")
            try:
                mrio = _load_fast_pickle(fast_cache_path)
            except Exception as exc:
                ts = int(time.time())
                bad = data_dir / f"mrio_system.fast.invalid.{ts}.pkl"
                try:
                    fast_cache_path.rename(bad)
                except Exception:
                    pass
                if verbose:
                    print(f"Warning: failed fast-cache load ({type(exc).__name__}: {exc}); falling back to text cache.")

        if mrio_cache_dir.exists():
            if mrio is None:
                if verbose:
                    print(f"Loading cached EXIOBASE IOSystem from {mrio_cache_dir} ...")
                try:
                    mrio = pymrio.load(mrio_cache_dir)
                    # Seed/update binary fast cache for subsequent runs.
                    _save_fast_pickle(fast_cache_path, mrio, verbose=verbose)
                except FileNotFoundError:
                    ts = int(time.time())
                    mrio_cache_dir.rename(data_dir / f"mrio_system.invalid.{ts}")
                    mrio = None
            if mrio is None:
                # fall through to zip parsing path below
                pass
        if not mrio_cache_dir.exists() or mrio is None:
             # Download if needed (MRIO wrapper around per-year EXIOBASE zip download)
             zip_path = ensure_exiobase_year_zip(
                 year=int(year),
                 data_dir=data_dir,
                 system="pxp",
                 force_download=force_download,
                 allow_download_missing=allow_download_missing,
                 resume_downloads=resume_downloads,
                 show_download_progress=show_download_progress,
                 download_chunk_mb=download_chunk_mb,
                 download_timeout_seconds=download_timeout_seconds,
                 download_retries=download_retries,
                 verbose=verbose,
             )

             if verbose:
                 print(f"Parsing EXIOBASE zip {zip_path} (this can take a while)...")
             mrio = pymrio.parse_exiobase3(zip_path)
             # If a previous partial run created a directory at this path, move it aside.
             if mrio_cache_dir.exists():
                 ts = int(time.time())
                 mrio_cache_dir.rename(data_dir / f"mrio_system.stale.{ts}")
             if verbose:
                 print(f"Saving parsed EXIOBASE IOSystem to {mrio_cache_dir} ...")
             mrio.save(mrio_cache_dir)
             # Also persist fast cache so future runs bypass large text parsing.
             _save_fast_pickle(fast_cache_path, mrio, verbose=verbose)
             if delete_source_zip_after_parse and _is_complete_exio_parsed_cache(mrio_cache_dir):
                 try:
                     zip_path.unlink(missing_ok=True)
                     if verbose:
                         print(f"Deleted EXIOBASE source zip after successful parse: {zip_path}")
                 except Exception as exc:
                     if verbose:
                         print(f"Warning: failed to delete EXIOBASE source zip {zip_path}: {exc}")

        y_cols = [c for c in mrio.Y.columns if "EXP" not in c]
        y_full = mrio.Y[y_cols].sum(axis=1)
    
    # Ensure Ghosh is calculated
    if calc_all and getattr(mrio, "G", None) is None:
        if verbose:
            print("Computing MRIO derived matrices (calc_all)...")
        mrio.calc_all(include_ghosh=include_ghosh)
        
    return mrio, y_full
