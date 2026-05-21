from __future__ import annotations

from contextlib import contextmanager
import json
import os
import pickle
import time
import zipfile
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import pymrio
from scipy.sparse import csr_matrix, diags, load_npz, save_npz
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


def resolve_exiobase_year_dir(year: int, *, data_root: str | Path | None = None) -> Path:
    """Return the shared EXIOBASE cache directory for a given year.

    The repo keeps one year directory per EXIOBASE release year and stores
    system-specific files ("pxp", "ixi") inside that directory.
    """
    root = Path(data_root) if data_root is not None else default_data_root()
    return root / f"exiobase3_{int(year)}"


def resolve_exiobase_cache_paths(
    year: int,
    *,
    system: str = "pxp",
    data_root: str | Path | None = None,
) -> dict[str, Path]:
    """Return the standard EXIOBASE cache paths used by :func:`load_mrio`.

    The returned mapping includes the shared year directory, the raw zip path,
    the system-specific text cache, and the fast pickle cache.
    """
    system = str(system).strip().lower()
    if system not in {"pxp", "ixi"}:
        raise ValueError(f"Unsupported exiobase system '{system}'. Use 'pxp' or 'ixi'.")
    data_dir = resolve_exiobase_year_dir(int(year), data_root=data_root)
    paths = _exio_cache_paths(data_dir, int(year), system)
    paths["data_dir"] = data_dir
    paths["zip"] = data_dir / f"IOT_{int(year)}_{system}.zip"
    return paths


def resolve_exiobase_matrix_cache_paths(
    year: int,
    *,
    system: str = "pxp",
    data_root: str | Path | None = None,
) -> dict[str, Path]:
    """Return the compact EXIOBASE matrix-cache paths used by :func:`load_mrio_matrices`."""
    system = str(system).strip().lower()
    if system not in {"pxp", "ixi"}:
        raise ValueError(f"Unsupported exiobase system '{system}'. Use 'pxp' or 'ixi'.")
    data_dir = resolve_exiobase_year_dir(int(year), data_root=data_root)
    cache_dir = data_dir / f"matrix_cache_{int(year)}_{system}"
    return {
        "data_dir": data_dir,
        "matrix_cache_dir": cache_dir,
        "Z": cache_dir / "Z.npz",
        "x": cache_dir / "x.pkl.gz",
        "y": cache_dir / "y.pkl.gz",
        "index": cache_dir / "index.csv.gz",
        "metadata": cache_dir / "metadata.json",
        "zip": data_dir / f"IOT_{int(year)}_{system}.zip",
    }


def _coerce_series_like(values: Any, *, index: pd.Index | None = None, name: str | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] == 1:
            series = values.iloc[:, 0].copy()
        else:
            series = values.sum(axis=1)
    else:
        series = pd.Series(values)
    series = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if index is not None:
        series = pd.to_numeric(pd.Series(series).reindex(index), errors="coerce").fillna(0.0).astype(float)
    if name is not None:
        series.name = name
    return series


def _exio_index_frame(index: pd.Index) -> pd.DataFrame:
    if not isinstance(index, pd.MultiIndex):
        raise ValueError("Expected EXIOBASE matrices to use a MultiIndex basis index.")
    index_frame = index.to_frame(index=False)
    if index_frame.shape[1] < 2:
        raise ValueError("Expected EXIOBASE basis index to have at least two levels.")
    index_frame = index_frame.iloc[:, :2].copy()
    index_frame.columns = ["region", "sector"]
    return index_frame.astype(str)


def _exio_index_from_frame(index_frame: pd.DataFrame) -> pd.MultiIndex:
    required = {"region", "sector"}
    if not required.issubset(index_frame.columns):
        raise ValueError("Matrix cache index.csv.gz must contain region and sector columns.")
    frame = index_frame.loc[:, ["region", "sector"]].copy().astype(str)
    return pd.MultiIndex.from_frame(frame, names=["region", "sector"])


def _save_exio_matrix_cache(
    *,
    cache_dir: Path,
    Z: csr_matrix,
    x: pd.Series,
    y: pd.Series,
    index: pd.MultiIndex,
    metadata: dict[str, Any],
    verbose: bool = False,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_npz(cache_dir / "Z.npz", Z, compressed=True)
    legacy_a_path = cache_dir / "A.npz"
    if legacy_a_path.exists():
        legacy_a_path.unlink()
    x = _coerce_series_like(x, index=index, name=x.name or "x")
    y = _coerce_series_like(y, index=index, name=y.name or "y")
    x.to_pickle(cache_dir / "x.pkl.gz", compression="gzip")
    y.to_pickle(cache_dir / "y.pkl.gz", compression="gzip")
    _exio_index_frame(index).to_csv(cache_dir / "index.csv.gz", index=False, compression="gzip")
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    if verbose:
        print(f"Saved EXIOBASE matrix cache to {cache_dir}")


def _load_exio_matrix_cache(
    *,
    cache_dir: Path,
) -> tuple[csr_matrix, pd.Series, pd.Series, pd.MultiIndex, dict[str, Any]]:
    z_path = cache_dir / "Z.npz"
    legacy_a_path = cache_dir / "A.npz"
    x_path = cache_dir / "x.pkl.gz"
    y_path = cache_dir / "y.pkl.gz"
    index_path = cache_dir / "index.csv.gz"
    metadata_path = cache_dir / "metadata.json"

    matrix_path = z_path if z_path.exists() else legacy_a_path
    matrix = load_npz(matrix_path).tocsr().astype(float)
    x = _coerce_series_like(pd.read_pickle(x_path)).astype(float)
    y = _coerce_series_like(pd.read_pickle(y_path)).astype(float)
    index_frame = pd.read_csv(index_path)
    basis_index = _exio_index_from_frame(index_frame)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if matrix.shape != (len(basis_index), len(basis_index)):
        raise ValueError(f"Matrix cache shape mismatch: matrix.shape={matrix.shape}, index={len(basis_index)}")
    if len(x) != len(basis_index) or len(y) != len(basis_index):
        raise ValueError("Matrix cache vectors do not match the basis index length.")
    if tuple(metadata.get("matrix_shape", [])) != tuple(int(v) for v in matrix.shape):
        raise ValueError("Matrix cache metadata does not match the cached matrix shape.")
    if int(metadata.get("nnz", -1)) != int(matrix.nnz):
        raise ValueError("Matrix cache metadata does not match the cached matrix nnz.")

    x = x.reindex(basis_index).astype(float)
    y = y.reindex(basis_index).astype(float)
    if matrix_path == z_path:
        inv_x = (1.0 / x.replace(0.0, pd.NA)).fillna(0.0).astype(float)
        A = (matrix @ diags(inv_x.to_numpy(dtype=float), format="csr")).tocsr()
        storage_kind = "Z"
    else:
        A = matrix
        storage_kind = "legacy_A"

    metadata = dict(metadata)
    metadata["matrix_storage"] = storage_kind
    metadata["derived_matrix"] = "A"
    return A, x, y, basis_index, metadata


def _resolve_exiobase3_url(year: int, system: str = "pxp", doi: str = "10.5281/zenodo.3583070") -> str:
    year_token = f"IOT_{int(year)}_{system}.zip"
    fallback_url = f"https://zenodo.org/records/15689391/files/{year_token}"
    cfg = dict(EXIOBASE3_CONFIG)
    cfg["url_db_view"] = "https://doi.org/" + str(doi)
    try:
        exio_web_content = _get_url_datafiles(**cfg)
    except Exception:
        return fallback_url
    for url in exio_web_content.data_urls:
        if year_token in url:
            return url
    return fallback_url


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


def _exio_cache_paths(data_dir: Path, year: int, system: str) -> dict[str, Path]:
    tag = f"{int(year)}_{system}"
    return {
        "text": data_dir / f"mrio_system_{tag}",
        "fast": data_dir / f"mrio_system_{tag}.fast.pkl",
        # Legacy names kept for backward-compatible pxp migration/loading.
        "legacy_text": data_dir / "mrio_system",
        "legacy_text_alt": data_dir / "mrio_system.pkl",
        "legacy_fast": data_dir / "mrio_system.fast.pkl",
    }


@contextmanager
def _exclusive_cache_lock(
    lock_path: Path,
    *,
    label: str,
    timeout_seconds: float | None = 6 * 60 * 60,
    poll_seconds: float = 1.0,
    stale_seconds: float = 12 * 60 * 60,
    verbose: bool = False,
):
    start = time.time()
    next_log_at = start
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age_seconds = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue

            if age_seconds >= stale_seconds:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue

            waited = time.time() - start
            if timeout_seconds is not None and waited >= timeout_seconds:
                raise TimeoutError(f"Timed out waiting for cache lock {lock_path} ({label}).")
            if verbose and time.time() >= next_log_at:
                print(f"Waiting for cache lock {lock_path.name} ({label}) ...")
                next_log_at = time.time() + 30.0
            time.sleep(max(0.1, float(poll_seconds)))
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(f"pid={os.getpid()}\nstarted_at={int(time.time())}\nlabel={label}\n")
        break

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _load_cached_exiobase(
    *,
    data_dir: Path,
    system: str,
    mrio_cache_dir: Path,
    fast_cache_path: Path,
    legacy_cache_dir: Path,
    legacy_cache_dir_alt: Path,
    legacy_fast_cache_path: Path,
    load_fast_cache: bool = True,
    write_fast_cache: bool = True,
    verbose: bool = False,
):
    mrio = None

    fast_candidates = [fast_cache_path] if load_fast_cache else []
    if load_fast_cache and system == "pxp" and legacy_fast_cache_path.exists():
        fast_candidates.append(legacy_fast_cache_path)

    for fast_candidate in fast_candidates:
        if not fast_candidate.exists():
            continue
        if verbose:
            print(f"Loading EXIOBASE fast cache from {fast_candidate} ...")
        try:
            mrio = _load_fast_pickle(fast_candidate)
            if fast_candidate != fast_cache_path and not fast_cache_path.exists():
                try:
                    fast_candidate.rename(fast_cache_path)
                    if verbose:
                        print(f"Migrated loaded legacy fast cache to {fast_cache_path}")
                except Exception:
                    pass
            break
        except Exception as exc:
            ts = int(time.time())
            bad = data_dir / f"{fast_candidate.name}.invalid.{ts}"
            try:
                fast_candidate.rename(bad)
            except Exception:
                pass
            if verbose:
                print(f"Warning: failed fast-cache load ({type(exc).__name__}: {exc}); falling back to text cache.")

    text_candidates = [mrio_cache_dir]
    if system == "pxp":
        text_candidates.extend([legacy_cache_dir, legacy_cache_dir_alt])

    for text_candidate in text_candidates:
        if mrio is not None:
            break
        if not text_candidate.exists() or not text_candidate.is_dir():
            continue

        cache_para = text_candidate / "file_parameters.json"
        if not cache_para.exists():
            ts = int(time.time())
            try:
                text_candidate.rename(data_dir / f"{text_candidate.name}.invalid.{ts}")
            except Exception:
                pass
            continue

        if verbose:
            print(f"Loading cached EXIOBASE IOSystem from {text_candidate} ...")
        try:
            mrio = pymrio.load(text_candidate)
            if write_fast_cache:
                _save_fast_pickle(fast_cache_path, mrio, verbose=verbose)
        except FileNotFoundError:
            ts = int(time.time())
            try:
                text_candidate.rename(data_dir / f"{text_candidate.name}.invalid.{ts}")
            except Exception:
                pass
            mrio = None

    return mrio


def _validate_exiobase_system(mrio: pymrio.IOSystem, system: str) -> None:
    expected_counts = {"pxp": 200, "ixi": 163}
    expected = expected_counts[str(system)]

    z_matrix = getattr(mrio, "Z", None)
    if z_matrix is None or not isinstance(z_matrix.index, pd.MultiIndex):
        raise ValueError("Loaded EXIOBASE object has no MultiIndex Z table; cannot validate system.")

    actual = len(z_matrix.index.get_level_values(1).unique())
    if actual != expected:
        raise ValueError(
            f"Loaded EXIOBASE cache/archive does not match requested system '{system}': "
            f"found {actual} sector labels in Z, expected {expected}."
        )


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
    exio_full_cache: str = "read_write",
    exio_save_text_cache: bool = False,
    exio_system: str = "pxp",
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

    Notes
    -----
    EXIOBASE cache policy defaults to `zip + fast pickle`:
    - keep source zip locally for reproducibility/re-parse,
    - write/use `mrio_system_<year>_<system>.fast.pkl` for fast reloads,
    - do not write large text cache directories unless
      `exio_save_text_cache=True`.
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
        system = str(exio_system).strip().lower()
        if system not in {"pxp", "ixi"}:
            raise ValueError(f"Unsupported exiobase system '{exio_system}'. Use 'pxp' or 'ixi'.")
        cache_mode = str(exio_full_cache).strip().lower()
        if cache_mode not in {"read_write", "read_only", "write_only", "none"}:
            raise ValueError(
                "exio_full_cache must be one of: read_write, read_only, write_only, none"
            )
        load_fast_cache = cache_mode in {"read_write", "read_only"}
        write_fast_cache = cache_mode in {"read_write", "write_only"}

        data_dir = resolve_exiobase_year_dir(int(year), data_root=data_root)
        data_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Ensuring EXIOBASE 3 data for {year} ({system}) in {data_dir}...")

        paths = resolve_exiobase_cache_paths(int(year), system=system, data_root=data_root)
        mrio_cache_dir = paths["text"]
        fast_cache_path = paths["fast"]
        legacy_cache_dir = paths["legacy_text"]
        legacy_cache_dir_alt = paths["legacy_text_alt"]
        legacy_fast_cache_path = paths["legacy_fast"]

        mrio = None

        # Backward-compat migration for historical pxp cache names.
        if system == "pxp" and not mrio_cache_dir.exists():
            for cand in (legacy_cache_dir, legacy_cache_dir_alt):
                if cand.is_dir():
                    if verbose:
                        print(f"Found legacy EXIOBASE text cache dir at {cand}")
                    try:
                        cand.rename(mrio_cache_dir)
                        if verbose:
                            print(f"Migrated legacy text cache dir to {mrio_cache_dir}")
                    except Exception:
                        # Keep fallback path if rename fails.
                        mrio_cache_dir = cand
                    break

        if system == "pxp" and not fast_cache_path.exists() and legacy_fast_cache_path.exists():
            try:
                legacy_fast_cache_path.rename(fast_cache_path)
                if verbose:
                    print(f"Migrated legacy EXIOBASE fast cache to {fast_cache_path}")
            except Exception:
                if verbose:
                    print(f"Using legacy EXIOBASE fast cache at {legacy_fast_cache_path}")

        # Optional: force re-download/re-parse
        if force_download and mrio_cache_dir.exists():
            ts = int(time.time())
            mrio_cache_dir.rename(data_dir / f"mrio_system.forced.{ts}")
        if force_download and fast_cache_path.exists():
            ts = int(time.time())
            fast_cache_path.rename(data_dir / f"{fast_cache_path.name}.forced.{ts}")

        # In pxp mode, force refresh also invalidates remaining legacy fast cache.
        if force_download and system == "pxp" and legacy_fast_cache_path.exists():
            ts = int(time.time())
            legacy_fast_cache_path.rename(data_dir / f"{legacy_fast_cache_path.name}.forced.{ts}")

        # Check if cached text system exists to avoid loading invalid directories.
        cache_para = mrio_cache_dir / "file_parameters.json"
        if mrio_cache_dir.exists() and not cache_para.exists():
            ts = int(time.time())
            mrio_cache_dir.rename(data_dir / f"mrio_system.invalid.{ts}")

        mrio = _load_cached_exiobase(
            data_dir=data_dir,
            system=system,
            mrio_cache_dir=mrio_cache_dir,
            fast_cache_path=fast_cache_path,
            legacy_cache_dir=legacy_cache_dir,
            legacy_cache_dir_alt=legacy_cache_dir_alt,
            legacy_fast_cache_path=legacy_fast_cache_path,
            load_fast_cache=load_fast_cache,
            write_fast_cache=write_fast_cache,
            verbose=verbose,
        )

        if mrio is None:
            lock_path = data_dir / f".exiobase_{int(year)}_{system}.lock"
            with _exclusive_cache_lock(
                lock_path,
                label=f"EXIOBASE {int(year)} {system}",
                verbose=verbose,
            ):
                mrio = _load_cached_exiobase(
                    data_dir=data_dir,
                    system=system,
                    mrio_cache_dir=mrio_cache_dir,
                    fast_cache_path=fast_cache_path,
                    legacy_cache_dir=legacy_cache_dir,
                    legacy_cache_dir_alt=legacy_cache_dir_alt,
                    legacy_fast_cache_path=legacy_fast_cache_path,
                    load_fast_cache=load_fast_cache,
                    write_fast_cache=write_fast_cache,
                    verbose=verbose,
                )

                if mrio is None:
                     # Download if needed (MRIO wrapper around per-year EXIOBASE zip download)
                     zip_path = ensure_exiobase_year_zip(
                         year=int(year),
                         data_dir=paths["data_dir"],
                         system=system,
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

                     # Optional text cache: disabled by default to avoid large on-disk caches.
                     if exio_save_text_cache:
                         # If a previous partial run created a directory at this path, move it aside.
                         if mrio_cache_dir.exists():
                             ts = int(time.time())
                             mrio_cache_dir.rename(data_dir / f"{mrio_cache_dir.name}.stale.{ts}")
                         if verbose:
                             print(f"Saving parsed EXIOBASE IOSystem to {mrio_cache_dir} ...")
                         mrio.save(mrio_cache_dir)

                     # Also persist fast cache so future runs bypass large text parsing.
                     if write_fast_cache:
                         _save_fast_pickle(fast_cache_path, mrio, verbose=verbose)

                     # Keep source zip by default; only delete when explicitly requested.
                     # If text cache is enabled, require complete parsed cache; otherwise
                     # allow delete once the fast cache exists.
                     can_delete_zip = (
                         _is_complete_exio_parsed_cache(mrio_cache_dir)
                         if exio_save_text_cache
                         else (fast_cache_path.exists() if write_fast_cache else False)
                     )
                     if delete_source_zip_after_parse and can_delete_zip:
                         try:
                             zip_path.unlink(missing_ok=True)
                             if verbose:
                                 print(f"Deleted EXIOBASE source zip after successful parse: {zip_path}")
                         except Exception as exc:
                             if verbose:
                                 print(f"Warning: failed to delete EXIOBASE source zip {zip_path}: {exc}")

        _validate_exiobase_system(mrio, system)

        y_cols = [c for c in mrio.Y.columns if "EXP" not in c]
        y_full = mrio.Y[y_cols].sum(axis=1)
    
    # Ensure Ghosh is calculated
    if calc_all and getattr(mrio, "G", None) is None:
        if verbose:
            print("Computing MRIO derived matrices (calc_all)...")
        mrio.calc_all(include_ghosh=include_ghosh)
        
    return mrio, y_full


def load_mrio_matrices(
    year: int,
    *,
    force_download: bool = False,
    data_root: str | Path | None = None,
    calc_all: bool = False,
    verbose: bool = False,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    delete_source_zip_after_parse: bool = False,
    exio_system: str = "pxp",
) -> tuple[csr_matrix, pd.Series, pd.Series, pd.MultiIndex, dict[str, Any]]:
    """Load a compact EXIOBASE matrix cache for matrix-only workflows.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Technical-coefficient matrix.
    x : pandas.Series
        Output vector indexed by the EXIO basis index.
    y : pandas.Series
        Final-demand vector indexed by the EXIO basis index.
    index : pandas.MultiIndex
        Basis index with ``region`` and ``sector`` levels.
    metadata : dict
        Cache metadata describing what is stored and what is not.
    """
    system = str(exio_system).strip().lower()
    if system not in {"pxp", "ixi"}:
        raise ValueError(f"Unsupported exiobase system '{exio_system}'. Use 'pxp' or 'ixi'.")

    data_dir = resolve_exiobase_year_dir(int(year), data_root=data_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_exiobase_matrix_cache_paths(int(year), system=system, data_root=data_root)
    cache_dir = paths["matrix_cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    if force_download and cache_dir.exists():
        ts = int(time.time())
        cache_dir.rename(data_dir / f"{cache_dir.name}.forced.{ts}")
        cache_dir.mkdir(parents=True, exist_ok=True)

    if not force_download:
        try:
            A, x, y, basis_index, metadata = _load_exio_matrix_cache(cache_dir=cache_dir)
            metadata = dict(metadata)
            metadata["cache_status"] = "hit"
            metadata["cache_dir"] = str(cache_dir)
            return A, x, y, basis_index, metadata
        except FileNotFoundError:
            pass
        except Exception as exc:
            ts = int(time.time())
            try:
                cache_dir.rename(data_dir / f"{cache_dir.name}.invalid.{ts}")
            except Exception:
                pass
            if verbose:
                print(f"Warning: failed matrix-cache load ({type(exc).__name__}: {exc}); rebuilding from EXIOBASE zip.")

    zip_path = ensure_exiobase_year_zip(
        year=int(year),
        data_dir=paths["data_dir"],
        system=system,
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
        print(f"Parsing EXIOBASE zip {zip_path} for matrix cache ...")
    mrio = pymrio.parse_exiobase3(zip_path)
    _validate_exiobase_system(mrio, system)

    Z = pd.DataFrame(mrio.Z).astype(float)
    Y = pd.DataFrame(mrio.Y).astype(float)
    basis_index = Z.index if isinstance(Z.index, pd.MultiIndex) else pd.Index(Z.index)
    if not isinstance(basis_index, pd.MultiIndex):
        raise ValueError("Expected EXIOBASE Z to have a MultiIndex basis index.")
    if not isinstance(Y.index, pd.MultiIndex) or not isinstance(Y.columns, pd.MultiIndex):
        raise ValueError("Expected EXIOBASE Y to have MultiIndex rows and columns.")

    x = getattr(mrio, "x", None)
    x_series = _coerce_series_like(x, index=basis_index, name="x") if x is not None else None
    if x_series is None or x_series.empty or float(x_series.abs().sum()) == 0.0:
        x_from_r = pd.DataFrame(mrio.Z).astype(float).sum(axis=1) + pd.DataFrame(mrio.Y).astype(float).sum(axis=1)
        x_series = _coerce_series_like(x_from_r, index=basis_index, name="x")

    y_cols = [c for c in Y.columns if "EXP" not in str(c)]
    y_series = Y[y_cols].sum(axis=1).astype(float)
    y_series = _coerce_series_like(y_series, index=basis_index, name="y")

    Z_df = pd.DataFrame(Z, copy=False).reindex(index=basis_index, columns=basis_index).fillna(0.0).astype(float)
    Z_csr = csr_matrix(Z_df.to_numpy(dtype=float))

    density = float(Z_csr.nnz) / float(Z_csr.shape[0] * Z_csr.shape[1]) if Z_csr.shape[0] and Z_csr.shape[1] else 0.0
    metadata = {
        "cache_kind": "exiobase_matrix",
        "cache_status": "rebuilt",
        "system": system,
        "year": int(year),
        "matrix_shape": [int(Z_csr.shape[0]), int(Z_csr.shape[1])],
        "nnz": int(Z_csr.nnz),
        "density": float(density),
        "stored_objects": ["Z", "x", "y", "index"],
        "not_stored_objects": ["A", "full_pymrio_iosystem", "fast_pickle", "text_cache_dir", "extensions"],
        "source_zip": str(zip_path),
        "cache_dir": str(cache_dir),
        "index_columns": ["region", "sector"],
        "matrix_storage": "Z",
    }
    _save_exio_matrix_cache(cache_dir=cache_dir, Z=Z_csr, x=x_series, y=y_series, index=basis_index, metadata=metadata, verbose=verbose)

    A_loaded, x_loaded, y_loaded, basis_index_loaded, metadata_loaded = _load_exio_matrix_cache(cache_dir=cache_dir)
    metadata_loaded = dict(metadata_loaded)
    metadata_loaded["cache_status"] = "hit_after_rebuild"
    metadata_loaded["cache_dir"] = str(cache_dir)
    if delete_source_zip_after_parse:
        try:
            zip_path.unlink(missing_ok=True)
            if verbose:
                print(f"Deleted EXIOBASE source zip after successful matrix-cache validation: {zip_path}")
        except Exception as exc:
            if verbose:
                print(f"Warning: failed to delete EXIOBASE source zip {zip_path}: {exc}")
    if calc_all and verbose:
        print("calc_all is ignored for load_mrio_matrices(); the compact cache stores matrices only.")
    return A_loaded, x_loaded, y_loaded, basis_index_loaded, metadata_loaded
