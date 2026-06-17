# pyright: reportMissingImports=false
from __future__ import annotations

from contextlib import contextmanager
import io
import numpy as np
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Tuple, cast

import pandas as pd
import pymrio
from scipy.sparse import csr_matrix
from pymrio.tools.iodownloader import EXIOBASE3_CONFIG, HEADERS, MRIOMetaData, _get_url_datafiles

from ecolecon_mrio.download_utils import DownloadSpec, download_file_resumable, zip_contains_member
from ecolecon_mrio.wiod16 import load_wiod16
from ecolecon_mrio.mrio_sda import UnitMetadata


# Default EXIOBASE source DOI used for new downloads. Cache paths are
# intentionally unchanged in this patch; the DOI controls only source
# discovery/download and provenance metadata.
DEFAULT_EXIOBASE3_DOI = "10.5281/zenodo.20051562"


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
    # Return only the EXIOBASE source-zip cache path. Full parsed IOSystem
    # caches and fast pickle caches are intentionally not exposed.
    system = str(system).strip().lower()
    if system not in {"pxp", "ixi"}:
        raise ValueError(f"Unsupported exiobase system '{system}'. Use 'pxp' or 'ixi'.")
    data_dir = resolve_exiobase_year_dir(int(year), data_root=data_root)
    return {
        "data_dir": data_dir,
        "zip": data_dir / f"IOT_{int(year)}_{system}.zip",
    }



def exiobase3_monetary_unit(
    *,
    system: str = "pxp",
    override: str | None = None,
) -> UnitMetadata:
    """Return EXIOBASE monetary table unit metadata for reporting.

    EXIOBASE monetary IOT files are conventionally interpreted in million euros
    in this project.  The helper is intentionally modest: it exposes the unit and
    provenance needed by downstream CSV outputs without introducing conversion or
    symbolic unit algebra.  Use ``override`` only when a caller has externally
    verified a different monetary unit for a specific dataset extract.
    """

    if override is not None and str(override).strip():
        return UnitMetadata(unit=str(override).strip(), unit_source="explicit_override")
    system_clean = str(system).strip().lower()
    if system_clean not in {"pxp", "ixi"}:
        raise ValueError(f"Unsupported exiobase system {system!r}. Use 'pxp' or 'ixi'.")
    return UnitMetadata(unit="MEUR", unit_source="exiobase3_monetary_table_convention")


def _open_exio_zip_member(zf: zipfile.ZipFile, member_name: str):
    """Open an EXIOBASE text member by basename, allowing root or nested zip layouts."""
    wanted = str(member_name).strip()
    names = zf.namelist()
    if wanted in names:
        return zf.open(wanted)
    matches = [name for name in names if Path(name).name == wanted]
    if not matches:
        lower = wanted.lower()
        matches = [name for name in names if Path(name).name.lower() == lower]
    if not matches:
        shown = ", ".join(sorted(Path(name).name for name in names)[:30])
        raise FileNotFoundError(f"EXIOBASE zip is missing {wanted!r}; available basenames include: {shown}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda name: (name.count("/"), len(name), name))
    return zf.open(matches[0])


def _find_zip_member_by_basename(zf: zipfile.ZipFile, basename: str) -> str:
    """Return the zip member named `basename`, allowing one nested directory level.

    Official EXIOBASE source zips normally expose x.txt, Z.txt, Y.txt, and
    file_parameters.json at the archive root, but some local archives may have a
    top-level directory wrapper. The loader must not depend on a parsed text-cache
    directory or a derived A.txt cache.
    """
    basename = str(basename)
    names = zf.namelist()
    if basename in names:
        return basename
    matches = [name for name in names if Path(name).name == basename]
    if not matches:
        raise FileNotFoundError(
            f"EXIOBASE source zip {zf.filename!s} does not contain {basename!r}. "
            f"Available basename sample: {sorted({Path(n).name for n in names})[:20]}"
        )
    matches = sorted(matches, key=lambda name: (name.count("/"), len(name), name))
    return matches[0]


def _open_exio_zip_text(zf: zipfile.ZipFile, basename: str) -> io.TextIOWrapper:
    member = _find_zip_member_by_basename(zf, basename)
    return io.TextIOWrapper(zf.open(member), encoding="utf-8", newline="")


def _read_exio_x_from_zip(zip_path: Path) -> tuple[pd.Series, pd.MultiIndex]:
    with zipfile.ZipFile(zip_path) as zf:
        with _open_exio_zip_text(zf, "x.txt") as fh:
            frame = pd.read_csv(fh, sep="\t")
    required = {"region", "sector"}
    if not required.issubset(frame.columns):
        raise ValueError("EXIOBASE x.txt must contain region and sector columns.")
    value_col = "indout" if "indout" in frame.columns else frame.columns[-1]
    basis_index = pd.MultiIndex.from_frame(
        frame.loc[:, ["region", "sector"]].astype(str),
        names=["region", "sector"],
    )
    x_numeric = cast(pd.Series, pd.to_numeric(frame[value_col], errors="coerce"))
    x = x_numeric.fillna(0.0).astype(float)
    return pd.Series(x.to_numpy(dtype=float), index=basis_index, name="x"), basis_index



def load_exiobase3_output_for_year(
    year: int,
    *,
    system: str = "pxp",
    data_root: str | Path | None = None,
    force_download: bool = False,
    allow_download_missing: bool = False,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    doi: str = DEFAULT_EXIOBASE3_DOI,
    verbose: bool = False,
) -> tuple[pd.Series, pd.MultiIndex, dict[str, Any]]:
    """Load only EXIOBASE ``x.txt`` for one year.

    Basis compilation and observed-output projection need only the annual
    monetary output vector ``x_t`` and its canonical ``(region, sector)`` index.
    They do not need ``Z.txt``, ``Y.txt``, sparse ``A``, Leontief inverses, Ghosh
    objects, a full pymrio ``IOSystem``, or any derived matrix cache.  This public
    MRIO boundary keeps source-zip discovery/download policy in MRIO while
    reading exactly the one source member needed by those analysis stages.
    """

    # Normalize the requested EXIOBASE IO system before any file path is built;
    # pxp and ixi have different sector bases and must never be silently mixed.
    system_clean = str(system).strip().lower()

    # Reject unsupported systems loudly because a wrong system changes the
    # region-sector index and therefore all downstream basis/projection rows.
    if system_clean not in {"pxp", "ixi"}:
        raise ValueError(f"Unsupported exiobase system '{system}'. Use 'pxp' or 'ixi'.")

    # Resolve the standard MRIO source-zip path without exposing zip internals to
    # downstream analysis repos.
    paths = resolve_exiobase_cache_paths(int(year), system=system_clean, data_root=data_root)

    # Ensure the year directory exists; the only durable data cache created or
    # reused by this function is the raw EXIOBASE source zip in that directory.
    data_dir = paths["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    # Locate or optionally download the raw IOT zip.  This call validates the zip
    # container but does not parse a full pymrio object or write derived caches.
    zip_path = ensure_exiobase_year_zip(
        int(year),
        data_dir=data_dir,
        system=system_clean,
        force_download=force_download,
        allow_download_missing=allow_download_missing,
        resume_downloads=resume_downloads,
        show_download_progress=show_download_progress,
        download_chunk_mb=download_chunk_mb,
        download_timeout_seconds=download_timeout_seconds,
        download_retries=download_retries,
        doi=str(doi),
        verbose=verbose,
    )

    # Read only x.txt.  This is the critical low-RAM step: no Z/Y streaming and no
    # A derivation happen in stages that only need the output vector or index.
    x, basis_index = _read_exio_x_from_zip(zip_path)

    # Return explicit metadata so callers and manifests can audit that this was
    # an x-only source-zip read rather than a full-object or matrix-cache load.
    metadata: dict[str, Any] = {
        "cache_kind": "exiobase_zip_stream_x_only",
        "cache_status": "read_x_direct_from_zip_no_cache_write",
        "source_zip": str(zip_path),
        "system": system_clean,
        "year": int(year),
        "exio_doi": _normalize_exiobase_doi(doi),
        "index_columns": ["region", "sector"],
        "stored_objects": [],
        "not_stored_objects": [
            "Z",
            "A",
            "Y",
            "full_pymrio_iosystem",
            "fast_pickle",
            "text_cache_dir",
            "matrix_npz_cache",
        ],
        "source_zip_retained": True,
    }

    # The output vector carries the authoritative MultiIndex used by analysis_v2;
    # the metadata documents the source path and cache policy for reproducibility.
    return x, basis_index, metadata

def _exio_text_header_index(fh: io.TextIOBase, *, table_name: str) -> pd.MultiIndex:
    regions = next(fh).rstrip("\n\r").split("\t")[2:]
    sectors = next(fh).rstrip("\n\r").split("\t")[2:]
    next(fh)
    if len(regions) != len(sectors):
        raise ValueError(
            f"EXIOBASE {table_name} column-header length mismatch: "
            f"regions={len(regions)}, sectors={len(sectors)}"
        )
    return pd.MultiIndex.from_arrays([regions, sectors], names=["region", "sector"])


def _numeric_tail_from_exio_line(line: str) -> np.ndarray:
    # EXIO text rows start with two tab-separated labels, followed by numeric columns.
    try:
        _region, _sector, rest = line.rstrip("\n\r").split("\t", 2)
    except ValueError:
        return np.array([], dtype=float)
    return np.fromstring(rest.replace("\t", " "), sep=" ", dtype=float)


def _read_exio_Z_sparse_from_zip(zip_path: Path, *, basis_index: pd.MultiIndex) -> csr_matrix:
    """Read EXIOBASE Z.txt as CSR without materializing a dense matrix.

    The EXIOBASE source zip is the only on-disk cache. A.txt is not part of the
    required source archive and must not be required by this loader.
    """
    n = len(basis_index)
    counts = np.zeros(int(n), dtype=np.int64)

    with zipfile.ZipFile(zip_path) as zf:
        with _open_exio_zip_text(zf, "Z.txt") as fh:
            column_index = _exio_text_header_index(fh, table_name="Z.txt")
            if not column_index.equals(basis_index):
                raise ValueError("EXIOBASE Z.txt column labels do not match x.txt basis order.")
            rows_seen = 0
            for row_idx, line in enumerate(fh):
                if row_idx >= n:
                    break
                try:
                    region, sector, _rest = line.rstrip("\n\r").split("\t", 2)
                except ValueError:
                    raise ValueError(f"EXIOBASE Z.txt malformed row at position {row_idx}") from None
                expected = basis_index[row_idx]
                if (str(region), str(sector)) != (str(expected[0]), str(expected[1])):
                    raise ValueError(
                        "EXIOBASE Z.txt row labels do not match x.txt basis order: "
                        f"row {row_idx} has {(region, sector)!r}, expected {expected!r}."
                    )
                vals = _numeric_tail_from_exio_line(line)
                if vals.size != n:
                    raise ValueError(
                        f"EXIOBASE Z.txt row width mismatch at row {row_idx}: "
                        f"got {vals.size}, expected {n}."
                    )
                counts[row_idx] = int(np.count_nonzero(vals))
                rows_seen += 1
            if rows_seen != n:
                raise ValueError(f"EXIOBASE Z.txt row count mismatch: got {rows_seen}, expected {n}.")

    indptr = np.empty(int(n) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=float)

    with zipfile.ZipFile(zip_path) as zf:
        with _open_exio_zip_text(zf, "Z.txt") as fh:
            column_index = _exio_text_header_index(fh, table_name="Z.txt")
            if not column_index.equals(basis_index):
                raise ValueError("EXIOBASE Z.txt column labels do not match x.txt basis order.")
            for row_idx, line in enumerate(fh):
                if row_idx >= n:
                    break
                try:
                    region, sector, _rest = line.rstrip("\n\r").split("\t", 2)
                except ValueError:
                    raise ValueError(f"EXIOBASE Z.txt malformed row at position {row_idx}") from None
                expected = basis_index[row_idx]
                if (str(region), str(sector)) != (str(expected[0]), str(expected[1])):
                    raise ValueError(
                        "EXIOBASE Z.txt row labels do not match x.txt basis order: "
                        f"row {row_idx} has {(region, sector)!r}, expected {expected!r}."
                    )
                vals = _numeric_tail_from_exio_line(line)
                nz = np.flatnonzero(vals)
                start = int(indptr[row_idx])
                stop = int(indptr[row_idx + 1])
                indices[start:stop] = nz.astype(np.int32, copy=False)
                data[start:stop] = vals[nz]

    Z = csr_matrix((data, indices, indptr), shape=(int(n), int(n)))
    Z.eliminate_zeros()
    return Z


def _derive_A_from_Z_and_x(Z: csr_matrix, x: pd.Series, *, basis_index: pd.MultiIndex) -> csr_matrix:
    """Compute sparse technical coefficients A_ij = Z_ij / x_j.

    This is column scaling by the buyer-sector output vector. Zero-output columns
    are mapped to zero coefficients to avoid infinities.
    """
    x_numeric = cast(pd.Series, pd.to_numeric(pd.Series(x).reindex(basis_index), errors="coerce"))
    x_aligned = x_numeric.fillna(0.0)
    x_values = x_aligned.to_numpy(dtype=float)
    inv_x = np.zeros_like(x_values, dtype=float)
    np.divide(1.0, x_values, out=inv_x, where=np.isfinite(x_values) & (x_values != 0.0))

    # Reuse the just-parsed sparse Z storage as A; the caller never needs raw Z
    # after this point, and avoiding Z.copy() cuts load-time sparse peak memory.
    A = Z.astype(float, copy=False).tocsr()
    if A.nnz:
        A.data *= inv_x[A.indices]
    A.eliminate_zeros()
    return A


def _read_exio_y_direct_from_zip(zip_path: Path, *, basis_index: pd.MultiIndex) -> pd.Series:
    n = len(basis_index)
    with zipfile.ZipFile(zip_path) as zf:
        with _open_exio_zip_text(zf, "Y.txt") as fh:
            _regions = next(fh).rstrip("\n\r").split("\t")[2:]
            categories = next(fh).rstrip("\n\r").split("\t")[2:]
            keep = np.array(["export" not in str(cat).lower() for cat in categories], dtype=bool)
            next(fh)
            y_vals = np.zeros(n, dtype=float)
            rows_seen = 0
            for row_idx, line in enumerate(fh):
                if row_idx >= n:
                    break
                try:
                    region, sector, _rest = line.rstrip("\n\r").split("\t", 2)
                except ValueError:
                    raise ValueError(f"EXIOBASE Y.txt malformed row at position {row_idx}") from None
                expected = basis_index[row_idx]
                if (str(region), str(sector)) != (str(expected[0]), str(expected[1])):
                    raise ValueError(
                        "EXIOBASE Y.txt row labels do not match x.txt basis order: "
                        f"row {row_idx} has {(region, sector)!r}, expected {expected!r}."
                    )
                vals = _numeric_tail_from_exio_line(line)
                width = min(vals.size, keep.size)
                if width:
                    y_vals[row_idx] = float(vals[:width][keep[:width]].sum())
                rows_seen += 1
            if rows_seen != n:
                raise ValueError(f"EXIOBASE Y.txt row count mismatch: got {rows_seen}, expected {n}.")
    return pd.Series(y_vals, index=basis_index, name="y", dtype=float)


def _load_exio_matrices_direct_from_zip(zip_path: Path) -> tuple[csr_matrix, pd.Series, pd.Series, pd.MultiIndex, dict[str, Any]]:
    x, basis_index = _read_exio_x_from_zip(zip_path)
    Z = _read_exio_Z_sparse_from_zip(zip_path, basis_index=basis_index)
    A = _derive_A_from_Z_and_x(Z, x, basis_index=basis_index)
    y = _read_exio_y_direct_from_zip(zip_path, basis_index=basis_index)
    density = float(A.nnz) / float(A.shape[0] * A.shape[1]) if A.shape[0] and A.shape[1] else 0.0
    metadata = {
        "cache_kind": "exiobase_zip_stream",
        "cache_status": "parsed_z_x_y_direct_from_zip_no_cache_write",
        "matrix_source": "Z.txt/x.txt",
        "a_derivation": "A_ij = Z_ij / x_j; zero-output columns set to zero",
        "matrix_shape": [int(A.shape[0]), int(A.shape[1])],
        "z_nnz": int(Z.nnz),
        "nnz": int(A.nnz),
        "density": density,
        "stored_objects": [],
        "not_stored_objects": ["Z", "full_pymrio_iosystem", "fast_pickle", "text_cache_dir", "extensions"],
        "source_zip": str(zip_path),
        "index_columns": ["region", "sector"],
        "cache_matrices": False,
    }
    return A, x, y, basis_index, metadata




def _normalize_exiobase_doi(doi: str | None = None) -> str:
    """Return a bare DOI token, accepting either a DOI or a doi.org URL."""
    raw = str(doi if doi is not None and str(doi).strip() else DEFAULT_EXIOBASE3_DOI).strip()
    lowered = raw.casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lowered.startswith(prefix.casefold()):
            raw = raw[len(prefix):].strip()
            break
    return raw.rstrip("/")


def _zenodo_record_id_from_doi(doi: str) -> str | None:
    doi_clean = _normalize_exiobase_doi(doi)
    marker = "zenodo."
    lowered = doi_clean.casefold()
    pos = lowered.find(marker)
    if pos < 0:
        return None
    tail = doi_clean[pos + len(marker):]
    digits: list[str] = []
    for char in tail:
        if char.isdigit():
            digits.append(char)
        else:
            break
    return "".join(digits) or None


def _resolve_exiobase3_url(year: int, system: str = "pxp", doi: str = DEFAULT_EXIOBASE3_DOI) -> str:
    doi_clean = _normalize_exiobase_doi(doi)
    year_token = f"IOT_{int(year)}_{system}.zip"
    record_id = _zenodo_record_id_from_doi(doi_clean)
    if record_id is None:
        fallback_url = f"https://doi.org/{doi_clean}"
    else:
        fallback_url = f"https://zenodo.org/records/{record_id}/files/{year_token}"
    cfg = dict(EXIOBASE3_CONFIG)
    cfg["url_db_view"] = "https://doi.org/" + doi_clean
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
    doi: str = DEFAULT_EXIOBASE3_DOI,
    verbose: bool = False,
) -> Path:
    doi_clean = _normalize_exiobase_doi(doi)
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

    url = _resolve_exiobase3_url(int(year), system=system, doi=doi_clean)
    if verbose:
        print(f"EXIOBASE zip not found at {zip_path}; downloading {url} ...")

    downlog = MRIOMetaData._make_download_log(
        location=data_dir,
        description="Download log of EXIOBASE3",
        name="EXIO3",
        system=str(system),
        version=doi_clean,
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
    exio_full_cache: str = "none",
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
    EXIOBASE full-IOSystem cache policy is zip-only:
    - keep the source zip locally for reproducibility and sparse/direct reads,
    - do not write/read fast pickle caches,
    - do not write/read large parsed text-cache directories,
    - use `load_mrio_matrices()` for compact sparse A/x/y loading from the zip.
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
        raise RuntimeError(
            "Full EXIOBASE IOSystem loading via load_mrio(dataset='exiobase') is disabled. "
            "Use load_mrio_matrices() / eigenmode sparse loaders so only the source zip is cached "
            "and A/x/Y are streamed directly from the zip."
        )

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
    doi: str = DEFAULT_EXIOBASE3_DOI,
    delete_source_zip_after_parse: bool = False,
    exio_system: str = "pxp",
    cache_matrices: bool = False,
    read_existing_cache: bool = False,
) -> tuple[csr_matrix, pd.Series, pd.Series, pd.MultiIndex, dict[str, Any]]:
    """Load EXIOBASE matrices directly from the source zip, without derived caches.

    This is the zip-only, sparse, low-disk loader used by eigenmode workflows.
    It reads only x.txt, Z.txt, and Y.txt from the EXIOBASE zip, computes
    sparse technical coefficients as A_ij = Z_ij / x_j, and returns A/x/y. It never writes
    fast pickles, parsed text caches, compact matrix caches, or deletes the zip.
    """
    system = str(exio_system).strip().lower()
    if system not in {"pxp", "ixi"}:
        raise ValueError(f"Unsupported exiobase system '{exio_system}'. Use 'pxp' or 'ixi'.")
    _ = read_existing_cache  # deprecated no-op: zip-stream loading never reads derived matrix caches.
    if cache_matrices:
        raise ValueError("EXIOBASE matrix cache writes are disabled; only the source zip may be cached.")
    if delete_source_zip_after_parse:
        raise ValueError("delete_source_zip_after_parse=True is disabled; the EXIOBASE source zip must be retained.")

    paths = resolve_exiobase_cache_paths(int(year), system=system, data_root=data_root)
    data_dir = paths["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ensure_exiobase_year_zip(
        year=int(year),
        data_dir=data_dir,
        system=system,
        force_download=force_download,
        allow_download_missing=allow_download_missing,
        resume_downloads=resume_downloads,
        show_download_progress=show_download_progress,
        download_chunk_mb=download_chunk_mb,
        download_timeout_seconds=download_timeout_seconds,
        download_retries=download_retries,
        doi=str(doi),
        verbose=verbose,
    )

    if verbose:
        print(f"Streaming EXIOBASE x/Z/Y directly from {zip_path} (zip-only, no derived cache writes) ...")
    A_direct, x_direct, y_direct, basis_index_direct, metadata_direct = _load_exio_matrices_direct_from_zip(zip_path)
    metadata_direct = dict(metadata_direct)
    metadata_direct.update(
        {
            "system": system,
            "year": int(year),
            "exio_doi": _normalize_exiobase_doi(doi),
            "cache_status": "streamed_direct_from_zip",
            "zip_only_cache": True,
            "cache_matrices": False,
            "source_zip_retained": True,
        }
    )
    if calc_all and verbose:
        print("calc_all is ignored for load_mrio_matrices(); sparse A/x/y are returned directly.")
    return A_direct, x_direct, y_direct, basis_index_direct, metadata_direct
