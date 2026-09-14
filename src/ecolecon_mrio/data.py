# pyright: reportMissingImports=false
from __future__ import annotations

from contextlib import contextmanager
import io
import json
import numpy as np
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence, cast

import pandas as pd
import pymrio
from scipy.sparse import csr_matrix
from pymrio.tools.iodownloader import EXIOBASE3_CONFIG, HEADERS, MRIOMetaData, _get_url_datafiles

from ecolecon_mrio.download_utils import DownloadSpec, download_file_resumable, zip_contains_member
from ecolecon_mrio.iot_flow_semantics import (
    _validate_technical_coefficient_denominators,
    _validated_nonnegative_output,
)
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
    """Read EXIO ``x.txt`` without turning missing/malformed output into zero."""
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
    if not basis_index.is_unique:
        raise ValueError("EXIOBASE x.txt contains duplicate region-sector rows.")
    x_numeric_obj = pd.to_numeric(frame[value_col], errors="coerce")
    if not isinstance(x_numeric_obj, pd.Series):
        raise TypeError("EXIOBASE x.txt numeric conversion did not return a Series.")
    nonfinite = ~np.isfinite(x_numeric_obj.to_numpy(dtype="float64"))
    if bool(nonfinite.any()):
        bad = [str(basis_index[i]) for i in np.flatnonzero(nonfinite)[:10]]
        raise ValueError(
            "EXIOBASE x.txt contains blank, malformed, or non-finite output values; "
            f"sample={bad!r}. Missing monetary output must not become zero."
        )
    x = x_numeric_obj.astype(float)
    return pd.Series(x.to_numpy(dtype=float), index=basis_index, name="x"), basis_index

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


def _read_exio_Z_sparse_from_zip(zip_path: Path) -> tuple[csr_matrix, pd.MultiIndex]:
    """Read EXIOBASE ``Z.txt`` as CSR and derive its own canonical square-IOT basis."""
    with zipfile.ZipFile(zip_path) as zf:
        with _open_exio_zip_text(zf, "Z.txt") as fh:
            basis_index = _exio_text_header_index(fh, table_name="Z.txt")
            if not basis_index.is_unique:
                raise ValueError("EXIOBASE Z.txt contains duplicate region-sector columns.")
            n = len(basis_index)
            counts = np.zeros(n, dtype=np.int64)
            rows_seen = 0
            for row_idx, line in enumerate(fh):
                if row_idx >= n:
                    raise ValueError(f"EXIOBASE Z.txt row count exceeds square basis length {n}.")
                try:
                    region, sector, _rest = line.rstrip("\n\r").split("\t", 2)
                except ValueError:
                    raise ValueError(f"EXIOBASE Z.txt malformed row at position {row_idx}") from None
                expected = basis_index[row_idx]
                if (str(region), str(sector)) != (str(expected[0]), str(expected[1])):
                    raise ValueError(
                        "EXIOBASE Z.txt row labels do not match its column basis order: "
                        f"row {row_idx} has {(region, sector)!r}, expected {expected!r}."
                    )
                vals = _numeric_tail_from_exio_line(line)
                if vals.size != n:
                    raise ValueError(f"EXIOBASE Z.txt row width mismatch at row {row_idx}: got {vals.size}, expected {n}.")
                if not np.isfinite(vals).all():
                    raise ValueError(f"EXIOBASE Z.txt contains non-finite monetary values at row {row_idx}.")
                counts[row_idx] = int(np.count_nonzero(vals))
                rows_seen += 1
            if rows_seen != n:
                raise ValueError(f"EXIOBASE Z.txt row count mismatch: got {rows_seen}, expected {n}.")
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    data = np.empty(int(indptr[-1]), dtype=float)
    with zipfile.ZipFile(zip_path) as zf:
        with _open_exio_zip_text(zf, "Z.txt") as fh:
            second_basis = _exio_text_header_index(fh, table_name="Z.txt")
            if not second_basis.equals(basis_index):
                raise ValueError("EXIOBASE Z.txt header changed between sparse read passes.")
            for row_idx, line in enumerate(fh):
                vals = _numeric_tail_from_exio_line(line)
                nz = np.flatnonzero(vals)
                start, stop = int(indptr[row_idx]), int(indptr[row_idx + 1])
                indices[start:stop] = nz.astype(np.int32, copy=False)
                data[start:stop] = vals[nz]
    Z = csr_matrix((data, indices, indptr), shape=(n, n))
    Z.eliminate_zeros()
    return Z, basis_index

def derive_A_from_Z_and_x(
    Z: csr_matrix,
    x: pd.Series,
    *,
    basis_index: pd.MultiIndex | None = None,
    copy: bool = True,
) -> csr_matrix:
    """Derive sparse ``A_ij = Z_ij / x_j`` without hiding invalid denominators.

    A zero-output buyer column is defined as zero only when the matching Z column
    has zero intermediate-input mass. ``copy=False`` deliberately permits reuse of
    Z sparse storage after all source-account validation is complete.
    """
    index = basis_index if basis_index is not None else x.index
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise TypeError("A derivation requires a 2-level MRIO region-sector basis.")
    if Z.shape != (len(index), len(index)):
        raise ValueError(f"Z shape {Z.shape} does not match basis length {len(index)}.")
    if not isinstance(x, pd.Series) or not x.index.equals(index):
        raise ValueError("x index must exactly match the Z basis for A derivation.")
    if Z.nnz and not np.isfinite(Z.data).all():
        raise ValueError("Z contains non-finite monetary values.")
    z_shape = cast(tuple[int, int], Z.shape)
    column_mass = np.bincount(
        Z.indices,
        weights=np.abs(Z.data),
        minlength=z_shape[1],
    ).astype("float64", copy=False)
    values = _validate_technical_coefficient_denominators(
        x,
        expected_index=index,
        column_absolute_mass=column_mass,
        context="derive_A_from_Z_and_x",
    )
    inv_x = np.zeros_like(values, dtype="float64")
    np.divide(1.0, values, out=inv_x, where=values != 0.0)
    A = Z.astype(float, copy=copy).tocsr(copy=False)
    if A.nnz:
        A.data *= inv_x[A.indices]
    A.eliminate_zeros()
    return A

def _read_exio_Y_matrix_direct_from_zip(zip_path: Path) -> tuple[pd.DataFrame, pd.MultiIndex]:
    """Read complete EXIOBASE ``Y.txt`` and derive its producer row basis independently."""
    with zipfile.ZipFile(zip_path) as zf:
        with _open_exio_zip_text(zf, "Y.txt") as fh:
            regions = next(fh).rstrip("\n\r").split("\t")[2:]
            categories = next(fh).rstrip("\n\r").split("\t")[2:]
            next(fh)
            if len(regions) != len(categories):
                raise ValueError("EXIOBASE Y.txt column-header length mismatch.")
            columns = pd.MultiIndex.from_arrays([regions, categories], names=["region", "category"])
            labels: list[tuple[str, str]] = []
            rows: list[np.ndarray] = []
            for row_idx, line in enumerate(fh):
                try:
                    region, sector, _rest = line.rstrip("\n\r").split("\t", 2)
                except ValueError:
                    raise ValueError(f"EXIOBASE Y.txt malformed row at position {row_idx}") from None
                vals = _numeric_tail_from_exio_line(line)
                if vals.size != len(columns):
                    raise ValueError(f"EXIOBASE Y.txt row width mismatch at row {row_idx}: got {vals.size}, expected {len(columns)}.")
                if not np.isfinite(vals).all():
                    raise ValueError(f"EXIOBASE Y.txt contains non-finite monetary values at row {row_idx}.")
                labels.append((str(region), str(sector)))
                rows.append(vals)
    basis_index = cast(
        pd.MultiIndex,
        pd.MultiIndex.from_tuples(labels, names=["region", "sector"]),
    )
    if not basis_index.is_unique:
        raise ValueError("EXIOBASE Y.txt contains duplicate producer region-sector rows.")
    data = np.vstack(rows) if rows else np.empty((0, len(columns)), dtype=float)
    return pd.DataFrame(data, index=basis_index, columns=columns, dtype="float64"), basis_index


def derive_nonexport_final_demand_from_Y(Y: pd.DataFrame) -> pd.Series:
    """Derive summed non-export final demand, rejecting malformed or missing cells."""
    if not isinstance(Y, pd.DataFrame):
        raise TypeError("Y must be a pandas DataFrame.")
    try:
        numeric = Y.to_numpy(dtype="float64", copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Y contains non-numeric final-demand values.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("Y contains missing, malformed, or non-finite final-demand values.")
    labels = (
        Y.columns.get_level_values("category")
        if isinstance(Y.columns, pd.MultiIndex) and "category" in Y.columns.names
        else Y.columns
    )

    def _is_export(label: object) -> bool:
        folded = str(label).strip().casefold()
        return "export" in folded or re.search(r"(^|[^a-z])exp([^a-z]|$)", folded) is not None

    keep = np.asarray([not _is_export(label) for label in labels], dtype=bool)
    return pd.Series(
        numeric[:, keep].sum(axis=1),
        index=Y.index,
        name="y",
        dtype="float64",
    )

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




def _doi_token_from_text(value: object) -> str | None:
    """Extract a Zenodo DOI token from arbitrary download-log text."""
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"10\.5281/zenodo\.\d+", text, flags=re.IGNORECASE)
    if match is None:
        return None
    return _normalize_exiobase_doi(match.group(0))


def _collect_exiobase_doi_tokens(value: Any) -> list[str]:
    """Return DOI-like version tokens found in pymrio download-log JSON."""
    tokens: list[str] = []
    if isinstance(value, dict):
        # Prefer explicit version/DOI-style fields, but recurse through the log
        # because pymrio metadata schemas can vary across releases.
        preferred_keys = {"version", "doi", "source_doi", "exio_doi"}
        for key, child in value.items():
            if str(key).casefold() in preferred_keys:
                token = _doi_token_from_text(child)
                if token is not None:
                    tokens.append(token)
        for child in value.values():
            tokens.extend(_collect_exiobase_doi_tokens(child))
    elif isinstance(value, list):
        for child in value:
            tokens.extend(_collect_exiobase_doi_tokens(child))
    elif isinstance(value, (str, int, float)):
        token = _doi_token_from_text(value)
        if token is not None:
            tokens.append(token)
    return list(dict.fromkeys(tokens))


def _exiobase_download_log_candidates(data_dir: str | Path) -> list[Path]:
    """Candidate pymrio download-log paths, with download_log.json first."""
    root = Path(data_dir)
    candidates: list[Path] = [root / "download_log.json"]
    candidates.extend(sorted(root.glob("*download*log*.json")))
    candidates.extend(sorted(root.glob("*metadata*.json")))
    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _read_exiobase_download_log_doi(data_dir: str | Path) -> tuple[str | None, Path | None]:
    """Read the recorded EXIOBASE DOI/version from the existing download log."""
    for path in _exiobase_download_log_candidates(data_dir):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tokens = _collect_exiobase_doi_tokens(payload)
        if tokens:
            return tokens[0], path
    return None, None


def _write_exiobase_download_log_json(
    *,
    data_dir: str | Path,
    year: int,
    system: str,
    doi: str,
    url: str,
    zip_path: str | Path,
) -> None:
    """Ensure download_log.json carries the DOI used to create the cached zip."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "download_log.json"
    doi_clean = _normalize_exiobase_doi(doi)
    payload: dict[str, Any] = {}
    if log_path.exists():
        try:
            loaded = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload.update(loaded)
        except Exception:
            payload["previous_log_unparseable"] = True
    payload.update(
        {
            "name": payload.get("name", "EXIO3"),
            "system": str(system),
            "year": int(year),
            "version": doi_clean,
            "doi": doi_clean,
            "url": str(url),
            "source_zip": Path(zip_path).name,
        }
    )
    log_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_existing_exiobase_zip_doi(
    *,
    zip_path: str | Path,
    data_dir: str | Path,
    requested_doi: str,
) -> None:
    """Fail loudly if an existing EXIOBASE zip lacks matching DOI provenance."""
    requested_clean = _normalize_exiobase_doi(requested_doi)
    recorded_doi, log_path = _read_exiobase_download_log_doi(data_dir)
    if recorded_doi is None:
        checked = ", ".join(str(p) for p in _exiobase_download_log_candidates(data_dir))
        raise FileNotFoundError(
            "Existing EXIOBASE zip found, but no readable download_log.json DOI/version provenance was found. "
            f"Zip: {zip_path}. Requested DOI: {requested_clean}. Checked: {checked}. "
            "Use force_download=True / --exio-force-download to replace the cached zip."
        )
    recorded_clean = _normalize_exiobase_doi(recorded_doi)
    if recorded_clean != requested_clean:
        raise ValueError(
            "Existing EXIOBASE zip DOI/version mismatch. "
            f"Zip: {zip_path}. Requested DOI: {requested_clean}. Recorded DOI: {recorded_clean}. "
            f"Download log: {log_path}. Use force_download=True / --exio-force-download to replace the cached zip."
        )

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
        zip_path.unlink()

    if zip_path.exists() and not zip_contains_member(zip_path, "file_parameters.json"):
        ts = int(time.time())
        zip_path.rename(data_dir / f"IOT_{int(year)}_{system}.zip.corrupt.{ts}")

    if zip_path.exists():
        _validate_existing_exiobase_zip_doi(
            zip_path=zip_path,
            data_dir=data_dir,
            requested_doi=doi_clean,
        )
        return zip_path

    if not allow_download_missing:
        raise FileNotFoundError(f"EXIOBASE zip not found in local cache and downloads are disabled: {zip_path}")

    url = _resolve_exiobase3_url(int(year), system=system, doi=doi_clean)
    if verbose:
        print(f"EXIOBASE zip not found at {zip_path}; downloading {url} ...")

    downlog = MRIOMetaData._make_download_log(
        location=cast(Any, data_dir),
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
    _write_exiobase_download_log_json(
        data_dir=data_dir,
        year=int(year),
        system=str(system),
        doi=doi_clean,
        url=url,
        zip_path=zip_path,
    )
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
) -> pymrio.IOSystem:
    """
    Load MRIO data using native pymrio download and parsing capabilities.
    Returns the requested full pymrio IOSystem.

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

    Notes
    -----
    EXIOBASE full-IOSystem cache policy is zip-only:
    - keep the source zip locally for reproducibility and sparse/direct reads,
    - do not write/read fast pickle caches,
    - do not write/read large parsed text-cache directories,
    - use `load_mrio_iot_matrices()` for selective primitive Z/x/Y loading.
    """
    if dataset == "test":
        if verbose:
            print("Loading internal pymrio test dataset (no download)...")
        mrio = pymrio.load_test()

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

    elif dataset == "wiod16":
        root = Path(data_root) if data_root is not None else default_data_root()
        data_dir = root / "wiod16"
        if not allow_download_missing:
            cache_dir = data_dir / f"mrio_system_wiod16_{year}"
            zip_path = data_dir / "WIOTS_in_STATA.zip"
            if not cache_dir.exists() and not zip_path.exists():
                raise FileNotFoundError(f"WIOD16 cache missing for year={year} and downloads are disabled: {data_dir}")
        return load_wiod16(
            year=year,
            data_dir=data_dir,
            force_download=force_download,
            calc_all=calc_all,
            include_ghosh=include_ghosh,
            verbose=verbose,
        )

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
            mrio = cast(pymrio.IOSystem, pymrio.load(cache_path))
            if calc_all and getattr(mrio, "G", None) is None:
                if verbose:
                    print("Computing MRIO derived matrices (calc_all)...")
                mrio.calc_all(include_ghosh=include_ghosh)
            return mrio

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

    elif str(dataset).strip().lower() in {"exiobase", "exiobase3"}:
        raise RuntimeError(
            "Full EXIOBASE IOSystem loading via load_mrio(dataset='exiobase') is disabled. "
            "Use load_mrio_iot_matrices(..., matrices=(\"Z\", \"x\", \"Y\")) so only "
            "the DOI-checked source zip is cached and primitive source accounts are streamed directly."
        )
    else:
        raise ValueError(
            f"Unsupported MRIO dataset {dataset!r}; supported full-object datasets are "
            "'test', 'oecd', 'wiod', and 'wiod16'. EXIOBASE uses load_mrio_iot_matrices()."
        )

    # Ensure Ghosh is calculated
    if calc_all and getattr(mrio, "G", None) is None:
        if verbose:
            print("Computing MRIO derived matrices (calc_all)...")
        mrio.calc_all(include_ghosh=include_ghosh)

    return mrio

def _normalize_iot_matrix_requests(matrices: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(name).strip() for name in matrices)
    if not requested:
        raise ValueError("matrices must request at least one of 'Z', 'x', or 'Y'.")
    if len(set(requested)) != len(requested):
        raise ValueError("matrices must not contain duplicate requests.")
    invalid = [name for name in requested if name not in {"Z", "x", "Y"}]
    if invalid:
        raise ValueError(f"Unsupported IOT matrix request(s): {invalid!r}; allowed: 'Z', 'x', 'Y'.")
    return requested


def _validate_same_iot_basis(
    reference: pd.MultiIndex | None,
    candidate: pd.MultiIndex,
    *,
    source: str,
) -> pd.MultiIndex:
    """Validate one canonical primitive producer basis."""
    if not isinstance(candidate, pd.MultiIndex) or candidate.nlevels != 2:
        raise TypeError(f"{source} must expose a 2-level region-sector basis.")
    if list(candidate.names) != ["region", "sector"]:
        candidate = candidate.set_names(["region", "sector"])
    if not candidate.is_unique:
        raise ValueError(f"{source} basis contains duplicate region-sector rows.")
    if reference is not None and not candidate.equals(reference):
        raise ValueError(f"{source} basis does not exactly match the other requested IOT source accounts.")
    return candidate if reference is None else reference


def _validate_requested_iot_primitives(
    result: dict[str, csr_matrix | pd.Series | pd.DataFrame],
    basis: pd.MultiIndex,
    *,
    context: str,
) -> None:
    """Enforce one provider-independent primitive Z/x/Y result contract."""
    if "Z" in result:
        Z = result["Z"]
        if not isinstance(Z, csr_matrix):
            raise TypeError(f"{context}.Z must be scipy.sparse.csr_matrix.")
        if Z.shape != (len(basis), len(basis)):
            raise ValueError(f"{context}.Z shape does not match the canonical basis.")
        if Z.nnz and not np.isfinite(Z.data).all():
            raise ValueError(f"{context}.Z contains non-finite monetary values.")
    if "x" in result:
        x = result["x"]
        if not isinstance(x, pd.Series):
            raise TypeError(f"{context}.x must be a pandas Series.")
        _validated_nonnegative_output(
            x,
            expected_index=basis,
            context=f"{context}.x",
        )
    if "Y" in result:
        Y = result["Y"]
        if not isinstance(Y, pd.DataFrame):
            raise TypeError(f"{context}.Y must be a pandas DataFrame.")
        if not Y.index.equals(basis):
            raise ValueError(f"{context}.Y row index must exactly match the canonical basis.")
        if not Y.columns.is_unique:
            raise ValueError(f"{context}.Y final-demand columns must be unique.")
        try:
            values = Y.to_numpy(dtype="float64", copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}.Y contains non-numeric final-demand values.") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{context}.Y contains non-finite final-demand values.")


def load_mrio_iot_matrices(
    dataset: str,
    year: int,
    *,
    matrices: Sequence[str],
    force_download: bool = False,
    data_root: str | Path | None = None,
    wiod_wiot_file: str | Path | None = None,
    verbose: bool = False,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    doi: str = DEFAULT_EXIOBASE3_DOI,
    exio_system: str = "pxp",
) -> tuple[dict[str, csr_matrix | pd.Series | pd.DataFrame], pd.MultiIndex, dict[str, Any]]:
    """Load only requested primitive square-IOT source accounts ``Z``, ``x`` and/or ``Y``.

    The public loader never returns or derives ``A`` or summed ``y``. EXIOBASE reads
    only the requested source ZIP members. Other supported providers currently use
    their existing full-object parser internally and expose the same primitive result contract.
    """
    requested = _normalize_iot_matrix_requests(matrices)
    dataset_norm = str(dataset).strip().lower()
    result: dict[str, csr_matrix | pd.Series | pd.DataFrame] = {}
    basis: pd.MultiIndex | None = None
    metadata: dict[str, Any]

    if dataset_norm in {"exiobase", "exiobase3"}:
        system = str(exio_system).strip().lower()
        if system not in {"pxp", "ixi"}:
            raise ValueError(f"Unsupported exiobase system {exio_system!r}. Use 'pxp' or 'ixi'.")
        paths = resolve_exiobase_cache_paths(int(year), system=system, data_root=data_root)
        data_dir = paths["data_dir"]
        data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = ensure_exiobase_year_zip(
            year=int(year), data_dir=data_dir, system=system,
            force_download=force_download, allow_download_missing=allow_download_missing,
            resume_downloads=resume_downloads, show_download_progress=show_download_progress,
            download_chunk_mb=download_chunk_mb, download_timeout_seconds=download_timeout_seconds,
            download_retries=download_retries, doi=str(doi), verbose=verbose,
        )
        for name in requested:
            if name == "Z":
                Z, candidate = _read_exio_Z_sparse_from_zip(zip_path)
                basis = _validate_same_iot_basis(basis, candidate, source="Z.txt")
                result["Z"] = Z
            elif name == "x":
                x, candidate = _read_exio_x_from_zip(zip_path)
                basis = _validate_same_iot_basis(basis, candidate, source="x.txt")
                result["x"] = x
            else:
                Y, candidate = _read_exio_Y_matrix_direct_from_zip(zip_path)
                basis = _validate_same_iot_basis(basis, candidate, source="Y.txt")
                result["Y"] = Y
        assert basis is not None
        metadata = {
            "dataset": "exiobase",
            "year": int(year),
            "system": system,
            "source_zip": str(zip_path),
            "exio_doi": _normalize_exiobase_doi(doi),
            "requested_matrices": list(requested),
            "cache_kind": "exiobase_zip_stream_iot",
            "zip_only_cache": True,
            "source_zip_retained": True,
            "index_columns": ["region", "sector"],
        }
        _validate_requested_iot_primitives(result, basis, context="exiobase")
        return result, basis, metadata

    mrio = load_mrio(
        dataset_norm, int(year), force_download=force_download, data_root=data_root,
        wiod_wiot_file=wiod_wiot_file, calc_all=False, include_ghosh=False,
        verbose=verbose, allow_download_missing=allow_download_missing,
        resume_downloads=resume_downloads, show_download_progress=show_download_progress,
        download_chunk_mb=download_chunk_mb, download_timeout_seconds=download_timeout_seconds,
        download_retries=download_retries,
    )
    # Some pymrio providers (including load_test()) expose primitive Z/Y before x.
    # Calculate only the missing core system when x was explicitly requested; do not
    # calculate extensions and do not route EXIOBASE through this full-object adapter.
    if "x" in requested and getattr(mrio, "x", None) is None:
        try:
            mrio.calc_system(include_ghosh=False)
        except Exception as exc:
            raise RuntimeError(
                f"{dataset_norm} full-object adapter could not calculate requested output x."
            ) from exc
        if getattr(mrio, "x", None) is None:
            raise RuntimeError(
                f"{dataset_norm} full-object adapter still has no output x after calc_system()."
            )
    if "Z" in requested:
        Z_df = pd.DataFrame(mrio.Z)
        candidate = pd.MultiIndex.from_tuples(list(Z_df.index), names=["region", "sector"]) if not isinstance(Z_df.index, pd.MultiIndex) else Z_df.index.set_names(["region", "sector"])
        basis = _validate_same_iot_basis(basis, candidate, source=f"{dataset_norm}.Z")
        if not Z_df.columns.equals(Z_df.index):
            raise ValueError(f"{dataset_norm} Z must be square with identical row/column basis.")
        result["Z"] = csr_matrix(Z_df.to_numpy(dtype=float))
    if "x" in requested:
        x_obj = mrio.x
        if isinstance(x_obj, pd.DataFrame) and x_obj.shape[1] != 1:
            raise ValueError(f"{dataset_norm} x must have exactly one output column.")
        x_series = x_obj.iloc[:, 0] if isinstance(x_obj, pd.DataFrame) else pd.Series(x_obj)
        candidate = x_series.index if isinstance(x_series.index, pd.MultiIndex) else pd.MultiIndex.from_tuples(list(x_series.index), names=["region", "sector"])
        candidate = candidate.set_names(["region", "sector"])
        basis = _validate_same_iot_basis(basis, candidate, source=f"{dataset_norm}.x")
        x_numeric = cast(pd.Series, pd.to_numeric(x_series, errors="raise"))
        result["x"] = pd.Series(
            x_numeric.to_numpy(dtype="float64"), index=candidate, name="x"
        )
    if "Y" in requested:
        Y_df = pd.DataFrame(mrio.Y)
        candidate = Y_df.index if isinstance(Y_df.index, pd.MultiIndex) else pd.MultiIndex.from_tuples(list(Y_df.index), names=["region", "sector"])
        candidate = candidate.set_names(["region", "sector"])
        basis = _validate_same_iot_basis(basis, candidate, source=f"{dataset_norm}.Y")
        Y_df.index = candidate
        result["Y"] = Y_df.astype(float)
    if basis is None:
        raise RuntimeError("requested IOT matrices did not establish a basis")
    _validate_requested_iot_primitives(result, basis, context=dataset_norm)
    metadata = {
        "dataset": dataset_norm,
        "year": int(year),
        "requested_matrices": list(requested),
        "loader": "pymrio_full_object_adapter",
        "index_columns": ["region", "sector"],
    }
    return result, basis, metadata


# transition-source-api: begin
# Public, low-memory source-access helpers shared by downstream scientific repos.
# These functions deliberately contain no knowledge of a caller's repository layout.
def is_exiobase_dataset(dataset: str) -> bool:
    """Return whether ``dataset`` selects the canonical EXIOBASE source family."""

    return str(dataset).strip().lower() in {"exiobase", "exiobase3"}


def get_mrio_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve an explicit data root or MRIO's configured default data root."""

    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    return default_data_root().resolve()


def mrio_source_input_paths_for_year(
    *,
    dataset: str,
    year: int,
    data_root: str | Path | None = None,
    exio_system: str = "pxp",
) -> list[Path]:
    """Return source files that should invalidate a downstream scientific manifest.

    EXIOBASE has a stable retained source-ZIP boundary. Other providers currently
    expose no equivalent source-file contract through this helper and therefore
    return an empty list rather than guessing provider-specific cache paths.
    """

    if not is_exiobase_dataset(dataset):
        return []
    paths = resolve_exiobase_cache_paths(
        int(year),
        system=str(exio_system),
        data_root=get_mrio_data_root(data_root),
    )
    return [Path(paths["zip"])]


def ensure_exiobase_source_zip(
    *,
    year: int,
    data_root: str | Path | None = None,
    exio_system: str = "pxp",
    exio_doi: str | None = None,
    allow_download_missing: bool = False,
    force_download: bool = False,
    refresh_mismatched: bool = False,
    verbose: bool = False,
) -> Path:
    """Ensure one DOI-audited EXIOBASE source ZIP at the canonical MRIO data root.

    ``refresh_mismatched`` is an explicit destructive-recovery opt-in: a DOI/version
    mismatch is otherwise fail-closed. The retry delegates to the existing
    ``ensure_exiobase_year_zip`` force-download semantics.
    """

    data_root_resolved = get_mrio_data_root(data_root)
    data_dir = resolve_exiobase_year_dir(int(year), data_root=data_root_resolved)
    kwargs: dict[str, Any] = {
        "data_dir": data_dir,
        "system": str(exio_system),
        "force_download": bool(force_download),
        "allow_download_missing": bool(allow_download_missing or force_download),
        "verbose": bool(verbose),
    }
    if exio_doi is not None and str(exio_doi).strip():
        kwargs["doi"] = str(exio_doi).strip()
    try:
        return ensure_exiobase_year_zip(int(year), **kwargs)
    except ValueError as exc:
        if not refresh_mismatched or "DOI/version mismatch" not in str(exc):
            raise
        retry = dict(kwargs)
        retry["force_download"] = True
        retry["allow_download_missing"] = True
        if verbose:
            print(
                f"Refreshing EXIOBASE year {int(year)} after DOI/version mismatch: {exc}",
                flush=True,
            )
        return ensure_exiobase_year_zip(int(year), **retry)


def load_mrio_output_vector(
    dataset: str,
    year: int,
    *,
    data_root: str | Path | None = None,
    exio_system: str = "pxp",
    exio_doi: str | None = None,
    allow_download_missing_exiobase_source_zips: bool = False,
    exio_force_download: bool = False,
    verbose: bool = False,
) -> pd.Series:
    """Load only the canonical annual output vector ``x`` for one MRIO year."""

    kwargs: dict[str, Any] = {
        "matrices": ("x",),
        "data_root": get_mrio_data_root(data_root),
        "force_download": bool(exio_force_download),
        "verbose": bool(verbose),
        "allow_download_missing": bool(
            allow_download_missing_exiobase_source_zips or exio_force_download
        ),
    }
    if is_exiobase_dataset(dataset):
        kwargs["exio_system"] = str(exio_system)
        if exio_doi is not None and str(exio_doi).strip():
            kwargs["doi"] = str(exio_doi).strip()
    matrices, basis_index, _metadata = load_mrio_iot_matrices(
        str(dataset), int(year), **kwargs
    )
    output = matrices["x"]
    if not isinstance(output, pd.Series):
        raise TypeError("MRIO x-only loader must return a pandas Series.")
    if not output.index.equals(basis_index):
        raise ValueError("MRIO x-only loader returned an output index that differs from its basis.")
    return output.astype(float, copy=False)


def load_mrio_index(
    dataset: str,
    year: int,
    *,
    data_root: str | Path | None = None,
    exio_system: str = "pxp",
    exio_doi: str | None = None,
    allow_download_missing_exiobase_source_zips: bool = False,
    exio_force_download: bool = False,
    verbose: bool = False,
) -> pd.MultiIndex:
    """Load the canonical region-sector basis without transaction matrices."""

    output = load_mrio_output_vector(
        str(dataset),
        int(year),
        data_root=data_root,
        exio_system=str(exio_system),
        exio_doi=exio_doi,
        allow_download_missing_exiobase_source_zips=bool(
            allow_download_missing_exiobase_source_zips
        ),
        exio_force_download=bool(exio_force_download),
        verbose=bool(verbose),
    )
    if not isinstance(output.index, pd.MultiIndex) or output.index.nlevels != 2:
        raise TypeError("MRIO output vector index must be a 2-level region-sector MultiIndex.")
    return output.index.set_names(["region", "sector"])
# transition-source-api: end


def load_exiobase3_iot_matrices_in_price_basis(
    year: int,
    *,
    matrices: Sequence[str],
    price_base_year: int | None = None,
    price_factor_years: Sequence[int] | None = None,
    force_download: bool = False,
    data_root: str | Path | None = None,
    verbose: bool = False,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    doi: str | None = None,
    exio_system: str = "pxp",
    price_factor_storage_folder: str | Path | None = None,
    price_factor_force_download: bool = False,
    price_factor_timeout_s: int = 120,
) -> tuple[dict[str, csr_matrix | pd.Series | pd.DataFrame], pd.MultiIndex, dict[str, Any]]:
    """Load selective EXIOBASE3 primitives in current prices or one common-EUR basis.

    ``price_base_year=None`` is a selective pass-through to the canonical raw
    ``load_mrio_iot_matrices`` EXIOBASE source-account boundary. Supplying a base
    year builds the audited 49-region country-GDP/common-EUR factors and applies
    the existing row transformation only to the requested primitive ``Z``/``x``/``Y``
    accounts. For multi-year work, ``price_factor_years`` declares the common
    factor-panel domain that every year state must reuse; the base year is included
    automatically. Derived ``A``/``y`` and physical extensions remain outside this API.
    """

    requested = _normalize_iot_matrix_requests(matrices)
    doi_value = (
        DEFAULT_EXIOBASE3_DOI
        if doi is None or not str(doi).strip()
        else str(doi).strip()
    )
    factor_years: tuple[int, ...] | None = None
    if price_base_year is None:
        if price_factor_years is not None:
            raise ValueError("price_factor_years requires price_base_year")
    else:
        factor_years = (
            (int(year),)
            if price_factor_years is None
            else tuple(sorted({int(item) for item in price_factor_years}))
        )
        if not factor_years:
            raise ValueError("price_factor_years must contain at least one year when provided")
        if int(year) not in factor_years:
            raise ValueError(
                f"loaded EXIOBASE year {int(year)} is outside declared price_factor_years {factor_years!r}"
            )

    loaded, basis, metadata = load_mrio_iot_matrices(
        "exiobase",
        int(year),
        matrices=requested,
        force_download=bool(force_download),
        data_root=data_root,
        verbose=bool(verbose),
        allow_download_missing=bool(allow_download_missing),
        resume_downloads=bool(resume_downloads),
        show_download_progress=bool(show_download_progress),
        download_chunk_mb=int(download_chunk_mb),
        download_timeout_seconds=download_timeout_seconds,
        download_retries=int(download_retries),
        doi=doi_value,
        exio_system=str(exio_system),
    )
    valuation_metadata: dict[str, Any] = dict(metadata or {})
    if price_base_year is None:
        valuation_metadata.update(
            {"valuation_basis": "current_price", "price_base_year": None}
        )
        return loaded, basis, valuation_metadata

    from ecolecon_mrio.exiobase_price_factors import build_exiobase3_eur_price_factors
    from ecolecon_mrio.price_adjustment import adjust_mrio_iot_prices, expand_region_price_factors

    assert factor_years is not None
    factor_table, factor_metadata = build_exiobase3_eur_price_factors(
        years=factor_years,
        base_year=int(price_base_year),
        storage_folder=price_factor_storage_folder,
        force_download=bool(price_factor_force_download),
        timeout_s=int(price_factor_timeout_s),
    )
    expected_panel_years = tuple(sorted(set(factor_years) | {int(price_base_year)}))
    if "year" not in factor_table.columns:
        raise ValueError("EXIOBASE price-factor table is missing required year column")
    table_panel_years = tuple(sorted({int(item) for item in factor_table["year"].tolist()}))
    if table_panel_years != expected_panel_years:
        raise ValueError(
            "EXIOBASE price-factor table year domain differs from the declared factor panel: "
            f"expected={expected_panel_years!r}, observed={table_panel_years!r}"
        )
    raw_returned_years = factor_metadata.get("returned_years")
    if not isinstance(raw_returned_years, (list, tuple)):
        raise ValueError("EXIOBASE price-factor metadata must record returned_years")
    metadata_panel_years = tuple(int(item) for item in raw_returned_years)
    if metadata_panel_years != expected_panel_years:
        raise ValueError(
            "EXIOBASE price-factor metadata year domain differs from the declared factor panel: "
            f"expected={expected_panel_years!r}, observed={metadata_panel_years!r}"
        )
    raw_factor_sha = factor_metadata.get("factor_table_sha256")
    factor_sha = str(raw_factor_sha).strip().lower() if isinstance(raw_factor_sha, str) else ""
    if len(factor_sha) != 64 or any(char not in "0123456789abcdef" for char in factor_sha):
        raise ValueError("EXIOBASE price-factor metadata must contain a 64-hex factor_table_sha256")

    selected = factor_table.loc[
        factor_table["year"].astype(int).eq(int(year)),
        ["exio_region", "price_adjustment_factor"],
    ].copy()
    if selected.empty or bool(selected["exio_region"].duplicated().any()):
        raise ValueError(
            f"EXIOBASE price-factor table must contain one unique region factor for year {year}"
        )
    region_factors = selected.set_index("exio_region")["price_adjustment_factor"].astype(float)
    basis_regions = {str(region) for region in basis.get_level_values("region")}
    factor_regions = {str(region) for region in region_factors.index}
    if factor_regions != basis_regions:
        raise ValueError(
            "EXIOBASE price-factor regions must exactly match the loaded MRIO region basis; "
            f"missing={sorted(basis_regions - factor_regions)}, "
            f"extra={sorted(factor_regions - basis_regions)}"
        )
    row_factors = expand_region_price_factors(basis, region_factors)
    adjusted = cast(
        dict[str, csr_matrix | pd.Series | pd.DataFrame],
        adjust_mrio_iot_prices(
            loaded,
            basis_index=basis,
            row_factors=row_factors,
        ),
    )
    _validate_requested_iot_primitives(
        adjusted,
        basis,
        context="exiobase_common_eur",
    )
    valuation_metadata.update(
        {
            "valuation_basis": "country_gdp_deflator_common_eur",
            "price_base_year": int(price_base_year),
            "price_factor_panel_years": list(expected_panel_years),
            "price_factor_metadata": factor_metadata,
            "price_factor_table_sha256": factor_sha,
        }
    )
    return adjusted, basis, valuation_metadata
