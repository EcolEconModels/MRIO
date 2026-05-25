from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ecolecon_mrio.data import (
    load_mrio_matrices,
    resolve_exiobase_cache_paths,
    resolve_exiobase_matrix_cache_paths,
)


def _fmt_mb(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{num_bytes / (1024.0 * 1024.0):.1f} MB"


def _dir_size(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += int(item.stat().st_size)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Report EXIOBASE matrix cache sizes and load time.")
    parser.add_argument("year", type=int, help="EXIOBASE year")
    parser.add_argument("--system", default="pxp", choices=["pxp", "ixi"], help="EXIOBASE system")
    parser.add_argument("--data-root", default=None, help="Cache root directory")
    parser.add_argument("--write-cache", action="store_true", help="Create/update the compact Z/x/y matrix cache if missing")
    parser.add_argument("--ignore-existing-cache", action="store_true", help="Parse from source zip instead of reading an existing matrix cache")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser() if args.data_root is not None else None
    full_paths = resolve_exiobase_cache_paths(args.year, system=args.system, data_root=data_root)
    matrix_paths = resolve_exiobase_matrix_cache_paths(args.year, system=args.system, data_root=data_root)

    source_zip_size = _dir_size(full_paths["zip"])
    fast_pickle_size = _dir_size(full_paths["fast"])
    matrix_cache_size = _dir_size(matrix_paths["matrix_cache_dir"])
    z_size = _dir_size(matrix_paths["Z"])

    start = time.perf_counter()
    A, x, y, basis_index, metadata = load_mrio_matrices(
        args.year,
        data_root=data_root,
        exio_system=args.system,
        allow_download_missing=False,
        cache_matrices=bool(args.write_cache),
        read_existing_cache=not bool(args.ignore_existing_cache),
        verbose=False,
    )
    load_seconds = time.perf_counter() - start

    print(f"source zip: {_fmt_mb(source_zip_size)}")
    print(f"full fast pickle: {_fmt_mb(fast_pickle_size)}")
    print(f"matrix cache total: {_fmt_mb(matrix_cache_size)}")
    print(f"Z.npz: {_fmt_mb(z_size)}")
    print(f"A shape: {A.shape[0]} x {A.shape[1]}")
    print(f"A nnz: {int(A.nnz)}")
    print(f"A density: {float(metadata.get('density', 0.0)):.6g}")
    print(f"matrix cache load time: {load_seconds:.3f}s")
    print(f"index size: {len(basis_index)}")
    print(f"x size: {len(x)}")
    print(f"y size: {len(y)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
