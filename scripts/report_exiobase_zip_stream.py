from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ecolecon_mrio.data import load_mrio_matrices, resolve_exiobase_cache_paths


def _fmt_mb(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{num_bytes / (1024.0 * 1024.0):.1f} MB"


def _size(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    return int(path.stat().st_size)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report EXIOBASE zip-stream sparse load characteristics.")
    parser.add_argument("year", type=int, help="EXIOBASE year")
    parser.add_argument("--system", default="pxp", choices=["pxp", "ixi"], help="EXIOBASE system")
    parser.add_argument("--data-root", default=None, help="EXIOBASE cache root")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser() if args.data_root is not None else None
    paths = resolve_exiobase_cache_paths(args.year, system=args.system, data_root=data_root)
    zip_path = paths["zip"]

    start = time.perf_counter()
    A, x, y, basis_index, metadata = load_mrio_matrices(
        args.year,
        data_root=data_root,
        exio_system=args.system,
        allow_download_missing=False,
        cache_matrices=False,
        read_existing_cache=False,
        verbose=False,
    )
    load_seconds = time.perf_counter() - start

    print(f"source zip: {_fmt_mb(_size(zip_path))} ({zip_path})")
    print("derived matrix cache: disabled")
    print("full pymrio/fast pickle cache: disabled")
    print(f"A source: {metadata.get('matrix_source', 'unknown')}")
    print(f"A shape: {A.shape[0]} x {A.shape[1]}")
    print(f"A nnz: {int(A.nnz)}")
    print(f"A density: {float(metadata.get('density', 0.0)):.6g}")
    print(f"zip-stream load time: {load_seconds:.3f}s")
    print(f"index size: {len(basis_index)}")
    print(f"x size: {len(x)}")
    print(f"y size: {len(y)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
