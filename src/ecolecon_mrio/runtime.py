"""Runtime helpers for resource-bounded MRIO computations.

The functions in this module are operational rather than economic.  They exist so
MRIO-owned command-line tools can enforce the same low-memory discipline used by
analysis repos without importing transition-specific scripts.  Library functions
should not silently impose a process memory cap; scripts call these helpers before
large EXIOBASE zip streams or sparse Leontief factorizations begin.
"""

from __future__ import annotations

from datetime import datetime
import os
import sys
from typing import Final


_THREAD_ENV_VARS: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def tick(message: str) -> None:
    """Print a flushed timestamped progress message for long MRIO steps.

    Sparse EXIOBASE parsing and SuperLU factorization can run for minutes on a
    laptop.  A short flushed milestone tells the user which scientific or
    resource-sensitive phase is active without creating persistent log files.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def enforce_memory_limit_gb(max_gb: float = 4.0) -> None:
    """Apply a best-effort virtual address-space cap to the current process.

    The cap is a guard against accidental dense MRIO materialization.  It is
    deliberately best-effort because ``resource.RLIMIT_AS`` is POSIX-specific; on
    unsupported platforms the function prints a diagnostic and continues.
    """

    if float(max_gb) <= 0.0:
        raise ValueError(f"Memory limit must be positive, got {max_gb!r} GB")
    try:
        import resource  # POSIX-only; unavailable on some platforms.
    except ImportError:
        tick(f"memory limit not enforced: resource module unavailable; requested {float(max_gb):.2f} GB")
        return

    requested_bytes = int(float(max_gb) * 1024**3)
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = requested_bytes if hard_limit < 0 else min(requested_bytes, hard_limit)
    if soft_limit > 0:
        new_soft = min(new_soft, soft_limit)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard_limit))
    except (OSError, ValueError) as exc:
        tick(f"memory limit not enforced: {exc}; requested {float(max_gb):.2f} GB")
        return
    tick(f"memory limit active: soft address-space cap {new_soft / 1024**3:.2f} GB")


def memory_snapshot_text() -> str:
    """Return a compact best-effort memory snapshot for diagnostics.

    The value is intended for error messages around sparse factorization.  It is
    not a scientific output and may be unavailable on some platforms.
    """

    try:
        import resource
    except ImportError:
        return f"rss unavailable on {sys.platform}"
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB; macOS reports bytes.  The repos are mostly
    # run on Linux/WSL2, but the branch keeps the diagnostic portable enough for
    # non-Linux development machines.
    raw = float(usage.ru_maxrss)
    rss_mb = raw / 1024.0 if sys.platform != "darwin" else raw / (1024.0**2)
    return f"peak_rss={rss_mb:.1f} MB"


def configure_low_memory_runtime(max_gb: float = 4.0) -> None:
    """Set MRIO low-memory defaults before expensive sparse work.

    Thread-count environment variables are best set before NumPy/SciPy import.
    Setting defaults here still documents the runtime contract for MRIO scripts
    and prevents child processes from oversubscribing BLAS threads.
    """

    for name in _THREAD_ENV_VARS:
        os.environ.setdefault(name, "1")
    tick("low-memory runtime defaults: BLAS/OpenMP thread env vars default to 1 if unset")
    enforce_memory_limit_gb(max_gb=float(max_gb))
