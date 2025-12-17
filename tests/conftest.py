from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src = project_root / "src"
    if src.exists():
        sys.path.insert(0, str(src))
