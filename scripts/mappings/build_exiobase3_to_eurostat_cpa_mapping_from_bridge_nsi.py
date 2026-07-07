#!/usr/bin/env python3
"""Deprecated compatibility wrapper for the official-concordance EXIOBASE3->CPA generator."""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

_CANONICAL_SCRIPT = Path(__file__).with_name("build_exiobase3_to_eurostat_cpa_mapping_from_official_concordances.py")
_CANONICAL_MODULE_NAME = "ecolecon_mrio_exio3_to_cpa_official_concordances_cli"

_spec = importlib.util.spec_from_file_location(_CANONICAL_MODULE_NAME, _CANONICAL_SCRIPT)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load canonical official-concordance generator: {_CANONICAL_SCRIPT}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_CANONICAL_MODULE_NAME] = _module
_spec.loader.exec_module(_module)

if __name__ == "__main__":
    warnings.warn(
        "scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_bridge_nsi.py is deprecated; "
        "use scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_official_concordances.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise SystemExit(_module.main())

globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})
