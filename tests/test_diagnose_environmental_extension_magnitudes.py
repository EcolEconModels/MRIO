from __future__ import annotations

# environmental-magnitude-diagnostic-v1

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_environmental_extension_magnitudes.py"
spec = importlib.util.spec_from_file_location("diagnose_environmental_extension_magnitudes", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_import_collapse_exposes_signed_net_cancellation() -> None:
    index = pd.Index(["O1", "O2"], dtype="object")
    mapped_f = pd.DataFrame({"A": [100.0, 0.0]}, index=index)
    mapped_x = pd.DataFrame({"A": [100.0, 100.0]}, index=index)
    mapped_u = pd.DataFrame({"A": [100.0, -100.0 + 1e-14]}, index=index)
    summary, coefficients, contributions = module._collapse_import_origin_magnitudes(
        mapped_f=mapped_f,
        mapped_x=mapped_x,
        mapped_u=mapped_u,
    )
    row = summary.set_index("sector").loc["A"]
    assert float(row["gross_abs_import_use"]) == pytest.approx(200.0)
    assert float(row["cancellation_ratio"]) < 1e-12
    assert abs(float(row["donor_import_intensity"])) > 1e12
    assert float(coefficients.loc["O1", "A"]) == pytest.approx(1.0)
    assert float(coefficients.loc["O2", "A"]) == pytest.approx(0.0)
    assert float(contributions.loc["O1", "A"]) == pytest.approx(100.0)


def test_import_collapse_normal_case_is_origin_weighted_ratio() -> None:
    index = pd.Index(["O1", "O2"], dtype="object")
    mapped_f = pd.DataFrame({"A": [10.0, 20.0]}, index=index)
    mapped_x = pd.DataFrame({"A": [100.0, 100.0]}, index=index)
    mapped_u = pd.DataFrame({"A": [50.0, 50.0]}, index=index)
    summary, _coefficients, _contributions = module._collapse_import_origin_magnitudes(
        mapped_f=mapped_f,
        mapped_x=mapped_x,
        mapped_u=mapped_u,
    )
    row = summary.set_index("sector").loc["A"]
    assert float(row["origin_weighted_numerator"]) == pytest.approx(15.0)
    assert float(row["net_import_use"]) == pytest.approx(100.0)
    assert float(row["cancellation_ratio"]) == pytest.approx(1.0)
    assert float(row["donor_import_intensity"]) == pytest.approx(0.15)


def test_import_collapse_keeps_exact_zero_net_intensity_unavailable() -> None:
    index = pd.Index(["O1", "O2"], dtype="object")
    mapped_f = pd.DataFrame({"A": [10.0, 20.0]}, index=index)
    mapped_x = pd.DataFrame({"A": [100.0, 100.0]}, index=index)
    mapped_u = pd.DataFrame({"A": [1.0, -1.0]}, index=index)
    summary, _coefficients, _contributions = module._collapse_import_origin_magnitudes(
        mapped_f=mapped_f,
        mapped_x=mapped_x,
        mapped_u=mapped_u,
    )
    row = summary.set_index("sector").loc["A"]
    assert float(row["net_import_use"]) == 0.0
    assert float(row["gross_abs_import_use"]) == pytest.approx(2.0)
    assert float(row["cancellation_ratio"]) == 0.0
    assert pd.isna(row["donor_import_intensity"])


def test_import_collapse_marks_active_nonpositive_output_origin_invalid() -> None:
    index = pd.Index(["O1", "O2"], dtype="object")
    mapped_f = pd.DataFrame({"A": [10.0, 20.0]}, index=index)
    mapped_x = pd.DataFrame({"A": [100.0, 0.0]}, index=index)
    mapped_u = pd.DataFrame({"A": [1.0, 2.0]}, index=index)
    summary, _coefficients, _contributions = module._collapse_import_origin_magnitudes(
        mapped_f=mapped_f,
        mapped_x=mapped_x,
        mapped_u=mapped_u,
    )
    row = summary.set_index("sector").loc["A"]
    assert int(row["n_invalid_active_origins"]) == 1
    assert pd.isna(row["origin_weighted_numerator"])
    assert pd.isna(row["donor_import_intensity"])


def test_reader_comparison_keeps_label_shape_distinct_from_numeric_equality() -> None:
    basis = pd.MultiIndex.from_tuples(
        [("AT", "A"), ("AT", "B")], names=["region", "sector"]
    )
    production = pd.Series([1.0, 2.0], index=basis, dtype="float64")
    robust = pd.DataFrame(
        [[1.0, 2.0]], index=pd.Index(["CO2 | combustion"]), columns=basis
    )
    result = module._reader_comparison(
        extension="air",
        stressor="CO2",
        unit="kg",
        production_values=production,
        robust_frame=robust,
        robust_error="",
    )
    assert result["match_state"] == "PREFIX_UNIQUE"
    assert result["numeric_equal"] is True
    assert result["label_exact"] is False
    assert float(result["max_abs_diff"]) == 0.0


def test_flags_report_cancellation_without_changing_any_validity_state() -> None:
    row = pd.Series(
        {
            "donor_import_intensity": 1e16,
            "net_import_use": 1e-12,
            "gross_abs_import_use": 100.0,
            "cancellation_ratio": 1e-14,
            "n_invalid_active_origins": 0,
        }
    )
    flags = module._flags(
        row,
        intensity_threshold=1e9,
        cancellation_threshold=1e-10,
        import_row=True,
    )
    assert flags == "EXTREME_ABS_INTENSITY;SIGNED_NET_CANCELLATION"
    args = module.build_parser().parse_args(
        ["--scope-code", "EU27_2020", "--split-root", "split", "--outdir", "audit"]
    )
    assert args.intensity_threshold == pytest.approx(1e9)
    assert args.cancellation_ratio_threshold == pytest.approx(1e-10)


def test_flags_report_positive_target_without_donor_import_use() -> None:
    row = pd.Series(
        {
            "donor_import_intensity": float("nan"),
            "target_monetary_output": 1.06,
            "net_import_use": 0.0,
            "gross_abs_import_use": 0.0,
            "cancellation_ratio": float("nan"),
            "n_invalid_active_origins": 0,
        }
    )
    flags = module._flags(
        row,
        intensity_threshold=1e9,
        cancellation_threshold=1e-10,
        import_row=True,
    )
    assert flags == "TARGET_POSITIVE_DONOR_IMPORT_USE_ZERO"


def test_total_support_coverage_retains_known_value_under_signed_import_use() -> None:
    keys = {
        "sector": ["CPA_E36", "p16"],
        "extension": ["air", "air"],
        "stressor": ["CO2", "CO2"],
        "unit": ["kg", "kg"],
    }
    dom = pd.DataFrame(
        {
            **keys,
            "observed_value": [100.0, 50.0],
            "observed_monetary_output": [1138.22, 863.4264476632452],
            "observed_intensity": [100.0 / 1138.22, 50.0 / 863.4264476632452],
        }
    )
    imp = pd.DataFrame(
        {
            **keys,
            "observed_value": [float("nan"), -20.0],
            "observed_monetary_output": [1.06, -813.7604651384088],
            "observed_intensity": [float("nan"), 20.0 / 813.7604651384088],
        }
    )
    out = module._total_support_coverage(dom, imp).set_index("sector")

    assert float(out.at["CPA_E36", "supported_value"]) == pytest.approx(100.0)
    assert float(out.at["CPA_E36", "coverage_fraction"]) == pytest.approx(
        1138.22 / (1138.22 + 1.06)
    )
    assert float(out.at["CPA_E36", "unsupported_monetary_mass"]) == pytest.approx(1.06)

    assert float(out.at["p16", "target_total_use"]) == pytest.approx(
        863.4264476632452 - 813.7604651384088
    )
    assert float(out.at["p16", "supported_value"]) == pytest.approx(30.0)
    assert float(out.at["p16", "coverage_fraction"]) == pytest.approx(1.0)
