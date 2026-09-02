from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ecolecon_mrio import exiobase3
from ecolecon_mrio.region_aggregation import RegionScope, load_mrio_iot_scope_slice
from ecolecon_mrio.region_aggregation import RegionScope, load_mrio_iot_scope_slice
from ecolecon_mrio.region_aggregation import RegionScope, load_mrio_iot_scope_slice
from ecolecon_mrio.region_aggregation import RegionScope, load_mrio_iot_scope_slice
from ecolecon_mrio.region_aggregation import RegionScope, load_mrio_iot_scope_slice
from ecolecon_mrio.region_aggregation import RegionScope, load_mrio_iot_scope_slice
from ecolecon_mrio.data import DEFAULT_EXIOBASE3_DOI


def _write_tiny_exio_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("file_parameters.json", "{}")
        zf.writestr(
            "x.txt",
            "region\tsector\tindout\n"
            "AT\tA\t100\n"
            "AT\tB\t200\n"
            "DE\tA\t300\n"
            "DE\tB\t400\n",
        )
        zf.writestr(
            "Z.txt",
            "\t\tAT\tAT\tDE\tDE\n"
            "\t\tA\tB\tA\tB\n"
            "\t\tunit\tunit\tunit\tunit\n"
            "AT\tA\t1\t2\t3\t4\n"
            "AT\tB\t5\t6\t7\t8\n"
            "DE\tA\t9\t10\t11\t12\n"
            "DE\tB\t13\t14\t15\t16\n",
        )
        zf.writestr(
            "Y.txt",
            "\t\tAT\tAT\tDE\n"
            "\t\tHFCE\tGFCF\tHFCE\n"
            "\t\tunit\tunit\tunit\n"
            "AT\tA\t100\t1\t9\n"
            "AT\tB\t200\t2\t8\n"
            "DE\tA\t300\t3\t7\n"
            "DE\tB\t400\t4\t6\n",
        )


def _prepare_cache_root(tmp_path: Path) -> Path:
    cache_root = tmp_path
    data_dir = cache_root / "exiobase3_2020"
    _write_tiny_exio_zip(data_dir / "IOT_2020_pxp.zip")
    (data_dir / "download_log.json").write_text(
        json.dumps({"version": DEFAULT_EXIOBASE3_DOI}) + "\n",
        encoding="utf-8",
    )
    return cache_root


@pytest.fixture(autouse=True)
def _identity_sector_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exiobase3, "_exio_label_to_code_lookup", lambda system: {"A": "CPA_A", "B": "CPA_B"})


def test_region_slice_streams_domestic_block_without_cache_or_full_mrio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = _prepare_cache_root(tmp_path)

    out = load_mrio_iot_scope_slice(
        "exiobase",
        2020,
        scope=RegionScope(code="AT", members=("AT",)),
        exio_system="pxp",
        flow="internal",
        data_root=cache_root,
        doi=DEFAULT_EXIOBASE3_DOI,
    )

    assert list(out.Z.index) == ["CPA_A", "CPA_B"]
    pd.testing.assert_frame_equal(
        out.Z,
        pd.DataFrame(
            [[1.0, 2.0], [5.0, 6.0]],
            index=pd.Index(["CPA_A", "CPA_B"], name="sector"),
            columns=pd.Index(["CPA_A", "CPA_B"]),
        ),
    )
    pd.testing.assert_frame_equal(
        out.Y,
        pd.DataFrame(
            [[100.0, 1.0], [200.0, 2.0]],
            index=pd.Index(["CPA_A", "CPA_B"], name="sector"),
            columns=pd.Index(["HFCE", "GFCF"]),
        ),
    )
    pd.testing.assert_series_equal(
        out.x_scope,
        pd.Series([100.0, 200.0], index=pd.Index(["CPA_A", "CPA_B"], name="sector"), name="x_scope"),
    )
    assert not (cache_root / "exiobase3_2020" / "slices").exists()


def test_region_slice_streams_import_and_total_flows(tmp_path: Path) -> None:
    cache_root = _prepare_cache_root(tmp_path)

    imports = load_mrio_iot_scope_slice(
        "exiobase",
        2020,
        scope=RegionScope(code="AT", members=("AT",)),
        exio_system="pxp",
        flow="external_imports",
        data_root=cache_root,
        doi=DEFAULT_EXIOBASE3_DOI,
    )
    total = load_mrio_iot_scope_slice(
        "exiobase",
        2020,
        scope=RegionScope(code="AT", members=("AT",)),
        exio_system="pxp",
        flow="total",
        data_root=cache_root,
        doi=DEFAULT_EXIOBASE3_DOI,
    )

    assert imports.Z.loc["CPA_A", "CPA_A"] == 9.0
    assert imports.Z.loc["CPA_B", "CPA_B"] == 14.0
    assert imports.Y.loc["CPA_A", "HFCE"] == 300.0
    assert total.Z.loc["CPA_A", "CPA_A"] == 10.0
    assert total.Z.loc["CPA_B", "CPA_B"] == 20.0
    assert total.Y.loc["CPA_A", "GFCF"] == 4.0


def _write_region_integrity_zip(
    path: Path, *, x_b: str = "200", y_b_row: str = "AT\tB\t20\n"
) -> None:
    """Write a two-sector domestic slice for monetary source-integrity tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("x.txt", f"region\tsector\tindout\nAT\tA\t100\nAT\tB\t{x_b}\n")
        zf.writestr(
            "Z.txt",
            "\t\tAT\tAT\n\t\tA\tB\n\t\tunit\tunit\n"
            "AT\tA\t1\t2\nAT\tB\t3\t4\n",
        )
        zf.writestr(
            "Y.txt",
            "\t\tAT\n\t\tHFCE\n\t\tunit\n"
            "AT\tA\t10\n"
            + y_b_row,
        )


def test_region_slice_rejects_blank_output_value(tmp_path: Path) -> None:
    """The fuels region loader must not turn a blank producer output into zero."""
    source = tmp_path / "blank_x.zip"
    _write_region_integrity_zip(source, x_b="")
    with pytest.raises(ValueError, match="x.txt contains blank"):
        exiobase3._load_exio_scope_slice_direct_from_zip(
            year=2020, system="pxp", scope=RegionScope(code="AT", members=("AT",)),
            source_zip=source, flow="internal"
        )


def test_region_slice_rejects_missing_Y_sector_row(tmp_path: Path) -> None:
    """Z/Y source row bases must match instead of filling a missing Y row with zero."""
    source = tmp_path / "missing_y_row.zip"
    _write_region_integrity_zip(source, y_b_row="")
    with pytest.raises(ValueError, match="Y.txt row basis does not match Z.txt"):
        exiobase3._load_exio_scope_slice_direct_from_zip(
            year=2020, system="pxp", scope=RegionScope(code="AT", members=("AT",)),
            source_zip=source, flow="internal"
        )


def test_region_slice_rejects_nonfinite_selected_Y_value(tmp_path: Path) -> None:
    """A non-finite destination-use value cannot enter fuels donor profiles."""
    source = tmp_path / "nonfinite_y.zip"
    _write_region_integrity_zip(source, y_b_row="AT\tB\tnan\n")
    with pytest.raises(ValueError, match="Y.txt contains non-finite monetary values"):
        exiobase3._load_exio_scope_slice_direct_from_zip(
            year=2020, system="pxp", scope=RegionScope(code="AT", members=("AT",)),
            source_zip=source, flow="internal"
        )


# Region-scope API regressions: geography aggregation is independent of sector splitting.
def _write_three_region_exio_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    regions = ("AT", "DE", "FR")
    sectors = ("A", "B")
    z_columns = [(region, sector) for region in regions for sector in sectors]
    y_columns = [(region, category) for region in regions for category in ("HFCE", "GFCF")]

    z_lines = [
        "\t\t" + "\t".join(region for region, _ in z_columns),
        "\t\t" + "\t".join(sector for _, sector in z_columns),
        "\t\t" + "\t".join("unit" for _ in z_columns),
    ]
    y_lines = [
        "\t\t" + "\t".join(region for region, _ in y_columns),
        "\t\t" + "\t".join(category for _, category in y_columns),
        "\t\t" + "\t".join("unit" for _ in y_columns),
    ]
    x_lines = ["region\tsector\tindout"]
    for origin_idx, region in enumerate(regions):
        for sector_idx, sector in enumerate(sectors):
            z_values = [
                float(100 * origin_idx + 10 * sector_idx + column_idx + 1)
                for column_idx in range(len(z_columns))
            ]
            y_values = [
                float(1000 * origin_idx + 100 * sector_idx + column_idx + 1)
                for column_idx in range(len(y_columns))
            ]
            z_lines.append(
                f"{region}\t{sector}\t" + "\t".join(f"{value:g}" for value in z_values)
            )
            y_lines.append(
                f"{region}\t{sector}\t" + "\t".join(f"{value:g}" for value in y_values)
            )
            output = 100.0 + 200.0 * origin_idx + 100.0 * sector_idx
            x_lines.append(f"{region}\t{sector}\t{output:g}")

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("file_parameters.json", "{}")
        zf.writestr("x.txt", "\n".join(x_lines) + "\n")
        zf.writestr("Z.txt", "\n".join(z_lines) + "\n")
        zf.writestr("Y.txt", "\n".join(y_lines) + "\n")


def _prepare_scope_cache_root(tmp_path: Path) -> Path:
    data_dir = tmp_path / "exiobase3_2020"
    _write_three_region_exio_zip(data_dir / "IOT_2020_pxp.zip")
    (data_dir / "download_log.json").write_text(
        json.dumps({"version": DEFAULT_EXIOBASE3_DOI}) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_compile_region_aggregation_builds_disjoint_scope_and_complement() -> None:
    """Protect one-to-one geography membership and source-derived RoW construction."""
    from ecolecon_mrio.region_aggregation import compile_region_aggregation

    mapping = pd.DataFrame(
        [
            {
                "scope_code": "AT_DE",
                "scope_label": "Austria and Germany",
                "member_region": "DE",
                "member_order": 2,
                "enabled": True,
                "notes": "",
            },
            {
                "scope_code": "AT_DE",
                "scope_label": "Austria and Germany",
                "member_region": "AT",
                "member_order": 1,
                "enabled": True,
                "notes": "",
            },
        ]
    )
    plan = compile_region_aggregation(
        source_regions=("AT", "DE", "FR"),
        mapping=mapping,
        scopes=("AT_DE",),
        include_complement=True,
        complement_code="RoW",
    )
    assert [scope.code for scope in plan.scopes] == ["AT_DE", "RoW"]
    assert plan.scopes[0].members == ("AT", "DE")
    assert plan.scopes[1].members == ("FR",)
    assert plan.source_to_scope.to_dict() == {"AT": "AT_DE", "DE": "AT_DE", "FR": "RoW"}


def test_compile_region_aggregation_rejects_overlap_and_unknown_members() -> None:
    """Reject geographic disaggregation and source-region codes absent from the donor."""
    from ecolecon_mrio.region_aggregation import compile_region_aggregation

    overlapping = pd.DataFrame(
        [
            ["S1", "one", "AT", 1, True, ""],
            ["S2", "two", "AT", 1, True, ""],
        ],
        columns=["scope_code", "scope_label", "member_region", "member_order", "enabled", "notes"],  # pyright: ignore[reportArgumentType]
    )
    with pytest.raises(ValueError, match="at most one selected scope"):
        compile_region_aggregation(source_regions=("AT", "DE"), mapping=overlapping)

    unknown = overlapping.iloc[[0]].copy()
    unknown.loc[:, "member_region"] = "ZZ"
    with pytest.raises(ValueError, match="absent from the selected source release"):
        compile_region_aggregation(source_regions=("AT", "DE"), mapping=unknown)


def test_checked_in_eu27_scope_mapping_has_expected_membership() -> None:
    """Lock the checked EU27_2020 helper to its 27 canonical EXIOBASE members."""
    from ecolecon_mrio.region_aggregation import compile_region_aggregation

    eu27 = (
        "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR",
        "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO",
        "SE", "SI", "SK",
    )
    plan = compile_region_aggregation(
        source_regions=(*eu27, "US"),
        mapping=Path("scripts/pipelines/region_scopes/eu27_2020.csv"),
        scopes=("EU27_2020",),
        include_complement=True,
    )
    assert plan.scopes[0].members == eu27
    assert plan.scopes[1].code == "RoW"
    assert plan.scopes[1].members == ("US",)


def test_scope_slice_streams_internal_external_and_total_boundaries(tmp_path: Path) -> None:
    """Protect scope trade boundaries, additive totals, and flow-invariant production x."""
    from ecolecon_mrio.region_aggregation import RegionScope, load_mrio_iot_scope_slice

    cache_root = _prepare_scope_cache_root(tmp_path)
    scope = RegionScope(code="AT_DE", members=("AT", "DE"), label="Austria and Germany")
    internal = load_mrio_iot_scope_slice("exiobase", 2020, scope=scope, data_root=cache_root, doi=DEFAULT_EXIOBASE3_DOI, flow="internal")
    external = load_mrio_iot_scope_slice("exiobase", 2020, scope=scope, data_root=cache_root, doi=DEFAULT_EXIOBASE3_DOI, flow="external_imports")
    total = load_mrio_iot_scope_slice("exiobase", 2020, scope=scope, data_root=cache_root, doi=DEFAULT_EXIOBASE3_DOI, flow="total")

    assert internal.Z.loc["CPA_A", "CPA_A"] == 208.0
    assert external.Z.loc["CPA_A", "CPA_A"] == 404.0
    assert total.Z.loc["CPA_A", "CPA_A"] == 612.0
    assert internal.Y.loc["CPA_A", "HFCE"] == 2008.0
    assert external.Y.loc["CPA_A", "HFCE"] == 4004.0
    assert total.Y.loc["CPA_A", "HFCE"] == 6012.0
    assert internal.x_scope.loc["CPA_A"] == 400.0
    pd.testing.assert_series_equal(internal.x_scope, external.x_scope)
    pd.testing.assert_series_equal(internal.x_scope, total.x_scope)
    pd.testing.assert_frame_equal(internal.Z.add(external.Z), total.Z)
    pd.testing.assert_frame_equal(internal.Y.add(external.Y), total.Y)
    assert not (cache_root / "exiobase3_2020" / "slices").exists()


def test_all_regions_scope_has_zero_external_imports(tmp_path: Path) -> None:
    """An all-source scope has a well-defined zero external-import monetary account."""
    from ecolecon_mrio.region_aggregation import RegionScope, load_mrio_iot_scope_slice

    cache_root = _prepare_scope_cache_root(tmp_path)
    out = load_mrio_iot_scope_slice(
        "exiobase",
        2020,
        data_root=cache_root,
        doi=DEFAULT_EXIOBASE3_DOI,
        scope=RegionScope(code="ALL", members=("AT", "DE", "FR")),
        flow="external_imports",
    )
    assert float(out.Z.to_numpy(dtype="float64").sum()) == 0.0
    assert float(out.Y.to_numpy(dtype="float64").sum()) == 0.0
    assert out.x_scope.loc["CPA_A"] == 900.0


def test_region_scope_api_is_exported_from_package() -> None:
    """Protect the accepted public geography symbols at the package root."""
    from ecolecon_mrio import (
        MrioScopeSlice,
        RegionAggregationPlan,
        RegionScope,
        compile_region_aggregation,
        load_mrio_iot_scope_slice,
    )

    assert MrioScopeSlice is not None
    assert RegionAggregationPlan is not None
    assert RegionScope is not None
    assert callable(compile_region_aggregation)
    assert callable(load_mrio_iot_scope_slice)



def test_scope_slice_rejects_mismatched_destination_member_basis(tmp_path: Path) -> None:
    source = tmp_path / "bad_destination_basis.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("x.txt", "region\tsector\tindout\nAT\tA\t1\nAT\tB\t2\nDE\tA\t3\nDE\tB\t4\n")
        zf.writestr(
            "Z.txt",
            "\t\tAT\tAT\tDE\tDE\n\t\tA\tB\tB\tA\n\t\tu\tu\tu\tu\n"
            "AT\tA\t1\t0\t0\t0\nAT\tB\t0\t1\t0\t0\nDE\tA\t0\t0\t1\t0\nDE\tB\t0\t0\t0\t1\n",
        )
        zf.writestr(
            "Y.txt",
            "\t\tAT\tDE\n\t\tHFCE\tHFCE\n\t\tu\tu\n"
            "AT\tA\t0\t0\nAT\tB\t0\t0\nDE\tA\t0\t0\nDE\tB\t0\t0\n",
        )
    with pytest.raises(ValueError, match="destination label basis differs"):
        exiobase3._load_exio_scope_slice_direct_from_zip(
            year=2020,
            system="pxp",
            scope=RegionScope(code="AT_DE", members=("AT", "DE")),
            source_zip=source,
            flow="internal",
        )


def test_scope_slice_rejects_malformed_producer_origin_basis(tmp_path: Path) -> None:
    source = tmp_path / "bad_origin_basis.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("x.txt", "region\tsector\tindout\nAT\tA\t1\nAT\tB\t2\nDE\tA\t3\nDE\tB\t4\n")
        zf.writestr(
            "Z.txt",
            "\t\tAT\tAT\tDE\tDE\n\t\tA\tB\tA\tB\n\t\tu\tu\tu\tu\n"
            "AT\tA\t1\t0\t0\t0\nAT\tB\t0\t1\t0\t0\nDE\tB\t0\t0\t0\t1\nDE\tA\t0\t0\t1\t0\n",
        )
        zf.writestr(
            "Y.txt",
            "\t\tAT\tDE\n\t\tHFCE\tHFCE\n\t\tu\tu\n"
            "AT\tA\t0\t0\nAT\tB\t0\t0\nDE\tB\t0\t0\nDE\tA\t0\t0\n",
        )
    with pytest.raises(ValueError, match="producer sector basis differs"):
        exiobase3._load_exio_scope_slice_direct_from_zip(
            year=2020,
            system="pxp",
            scope=RegionScope(code="AT", members=("AT",)),
            source_zip=source,
            flow="internal",
        )


def test_scope_slice_rejects_extra_x_sector(tmp_path: Path) -> None:
    source = tmp_path / "extra_x.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("x.txt", "region\tsector\tindout\nAT\tA\t1\nAT\tB\t2\nAT\tC\t3\n")
        zf.writestr(
            "Z.txt",
            "\t\tAT\tAT\n\t\tA\tB\n\t\tu\tu\nAT\tA\t1\t0\nAT\tB\t0\t1\n",
        )
        zf.writestr(
            "Y.txt",
            "\t\tAT\n\t\tHFCE\n\t\tu\nAT\tA\t0\nAT\tB\t0\n",
        )
    with pytest.raises(ValueError, match="x.txt sector basis does not exactly match"):
        exiobase3._load_exio_scope_slice_direct_from_zip(
            year=2020,
            system="pxp",
            scope=RegionScope(code="AT", members=("AT",)),
            source_zip=source,
            flow="internal",
        )


def test_scope_loader_forwards_download_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    source = tmp_path / "IOT_2020_pxp.zip"
    _write_tiny_exio_zip(source)

    def fake_ensure(*args, **kwargs):
        captured.update(kwargs)
        return source

    monkeypatch.setattr("ecolecon_mrio.data.ensure_exiobase_year_zip", fake_ensure)
    load_mrio_iot_scope_slice(
        "exiobase",
        2020,
        scope=RegionScope(code="AT", members=("AT",)),
        flow="internal",
        data_root=tmp_path,
        allow_download_missing=False,
        resume_downloads=False,
        show_download_progress=False,
        download_chunk_mb=7,
        download_timeout_seconds=12.0,
        download_retries=2,
    )
    assert captured["allow_download_missing"] is False
    assert captured["resume_downloads"] is False
    assert captured["show_download_progress"] is False
    assert captured["download_chunk_mb"] == 7
    assert captured["download_timeout_seconds"] == 12.0
    assert captured["download_retries"] == 2
