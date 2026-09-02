import json

import pandas as pd

from ecolecon_mrio.panel_coverage import (
    attempt_mrio_year,
    build_year_coverage_manifest,
    select_contiguous_year_block,
)


class _DummyMrio:
    def __init__(self):
        self.A = pd.DataFrame([[0.1, 0.0], [0.0, 0.2]], index=["a", "b"], columns=["a", "b"])
        self.x = pd.Series([10.0, 5.0], index=["a", "b"])
        self.Y = pd.DataFrame({"final_demand": [1.0, 2.0]}, index=["a", "b"])


def test_attempt_mrio_year_success_and_validator(monkeypatch):
    def _fake_coverage_load_mrio(*, dataset, year, **kwargs):
        mrio = _DummyMrio()
        return mrio, mrio.Y.sum(axis=1)

    def _validator(year, mrio, y_full):
        return {"status": "ok", "reason": "ready", "payload": {"n": int(len(y_full)), "year": year}}

    import ecolecon_mrio.panel_coverage as pc

    monkeypatch.setattr(pc, "_coverage_load_mrio", _fake_coverage_load_mrio)
    rec = attempt_mrio_year("exiobase", 2020, validators={"ready": _validator})
    assert rec.load_status == "ok"
    assert rec.calc_all_status == "ok"
    assert rec.validator_statuses["ready"] == "ok"
    assert rec.validator_reasons["ready"] == "ready"
    assert rec.validator_payloads["ready"]["n"] == 2


def test_attempt_mrio_year_failure_records_error(monkeypatch):
    def _fake_load_mrio(*, dataset, year, **kwargs):
        raise RuntimeError("boom")

    import ecolecon_mrio.panel_coverage as pc

    monkeypatch.setattr(pc, "_coverage_load_mrio", _fake_load_mrio)
    rec = attempt_mrio_year("exiobase", 2021)
    assert rec.load_status == "load_error"
    assert "RuntimeError" in rec.load_reason
    assert rec.calc_all_status == "not_attempted"


def test_build_year_coverage_manifest_flattens_validators(monkeypatch):
    def _fake_load_mrio(*, dataset, year, **kwargs):
        return _DummyMrio()

    def _validator(year, mrio, y_full):
        return {"status": "warn", "reason": "partial", "payload": {"labels": list(y_full.index)}}

    import ecolecon_mrio.panel_coverage as pc

    monkeypatch.setattr(pc, "load_mrio", _fake_load_mrio)
    df = build_year_coverage_manifest("test", [2021, 2020], validators={"cons": _validator})
    assert list(df["year"]) == [2020, 2021]
    assert {"cons__status", "cons__reason", "cons__payload"}.issubset(df.columns)
    payload = json.loads(df.loc[df["year"] == 2020, "cons__payload"].iloc[0])
    assert payload["labels"] == ["a", "b"]


def test_select_contiguous_year_block_prefers_latest_then_longest():
    df = pd.DataFrame(
        {
            "year": [2000, 2001, 2003, 2004, 2005, 2008],
            "load_status": ["ok"] * 6,
            "cons__status": ["ok"] * 6,
        }
    )
    sel = select_contiguous_year_block(df, required_status_columns=["cons__status"], min_years=2)
    assert sel["selection_status"] == "ok"
    assert sel["selected_years"] == [2003, 2004, 2005]


def test_select_contiguous_year_block_no_block_meets_threshold_reports_best_run():
    df = pd.DataFrame(
        {
            "year": [2010, 2011, 2013],
            "load_status": ["ok", "ok", "ok"],
            "cons__status": ["ok", "ok", "ok"],
        }
    )
    sel = select_contiguous_year_block(df, required_status_columns=["cons__status"], min_years=3)
    assert sel["selection_status"] == "no_block_meets_min_years"
    assert sel["best_available_run"] == [2010, 2011]


def test_build_year_coverage_manifest_integration_test_dataset():
    def _simple_validator(year, mrio, y_full):
        return {
            "status": "ok",
            "reason": "has_core_objects",
            "payload": {"has_A": getattr(mrio, "A", None) is not None, "y_len": int(len(y_full))},
        }

    df = build_year_coverage_manifest(
        "test",
        [2020, 2021],
        force_download_missing=False,
        validators={"core": _simple_validator},
        verbose=False,
    )
    assert len(df) == 2
    assert (df["load_status"] == "ok").all()
    assert (df["core__status"] == "ok").all()


def test_attempt_mrio_year_passes_download_control_kwargs(monkeypatch):
    captured = {}

    def _fake_load_mrio(*, dataset, year, **kwargs):
        captured.update(kwargs)
        return _DummyMrio(), pd.Series([1.0, 2.0], index=['a', 'b'])

    import ecolecon_mrio.panel_coverage as pc

    monkeypatch.setattr(pc, '_coverage_load_mrio', _fake_load_mrio)
    rec = attempt_mrio_year(
        'exiobase',
        2022,
        calc_all=False,
        allow_download_missing=False,
        resume_downloads=False,
        show_download_progress=False,
        download_chunk_mb=4,
        download_timeout_seconds=12.5,
        download_retries=7,
        delete_source_zip_after_parse=True,
    )
    assert rec.load_status == 'ok'
    assert captured['allow_download_missing'] is False
    assert captured['resume_downloads'] is False
    assert captured['show_download_progress'] is False
    assert captured['download_chunk_mb'] == 4
    assert captured['download_timeout_seconds'] == 12.5
    assert captured['download_retries'] == 7
    assert captured['delete_source_zip_after_parse'] is True
