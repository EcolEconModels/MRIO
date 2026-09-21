from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import zipfile

import pytest

from ecolecon_mrio import data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_exiobase_box_beta.py"
SPEC = importlib.util.spec_from_file_location("install_exiobase_box_beta", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def _zip(path: Path, *, marker: str = "beta") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("file_parameters.json", "{}")
        zf.writestr("marker.txt", marker)
    return path


def test_public_box_bootstrap_parser_and_direct_url() -> None:
    page = r'''<html><script>
    Box.postStreamData = {"/app-api/enduserapp/item/d_1":{"items":[
      {"typedID":"f_123","type":"file","id":123,"name":"IOT_2022_pxp.zip"},
      {"typedID":"f_456","type":"file","id":456,"name":"IOT_2024_pxp.zip"}
    ]}};
    </script></html>'''
    files = installer.box_file_index(installer.parse_box_post_stream_data(page))
    assert files == {"IOT_2022_pxp.zip": "123", "IOT_2024_pxp.zip": "456"}
    assert installer.box_direct_download_url(
        "https://xio.app.box.com/s/sharedtoken", "123"
    ) == (
        "https://xio.app.box.com/index.php?rm=box_download_shared_file"
        "&shared_name=sharedtoken&file_id=f_123"
    )


def test_public_box_index_enumerates_every_declared_page(monkeypatch: pytest.MonkeyPatch) -> None:
    def page_html(page_number: int, items: list[dict[str, object]]) -> str:
        return (
            "<script>Box.postStreamData = "
            + json.dumps(
                {
                    "/app-api/enduserapp/shared-folder": {
                        "pageCount": 2,
                        "pageNumber": page_number,
                        "items": items,
                    }
                }
            )
            + ";</script>"
        )

    class Response:
        def __init__(self, text: str, cookie: str) -> None:
            self.text = text
            self.cookies = type("Cookies", (), {"get_dict": lambda _self: {cookie: "1"}})()

        def raise_for_status(self) -> None:
            return None

    calls: list[str] = []

    def fake_get(url: str, **_kwargs: object) -> Response:
        calls.append(url)
        page = int(parse_qs(urlparse(url).query).get("page", ["1"])[0])
        if page == 1:
            return Response(
                page_html(1, [{"type": "file", "id": 2022, "name": "IOT_2022_pxp.zip"}]),
                "first",
            )
        if page == 2:
            return Response(
                page_html(2, [{"type": "file", "id": 2012, "name": "IOT_2012_pxp.zip"}]),
                "second",
            )
        raise AssertionError(f"unexpected page request: {url}")

    monkeypatch.setattr(installer.requests, "get", fake_get)
    files, cookies = installer.fetch_public_box_index("https://xio.app.box.com/s/sharedtoken")

    assert files == {"IOT_2022_pxp.zip": "2022", "IOT_2012_pxp.zip": "2012"}
    assert cookies == {"first": "1", "second": "1"}
    assert calls == [
        "https://xio.app.box.com/s/sharedtoken",
        "https://xio.app.box.com/s/sharedtoken?page=2",
    ]


def test_local_override_is_accepted_without_doi_download_log(tmp_path: Path) -> None:
    year = 2022
    data_dir = data.resolve_exiobase_year_dir(year, data_root=tmp_path)
    source = _zip(data_dir / "IOT_2022_pxp.zip")
    manifest = data.write_exiobase_local_source_manifest(
        data_dir=data_dir,
        zip_path=source,
        year=year,
        system="pxp",
        release_label="3.11.2-beta",
        source_url="https://xio.app.box.com/s/example",
        source_item_id="123",
        locator_doi=data.DEFAULT_EXIOBASE3_DOI,
    )

    assert data.ensure_exiobase_year_zip(
        year,
        data_dir=data_dir,
        system="pxp",
        allow_download_missing=False,
    ) == source
    inputs = data.mrio_source_input_paths_for_year(
        dataset="exiobase",
        year=year,
        data_root=tmp_path,
        exio_system="pxp",
    )
    assert inputs == [source, manifest]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["release_label"] == "3.11.2-beta"
    assert len(payload["source_zip_sha256"]) == 64


def test_local_override_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    year = 2022
    data_dir = data.resolve_exiobase_year_dir(year, data_root=tmp_path)
    source = _zip(data_dir / "IOT_2022_pxp.zip")
    data.write_exiobase_local_source_manifest(
        data_dir=data_dir,
        zip_path=source,
        year=year,
        system="pxp",
        release_label="3.11.2-beta",
        source_url="https://xio.app.box.com/s/example",
    )
    with zipfile.ZipFile(source, "a") as zf:
        zf.writestr("changed.txt", "changed")

    with pytest.raises(ValueError, match="byte hash mismatch"):
        data.ensure_exiobase_year_zip(
            year,
            data_dir=data_dir,
            system="pxp",
            allow_download_missing=False,
        )


def test_force_download_explicitly_removes_local_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    year = 2022
    data_dir = data.resolve_exiobase_year_dir(year, data_root=tmp_path)
    source = _zip(data_dir / "IOT_2022_pxp.zip")
    manifest = data.write_exiobase_local_source_manifest(
        data_dir=data_dir,
        zip_path=source,
        year=year,
        system="pxp",
        release_label="3.11.2-beta",
        source_url="https://xio.app.box.com/s/example",
    )

    def fake_download(spec, **_kwargs):
        _zip(Path(spec.dest_path), marker="doi")
        return type("Result", (), {"status": "done", "reason": "test", "bytes_written": 1})()

    monkeypatch.setattr(data, "download_file_resumable", fake_download)
    monkeypatch.setattr(data, "_resolve_exiobase3_url", lambda *_a, **_k: "https://example.test/IOT_2022_pxp.zip")
    monkeypatch.setattr(data.MRIOMetaData, "_make_download_log", lambda **_k: type(
        "Log", (), {"_add_fileio": lambda *_a: None, "save": lambda *_a: None}
    )())

    out = data.ensure_exiobase_year_zip(
        year,
        data_dir=data_dir,
        system="pxp",
        force_download=True,
        allow_download_missing=True,
    )
    assert out == source
    assert not manifest.exists()
    assert zipfile.ZipFile(source).read("marker.txt") == b"doi"
