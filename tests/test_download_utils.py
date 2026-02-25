from __future__ import annotations

from pathlib import Path

from ecolecon_mrio.download_utils import DownloadSpec, download_file_resumable


class _FakeResp:
    def __init__(self, status_code: int, payload: bytes, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self._payload), chunk_size):
            yield self._payload[i : i + chunk_size]

    def close(self):
        return None


def test_download_file_resumable_skips_existing_valid(tmp_path):
    dest = tmp_path / 'x.bin'
    dest.write_bytes(b'abc')
    spec = DownloadSpec(url='http://example.test/x.bin', dest_path=dest, integrity_check=lambda p: p.read_bytes() == b'abc')
    res = download_file_resumable(spec, progress=False)
    assert res.status == 'skipped_existing'
    assert dest.read_bytes() == b'abc'


def test_download_file_resumable_resumes_with_range(monkeypatch, tmp_path):
    dest = tmp_path / 'x.bin'
    part = tmp_path / 'x.bin.part'
    part.write_bytes(b'abc')
    calls: list[dict] = []

    def _fake_get(url, stream=True, headers=None, cookies=None, timeout=None):
        calls.append({'url': url, 'headers': dict(headers or {})})
        assert headers and headers.get('Range') == 'bytes=3-'
        return _FakeResp(
            206,
            b'def',
            headers={
                'Content-Length': '3',
                'Content-Range': 'bytes 3-5/6',
            },
        )

    import ecolecon_mrio.download_utils as du

    monkeypatch.setattr(du.requests, 'get', _fake_get)
    spec = DownloadSpec(url='http://example.test/x.bin', dest_path=dest)
    res = download_file_resumable(spec, progress=False, resume=True)
    assert res.status == 'resumed_and_completed'
    assert res.resumed is True
    assert dest.read_bytes() == b'abcdef'
    assert not part.exists()
    assert len(calls) == 1


def test_download_file_resumable_restarts_if_no_range_support(monkeypatch, tmp_path):
    dest = tmp_path / 'y.bin'
    part = tmp_path / 'y.bin.part'
    part.write_bytes(b'old')
    calls = {'n': 0}

    def _fake_get(url, stream=True, headers=None, cookies=None, timeout=None):
        calls['n'] += 1
        if calls['n'] == 1:
            assert headers and headers.get('Range') == 'bytes=3-'
            # server ignores Range and returns full object
            return _FakeResp(200, b'abcdef', headers={'Content-Length': '6'})
        assert not headers or 'Range' not in headers
        return _FakeResp(200, b'abcdef', headers={'Content-Length': '6'})

    import ecolecon_mrio.download_utils as du

    monkeypatch.setattr(du.requests, 'get', _fake_get)
    spec = DownloadSpec(url='http://example.test/y.bin', dest_path=dest)
    res = download_file_resumable(spec, progress=False, resume=True)
    assert res.status == 'downloaded'
    assert res.range_supported is False
    assert dest.read_bytes() == b'abcdef'
    assert not part.exists()
    assert calls['n'] == 2
