import hashlib
import threading
import time
from pathlib import Path

import pytest

from fast_langdetect.infer import FastLangdetectError, ModelDownloader, ModelLoader


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def test_downloader_replaces_invalid_cached_model(monkeypatch, tmp_path):
    model_data = b"valid model"
    expected_md5 = _md5(model_data)
    target = tmp_path / "lid.176.bin"
    target.write_bytes(b"corrupt model")

    def fake_download(*, folder, filename, **kwargs):
        assert "md5" not in kwargs
        Path(folder, filename).write_bytes(model_data)

    monkeypatch.setattr("fast_langdetect.infer.download", fake_download)

    ModelDownloader.download(
        "https://example.invalid/model.bin",
        target,
        expected_md5=expected_md5,
    )

    assert target.read_bytes() == model_data
    assert not list(tmp_path.glob("*.tmp"))
    assert not (tmp_path / "lid.176.bin.lock").exists()


def test_downloader_does_not_publish_failed_integrity_download(monkeypatch, tmp_path):
    target = tmp_path / "lid.176.bin"
    expected_md5 = _md5(b"valid model")

    def fake_download(*, folder, filename, **kwargs):
        Path(folder, filename).write_bytes(b"wrong model")

    monkeypatch.setattr("fast_langdetect.infer.download", fake_download)

    with pytest.raises(FastLangdetectError, match="integrity check"):
        ModelDownloader.download(
            "https://example.invalid/model.bin",
            target,
            expected_md5=expected_md5,
        )

    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert not (tmp_path / "lid.176.bin.lock").exists()


def test_downloader_times_out_when_cache_lock_is_held(monkeypatch, tmp_path):
    target = tmp_path / "lid.176.bin"
    (tmp_path / "lid.176.bin.lock").mkdir()
    monkeypatch.setattr("fast_langdetect.infer._MODEL_DOWNLOAD_LOCK_POLL_SECONDS", 0)

    with pytest.raises(FastLangdetectError, match="model cache lock"):
        ModelDownloader.download(
            "https://example.invalid/model.bin",
            target,
            expected_md5=_md5(b"valid model"),
            lock_timeout=0,
        )


def test_loader_downloads_only_once_for_concurrent_callers(monkeypatch, tmp_path):
    model_data = b"valid model"
    expected_md5 = _md5(model_data)
    target = tmp_path / "lid.176.bin"
    download_calls = []
    load_calls = []
    release_download = threading.Event()

    monkeypatch.setattr("fast_langdetect.infer.FASTTEXT_LARGE_MODEL_MD5", expected_md5)

    def fake_download(*, folder, filename, **kwargs):
        download_calls.append(filename)
        Path(folder, filename).write_bytes(model_data)
        release_download.wait(timeout=2)

    def fake_load_local(self, model_path):
        load_calls.append(model_path)
        return object()

    monkeypatch.setattr("fast_langdetect.infer.download", fake_download)
    monkeypatch.setattr(ModelLoader, "load_local", fake_load_local)

    loaders = [ModelLoader(), ModelLoader()]
    errors = []

    def load_model(loader):
        try:
            loader.load_with_download(target)
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=load_model, args=(loaders[0],))
    second = threading.Thread(target=load_model, args=(loaders[1],))

    first.start()
    while not download_calls:
        time.sleep(0.01)
    second.start()
    release_download.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert len(download_calls) == 1
    assert len(load_calls) == 2
    assert target.read_bytes() == model_data
