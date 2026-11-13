import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import fast_langdetect.infer as infer
from fast_langdetect import LangDetectConfig, LangDetector, detect


def test_auto_falls_back_to_lite_on_windows_memory_error(monkeypatch, tmp_path):
    placeholder = b"full model placeholder"
    (tmp_path / infer.FASTTEXT_LARGE_MODEL_NAME).write_bytes(placeholder)
    # Make the cached placeholder pass the checksum so no download is attempted.
    monkeypatch.setattr(infer, "FASTTEXT_LARGE_MODEL_MD5", hashlib.md5(placeholder).hexdigest())
    real_load_model = infer.fasttext.load_model

    def load_model(path):
        # Every Windows strategy (direct, relative, temp copy) loads a .bin path for the full model.
        if str(path).endswith(".bin"):
            raise MemoryError("simulated out of memory")
        return real_load_model(path)

    monkeypatch.setattr(infer.platform, "system", lambda: "Windows")
    monkeypatch.setattr(infer.fasttext, "load_model", load_model)

    detector = LangDetector(LangDetectConfig(cache_dir=str(tmp_path)))

    assert detector.detect("Hello world", model="auto")[0]["lang"] == "en"
    with pytest.raises(MemoryError):
        detector.detect("Hello world", model="full")


def test_detect_with_config_warns_about_reloading():
    with pytest.warns(DeprecationWarning, match="LangDetector"):
        detect("Hello world", model="lite", config=LangDetectConfig())


class _FakeModel:
    def predict(self, text, k=1, threshold=0.0):
        return (["__label__en"], [0.99])


def test_config_model_url_overrides_download_source(monkeypatch, tmp_path):
    model_data = b"mirror model"
    requested_urls = []

    monkeypatch.setattr(infer, "FASTTEXT_LARGE_MODEL_MD5", hashlib.md5(model_data).hexdigest())

    def fake_download(*, url, folder, filename, **kwargs):
        requested_urls.append(url)
        Path(folder, filename).write_bytes(model_data)

    monkeypatch.setattr(infer, "download", fake_download)
    monkeypatch.setattr(infer.ModelLoader, "load_local", lambda self, path: _FakeModel())

    detector = LangDetector(
        LangDetectConfig(
            cache_dir=str(tmp_path),
            model_url="https://mirror.example/lid.176.bin",
        )
    )

    assert detector.detect("Hello world", model="full")[0]["lang"] == "en"
    assert requested_urls == ["https://mirror.example/lid.176.bin"]


def test_model_url_env_overrides_default():
    env = os.environ.copy()
    env["FTLANG_MODEL_URL"] = "https://mirror.example/lid.176.bin"
    src_dir = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(src_dir), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import fast_langdetect.infer as i; print(i.FASTTEXT_LARGE_MODEL_URL)",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert result.stdout.strip() == "https://mirror.example/lid.176.bin"
