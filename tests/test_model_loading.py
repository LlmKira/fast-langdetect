import hashlib

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
