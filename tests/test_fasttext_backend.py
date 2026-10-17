import sys
import types

import pytest

from fast_langdetect.infer import ModelLoadError, ModelLoader


class _DeprecatedFastText:
    @staticmethod
    def train_supervised(*args, **kwargs):
        raise AssertionError("should not train during backend validation")

    @staticmethod
    def load_model(path):
        return {"path": path}


class _PredictOnlyFastText:
    @staticmethod
    def load_model(path):
        return {"path": path}


@pytest.fixture
def model_path(monkeypatch, tmp_path):
    monkeypatch.setattr("fast_langdetect.infer.platform.system", lambda: "Linux")
    path = tmp_path / "lid.176.ftz"
    path.write_bytes(b"model")
    return path


def _use_numpy(monkeypatch, version):
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace(__version__=version))


def test_loader_rejects_deprecated_fasttext_under_numpy2(monkeypatch, model_path):
    monkeypatch.setattr("fast_langdetect.infer.fasttext", _DeprecatedFastText)
    _use_numpy(monkeypatch, "2.1.3")

    with pytest.raises(ModelLoadError, match="deprecated `fasttext` PyPI package"):
        ModelLoader().load_local(model_path)


def test_loader_accepts_deprecated_fasttext_under_numpy1(monkeypatch, model_path):
    monkeypatch.setattr("fast_langdetect.infer.fasttext", _DeprecatedFastText)
    _use_numpy(monkeypatch, "1.26.4")

    assert ModelLoader().load_local(model_path) == {"path": str(model_path)}


def test_loader_accepts_predict_only_fasttext_backend(monkeypatch, model_path):
    monkeypatch.setattr("fast_langdetect.infer.fasttext", _PredictOnlyFastText)
    _use_numpy(monkeypatch, "2.1.3")

    assert ModelLoader().load_local(model_path) == {"path": str(model_path)}
