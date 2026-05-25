import pytest

from fast_langdetect.infer import ModelLoadError, ModelLoader


class _DeprecatedFastText:
    @staticmethod
    def train_supervised(*args, **kwargs):
        raise AssertionError("should not train during backend validation")

    @staticmethod
    def load_model(path):
        raise AssertionError("legacy fasttext backend should fail before load_model")


class _PredictOnlyFastText:
    @staticmethod
    def load_model(path):
        return {"path": path}


def test_loader_rejects_deprecated_fasttext_backend(monkeypatch, tmp_path):
    model_path = tmp_path / "lid.176.ftz"
    model_path.write_bytes(b"model")
    monkeypatch.setattr("fast_langdetect.infer.fasttext", _DeprecatedFastText)

    with pytest.raises(ModelLoadError, match="deprecated `fasttext` PyPI package"):
        ModelLoader().load_local(model_path)


def test_loader_accepts_predict_only_fasttext_backend(monkeypatch, tmp_path):
    model_path = tmp_path / "lid.176.ftz"
    model_path.write_bytes(b"model")
    monkeypatch.setattr("fast_langdetect.infer.fasttext", _PredictOnlyFastText)
    monkeypatch.setattr("fast_langdetect.infer.platform.system", lambda: "Linux")

    assert ModelLoader().load_local(model_path) == {"path": str(model_path)}
