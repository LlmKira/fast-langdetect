# -*- coding: utf-8 -*-
"""Regression tests for the deprecated ``fasttext`` co-install scenario.

When the deprecated ``fasttext`` PyPI distribution is installed alongside
``fasttext-predict`` (which ``fast-langdetect`` requires), the two
distributions own the same ``fasttext/`` package directory and whichever
installed last wins on disk. The deprecated dist's ``_FastText.predict``
ends with ``np.array(probs, copy=False)``, which raises under NumPy >= 2.0.

These tests simulate that broken state (without actually installing the
deprecated dist) and verify that ``infer._patch_fasttext_for_numpy2``
repairs it.
"""

import importlib.metadata

import numpy as np
import pytest


class _StubBinding:
    """Stub for the C++ pybind binding stored at ``_FastText.f``."""

    def predict(self, text, k, threshold, on_unicode_error):
        return [(0.95, "__label__en"), (0.05, "__label__de")]


class _StubFastText:
    """Stand-in for ``fasttext._FastText`` instances."""

    def __init__(self):
        self.f = _StubBinding()


def _broken_predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
    """The exact failure mode of deprecated ``fasttext`` under NumPy >= 2.0."""
    predictions = self.f.predict(text + "\n", k, threshold, on_unicode_error)
    probs, labels = zip(*predictions)
    # Raises under NumPy >= 2.0 because a copy cannot be avoided.
    return labels, np.array(probs, copy=False)


@pytest.fixture
def simulated_broken_fasttext(monkeypatch):
    """Pretend the deprecated ``fasttext`` dist is installed with broken predict."""
    import fasttext.FastText as _ft_module

    cls = _ft_module._FastText
    real_distribution = importlib.metadata.distribution

    def fake_distribution(name):
        if name == "fasttext":
            return object()  # Only existence is checked, content is unused.
        return real_distribution(name)

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)
    monkeypatch.setattr(cls, "predict", _broken_predict, raising=False)
    return cls


def test_broken_predict_reproduces_upstream_bug(simulated_broken_fasttext):
    """Sanity check: the simulated broken predict raises the documented error."""
    if int(np.__version__.split(".")[0]) < 2:
        pytest.skip("Only reproducible under NumPy >= 2.0")
    instance = _StubFastText()
    with pytest.raises(ValueError, match="copy"):
        _broken_predict(instance, "hello world")


def test_patch_repairs_broken_predict(simulated_broken_fasttext):
    """The patch must replace broken predict with a NumPy-2-safe version."""
    from fast_langdetect.infer import _patch_fasttext_for_numpy2

    assert _patch_fasttext_for_numpy2() is True

    instance = _StubFastText()
    labels, scores = simulated_broken_fasttext.predict(instance, "hello world")
    assert labels == ("__label__en", "__label__de")
    assert isinstance(scores, np.ndarray)
    assert scores.tolist() == [0.95, 0.05]


def test_patch_is_noop_when_deprecated_fasttext_absent(monkeypatch):
    """If only ``fasttext-predict`` is installed, the patch is a no-op."""
    # `fast_langdetect` must already be importable (so the real
    # `distribution()` is not yet patched). After import, swap it for a
    # version that pretends `fasttext` is uninstalled.
    from fast_langdetect.infer import _patch_fasttext_for_numpy2

    real_distribution = importlib.metadata.distribution

    def fake_distribution(name):
        if name == "fasttext":
            raise importlib.metadata.PackageNotFoundError("fasttext")
        return real_distribution(name)

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)

    assert _patch_fasttext_for_numpy2() is False
