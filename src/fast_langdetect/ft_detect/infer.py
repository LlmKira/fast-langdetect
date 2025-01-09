# -*- coding: utf-8 -*-
# @Time    : 2024/01/17 下午08:30
# @Author  : sudoskys
# @File    : infer.py
# @Software: PyCharm
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import fasttext
from robust_downloader import download

logger = logging.getLogger(__name__)
CACHE_DIRECTORY = os.getenv("FTLANG_CACHE", "/tmp/fasttext-langdetect")
LOCAL_SMALL_MODEL_PATH = Path(__file__).parent / "resources" / "lid.176.ftz"

FASTTEXT_LARGE_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_LARGE_MODEL_NAME = "lid.176.bin"


class DetectError(Exception):
    pass


class ModelManager:
    def __init__(self):
        self._models = {}

    def get_model(self, key: str):
        """Retrieve cached model."""
        return self._models.get(key)

    def cache_model(self, key: str, model) -> None:
        """Cache loaded FastText model."""
        self._models[key] = model


_model_cache = ModelManager()


def download_model(
        download_url: str,
        save_path: Path,
        proxy: Optional[str] = None
) -> None:
    """
    Download FastText model file if it doesn't exist.
    :param download_url: URL to download the model from
    :param save_path: Path to save the downloaded model
    :param proxy: Proxy URL for downloading the model
    :raises DetectError: If download fails
    """
    if save_path.exists():
        logger.info(f"fast-langdetect:Model already exists at {save_path}. Skipping download.")
        return

    logger.info(f"fast-langdetect:Downloading FastText model from {download_url} to {save_path}")
    try:
        download(
            url=download_url,
            folder=str(save_path.parent),
            filename=save_path.name,
            proxy=proxy,
            retry_max=3,
            timeout=30,
        )
    except Exception as e:
        logger.error(f"fast-langdetect:Failed to download FastText model from {download_url}: {e}")
        raise DetectError(f"Unable to download model from {download_url}")


def load_fasttext_model(
        model_path: Path,
        download_url: Optional[str] = None,
        proxy: Optional[str] = None,
):
    """
    Load a FastText model, downloading it if necessary.
    :param model_path: Path to the FastText model file
    :param download_url: URL to download the model from
    :param proxy: Proxy URL for downloading the model
    :return: FastText model
    :raises DetectError: If model loading fails
    """
    if not model_path.exists() and download_url:
        # Attempt to download the model
        download_model(download_url, model_path, proxy)

    if not model_path.exists():
        raise DetectError(f"FastText model file not found at {model_path}")

    try:
        # Load FastText model
        return fasttext.load_model(str(model_path))
    except Exception as e:
        logger.error(f"fast-langdetect:Failed to load FastText model from {model_path}: {e}")
        raise DetectError(f"Failed to load FastText model: {e}")


def load_model(
        low_memory: bool = False,
        download_proxy: Optional[str] = None,
        use_strict_mode: bool = False,
):
    """
    Load a FastText model based on memory preference.
    :param low_memory: Whether to use a memory-efficient model
    :param download_proxy: Proxy URL for downloading the model
    :param use_strict_mode: If enabled, strictly loads large model or raises error if it fails
    """
    # Model path selection
    if low_memory:
        cache_key = "low_memory"
        model_path = LOCAL_SMALL_MODEL_PATH
    else:
        cache_key = "high_memory"
        model_path = Path(CACHE_DIRECTORY) / FASTTEXT_LARGE_MODEL_NAME

    # Check cache
    cached_model = _model_cache.get_model(cache_key)
    if cached_model:
        return cached_model

    # Load appropriate model
    try:
        if low_memory:
            model = load_fasttext_model(model_path)
        else:
            model = load_fasttext_model(model_path, download_url=FASTTEXT_LARGE_MODEL_URL, proxy=download_proxy)
        _model_cache.cache_model(cache_key, model)
        return model
    except Exception as e:
        logger.error(f"fast-langdetect:Failed to load model ({'low' if low_memory else 'high'} memory): {e}")
        if use_strict_mode:
            raise DetectError("Failed to load FastText model.") from e
        elif not low_memory:
            logger.info("Falling back to low-memory model...")
            return load_model(low_memory=True, use_strict_mode=True)
        raise e


def detect(
        text: str,
        *,
        low_memory: bool = True,
        model_download_proxy: Optional[str] = None,
        use_strict_mode: bool = False,
) -> Dict[str, Union[str, float]]:
    """
    Detect the language of a text using FastText.
    This function assumes to be given a single line of text. We split words on whitespace (space, newline, tab, vertical tab) and the control characters carriage return, formfeed and the null character.
    If the model is not supervised, this function will throw a ValueError.
    :param text: The text for language detection
    :param low_memory: Whether to use a memory-efficient model
    :param model_download_proxy: Download proxy for the model if needed
    :param use_strict_mode: If it was enabled, strictly loads large model or raises error if it fails
    :return: A dictionary with detected language and confidence score
    :raises LanguageDetectionError: If detection fails
    """
    model = load_model(
        low_memory=low_memory,
        download_proxy=model_download_proxy,
        use_strict_mode=use_strict_mode,
    )
    try:
        labels, scores = model.predict(text)
        language_label = labels[0].replace("__label__", "")
        confidence_score = min(float(scores[0]), 1.0)
        return {"lang": language_label, "score": confidence_score}
    except Exception as e:
        logger.error(f"fast-langdetect:Error during language detection: {e}")
        raise DetectError("Language detection failed.")


def detect_multilingual(
        text: str,
        *,
        low_memory: bool = True,
        model_download_proxy: Optional[str] = None,
        k: int = 5,
        threshold: float = 0.0,
        use_strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Detect the top-k probable languages for a given text.
    """
    model = load_model(
        low_memory=low_memory,
        download_proxy=model_download_proxy,
        use_strict_mode=use_strict_mode,
    )
    # Multilingual detection using FastText
    try:
        labels, scores = model.predict(text, k=k, threshold=threshold)
        results = [
            {"lang": label.replace("__label__", ""), "score": min(float(score), 1.0)}
            for label, score in zip(labels, scores)
        ]
        return sorted(results, key=lambda x: x["score"], reverse=True)
    except Exception as e:
        logger.error(f"fast-langdetect:Error during multilingual detection: {e}")
        raise DetectError("Multilingual detection failed.")
