# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 下午8:30
# @Author  : sudoskys
# @File    : infer.py
# @Software: PyCharm
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Union, List, Optional, Any

import fasttext
from robust_downloader import download

logger = logging.getLogger(__name__)
CACHE_DIRECTORY = os.getenv("FTLANG_CACHE", "/tmp/fasttext-langdetect")
LOCAL_SMALL_MODEL_PATH = Path(__file__).parent / "resources" / "lid.176.ftz"

# Suppress FastText output if possible
try:
    fasttext.FastText.eprint = lambda *args, **kwargs: None
except Exception:
    pass


class ModelType(Enum):
    LOW_MEMORY = "low_mem"
    HIGH_MEMORY = "high_mem"


class ModelCache:
    def __init__(self):
        self._models = {}

    def get_model(self, model_type: ModelType) -> Optional["fasttext.FastText._FastText"]:
        return self._models.get(model_type)

    def set_model(self, model_type: ModelType, model: "fasttext.FastText._FastText"):
        self._models[model_type] = model


_model_cache = ModelCache()


class DetectError(Exception):
    """Custom exception for language detection errors."""
    pass


def load_model(low_memory: bool = False,
               download_proxy: Optional[str] = None,
               use_strict_mode: bool = False) -> "fasttext.FastText._FastText":
    """
    Load the FastText model based on memory preference.

    :param low_memory: Indicates whether to load a smaller, memory-efficient model
    :param download_proxy: Proxy to use for downloading the large model if necessary
    :param use_strict_mode: If enabled, strictly loads large model or raises error if it fails
    :return: Loaded FastText model
    :raises DetectError: If the model cannot be loaded
    """
    model_type = ModelType.LOW_MEMORY if low_memory else ModelType.HIGH_MEMORY

    # If the model is already loaded, return it
    cached_model = _model_cache.get_model(model_type)
    if cached_model:
        return cached_model

    def load_local_small_model():
        """Try to load the local small model."""
        try:
            _loaded_model = fasttext.load_model(str(LOCAL_SMALL_MODEL_PATH))
            _model_cache.set_model(ModelType.LOW_MEMORY, _loaded_model)
            return _loaded_model
        except Exception as e:
            logger.error(f"Failed to load the local small model '{LOCAL_SMALL_MODEL_PATH}': {e}")
            raise DetectError("Unable to load low-memory model from local resources.")

    def load_large_model():
        """Try to load the large model."""
        try:
            loaded_model = fasttext.load_model(str(model_path))
            _model_cache.set_model(ModelType.HIGH_MEMORY, loaded_model)
            return loaded_model
        except Exception as e:
            logger.error(f"Failed to load the large model '{model_path}': {e}")
        return None

    if low_memory:
        # Attempt to load the local small model
        return load_local_small_model()

    # Path for the large model
    large_model_name = "lid.176.bin"
    model_path = Path(CACHE_DIRECTORY) / large_model_name

    # If the large model is already present, load it
    if model_path.exists():
        # Model cant be dir
        if model_path.is_dir():
            try:
                model_path.rmdir()
            except Exception as e:
                logger.error(f"Failed to remove the directory '{model_path}': {e}")
                raise DetectError(f"Unexpected directory found in large model file path '{model_path}': {e}")
        # Attempt to load large model
        loaded_model = load_large_model()
        if loaded_model:
            return loaded_model

    # If the large model is not present, attempt to download (only if necessary)
    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    try:
        logger.info(f"Downloading large model from {model_url} to {model_path}")
        download(
            url=model_url,
            folder=CACHE_DIRECTORY,
            filename=large_model_name,
            proxy=download_proxy,
            retry_max=3,
            timeout=20
        )
        # Try loading the model again after download
        loaded_model = load_large_model()
        if loaded_model:
            return loaded_model
    except Exception as e:
        logger.error(f"Failed to download the large model: {e}")

    # Handle fallback logic for strict and non-strict modes
    if use_strict_mode:
        raise DetectError("Strict mode enabled: Unable to download or load the large model.")
    else:
        logger.info("Attempting to fall back to local small model.")
        return load_local_small_model()


def detect(text: str, *,
           low_memory: bool = True,
           model_download_proxy: Optional[str] = None,
           use_strict_mode: bool = False
           ) -> Dict[str, Union[str, float]]:
    """
    Detect the language of a text using FastText.
    This function assumes to be given a single line of text. We split words on whitespace (space, newline, tab, vertical tab) and the control characters carriage return, formfeed and the null character.
    If the model is not supervised, this function will throw a ValueError.
    :param text: The text for language detection
    :param low_memory: Whether to use a memory-efficient model
    :param model_download_proxy: Download proxy for the model if needed
    :param use_strict_mode: If enabled, strictly loads large model or raises error if it fails
    :return: A dictionary with detected language and confidence score
    :raises LanguageDetectionError: If detection fails
    """
    model = load_model(low_memory=low_memory, download_proxy=model_download_proxy, use_strict_mode=use_strict_mode)
    labels, scores = model.predict(text)
    language_label = labels[0].replace("__label__", '')
    confidence_score = min(float(scores[0]), 1.0)
    return {
        "lang": language_label,
        "score": confidence_score,
    }


def detect_multilingual(text: str, *,
                        low_memory: bool = True,
                        model_download_proxy: Optional[str] = None,
                        k: int = 5,
                        threshold: float = 0.0,
                        on_unicode_error: str = "strict",
                        use_strict_mode: bool = False
                        ) -> List[Dict[str, Any]]:
    """
    Detect multiple potential languages and their probabilities in a given text.
    k controls the number of returned labels. A choice of 5, will return the 5 most probable labels. By default, this returns only the most likely label and probability. threshold filters the returned labels by a threshold on probability. A choice of 0.5 will return labels with at least 0.5 probability. k and threshold will be applied together to determine the returned labels.
    This function assumes to be given a single line of text. We split words on whitespace (space, newline, tab, vertical tab) and the control characters carriage return, formfeed, and the null character.
    If the model is not supervised, this function will throw a ValueError.

    :param text: The text for language detection
    :param low_memory: Whether to use a memory-efficient model
    :param model_download_proxy: Proxy for downloading the model
    :param k: Number of top language predictions to return
    :param threshold: Minimum score threshold for predictions
    :param on_unicode_error: Error handling for Unicode errors
    :param use_strict_mode: If enabled, strictly loads large model or raises error if it fails
    :return: A list of dictionaries, each containing a language and its confidence score
    :raises LanguageDetectionError: If detection fails
    """
    model = load_model(low_memory=low_memory, download_proxy=model_download_proxy, use_strict_mode=use_strict_mode)
    labels, scores = model.predict(text, k=k, threshold=threshold, on_unicode_error=on_unicode_error)
    results = []
    for label, score in zip(labels, scores):
        language_label = label.replace("__label__", '')
        confidence_score = min(float(score), 1.0)
        results.append({
            "lang": language_label,
            "score": confidence_score,
        })
    return sorted(results, key=lambda x: x['score'], reverse=True)
