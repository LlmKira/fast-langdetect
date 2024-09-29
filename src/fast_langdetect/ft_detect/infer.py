# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 下午8:30
# @Author  : sudoskys
# @File    : infer.py
# @Software: PyCharm
import logging
import os
from pathlib import Path
from typing import Dict, Union, List, Optional, Any

import fasttext
from robust_downloader import download

logger = logging.getLogger(__name__)
MODELS = {"low_mem": None, "high_mem": None}
CACHE_DIRECTORY = os.getenv("FTLANG_CACHE", "/tmp/fasttext-langdetect")
LOCAL_SMALL_MODEL_PATH = Path(__file__).parent / "resources" / "lid.176.ftz"

# Suppress FastText output if possible
try:
    fasttext.FastText.eprint = lambda *args, **kwargs: None
except Exception:
    pass


class DetectError(Exception):
    """Custom exception for language detection errors."""
    pass


def load_model(low_memory: bool = False, download_proxy: Optional[str] = None) -> fasttext.FastText:
    """
    Load the FastText model based on memory preference.

    :param low_memory: Indicates whether to load a smaller, memory-efficient model
    :param download_proxy: Proxy to use for downloading the large model if necessary
    :return: Loaded FastText model
    :raises LanguageDetectionError: If the model cannot be loaded
    """
    model_dict_key = "low_mem" if low_memory else "high_mem"
    if MODELS[model_dict_key]:
        return MODELS[model_dict_key]

    if low_memory:
        model_path = LOCAL_SMALL_MODEL_PATH
    else:
        model_path = Path(CACHE_DIRECTORY) / "lid.176.bin"
        if not model_path.exists():
            model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            try:
                logger.info(f"Downloading large model from {model_url} to {model_path}")
                download(
                    url=model_url,
                    folder=CACHE_DIRECTORY,
                    filename="lid.176.bin",
                    proxy=download_proxy,
                    retry_max=3,
                    timeout=20
                )
            except Exception as e:
                logger.error(f"Failed to download the large model: {e}")
                raise DetectError("Unable to download the large model due to network issues.")

    try:
        loaded_model = fasttext.load_model(str(model_path))
        MODELS[model_dict_key] = loaded_model
        return loaded_model
    except Exception as e:
        logger.error(f"Failed to load the model '{model_path}': {e}")
        model_type = "local small" if low_memory else "large"
        raise DetectError(f"Unable to load the {model_type} model due to an error.")


def detect(text: str, *,
           low_memory: bool = True,
           model_download_proxy: Optional[str] = None
           ) -> Dict[str, Union[str, float]]:
    """
    Detect the language of a text using FastText.

    :param text: The text for language detection
    :param low_memory: Whether to use a memory-efficient model
    :param model_download_proxy: Download proxy for the model if needed
    :return: A dictionary with detected language and confidence score
    :raises LanguageDetectionError: If detection fails
    """
    model = load_model(low_memory=low_memory, download_proxy=model_download_proxy)
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
                        on_unicode_error: str = "strict"
                        ) -> List[Dict[str, Any]]:
    """
    Detect multiple potential languages and their probabilities in a given text.

    :param text: The text for language detection
    :param low_memory: Whether to use a memory-efficient model
    :param model_download_proxy: Proxy for downloading the model
    :param k: Number of top language predictions to return
    :param threshold: Minimum score threshold for predictions
    :param on_unicode_error: Error handling for Unicode errors
    :return: A list of dictionaries, each containing a language and its confidence score
    :raises LanguageDetectionError: If detection fails
    """
    model = load_model(low_memory=low_memory, download_proxy=model_download_proxy)
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
