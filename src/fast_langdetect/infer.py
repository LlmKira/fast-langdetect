# -*- coding: utf-8 -*-
"""
FastText based language detection module.
"""

import hashlib
import logging
import os
import tempfile
import platform
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import fasttext
from robust_downloader import download

logger = logging.getLogger(__name__)

# 使用系统临时目录作为默认位置
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "fasttext-langdetect"
CACHE_DIRECTORY = os.getenv("FTLANG_CACHE", str(DEFAULT_CACHE_DIR))

LOCAL_SMALL_MODEL_PATH = Path(__file__).parent / "resources" / "lid.176.ftz"
FASTTEXT_LARGE_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)
FASTTEXT_LARGE_MODEL_NAME = "lid.176.bin"
VERIFY_FASTTEXT_LARGE_MODEL = "01810bc59c6a3d2b79c79e6336612f65"


class DetectError(Exception):
    """Base exception for language detection errors."""

    pass


class ModelVerifier:
    """Model file verification utilities."""

    @staticmethod
    def calculate_md5(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
        """
        Calculate MD5 hash of a file.

        :param file_path: Path to the file
        :param chunk_size: Size of chunks to read

        :return: MD5 hash string
        """
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def verify(file_path: Union[str, Path], expected_md5: str) -> bool:
        """
        Verify file integrity using MD5 hash.

        :param file_path: Path to the file
        :param expected_md5: Expected MD5 hash

        :return: True if hash matches, False otherwise
        """
        return ModelVerifier.calculate_md5(file_path) == expected_md5


class ModelDownloader:
    """Model download handler."""

    @staticmethod
    def download(url: str, save_path: Path, proxy: Optional[str] = None) -> None:
        """
        Download model file if not exists.

        :param url: URL to download from
        :param save_path: Path to save the model
        :param proxy: Optional proxy URL

        :raises:
            DetectError: If download fails
        """
        if save_path.exists():
            logger.info(f"fast-langdetect: Model exists at {save_path}")
            return

        logger.info(f"fast-langdetect: Downloading model from {url}")
        try:
            download(
                url=url,
                folder=str(save_path.parent),
                filename=save_path.name,
                proxy=proxy,
                retry_max=2,
                sleep_max=5,
                timeout=7,
            )
        except Exception as e:
            raise DetectError(f"fast-langdetect: Download failed: {e}")


class ModelLoader:
    """Model loading and caching handler."""

    def __init__(self):
        """Initialize model loader with verifier and downloader."""
        self._models = {}
        self._verifier = ModelVerifier()
        self._downloader = ModelDownloader()

    def get_cached(self, key: str) -> Optional[Any]:
        """
        Get model from cache.
        
        :param key: Cache key for the model
        :return: Cached model if exists, None otherwise
        """
        return self._models.get(key)

    def cache(self, key: str, model: Any) -> None:
        """
        Cache a model instance.

        :param key: Cache key for the model
        :param model: Model instance to cache
        """
        self._models[key] = model

    def load(
        self,
        model_path: Path,
        model_url: Optional[str] = None,
        proxy: Optional[str] = None,
        verify_hash: Optional[str] = None,
    ) -> Any:
        """
        Load model with optional download and verification.

        :param model_path: Path to the model file
        :param model_url: URL to download model if not exists
        :param proxy: Optional proxy for download
        :param verify_hash: Optional hash for verification

        :return: Loaded model instance

        :raises:
            DetectError: If model loading fails
        """
        if verify_hash and model_path.exists():
            if not self._verifier.verify(model_path, verify_hash):
                logger.warning(f"fast-langdetect: MD5 verification failed for {model_path}")

        if not model_path.exists() and model_url:
            self._downloader.download(model_url, model_path, proxy)

        if platform.system() == 'Windows':
            try:
                return self._load_windows_compatible(model_path)
            except Exception as e:
                raise DetectError(f"Failed to load model on Windows: {e}")
        else:
            return self._load_unix(model_path)

    def _load_windows_compatible(self, model_path: Path) -> Any:
        """Handle Windows path compatibility issues."""
        if re.match(r'^[A-Za-z0-9_/\\:.]*$', str(model_path)):
            return fasttext.load_model(str(model_path))
            
        # Create a temporary file to handle special characters in the path
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copy2(model_path, tmp.name)
            try:
                model = fasttext.load_model(tmp.name)
            finally:
                os.unlink(tmp.name)
        return model

    def _load_unix(self, model_path: Path) -> Any:
        """Load model on Unix-like systems."""
        try:
            return fasttext.load_model(str(model_path))
        except Exception as e:
            raise DetectError(f"fast-langdetect: Failed to load model: {e}")


class LangDetectConfig:
    """
    Configuration for language detection.

    :param cache_dir: Directory for storing downloaded models
    :param model_url: URL for downloading large model
    :param small_model_path: Path to small model file
    :param proxy: HTTP proxy for downloads
    :param allow_fallback: Whether to fallback to small model
    :param verify_hash: Hash for model verification
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        model_url: Optional[str] = None,
        small_model_path: Optional[str] = None,
        proxy: Optional[str] = None,
        allow_fallback: bool = True,
        verify_hash: Optional[str] = None,
    ):
        self.cache_dir = cache_dir or CACHE_DIRECTORY
        self.model_url = model_url or FASTTEXT_LARGE_MODEL_URL
        self.small_model_path = small_model_path or str(LOCAL_SMALL_MODEL_PATH)
        self.proxy = proxy
        self.allow_fallback = allow_fallback
        self.verify_hash = verify_hash or VERIFY_FASTTEXT_LARGE_MODEL


class LangDetector:
    """Language detector using FastText models."""

    def __init__(self, config: Optional[LangDetectConfig] = None):
        """
        Initialize language detector.

        :param config: Optional configuration for the detector
        """
        self.config = config or LangDetectConfig()
        self._model_loader = ModelLoader()

    def _get_model(self, low_memory: bool = True) -> Any:
        """
        Get or load appropriate model.

        :param low_memory: Whether to use memory-efficient model

        :return: Loaded model instance

        :raises:
            DetectError: If model loading fails
        """
        cache_key = "low_memory" if low_memory else "high_memory"
        if model := self._model_loader.get_cached(cache_key):
            return model

        try:
            if low_memory:
                model = self._model_loader.load(Path(self.config.small_model_path))
            else:
                model_path = Path(self.config.cache_dir) / FASTTEXT_LARGE_MODEL_NAME
                model = self._model_loader.load(
                    model_path,
                    self.config.model_url,
                    self.config.proxy,
                    self.config.verify_hash,
                )
            self._model_loader.cache(cache_key, model)
            return model
        except Exception as e:
            if not low_memory and self.config.allow_fallback:
                logger.info("fast-langdetect: Falling back to low-memory model...")
                return self._get_model(low_memory=True)
            raise DetectError("Failed to load model") from e

    def detect(
        self, text: str, low_memory: bool = True
    ) -> Dict[str, Union[str, float]]:
        """
        Detect primary language of text.

        :param text: Input text
        :param low_memory: Whether to use memory-efficient model

        :return: Dictionary with language and confidence score

        :raises:
            DetectError: If detection fails
        """
        model = self._get_model(low_memory)
        try:
            labels, scores = model.predict(text)
            return {
                "lang": labels[0].replace("__label__", ""),
                "score": min(float(scores[0]), 1.0),
            }
        except Exception as e:
            logger.error(f"fast-langdetect: Language detection error: {e}")
            raise DetectError("Language detection failed") from e

    def detect_multilingual(
        self,
        text: str,
        low_memory: bool = False,
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Detect multiple possible languages in text.

        :param text: Input text
        :param low_memory: Whether to use memory-efficient model
        :param k: Number of top languages to return
        :param threshold: Minimum confidence threshold

        :return: List of dictionaries with languages and scores

        :raises:
            DetectError: If detection fails
        """
        model = self._get_model(low_memory)
        try:
            labels, scores = model.predict(text, k=k, threshold=threshold)
            results = [
                {
                    "lang": label.replace("__label__", ""),
                    "score": min(float(score), 1.0),
                }
                for label, score in zip(labels, scores)
            ]
            return sorted(results, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            logger.error(f"fast-langdetect: Multilingual detection error: {e}")
            raise DetectError("Multilingual detection failed.")


# Global instance for simple usage
_default_detector = LangDetector()


def detect(
    text: str,
    *,
    low_memory: bool = True,
    model_download_proxy: Optional[str] = None,
    use_strict_mode: bool = False,
) -> Dict[str, Union[str, float]]:
    """
    Simple interface for language detection.

    Before passing a text to this function, you remove all the newline characters.

    Too long or too short text will effect the accuracy of the prediction.

    :param text: Input text without newline characters
    :param low_memory: Whether to use memory-efficient model
    :param model_download_proxy: Optional proxy for model download
    :param use_strict_mode: Disable fallback to small model

    :return: Dictionary with language and confidence score
    """
    if "\n" in text or len(text) > 1000:
        logger.warning(
            "fast-langdetect: Text contains newline characters or is too long. "
            "You should only pass a single sentence for accurate prediction."
        )
    if model_download_proxy or use_strict_mode:
        config = LangDetectConfig(
            proxy=model_download_proxy, allow_fallback=not use_strict_mode
        )
        detector = LangDetector(config)
        return detector.detect(text, low_memory=low_memory)
    return _default_detector.detect(text, low_memory=low_memory)


def detect_multilingual(
    text: str,
    *,
    low_memory: bool = False,
    model_download_proxy: Optional[str] = None,
    k: int = 5,
    threshold: float = 0.0,
    use_strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Simple interface for multi-language detection.

    Before passing a text to this function, you remove all the newline characters.

    Too long or too short text will effect the accuracy of the prediction.

    :param text: Input text without newline characters
    :param low_memory: Whether to use memory-efficient model
    :param model_download_proxy: Optional proxy for model download
    :param k: Number of top languages to return
    :param threshold: Minimum confidence threshold
    :param use_strict_mode: Disable fallback to small model

    :return: List of dictionaries with languages and scores
    """
    if "\n" in text or len(text) > 100:
        logger.warning(
            "fast-langdetect: Text contains newline characters or is too long. "
            "You should only pass a single sentence for accurate prediction."
        )
    if model_download_proxy or use_strict_mode:
        config = LangDetectConfig(
            proxy=model_download_proxy, allow_fallback=not use_strict_mode
        )
        detector = LangDetector(config)
        return detector.detect_multilingual(
            text, low_memory=low_memory, k=k, threshold=threshold
        )
    return _default_detector.detect_multilingual(
        text, low_memory=low_memory, k=k, threshold=threshold
    )
