# -*- coding: utf-8 -*-
"""
FastText based language detection module.
"""

import hashlib
import logging
import os
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import fasttext
from robust_downloader import download

logger = logging.getLogger(__name__)

# Use system temporary directory as default cache directory
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "fasttext-langdetect"
CACHE_DIRECTORY = os.getenv("FTLANG_CACHE", str(DEFAULT_CACHE_DIR))
FASTTEXT_LARGE_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)
FASTTEXT_LARGE_MODEL_NAME = "lid.176.bin"
_LOCAL_SMALL_MODEL_PATH = Path(__file__).parent / "resources" / "lid.176.ftz"


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
        self._verifier = ModelVerifier()
        self._downloader = ModelDownloader()

    def load_local(self, model_path: Path, verify_hash: Optional[str] = None) -> Any:
        """Load model from local file."""
        if verify_hash and model_path.exists():
            if not self._verifier.verify(model_path, verify_hash):
                logger.warning(
                    f"fast-langdetect: MD5 verification failed for {model_path}. "
                    "This may affect prediction accuracy."
                )

        if not model_path.exists():
            raise DetectError(f"Model file not found: {model_path}")

        if platform.system() == "Windows":
            return self._load_windows_compatible(model_path)
        return self._load_unix(model_path)

    def load_with_download(self, model_path: Path, proxy: Optional[str] = None) -> Any:
        """Internal method to load model with download if needed."""
        if not model_path.exists():
            self._downloader.download(FASTTEXT_LARGE_MODEL_URL, model_path, proxy)
        return self.load_local(model_path)

    def _load_windows_compatible(self, model_path: Path) -> Any:
        """
        Handle Windows path compatibility issues when loading FastText models.
        
        Attempts multiple strategies in order:
        1. Direct loading if path contains only safe characters
        2. Loading via relative path if possible
        3. Copying to temporary file as last resort
        
        :param model_path: Path to the model file
        :return: Loaded FastText model
        :raises DetectError: If all loading strategies fail
        """
        model_path_str = str(model_path.resolve())

        # Try to load model directly
        try:
            return fasttext.load_model(model_path_str)
        except Exception as e:
            logger.debug(f"fast-langdetect: Load model failed: {e}")

        # Try to load model using relative path
        try:
            cwd = Path.cwd()
            rel_path = os.path.relpath(model_path, cwd)
            return fasttext.load_model(rel_path)
        except Exception as e:
            logger.debug(f"fast-langdetect: Failed to load model using relative path: {e}")

        # Use temporary file as last resort
        logger.debug(f"fast-langdetect: Using temporary file to load model: {model_path}")
        tmp_path = None
        try:
            # Use NamedTemporaryFile to create a temporary file
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.bin')
            os.close(tmp_fd)  # Close file descriptor

            # Copy model file to temporary location
            shutil.copy2(model_path, tmp_path)
            return fasttext.load_model(tmp_path)
        except Exception as e:
            raise DetectError(f"Failed to load model using temporary file: {e}")
        finally:
            # Clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except (OSError, PermissionError) as e:
                    logger.warning(f"fast-langdetect: Failed to delete temporary file {tmp_path}: {e}")
                    # Plan to delete on next reboot on Windows
                    if platform.system() == "Windows":
                        try:
                            import _winapi
                            _winapi.MoveFileEx(tmp_path, None, _winapi.MOVEFILE_DELAY_UNTIL_REBOOT)
                        except (ImportError, AttributeError, OSError) as we:
                            logger.warning(f"fast-langdetect: Failed to schedule file deletion: {we}")

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
    :param custom_model_path: Path to custom model file (if using own model)
    :param proxy: HTTP proxy for downloads
    :param allow_fallback: Whether to fallback to small model
    :param disable_verify: Whether to disable MD5 verification
    :param normalize_input: Whether to normalize input text (e.g. lowercase for uppercase text)
    """

    def __init__(
            self,
            cache_dir: Optional[str] = None,
            custom_model_path: Optional[str] = None,
            proxy: Optional[str] = None,
            allow_fallback: bool = True,
            disable_verify: bool = False,
            verify_hash: Optional[str] = None,
            normalize_input: bool = True,
    ):
        self.cache_dir = cache_dir or CACHE_DIRECTORY
        self.custom_model_path = custom_model_path
        self.proxy = proxy
        self.allow_fallback = allow_fallback
        # Only verify large model
        self.disable_verify = disable_verify
        self.verify_hash = verify_hash
        self.normalize_input = normalize_input
        if self.custom_model_path and not Path(self.custom_model_path).exists():
            raise FileNotFoundError(f"fast-langdetect: Target model file not found: {self.custom_model_path}")


class LangDetector:
    """Language detector using FastText models."""
    VERIFY_FASTTEXT_LARGE_MODEL = "01810bc59c6a3d2b79c79e6336612f65"

    def __init__(self, config: Optional[LangDetectConfig] = None):
        """
        Initialize language detector.

        :param config: Optional configuration for the detector
        """
        self._models = {}
        self.config = config or LangDetectConfig()
        self._model_loader = ModelLoader()

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """
        Check text for newline characters and length.

        :param text: Input text
        :return: Processed text
        """
        if len(text) > 100:
            logger.warning(
                "fast-langdetect: Text may be too long. "
                "Consider passing only a single sentence for accurate prediction."
            )
        if "\n" in text:
            logger.warning(
                "fast-langdetect: Newline characters will be removed. "
                "Input should not contain newline characters. or FastText will raise an error."
            )
            text = text.replace("\n", " ")
        return text

    @staticmethod
    def _normalize_text(text: str, should_normalize: bool = False) -> str:
        """
        Normalize text based on configuration.
        
        Currently, handles:
        - Removing newline characters for better prediction
        - Lowercasing uppercase text to prevent misdetection as Japanese
        
        :param text: Input text
        :param should_normalize: Whether normalization should be applied
        :return: Normalized text
        """
        # If not normalization is needed, return the processed text
        if not should_normalize:
            return text

        # Check if text is all uppercase or mostly uppercase
        # https://github.com/LlmKira/fast-langdetect/issues/14
        if text.isupper() or (
                len(re.findall(r'[A-Z]', text)) > 0.8 * len(re.findall(r'[A-Za-z]', text))
                and len(text) > 5
        ):
            return text.lower()

        return text

    def _get_model(self, low_memory: bool = True) -> Any:
        """Get or load appropriate model."""
        cache_key = "low_memory" if low_memory else "high_memory"
        if model := self._models.get(cache_key):
            return model

        try:
            if self.config.custom_model_path is not None:
                # Load Custom Model
                if self.config.disable_verify:
                    self.config.verify_hash = None
                model = self._model_loader.load_local(Path(self.config.custom_model_path))
            elif low_memory is True:
                self.config.verify_hash = None
                # Load Small Model
                model = self._model_loader.load_local(_LOCAL_SMALL_MODEL_PATH)
            else:
                if self.config.verify_hash is None and not self.config.disable_verify:
                    self.config.verify_hash = self.VERIFY_FASTTEXT_LARGE_MODEL
                # Download and Load Large Model
                model_path = Path(self.config.cache_dir) / FASTTEXT_LARGE_MODEL_NAME
                model = self._model_loader.load_with_download(
                    model_path,
                    self.config.proxy,
                )
            self._models[cache_key] = model
            return model
        except Exception as e:
            if low_memory is not True and self.config.allow_fallback:
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
        text = self._preprocess_text(text)
        normalized_text = self._normalize_text(text, self.config.normalize_input)
        try:
            labels, scores = model.predict(normalized_text)
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
        text = self._preprocess_text(text)
        normalized_text = self._normalize_text(text, self.config.normalize_input)
        try:
            labels, scores = model.predict(normalized_text, k=k, threshold=threshold)
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
        config: Optional[LangDetectConfig] = None,
) -> Dict[str, Union[str, float]]:
    """
    Simple interface for language detection.
    
    Too long or too short text will effect the accuracy of the prediction.

    :param text: Input text without newline characters
    :param low_memory: Whether to use memory-efficient model
    :param model_download_proxy: [DEPRECATED] Optional proxy for model download
    :param use_strict_mode: [DEPRECATED] Disable fallback to small model
    :param config: Optional LangDetectConfig object for advanced configuration

    :return: Dictionary with language and confidence score
    """
    # Provide config
    if config is not None:
        detector = LangDetector(config)
        return detector.detect(text, low_memory=low_memory)

    # Check if any custom parameters are provided
    has_custom_params = any([
        model_download_proxy is not None,
        use_strict_mode,
    ])
    if has_custom_params:
        # Show warning if using individual parameters
        logger.warning(
            "fast-langdetect: Using individual parameters is deprecated. "
            "Consider using LangDetectConfig for better configuration management. "
            "Will be removed in next major release. see https://github.com/LlmKira/fast-langdetect/pull/16"
        )
        custom_config = LangDetectConfig(
            proxy=model_download_proxy,
            allow_fallback=not use_strict_mode,
        )
        detector = LangDetector(custom_config)
        return detector.detect(text, low_memory=low_memory)

    # Use default detector
    return _default_detector.detect(text, low_memory=low_memory)


def detect_multilingual(
        text: str,
        *,
        low_memory: bool = False,
        model_download_proxy: Optional[str] = None,
        k: int = 5,
        threshold: float = 0.0,
        use_strict_mode: bool = False,
        config: Optional[LangDetectConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Simple interface for multi-language detection.

    Too long or too short text will effect the accuracy of the prediction.

    :param text: Input text without newline characters
    :param low_memory: Whether to use memory-efficient model
    :param k: Number of top languages to return
    :param threshold: Minimum confidence threshold
    :param model_download_proxy: [DEPRECATED] Optional proxy for model download
    :param use_strict_mode: [DEPRECATED] Disable fallback to small model
    :param config: Optional LangDetectConfig object for advanced configuration

    :return: List of dictionaries with languages and scores
    """
    # Use provided config or create new config
    if config is not None:
        detector = LangDetector(config)
        return detector.detect_multilingual(
            text, low_memory=low_memory, k=k, threshold=threshold
        )

    # Check if any custom parameters are provided
    has_custom_params = any([
        model_download_proxy is not None,
        use_strict_mode,
    ])
    if has_custom_params:
        # Show warning if using individual parameters
        logger.warning(
            "fast-langdetect: Using individual parameters is deprecated. "
            "Consider using LangDetectConfig for better configuration management. "
            "Will be removed in next major release. see https://github.com/LlmKira/fast-langdetect/pull/16"
        )
        custom_config = LangDetectConfig(
            proxy=model_download_proxy,
            allow_fallback=not use_strict_mode,
        )
        detector = LangDetector(custom_config)
        return detector.detect_multilingual(
            text, low_memory=low_memory, k=k, threshold=threshold
        )

    # Use default detector
    return _default_detector.detect_multilingual(
        text, low_memory=low_memory, k=k, threshold=threshold
    )
