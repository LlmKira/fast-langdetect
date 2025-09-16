# -*- coding: utf-8 -*-
"""
FastText based language detection module.
"""

import logging
import os
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal

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


class FastLangdetectError(Exception):
    """Base exception for library-specific failures."""
    pass


class ModelLoadError(FastLangdetectError):
    """Raised when a FastText model fails to load."""
    pass


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
            FastLangdetectError: If download fails
            FileNotFoundError: If a user-provided cache directory does not exist
        """
        if save_path.exists():
            logger.info(f"fast-langdetect: Model exists at {save_path}")
            return

        logger.info(f"fast-langdetect: Downloading model from {url}")
        # Ensure target directory handling is consistent across OSes
        parent_dir = save_path.parent
        default_cache_dir = Path(CACHE_DIRECTORY)
        if not parent_dir.exists():
            # Only auto-create when using the library's default cache dir
            if parent_dir == default_cache_dir:
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise FastLangdetectError(
                        f"fast-langdetect: Cannot create cache directory {parent_dir}: {e}"
                    ) from e
            else:
                # For user-specified cache_dir, do not fallback; raise
                raise FileNotFoundError(f"fast-langdetect: Cache directory not found: {parent_dir}")
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
            # Download failures are library-specific
            raise FastLangdetectError(f"fast-langdetect: Download failed: {e}") from e


class ModelLoader:
    """Model loading and caching handler."""

    def __init__(self):
        self._downloader = ModelDownloader()

    def load_local(self, model_path: Path) -> Any:
        """Load model from local file."""
        if not model_path.exists():
            # Missing path is a standard I/O error
            raise FileNotFoundError(f"Model file not found: {model_path}")

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
        :raises ModelLoadError: If all loading strategies fail
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
            raise ModelLoadError(f"Failed to load model using temporary file: {e}") from e
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
        except MemoryError as e:
            # Let MemoryError propagate up to be handled by _get_model
            raise e
        except Exception as e:
            raise ModelLoadError(f"fast-langdetect: Failed to load model: {e}") from e


class LangDetectConfig:
    """
    Configuration for language detection.

    :param cache_dir: Directory for storing downloaded models
    :param custom_model_path: Path to custom model file (if using own model)
    :param proxy: HTTP proxy for downloads
    :param normalize_input: Whether to normalize input text (e.g. lowercase for uppercase text)
    :param max_input_length: If set, truncate input to this many characters (always debug-log the change)
    :param model: Default model selection ('auto' | 'full' | 'lite') used when detect() is called without a model
    """

    def __init__(
            self,
            cache_dir: Optional[str] = None,
            custom_model_path: Optional[str] = None,
            proxy: Optional[str] = None,
            normalize_input: bool = True,
            max_input_length: Optional[int] = 80,
            model: Literal["lite", "full", "auto"] = "auto",
    ):
        self.cache_dir = cache_dir or CACHE_DIRECTORY
        self.custom_model_path = custom_model_path
        self.proxy = proxy
        self.normalize_input = normalize_input
        # Input handling
        self.max_input_length = max_input_length
        self.model: Literal["lite", "full", "auto"] = model
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

    def _preprocess_text(self, text: str) -> str:
        """
        Check text for newline characters and length.

        :param text: Input text
        :return: Processed text
        """
        # Always replace newline characters to avoid FastText errors (silent)
        if "\n" in text:
            text = text.replace("\n", " ")

        # Auto-truncate overly long input if configured
        if self.config.max_input_length is not None and len(text) > self.config.max_input_length:
            logger.info(
                f"fast-langdetect: Truncating input from {len(text)} to {self.config.max_input_length} characters; may reduce accuracy."
            )
            text = text[: self.config.max_input_length]
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

    def _get_model(self, low_memory: bool = True, *, fallback_on_memory_error: bool = False) -> Any:
        """Get or load appropriate model.

        :param low_memory: choose small (True) or large (False) model
        :param fallback_on_memory_error: override whether to fallback on MemoryError
        """
        cache_key = "low_memory" if low_memory else "high_memory"
        if model := self._models.get(cache_key):
            return model

        try:
            if self.config.custom_model_path is not None:
                # Load Custom Model
                model = self._model_loader.load_local(Path(self.config.custom_model_path))
            elif low_memory is True:
                # Load Small Model
                model = self._model_loader.load_local(_LOCAL_SMALL_MODEL_PATH)
            else:
                # Download and Load Large Model
                model_path = Path(self.config.cache_dir) / FASTTEXT_LARGE_MODEL_NAME
                model = self._model_loader.load_with_download(
                    model_path,
                    self.config.proxy,
                )
            self._models[cache_key] = model
            return model
        except MemoryError as e:
            if (not low_memory) and fallback_on_memory_error:
                logger.info("fast-langdetect: Falling back to low-memory model...")
                return self._get_model(low_memory=True, fallback_on_memory_error=False)
            # Preserve original MemoryError and traceback
            raise

    def detect(
            self,
            text: str,
            *,
            model: Optional[Literal["lite", "full", "auto"]] = None,
            k: int = 1,
            threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Detect language candidates. Always returns a list of results.

        :param text: Input text
        :param model: 'lite' | 'full' | 'auto' (auto falls back on MemoryError)
        :param k: Number of top languages to return
        :param threshold: Minimum confidence threshold
        :raises FastLangdetectError: For library-specific failures (e.g., invalid model)
        :raises Exception: Standard Python exceptions propagate, such as MemoryError, FileNotFoundError
        """
        # Determine model to use (config default if not provided)
        sel_model: Literal["lite", "full", "auto"]
        if model is None:
            sel_model = self.config.model
        else:
            if model not in {"lite", "full", "auto"}:  # type: ignore[comparison-overlap]
                raise FastLangdetectError(f"Invalid model: {model}")
            sel_model = model

        # Select model backend
        if sel_model == "lite":
            ft_model = self._get_model(low_memory=True, fallback_on_memory_error=False)
        elif sel_model == "full":
            ft_model = self._get_model(low_memory=False, fallback_on_memory_error=False)
        else:
            ft_model = self._get_model(low_memory=False, fallback_on_memory_error=True)

        text = self._preprocess_text(text)
        normalized_text = self._normalize_text(text, self.config.normalize_input)
        labels, scores = ft_model.predict(normalized_text, k=k, threshold=threshold)
        results = [
            {
                "lang": label.replace("__label__", ""),
                "score": min(float(score), 1.0),
            }
            for label, score in zip(labels, scores)
        ]
        return sorted(results, key=lambda x: x["score"], reverse=True)


# Global instance for simple usage
_default_detector = LangDetector()


def detect(
    text: str,
    *,
    model: Optional[Literal["lite", "full", "auto"]] = None,
    k: int = 1,
    threshold: float = 0.0,
    config: Optional[LangDetectConfig] = None,
) -> List[Dict[str, Union[str, float]]]:
    detector = LangDetector(config) if config is not None else _default_detector
    return detector.detect(text, model=model, k=k, threshold=threshold)
