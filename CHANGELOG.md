# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-10-17

- Packaging: On Python 3.14+, the `fasttext` backend is now `fasttext-predict-v2==1.1.0`, which ships 3.14 wheels, so installs no longer need a C++ compiler. Python 3.9–3.13 keep `fasttext-predict`. (#34)
- Packaging: Ship a `py.typed` marker so type checkers use the inline type hints. (#25)
- Packaging: Drop the direct `requests` dependency; it is still installed through `robust-downloader`.
- Reliability: Full-model downloads are checksum-verified and published atomically under a single-writer lock, so concurrent first use can no longer corrupt the shared cache. Corrupt cached models are replaced, and a lock left by a killed process is taken over after 600 s.
- Fix: `model="auto"` now falls back to the lite model on Windows when loading the full model raises `MemoryError`. Previously the error was wrapped in `ModelLoadError`.
- Fix: When the deprecated `fasttext` PyPI package overwrites the backend and NumPy 2 is installed, `detect()` raises `ModelLoadError` with repair instructions instead of an opaque NumPy `ValueError`. With NumPy 1 it keeps working. (#31)
- Deprecation: `detect(text, config=...)` reloads the model on every call and now emits a `DeprecationWarning`. Create `LangDetector(config)` once and reuse it. The parameter will be removed in 2.0.
- Documentation: Correct the input-handling docs (truncation is logged at INFO) and explain per-language sampling. (#24)

## [1.0.1] - 2026-05-06

- Packaging: Declare Python 3.14 support and publish package metadata classifiers for Python 3.9 through 3.14.
- CI: Add Python 3.14 to the test matrix.

## [0.4.0] - 2025-09-15

- Behavior: Always replace newline characters in input to prevent FastText errors. This adjustment is logged at DEBUG level only.
- Default input truncation: Truncate inputs to 80 characters by default for stable predictions. Configurable via `LangDetectConfig(max_input_length=...)`; set `None` to disable.
- Simplified config: Removed previously proposed `verbose` and `replace_newlines` options; newline replacement is unconditional and logging of adjustments is controlled by global logger level.
- Logging: Deprecated-parameter messages lowered from WARNING to INFO to reduce noise.
- Documentation: README now includes language code → name mapping guidance and an explicit model license note (CC BY-SA 3.0) alongside MIT for code.
