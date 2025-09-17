# fast-langdetect 🚀

[![PyPI version](https://badge.fury.io/py/fast-langdetect.svg)](https://badge.fury.io/py/fast-langdetect)
[![Downloads](https://pepy.tech/badge/fast-langdetect)](https://pepy.tech/project/fast-langdetect)
[![Downloads](https://pepy.tech/badge/fast-langdetect/month)](https://pepy.tech/project/fast-langdetect/)

## Overview

**`fast-langdetect`** is an ultra-fast and highly accurate language detection library based on FastText, a library developed by Facebook. Its incredible speed and accuracy make it 80x faster than conventional methods and deliver up to 95% accuracy.

- Supported Python `3.9` to `3.13`.
- Works offline with the lite model
- No `numpy` required (thanks to @dalf).

> ### Background
> 
> This project builds upon [zafercavdar/fasttext-langdetect](https://github.com/zafercavdar/fasttext-langdetect#benchmark) with enhancements in packaging.
> For more information about the underlying model, see the official FastText documentation: [Language Identification](https://fasttext.cc/docs/en/language-identification.html).

> ### Memory note
> 
> The lite model runs offline and is memory-friendly; the full model is larger and offers higher accuracy.
> 
> Approximate memory usage (RSS after load):
> - Lite: ~45–60 MB
> - Full: ~170–210 MB
> - Auto: tries full first, falls back to lite only on MemoryError.
> 
> Notes:
> - Measurements vary by Python version, OS, allocator, and import graph; treat these as practical ranges.
> - Validate on your system if constrained; see `examples/memory_usage_check.py` (credit: script by github@JackyHe398`).
> - Run memory checks in a clean terminal session. IDEs/REPLs may preload frameworks and inflate peak RSS (ru_maxrss),
>   leading to very large peaks with near-zero deltas.
> 
> Choose the model that best fits your constraints.

## Installation 💻

To install fast-langdetect, you can use either `pip` or `pdm`:

### Using pip

```bash
pip install fast-langdetect
```

### Using pdm

```bash
pdm add fast-langdetect
```

## Usage 🖥️

For higher accuracy, prefer the full model via `detect(text, model='full')`. For robust behavior under memory pressure, use `detect(text, model='auto')` which falls back to the lite model only on MemoryError.

### Prerequisites

- If the sample is too long or too short, the accuracy will be reduced.
- The model will be downloaded to system temporary directory by default. You can customize it by:
  - Setting `FTLANG_CACHE` environment variable
  - Using `LangDetectConfig(cache_dir="your/path")`

### Simple Usage (Recommended)

Call by model explicitly — clear and predictable, and use `k` to get multiple candidates. The function always returns a list of results:

```python
from fast_langdetect import detect

# Lite model (offline, smaller, faster) — never falls back
print(detect("Hello", model='lite', k=1))          # -> [{'lang': 'en', 'score': ...}]

# Full model (downloaded to cache, higher accuracy) — never falls back
print(detect("Hello", model='full', k=1))          # -> [{'lang': 'en', 'score': ...}]

# Auto mode: try full, fallback to lite only on MemoryError
print(detect("Hello", model='auto', k=1))          # -> [{'lang': 'en', 'score': ...}]

# Multilingual: top 3 candidates (always a list)
print(detect("Hello 世界 こんにちは", model='auto', k=3))
```

If you need a custom cache directory, pass `LangDetectConfig`:

```python
from fast_langdetect import LangDetectConfig, detect

cfg = LangDetectConfig(cache_dir="/custom/cache/path")
print(detect("Hello", model='full', config=cfg))

# Set a default model via config and let calls omit model
cfg_lite = LangDetectConfig(model="lite")
print(detect("Hello", config=cfg_lite))          # uses lite by default
print(detect("Bonjour", config=cfg_lite))        # uses lite by default
print(detect("Hello", model='full', config=cfg_lite))  # per-call override to full

```

### Native API (Recommended)

```python
from fast_langdetect import detect, LangDetector, LangDetectConfig

# Simple detection (uses config default if not provided; defaults to 'auto')
print(detect("Hello, world!", k=1))
# Output: [{'lang': 'en', 'score': 0.98}]

# Using full model for better accuracy
print(detect("Hello, world!", model='full', k=1))
# Output: [{'lang': 'en', 'score': 0.99}]

# Custom configuration
config = LangDetectConfig(cache_dir="/custom/cache/path", model="auto")  # Custom cache + default model
detector = LangDetector(config)

# Omit model to use config.model; pass model to override
result = detector.detect("Hello world", k=1)
print(result)  # [{'lang': 'en', 'score': 0.98}]

# Multiline text is handled automatically (newlines are replaced)
multiline_text = "Hello, world!\nThis is a multiline text."
print(detect(multiline_text, k=1))
# Output: [{'lang': 'en', 'score': 0.85}]

# Multi-language detection
results = detect(
    "Hello 世界 こんにちは",
    model='auto',
    k=3               # Return top 3 languages (auto model loading)
)
print(results)
# Output: [
#     {'lang': 'ja', 'score': 0.4}, 
#     {'lang': 'zh', 'score': 0.3}, 
#     {'lang': 'en', 'score': 0.2}
# ]
```

#### Fallback Policy (Keep It Simple)

- Only `MemoryError` triggers fallback (in `model='auto'`): when loading the full model runs out of memory, it falls back to the lite model.
- I/O/network/permission/path/integrity errors raise standard exceptions (e.g., `FileNotFoundError`, `PermissionError`) or library-specific errors where applicable — no silent fallback.
- `model='lite'` and `model='full'` never fallback by design.

#### Errors

- Base error: `FastLangdetectError` (library-specific failures).
- Model loading failures: `ModelLoadError`.
- Standard Python exceptions (e.g., `ValueError`, `TypeError`, `FileNotFoundError`, `MemoryError`) propagate when they are not library-specific.

### Convenient `detect_language` Function

```python
from fast_langdetect import detect_language

# Single language detection
print(detect_language("Hello, world!"))
# Output: EN

print(detect_language("Привет, мир!"))
# Output: RU

print(detect_language("你好，世界！"))
# Output: ZH
```

### Load Custom Models

```python
# Load model from local file
config = LangDetectConfig(custom_model_path="/path/to/your/model.bin")
detector = LangDetector(config)
result = detector.detect("Hello world", model='auto', k=1)
```

### Splitting Text by Language 🌐

For text splitting based on language, please refer to the [split-lang](https://github.com/DoodleBears/split-lang)
repository.


### Input Handling

You can control log verbosity and input normalization via `LangDetectConfig`:

```python
from fast_langdetect import LangDetectConfig, LangDetector

config = LangDetectConfig(
    max_input_length=80,    # default: auto-truncate long inputs for stable results
)
detector = LangDetector(config)
print(detector.detect("Some very long text..."))
```

- Newlines are always replaced with spaces to avoid FastText errors (silent, no log).
- When truncation happens, a WARNING is logged because it may reduce accuracy.
- `max_input_length=80` truncates overly long inputs; set `None` to disable if you prefer no truncation.

### Cache Directory Behavior

- Default cache: if `cache_dir` is not set, models are stored under a system temp-based directory specified by `FTLANG_CACHE` or an internal default. This directory is created automatically when needed.
- User-provided cache_dir: if you set `LangDetectConfig(cache_dir=...)` to a path that does not exist, the library raises `FileNotFoundError` instead of silently creating or using another location. Create the directory yourself if that’s intended.

### Advanced Options (Optional)

The constructor exposes a few advanced knobs (`proxy`, `normalize_input`, `max_input_length`). These are rarely needed for typical usage and can be ignored. Prefer `detect(..., model=...)` unless you know you need them.

### Language Codes → English Names

The detector returns fastText language codes (e.g., `en`, `zh`, `ja`, `pt-br`). To present user-friendly names, you can map codes to English names using a third-party library. Example using `langcodes`:

```python
# pip install langcodes
from langcodes import Language

OVERRIDES = {
    # fastText-specific or variant tags commonly used
    "yue": "Cantonese",
    "wuu": "Wu Chinese",
    "arz": "Egyptian Arabic",
    "ckb": "Central Kurdish",
    "kab": "Kabyle",
    "zh-cn": "Chinese (China)",
    "zh-tw": "Chinese (Taiwan)",
    "pt-br": "Portuguese (Brazil)",
}

def code_to_english_name(code: str) -> str:
    code = code.replace("_", "-").lower()
    if code in OVERRIDES:
        return OVERRIDES[code]
    try:
        # Display name in English; e.g. 'Portuguese (Brazil)'
        return Language.get(code).display_name("en")
    except Exception:
        # Try the base language (e.g., 'pt' from 'pt-br')
        base = code.split("-")[0]
        try:
            return Language.get(base).display_name("en")
        except Exception:
            return code

# Usage
from fast_langdetect import detect
result = detect("Olá mundo", model='full', k=1)
print(code_to_english_name(result[0]["lang"]))  # Portuguese (Brazil) or Portuguese
```

Alternatively, `pycountry` can be used for ISO 639 lookups (install with `pip install pycountry`), combined with a small override dict for non-standard tags like `pt-br`, `zh-cn`, `yue`, etc.

## Benchmark 📊

For detailed benchmark results, refer
to [zafercavdar/fasttext-langdetect#benchmark](https://github.com/zafercavdar/fasttext-langdetect#benchmark).

## References 📚

[1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification

```bibtex
@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}
```

[2] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, FastText.zip: Compressing text classification
models

```bibtex
@article{joulin2016fasttext,
  title={FastText.zip: Compressing text classification models},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1612.03651},
  year={2016}
}
```

## License 📄

- Code: Released under the MIT License (see `LICENSE`).
- Models: This package uses the pre-trained fastText language identification models (`lid.176.ftz` bundled for offline use and `lid.176.bin` downloaded as needed). These models are licensed under the Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0) license.
- Attribution: fastText language identification models by Facebook AI Research. See the fastText docs and license for details:
  - https://fasttext.cc/docs/en/language-identification.html
  - https://creativecommons.org/licenses/by-sa/3.0/
- Note: If you redistribute or modify the model files, you must comply with CC BY-SA 3.0. Inference usage via this library does not change the license of the model files themselves.
