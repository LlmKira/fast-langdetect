# fast-langdetect 🚀

[![PyPI version](https://badge.fury.io/py/fast-langdetect.svg)](https://badge.fury.io/py/fast-langdetect)
[![Downloads](https://pepy.tech/badge/fast-langdetect)](https://pepy.tech/project/fast-langdetect)
[![Downloads](https://pepy.tech/badge/fast-langdetect/month)](https://pepy.tech/project/fast-langdetect/)

## Overview

**`fast-langdetect`** is an ultra-fast and highly accurate language detection library based on FastText, a library developed by Facebook. Its incredible speed and accuracy make it 80x faster than conventional methods and deliver up to 95% accuracy.

- Supported Python `3.9` to `3.13`.
- Works offline with the lite model.
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

- If the sample is too long (generally over 80 characters) or too short, accuracy will be reduced.
- The model downloads to the system temporary directory by default. You can customize it by:
  - Setting `FTLANG_CACHE` environment variable
  - Using `LangDetectConfig(cache_dir="your/path")`

### Quick Start

```python
from fast_langdetect import detect

print(detect("Hello, world!", model="auto", k=1))
print(detect("Hello 世界 こんにちは", model="auto", k=3))
```

`detect` always returns a list of candidates ordered by score. Use `model="full"` for the best accuracy or `model="lite"` for an offline-only workflow.

### Custom Configuration

```python
from fast_langdetect import LangDetectConfig, LangDetector

config = LangDetectConfig(cache_dir="/custom/cache", model="lite")
detector = LangDetector(config)
print(detector.detect("Bonjour", k=1))
print(detector.detect("Hola", model="full", k=1))
```

Each `LangDetector` instance maintains its own in-memory model cache. Once loaded, models are reused for subsequent calls within the same instance. The global `detect()` function uses a shared default detector, so it also benefits from automatic caching.

Create a custom `LangDetector` instance when you need specific configuration (custom cache directory, input limits, etc.) or isolated model management.

#### 🌵 Fallback Policy 

Keep It Simple!

- Only `MemoryError` triggers fallback (in `model='auto'`): when loading the full model runs out of memory, it falls back to the lite model.
- I/O/network/permission/path/integrity errors raise standard exceptions (e.g., `FileNotFoundError`, `PermissionError`) or library-specific errors where applicable — no silent fallback.
- `model='lite'` and `model='full'` never fallback by design.

#### Errors

- Base error: `FastLangdetectError` (library-specific failures).
- Model loading failures: `ModelLoadError`.
- Standard Python exceptions (e.g., `ValueError`, `TypeError`, `FileNotFoundError`, `MemoryError`) propagate when they are not library-specific.

### Splitting Text by Language 🌐

For text splitting based on language, please refer to the [split-lang](https://github.com/DoodleBears/split-lang)
repository.


### Input Handling

You can control log verbosity and input normalization via `LangDetectConfig`:

```python
from fast_langdetect import LangDetectConfig, LangDetector

config = LangDetectConfig(max_input_length=200)
detector = LangDetector(config)
print(detector.detect("Some very long text..." * 5))
```

- Newlines are always replaced with spaces to avoid FastText errors (silent, no log).
- When truncation happens, a WARNING is logged because it may reduce accuracy.
- The default `max_input_length` is 80 characters (optimal for accuracy); increase it if you need longer samples, or set `None` to disable truncation entirely.

### Cache Directory Behavior

- Default cache: if `cache_dir` is not set, models are stored under a system temp-based directory specified by `FTLANG_CACHE` or an internal default. This directory is created automatically when needed.
- User-provided cache_dir: if you set `LangDetectConfig(cache_dir=...)` to a path that does not exist, the library raises `FileNotFoundError` instead of silently creating or using another location. Create the directory yourself if that’s intended.

### Advanced Options (Optional)

The constructor exposes a few advanced knobs (`proxy`, `normalize_input`, `max_input_length`). These are rarely needed for typical usage and can be ignored. Prefer `detect(..., model=...)` unless you know you need them.

### Language Codes → English Names

fastText reports BCP-47 style tags such as `en`, `zh-cn`, `pt-br`, `yue`. The detector keeps those codes so you can decide how to display them. Choose the approach that fits your product:

- **Small, fixed list?** Maintain a hand-written mapping and fall back to the raw code for anything unexpected.

```python
FASTTEXT_DISPLAY_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese (China)",
    "zh-tw": "Chinese (Taiwan)",
    "pt": "Portuguese",
    "pt-br": "Portuguese (Brazil)",
    "yue": "Cantonese",
    "wuu": "Wu Chinese",
    "arz": "Egyptian Arabic",
    "ckb": "Central Kurdish",
    "kab": "Kabyle",
}

def code_to_display_name(code: str) -> str:
    return FASTTEXT_DISPLAY_NAMES.get(code.lower(), code)

print(code_to_display_name("pt-br"))
print(code_to_display_name("de"))
```

- **Need coverage for all 176 fastText languages?** Use a language database library that understands subtags and scripts. Two popular libraries are `langcodes` and `pycountry`.

```python
# pip install langcodes
from langcodes import Language

LANG_OVERRIDES = {
    "pt-br": "Portuguese (Brazil)",
    "zh-cn": "Chinese (China)",
    "zh-tw": "Chinese (Taiwan)",
    "yue": "Cantonese",
}

def fasttext_to_name(code: str) -> str:
    normalized = code.replace("_", "-").lower()
    if normalized in LANG_OVERRIDES:
        return LANG_OVERRIDES[normalized]
    try:
        return Language.get(normalized).display_name("en")
    except Exception:
        base = normalized.split("-")[0]
        try:
            return Language.get(base).display_name("en")
        except Exception:
            return code

from fast_langdetect import detect
result = detect("Olá mundo", model="full", k=1)
print(fasttext_to_name(result[0]["lang"]))
```

`pycountry` works similarly (`pip install pycountry`). Use `pycountry.languages.lookup("pt")` for fuzzy matching or `pycountry.languages.get(alpha_2="pt")` for exact lookups, and pair it with a small override dictionary for non-standard tags such as `pt-br`, `zh-cn`, or dialect codes like `yue`.

```python
# pip install pycountry
import pycountry

FASTTEXT_OVERRIDES = {
    "pt-br": "Portuguese (Brazil)",
    "zh-cn": "Chinese (China)",
    "zh-tw": "Chinese (Taiwan)",
    "yue": "Cantonese",
}

def fasttext_to_name_pycountry(code: str) -> str:
    normalized = code.replace("_", "-").lower()
    if normalized in FASTTEXT_OVERRIDES:
        return FASTTEXT_OVERRIDES[normalized]
    try:
        return pycountry.languages.lookup(normalized).name
    except LookupError:
        base = normalized.split("-")[0]
        try:
            return pycountry.languages.lookup(base).name
        except LookupError:
            return code

from fast_langdetect import detect
result = detect("Olá mundo", model="full", k=1)
print(fasttext_to_name_pycountry(result[0]["lang"]))
```

### Load Custom Models

```python
from importlib import resources
from fast_langdetect import LangDetectConfig, LangDetector

with resources.path("fast_langdetect.resources", "lid.176.ftz") as model_path:
    config = LangDetectConfig(custom_model_path=str(model_path))
    detector = LangDetector(config)
    print(detector.detect("Hello world", k=1))
```

When using a custom model via `custom_model_path`, the `model` parameter in `detect()` calls is ignored since your custom model file is always loaded directly. The `model="lite"`, `model="full"`, and `model="auto"` parameters only apply when using the built-in models.

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
