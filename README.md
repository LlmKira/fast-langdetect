# fast-langdetect 🚀

[![PyPI version](https://badge.fury.io/py/fast-langdetect.svg)](https://badge.fury.io/py/fast-langdetect)
[![Downloads](https://pepy.tech/badge/fast-langdetect)](https://pepy.tech/project/fast-langdetect)
[![Downloads](https://pepy.tech/badge/fast-langdetect/month)](https://pepy.tech/project/fast-langdetect/)

## Overview

**fast-langdetect** provides ultra-fast and highly accurate language detection based on FastText, a library developed by
Facebook. This package is 80x faster than traditional methods and offers 95% accuracy.

It supports Python versions 3.9 to 3.12.

This project builds upon [zafercavdar/fasttext-langdetect](https://github.com/zafercavdar/fasttext-langdetect#benchmark)
with enhancements in packaging.

For more information on the underlying FastText model, refer to the official
documentation: [FastText Language Identification](https://fasttext.cc/docs/en/language-identification.html).

> [!NOTE]
> This library requires over 200MB of memory to use in low memory mode.

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

For optimal performance and accuracy in language detection, use `detect(text, low_memory=False)` to load the larger
model.

> The model will be downloaded to the `/tmp/fasttext-langdetect` directory upon first use.

### Native API (Recommended)

```python
from fast_langdetect import detect, detect_multilingual

# Single language detection
print(detect("Hello, world!"))
# Output: {'lang': 'en', 'score': 0.1520957201719284}

print(detect("Привет, мир!")["lang"])
# Output: ru

# Multi-language detection
print(detect_multilingual("Hello, world!你好世界!Привет, мир!"))
# Output: [
#     {'lang': 'ru', 'score': 0.39008623361587524},
#     {'lang': 'zh', 'score': 0.18235979974269867},
# ]
```

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

### Splitting Text by Language 🌐

For text splitting based on language, please refer to the [split-lang](https://github.com/DoodleBears/split-lang)
repository.

## Accuracy 🎯

For detailed benchmark results, refer
to [zafercavdar/fasttext-langdetect#benchmark](https://github.com/zafercavdar/fasttext-langdetect#benchmark).