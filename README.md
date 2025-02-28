# fast-langdetect 🚀

[![PyPI version](https://badge.fury.io/py/fast-langdetect.svg)](https://badge.fury.io/py/fast-langdetect)
[![Downloads](https://pepy.tech/badge/fast-langdetect)](https://pepy.tech/project/fast-langdetect)
[![Downloads](https://pepy.tech/badge/fast-langdetect/month)](https://pepy.tech/project/fast-langdetect/)

## Overview

**`fast-langdetect`** is an ultra-fast and highly accurate language detection library based on FastText, a library developed by Facebook. Its incredible speed and accuracy make it 80x faster than conventional methods and deliver up to 95% accuracy.

- Supported Python `3.9` to `3.13`.
- Works offline  in low memory mode
- No `numpy` required (thanks to @dalf).

> ### Background
> 
> This project builds upon [zafercavdar/fasttext-langdetect](https://github.com/zafercavdar/fasttext-langdetect#benchmark) with enhancements in packaging.
> For more information about the underlying model, see the official FastText documentation: [Language Identification](https://fasttext.cc/docs/en/language-identification.html).

> ### Possible memory usage
> 
> *This library requires at least **200MB memory** in low-memory mode.*

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

In scenarios **where accuracy is important**, you should not rely on the detection results of small models, use `low_memory=False` to download larger models!

### Prerequisites

- The "\n" character in the argument string must be removed before calling the function.
- If the sample is too long or too short, the accuracy will be reduced.
- The model will be downloaded to system temporary directory by default. You can customize it by:
  - Setting `FTLANG_CACHE` environment variable
  - Using `LangDetectConfig(cache_dir="your/path")`

### Native API (Recommended)

```python
from fast_langdetect import detect, detect_multilingual, LangDetector, LangDetectConfig, DetectError

# Simple detection
print(detect("Hello, world!"))
# Output: {'lang': 'en', 'score': 0.12450417876243591}

# Using large model for better accuracy
print(detect("Hello, world!", low_memory=False))
# Output: {'lang': 'en', 'score': 0.98765432109876}

# Custom configuration with fallback mechanism
config = LangDetectConfig(
    cache_dir="/custom/cache/path",  # Custom model cache directory
    allow_fallback=True             # Enable fallback to small model if large model fails
)
detector = LangDetector(config)

try:
    result = detector.detect("Hello world", low_memory=False)
    print(result)  # {'lang': 'en', 'score': 0.98}
except DetectError as e:
    print(f"Detection failed: {e}")

# How to deal with multiline text
multiline_text = """
Hello, world!
This is a multiline text.
But we need remove \n characters or it will raise a DetectError.
"""
multiline_text = multiline_text.replace("\n", " ")  
print(detect(multiline_text))
# Output: {'lang': 'en', 'score': 0.8509423136711121}

# Multi-language detection
results = detect_multilingual(
    "Hello 世界 こんにちは", 
    low_memory=False,  # Use large model for better accuracy
    k=3               # Return top 3 languages
)
print(results)
# Output: [
#     {'lang': 'ja', 'score': 0.4}, 
#     {'lang': 'zh', 'score': 0.3}, 
#     {'lang': 'en', 'score': 0.2}
# ]
```

#### Fallbacks

We provide a fallback mechanism: when `allow_fallback=True`, if the program fails to load the **large model** (`low_memory=False`), it will fall back to the offline **small model** to complete the prediction task.

```python
# Disable fallback - will raise error if large model fails to load
# But fallback disabled when custom_model_path is not None, because its a custom model, we will directly use it.
import tempfile
config = LangDetectConfig(
    allow_fallback=False, 
    custom_model_path=None,
    cache_dir=tempfile.gettempdir(),
    )
detector = LangDetector(config)

try:
    result = detector.detect("Hello world", low_memory=False)
except DetectError as e:
    print("Model loading failed and fallback is disabled")
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

### Load Custom Models

```python
# Load model from local file
config = LangDetectConfig(
    custom_model_path="/path/to/your/model.bin",  # Use local model file
    disable_verify=True                     # Skip MD5 verification
)
detector = LangDetector(config)
result = detector.detect("Hello world")
```

### Splitting Text by Language 🌐

For text splitting based on language, please refer to the [split-lang](https://github.com/DoodleBears/split-lang)
repository.

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
