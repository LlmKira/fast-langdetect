# fast-langdetect 🚀

[![PyPI version](https://badge.fury.io/py/fast-langdetect.svg)](https://badge.fury.io/py/fast-langdetect)
[![Downloads](https://pepy.tech/badge/fast-langdetect)](https://pepy.tech/project/fast-langdetect)
[![Downloads](https://pepy.tech/badge/fast-langdetect/month)](https://pepy.tech/project/fast-langdetect/)

## Overview

**fast-langdetect** provides ultra-fast and highly accurate language detection based on FastText, a library developed by
Facebook. This package is 80x faster than traditional methods and offers 95% accuracy.

It supports Python versions 3.9 to 3.12.

Support offline usage.

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

> [!NOTE]
> This function assumes to be given a single line of text. *You should remove `\n` characters before passing the text.*
> If the sample is too long or too short, the accuracy will decrease (for example, in the case of too short, Chinese
> will be predicted as Japanese).

```python

from fast_langdetect import detect, detect_multilingual

# Single language detection
print(detect("Hello, world!"))
# Output: {'lang': 'en', 'score': 0.12450417876243591}

# `use_strict_mode` determines whether the model loading process should enforce strict conditions before using fallback options.
# If `use_strict_mode` is set to True, we will load only the selected model, not the fallback model.
print(detect("Hello, world!", low_memory=False, use_strict_mode=True))

# How to deal with multiline text
multiline_text = """
Hello, world!
This is a multiline text.
But we need remove `\n` characters or it will raise an ValueError.
"""
multiline_text = multiline_text.replace("\n", "")  # NOTE:ITS IMPORTANT TO REMOVE \n CHARACTERS
print(detect(multiline_text))
# Output: {'lang': 'en', 'score': 0.8509423136711121}

print(detect("Привет, мир!")["lang"])
# Output: ru

# Multi-language detection
print(detect_multilingual("Hello, world!你好世界!Привет, мир!"))
# Output: [{'lang': 'ja', 'score': 0.32009604573249817}, {'lang': 'uk', 'score': 0.27781224250793457}, {'lang': 'zh', 'score': 0.17542070150375366}, {'lang': 'sr', 'score': 0.08751443773508072}, {'lang': 'bg', 'score': 0.05222449079155922}]

# Multi-language detection with low memory mode disabled
print(detect_multilingual("Hello, world!你好世界!Привет, мир!", low_memory=False))
# Output: [{'lang': 'ru', 'score': 0.39008623361587524}, {'lang': 'zh', 'score': 0.18235979974269867}, {'lang': 'ja', 'score': 0.08473210036754608}, {'lang': 'sr', 'score': 0.057975586503744125}, {'lang': 'en', 'score': 0.05422825738787651}]
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