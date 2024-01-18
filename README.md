# fast-langdetect

[![PyPI version](https://badge.fury.io/py/fast-langdetect.svg)](https://badge.fury.io/py/fast-langdetect)
[![Downloads](https://pepy.tech/badge/fast-langdetect)](https://pepy.tech/project/fast-langdetect)
[![Downloads](https://pepy.tech/badge/fast-langdetect/month)](https://pepy.tech/project/fast-langdetect/month)

Python 3.8-3.11 support only.

80x faster and 95% accurate language identification with Fasttext

This library is a wrapper for the language detection model trained on fasttext by Facebook. For more information, please
visit: https://fasttext.cc/docs/en/language-identification.html

This repository is patched
from [zafercavdar/fasttext-langdetect](https://github.com/zafercavdar/fasttext-langdetect#benchmark), adding
multi-language segmentation and better packaging
support.

Facilitates more accurate TTS implementation.

## Installation

```bash
pip install fast-langdetect
```

## Usage

**For more accurate language detection, please use `detect(text,low_memory=False)` to load the big model.**

**Model will be downloaded in `/tmp/fasttext-langdetect` directory when you first use it.**

```python
from fast_langdetect import detect_langs

print(detect_langs("Hello, world!"))
# [en:0.9999961853027344]

print(detect_langs("Привет, мир!"))
# [ru:0.9999961853027344]


print(detect_langs("你好，世界！"))
# [zh:0.9999961853027344]

```

## Advanced usage

```python
from fast_langdetect import detect, parse_sentence, detect_multilingual

print(detect("Hello, world!"))
# {'lang': 'en', 'score': 0.1520957201719284}

print(detect_multilingual("Hello, world!你好世界!Привет, мир!"))
# [{'lang': 'ru', 'score': 0.39008623361587524}, {'lang': 'zh', 'score': 0.18235979974269867}, {'lang': 'ja', 'score': 0.08473210036754608}, {'lang': 'sr', 'score': 0.057975586503744125}, {'lang': 'en', 'score': 0.05422825738787651}]

print(parse_sentence("你好世界！Hello, world！Привет, мир！"))
# [{'text': '你好世界！Hello, world！', 'lang': 'ZH', 'length': 18}, {'text': 'Привет, мир！', 'lang': 'UK', 'length': 12}, {'text': '', 'lang': 'EN', 'length': 0}]
```

## Accuracy

References to the [benchmark](https://github.com/zafercavdar/fasttext-langdetect#benchmark)
