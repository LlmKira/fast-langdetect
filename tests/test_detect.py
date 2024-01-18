# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 下午5:28
# @Author  : sudoskys
# @File    : test_detect.py
# @Software: PyCharm


def test_muti_detect():
    from fast_langdetect.ft_detect import detect_multilingual
    result = detect_multilingual("hello world", low_memory=True)
    assert result[0].get("lang") == "en", "ft_detect error"


def test_detect_totally():
    from fast_langdetect import detect_langs
    assert detect_langs("hello world") == "EN", "ft_detect error"
    assert detect_langs("你好世界") == "ZH", "ft_detect error"
    assert detect_langs("こんにちは世界") == "JA", "ft_detect error"


def test_parse():
    from fast_langdetect import parse_sentence
    assert parse_sentence("hello world") == [{"text": "hello world", "lang": "EN", "length": 11}], "ft_detect error"
