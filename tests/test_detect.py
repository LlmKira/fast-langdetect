# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 下午5:28

def test_muti_detect():
    from fast_langdetect import detect
    result = detect(
        "hello world",
        k=3,
    )
    assert result[0].get("lang") == "en", "ft_detect error"
    return True


def test_large():
    from fast_langdetect import detect
    result = detect("hello world", k=3)
    assert result[0].get("lang") == "en", "ft_detect error"
    result = detect("你好世界", k=3)
    assert result[0].get("lang") == "zh", "ft_detect error"


def test_detect():
    from fast_langdetect import detect
    assert detect("hello world", model="full")[0]["lang"] == "en", "ft_detect error"
    assert detect("你好世界", model="lite")[0]["lang"] == "zh", "ft_detect error"
    assert detect("こんにちは世界", model="full")[0]["lang"] == "ja", "ft_detect error"
    assert detect("안녕하세요 세계", model="lite")[0]["lang"] == "ko", "ft_detect error"
    assert detect("Bonjour le monde", model="full")[0]["lang"] == "fr", "ft_detect error"


def test_detect_totally():
    from fast_langdetect import detect_language
    assert detect_language("hello world") == "EN", "ft_detect error"
    assert detect_language("你好世界") == "ZH", "ft_detect error"
    assert detect_language("こんにちは世界") == "JA", "ft_detect error"
    assert detect_language("안녕하세요 세계") == "KO", "ft_detect error"
    assert detect_language("Bonjour le monde") == "FR", "ft_detect error"
    assert detect_language("Hallo Welt") == "DE", "ft_detect error"
    assert detect_language(
        "這些機構主辦的課程，多以基本電腦使用為主，例如文書處理、中文輸入、互聯網應用等"
    ) == "ZH", "ft_detect error"


def test_failed_example():
    from fast_langdetect import detect
    try:
        detect("hello world\nNEW LINE", model="lite")
    except Exception as e:
        assert isinstance(e, Exception), "ft_detect exception error"
