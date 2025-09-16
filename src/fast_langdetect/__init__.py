# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 下午4:00

from .infer import (
    LangDetector,
    LangDetectConfig,
    detect,
    FastLangdetectError,
    ModelLoadError,
)  # noqa: F401


def is_japanese(string):
    for ch in string:
        if 0x3040 < ord(ch) < 0x30FF:
            return True
    return False


def detect_language(sentence: str, *, low_memory: bool = True):
    """
    Detect language
    :param sentence: str sentence
    :param low_memory: bool (default: True) whether to use low memory mode
    :return: ZH, EN, JA, KO, FR, DE, ES, .... (two uppercase letters)
    """
    model = "lite" if low_memory else "full"
    res_list = detect(sentence, model=model, k=1)
    lang_code = res_list[0].get("lang").upper() if res_list else "EN"
    if lang_code == "JA" and not is_japanese(sentence):
        lang_code = "ZH"
    return lang_code
