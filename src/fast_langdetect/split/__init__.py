# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 下午4:23
# @Author  : sudoskys
# @File    : __init__.py.py
from typing import List

from .cut import CutSentence
from ..ft_detect import detect_langs

CUT = CutSentence()


def _merge_cell(result: List[dict]) -> List[dict]:
    _merged = []
    _cache = []
    last_lang = None
    for _result in result:
        if _result["lang"] == last_lang:
            _cache.append(_result["text"])
        else:
            if _cache:
                _length = sum([len(_c) for _c in _cache])
                _merged.append({"text": "".join(_cache), "lang": last_lang, "length": _length})
            _cache = [_result["text"]]
            last_lang = _result["lang"]
    if _cache:
        _length = sum([len(_c) for _c in _cache])
        _merged.append({"text": "".join(_cache), "lang": last_lang, "length": _length})
    return _merged


def parse_sentence(sentence: str,
                   merge_same: bool = True,
                   cell_limit: int = 150,
                   filter_space: bool = True,
                   low_memory: bool = True
                   ) -> list:
    """
    Parse sentence
    :param sentence:
    :param merge_same:
    :param cell_limit:
    :param filter_space:
    :param low_memory:
    :return:
    """

    cut_list = CUT.chinese_sentence_cut(sentence)
    _cut_list = []
    for _cut in cut_list:
        if len(_cut) > cell_limit:
            _text_list = [_cut[i:i + cell_limit] for i in range(0, len(_cut), cell_limit)]
            _cut_list.extend(_text_list)
        else:
            _cut_list.append(_cut)
    _result = []
    for _cut in _cut_list:
        _lang = detect_langs(_cut, low_memory=low_memory)
        if not filter_space:
            _result.append({"text": _cut, "lang": _lang, "length": len(_cut)})
        else:
            if _lang:
                _result.append({"text": _cut, "lang": _lang, "length": len(_cut)})
    if merge_same:
        _result = _merge_cell(_result)
    return _result
