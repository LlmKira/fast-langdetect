# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 下午4:23
# @Author  : sudoskys
# @File    : __init__.py.py
from typing import List

from .cut import CutSentence
from ..ft_detect import detect_text


class Parse(object):
    @staticmethod
    def _merge_cell(result: List[dict]) -> List[dict]:
        """
        合并句子
        :param result:
        :return:
        """
        _merged = []
        _cache = []
        last_lang = None
        for _result in result:
            if _result["lang"] == last_lang:
                _cache.append(_result["text"])
            else:
                if _cache:
                    # 计算列表内文本长度
                    _length = sum([len(_c) for _c in _cache])
                    _merged.append({"text": "".join(_cache), "lang": last_lang, "length": _length})
                _cache = [_result["text"]]
                last_lang = _result["lang"]
        if _cache:
            _length = sum([len(_c) for _c in _cache])
            _merged.append({"text": "".join(_cache), "lang": last_lang, "length": _length})
        return _merged

    def create_cell(self,
                    sentence: str,
                    merge_same: bool = True,
                    cell_limit: int = 150,
                    filter_space: bool = True
                    ) -> list:
        """
        分句，识别语言
        :param sentence: 句子
        :param merge_same: 是否合并相同语言的句子
        :param cell_limit: 单元最大长度
        :return:
        """
        cut = CutSentence()
        cut_list = cut.chinese_sentence_cut(sentence)
        _cut_list = []
        for _cut in cut_list:
            if len(_cut) > cell_limit:
                _text_list = [_cut[i:i + cell_limit] for i in range(0, len(_cut), cell_limit)]
                _cut_list.extend(_text_list)
            else:
                _cut_list.append(_cut)
        # 为每个句子标排语言
        _result = []
        for _cut in _cut_list:
            _lang = detect_text(_cut)
            if not filter_space:
                _result.append({"text": _cut, "lang": _lang, "length": len(_cut)})
            else:
                if _lang:
                    _result.append({"text": _cut, "lang": _lang, "length": len(_cut)})
        if merge_same:
            _result = self._merge_cell(_result)
        return _result


def parse_sentence(text: str,
                   *,
                   merge_same: bool = True,
                   cell_limit: int = 150,
                   filter_space: bool = True
                   ) -> list:
    """
    分句，识别语言
    :param text: Sentence to be parsed
    :param filter_space: Whether to strip the space
    :param merge_same: Whether to merge sentences of the same language
    :param cell_limit: Maximum length of a single unit
    :return: [{"text": "abcd", "lang": "EN", "length": 4}]
    """
    _parse = Parse()
    return _parse.create_cell(sentence=text, merge_same=merge_same, cell_limit=cell_limit, filter_space=filter_space)
