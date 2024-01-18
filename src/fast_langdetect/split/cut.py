# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 下午8:27
# @Author  : sudoskys
# @File    : cut.py
# @Software: PyCharm
import re


class CutSentence(object):
    @staticmethod
    def english_sentence_cut(text) -> list:
        list_ = list()
        for s_str in text.split('.'):
            if '?' in s_str:
                list_.extend(s_str.split('?'))
            elif '!' in s_str:
                list_.extend(s_str.split('!'))
            else:
                list_.append(s_str)
        return list_

    @staticmethod
    def chinese_sentence_cut(text) -> list:
        """
        中文断句
        """
        text = re.sub('([^\n])([.!?。！？]+)(?!\d|[a-zA-Z]|[^\s\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff\uac00-\ud7a3])',
                      r'\1\2\n', text)  # 根据句子末尾的标点符号进行切分
        text = re.sub('([。：:！？\?])([^’”])', r'\1\n\2', text)  # 根据中文断句符号进行切分
        text = re.sub('(\.{6})([^’”])', r'\1\n\2', text)  # 普通断句符号且后面没有引号
        text = re.sub('(\…{2})([^’”])', r'\1\n\2', text)  # 英文省略号且后面没有引号
        text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])', r'\1\n\2', text)  # 中文省略号且后面没有引号
        text = re.sub('(\. )([^a-zA-Z\d])', r'\1\n\2', text)  # 根据英文断句号+空格进行切分
        text = re.sub('\n\n+', '\n', text)  # 删除多余的换行符
        return text.split("\n")  # 断句号+引号且后面没有引号

    def cut_chinese_sentence(self, text):
        """
        中文断句
        """
        p = re.compile("“.*?”")
        listr = []
        index = 0
        for i in p.finditer(text):
            temp = ''
            start = i.start()
            end = i.end()
            for j in range(index, start):
                temp += text[j]
            if temp != '':
                temp_list = self.chinese_sentence_cut(temp)
                listr += temp_list
            temp = ''
            for k in range(start, end):
                temp += text[k]
            if temp != ' ':
                listr.append(temp)
            index = end
        return listr
