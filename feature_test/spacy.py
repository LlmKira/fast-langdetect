# -*- coding: utf-8 -*-
# @Time    : 2024/1/18 下午3:25
# @Author  : sudoskys
# @File    : spacy.py
# @Software: PyCharm
import spacy
import spacy_fastlang  # noqa: F401

nlp = spacy.blank("xx")
nlp.add_pipe("language_detector")

# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("language_detector")
doc = nlp('Life is like a box of chocolates. You never know what you are gonna get.')

assert doc._.language == 'en'
assert doc._.language_score >= 0.8
