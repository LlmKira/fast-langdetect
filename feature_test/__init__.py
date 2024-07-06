# -*- coding: utf-8 -*-
# @Time    : 2024/1/18 上午11:41
# @Author  : sudoskys


from fast_langdetect import detect, detect_multilingual, detect_language

# 测试繁体，简体，日文，英文，韩文，法文，德文，西班牙文

print(detect_multilingual("Hello, world!你好世界!Привет, мир!"))
# [{'lang': 'ja', 'score': 0.32009604573249817}, {'lang': 'uk', 'score': 0.27781224250793457}, {'lang': 'zh', 'score': 0.17542070150375366}, {'lang': 'sr', 'score': 0.08751443773508072}, {'lang': 'bg', 'score': 0.05222449079155922}]
print(detect("hello world"))

print(detect_language("Привет, мир!"))
print(detect_language("你好世界"))
print(detect_language("こんにちは世界"))
print(detect_language("안녕하세요 세계"))
print(detect_language("Bonjour le monde"))
print(detect_language("Hallo Welt"))
print(detect_language("Hola mundo"))
print(detect_language("這些機構主辦的課程，多以基本電腦使用為主，例如文書處理、中文輸入、互聯網應用等"))
