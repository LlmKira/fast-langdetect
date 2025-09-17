# -*- coding: utf-8 -*-
# @Time    : 2024/1/18 上午11:41
# @Author  : sudoskys
from fast_langdetect import (
    detect,
    detect_language,
    LangDetector,
    LangDetectConfig,
)

# 多语言候选（使用 full 模型，返回前 5 个候选）
print(detect("Hello, world!你好世界!Привет, мир!", model="full", k=5))

# 简单检测（返回列表，取前 1 个候选）
print(detect("hello world", k=1))
print(detect("你好世界", k=1))
print(detect_language("Привет, мир!"))
print(detect_language("你好世界"))
print(detect_language("こんにちは世界"))
print(detect_language("안녕하세요 세계"))
print(detect_language("Bonjour le monde"))
print(detect_language("Hallo Welt"))
print(detect_language("Hola mundo"))
print(
    detect_language(
        "這些機構主辦的課程，多以基本電腦使用為主，例如文書處理、中文輸入、互聯網應用等"
    )
)

# 当离线或无网络时，使用 full 模型可能抛出标准 I/O/网络异常或库内异常；lite 模型离线可用
try:
    print(
        detect(
            "Hello, world!你好世界!Привет, мир!",
            model="full",
            k=5,
            config=LangDetectConfig(),
        )
    )
except Exception as e:
    print(f"Detection failed: {e}")

# 使用自定义配置与实例化 Detector
config = LangDetectConfig()
detector = LangDetector(config)
# 使用大模型进行检测（无自动回退）
result = detector.detect("Hello world", model="full", k=1)
print(result)
