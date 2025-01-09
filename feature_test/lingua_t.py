from lingua import LanguageDetectorBuilder

from fast_langdetect import detect_language, detect_multilingual

low_mem_detector = (LanguageDetectorBuilder
                    .from_all_languages()
                    .with_low_accuracy_mode()
                    .with_preloaded_language_models()
                    .build())
detector = (LanguageDetectorBuilder
            .from_all_languages()
            .with_preloaded_language_models()
            .build())
ja_sentence = "こんにちは世界"
print(detect_language(ja_sentence))
print(low_mem_detector.detect_language_of(ja_sentence).iso_code_639_1.name)
print("===")
ko_sentence = "안녕하세요 세계"
print(detect_language(ko_sentence))
print(low_mem_detector.detect_language_of(ko_sentence).iso_code_639_1.name)
print("===")
fr_sentence = "Bonjour le monde"
print(detect_language(fr_sentence))
print(low_mem_detector.detect_language_of(fr_sentence).iso_code_639_1.name)
print("===")
de_sentence = "Hallo Welt"
print(detect_language(de_sentence))
print(low_mem_detector.detect_language_of(de_sentence).iso_code_639_1.name)
print("===")
zh_sentence = "這些機構主辦的課程，多以基本電腦使用為主，例如文書處理、中文輸入、互聯網應用等"
print(detect_language(zh_sentence))
print(low_mem_detector.detect_language_of(zh_sentence).iso_code_639_1.name)
print("===")
es_sentence = "Hola mundo"
print(detect_language(es_sentence))
print(low_mem_detector.detect_language_of(es_sentence).iso_code_639_1.name)
print("===")

sentence = "こんにちは世界"
for result in detector.detect_multiple_languages_of(sentence):
    print(result.language)
print("===")
sentence = """
こんにちは世界
안녕하세요 세계
Hallo Welt
這些機構主辦的課程，多以基本電腦使用為主，例如文書處理、中文輸入、互聯網應用等
Bonjour le monde
"""
langs = detect_multilingual(sentence.replace("\n", " "), low_memory=False)
for lang in langs:
    print(lang)
confidence_values = detector.compute_language_confidence_values(sentence)
for confidence in confidence_values:
    if confidence.value > 0:
        print(f"{confidence.language.iso_code_639_1.name}: {confidence.value:.2f}")
print("===")
for result in low_mem_detector.detect_multiple_languages_of(sentence):
    print(result.language.iso_code_639_1.name)
