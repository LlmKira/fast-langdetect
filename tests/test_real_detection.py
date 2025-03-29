"""Real environment tests for language detection."""

import pytest
from fast_langdetect import (
    detect,
    detect_multilingual,
    LangDetector,
    LangDetectConfig,
    DetectError,
)

# Test samples with known languages
SAMPLES = [
    ("Hello world", "en"),
    ("你好世界", "zh"),
    ("こんにちは世界", "ja"),
    ("Привет мир", "ru"),
    ("안녕하세요 세계", "ko"),
]

# Mixed language samples
MIXED_SAMPLES = [
    "Hello 世界 こんにちは",  # en-zh-ja
    "你好 world こんにちは",  # zh-en-ja
    "Bonjour 世界 hello",  # fr-zh-en
]


@pytest.mark.real
class TestRealDetection:
    """Test language detection with real FastText models."""

    @pytest.mark.parametrize("text,expected", SAMPLES)
    def test_basic_detection(self, text, expected):
        """Test basic language detection for various languages."""
        result = detect(text)
        print(result)
        assert result["lang"] == expected
        assert 0.1 <= result["score"] <= 1.0

    def test_multilingual_detection(self):
        """Test multilingual detection with mixed language text."""
        for text in MIXED_SAMPLES:
            results = detect_multilingual(text, k=3)
            assert len(results) == 3
            # 验证结果是按置信度排序的
            assert all(
                results[i]["score"] >= results[i + 1]["score"]
                for i in range(len(results) - 1)
            )

    def test_low_memory_mode(self):
        """Test detection works in low memory mode."""
        for text, expected in SAMPLES:
            result = detect(text, low_memory=True)
            assert result["lang"] == expected

    def test_strict_mode(self):
        """Test detection in strict mode."""
        result = detect(SAMPLES[0][0], use_strict_mode=True)
        assert result["lang"] == SAMPLES[0][1]

    def test_long_text(self):
        """Test detection with longer text."""
        long_text = " ".join([text for text, _ in SAMPLES])
        result = detect(long_text)
        assert "lang" in result
        assert "score" in result

    def test_very_short_text(self):
        """Test detection with very short text."""
        result = detect("Hi")
        assert "lang" in result
        assert "score" in result

    def test_custom_config(self):
        """Test detection with custom configuration."""
        config = LangDetectConfig(allow_fallback=False)
        detector = LangDetector(config)
        result = detector.detect(SAMPLES[0][0])
        assert result["lang"] == SAMPLES[0][1]

    def test_not_found_model(self):
        """Test fallback to small model when large model fails to load."""
        # 创建一个配置，指定一个不存在的大模型路径

        with pytest.raises(FileNotFoundError):
            config = LangDetectConfig(
                cache_dir="/nonexistent/path",
                custom_model_path="invalid_path",
                allow_fallback=True,
            )
            detector = LangDetector(config)
            detector.detect("Hello world", low_memory=False)

    def test_not_found_model_with_fallback(self):
        """Test fallback to small model when large model fails to load."""
        config = LangDetectConfig(
            cache_dir="/nonexistent/path",
            allow_fallback=True,
        )
        detector = LangDetector(config)
        result = detector.detect("Hello world", low_memory=False)
        assert result["lang"] == "en"
        assert 0.1 <= result["score"] <= 1.0

@pytest.mark.real
@pytest.mark.slow
class TestEdgeCases:
    """Test language detection edge cases with real models."""

    def test_empty_string(self):
        """Test detection with empty string."""
        result = detect("")
        assert "lang" in result
        assert "score" in result

    def test_special_characters(self):
        """Test detection with special characters."""
        texts = [
            "Hello! @#$%^&*()",
            "你好！@#￥%……&*（）",
            "こんにちは！＠＃＄％＾＆＊（）",
        ]
        for text in texts:
            result = detect(text)
            assert "lang" in result
            assert "score" in result

    def test_numbers_only(self):
        """Test detection with numbers only."""
        result = detect("12345")
        assert "lang" in result
        assert "score" in result

    def test_mixed_scripts(self):
        """Test detection with mixed scripts."""
        mixed_texts = [
            "Hello你好こんにちは",
            "12345 Hello 你好",
            "Hello! 你好! こんにちは!",
        ]
        for text in mixed_texts:
            results = detect_multilingual(text, k=3)
            assert len(results) == 3
