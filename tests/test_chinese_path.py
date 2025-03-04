import os
import tempfile
import shutil
import platform
from pathlib import Path
import pytest
from fast_langdetect.infer import LangDetectConfig, LangDetector

@pytest.mark.skipif(platform.system() != "Windows", reason="Windows path test")
def test_model_loading_with_chinese_path():
    # 创建包含中文字符的临时目录
    temp_dir = Path(tempfile.gettempdir()) / "测试_模型_路径"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 使用项目中已有的模型文件
        # 查找项目根目录
        project_root = Path(__file__).parent.parent
        model_path = project_root / "src" / "fast_langdetect" / "resources" / "lid.176.ftz"
        
        if not model_path.exists():
            pytest.skip(f"Model file does not exist: {model_path}")
            
        # 复制模型文件到中文路径
        chinese_model_path = temp_dir / "测试模型.ftz"
        shutil.copy2(model_path, chinese_model_path)
        
        # 正确使用自定义模型路径
        config = LangDetectConfig(
            custom_model_path=str(chinese_model_path),
            allow_fallback=False
            )
        detector = LangDetector(config)
        result = detector.detect("This is a test")
        
        assert "lang" in result
        assert "score" in result
    finally:
        # 清理
        shutil.rmtree(temp_dir, ignore_errors=True) 