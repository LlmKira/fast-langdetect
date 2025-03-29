import pytest

def pytest_configure(config):
    """注册自定义标记。"""
    config.addinivalue_line(
        "markers",
        "slow: Run in long progress"
    )
    config.addinivalue_line(
        "markers",
        "real: Test with real model"
    ) 