import pytest

def pytest_configure(config):
    """注册自定义标记。"""
    config.addinivalue_line(
        "markers",
        "slow: 标记需要较长时间运行的测试"
    )
    config.addinivalue_line(
        "markers",
        "real: 标记使用真实模型的测试"
    ) 