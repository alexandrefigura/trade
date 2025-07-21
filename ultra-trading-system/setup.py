from setuptools import setup, find_packages

setup(
    name="ultra-trading-system",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0",
        "python-binance>=1.0.17",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "redis>=4.3.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "aiohttp>=3.8.0",
        "websocket-client>=1.4.0",
    ],
    entry_points={
        "console_scripts": [
            "trade-system=trade_system.cli:main",
        ],
    },
)