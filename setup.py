"""
Setup para o Sistema de Trading Ultra-Otimizado
"""
from setuptools import setup, find_packages
import os

# Ler README se existir
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

# Ler requirements se existir
requirements = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    # Requirements mínimos
    requirements = [
        "numpy>=1.21.0",
        "numba>=0.54.0",
        "pandas>=1.3.0",
        "python-binance>=1.0.0",
        "pyyaml>=5.4.0",
        "aiohttp>=3.8.0",
        "redis>=4.0.0",
        "requests>=2.26.0",
    ]

setup(
    name="ultra-trading-system",
    version="5.2.0",
    author="Trading System Team",
    author_email="",
    description="Sistema de Trading Ultra-Otimizado com Paper Trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ultra-trading-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "trade-system=trade_system.cli:main",
            "trading=trade_system.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "trade_system": ["*.yaml", "*.yml"],
    },
)
