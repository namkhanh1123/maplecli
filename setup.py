"""Setup script for MapleCLI - Enhanced OpenAI-compatible CLI with code analysis"""
from setuptools import setup, find_packages

setup(
    name="maplecli",
    version="2.0.0",
    description="A secure, feature-rich CLI for OpenAI-compatible APIs with advanced code analysis",
    author="MapleCLI Team",
    author_email="team@maplecli.dev",
    url="https://github.com/maplecli/maplecli",
    py_modules=["main"],
    install_requires=[
        "rich>=13.0.0",
        "requests>=2.31.0",
        "aiofiles>=23.0.0",
        "pathlib2>=2.3.0; python_version<'3.4'",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "coverage>=7.0.0",
        ],
        "security": [
            "bandit>=1.7.0",
            "safety>=2.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "maplecli=main:main",
            "openaicli=main:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Terminals",
        "Topic :: Utilities",
        "Security :: Developers",
    ],
    keywords="openai cli chat code analysis ai security",
    project_urls={
        "Bug Reports": "https://github.com/maplecli/maplecli/issues",
        "Source": "https://github.com/maplecli/maplecli",
        "Documentation": "https://maplecli.readthedocs.io/",
    },
)
