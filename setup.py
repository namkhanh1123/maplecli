"""Setup script for OpenAI CLI"""
from setuptools import setup, find_packages

setup(
    name="openaicli",
    version="1.0.0",
    description="A beautiful CLI for OpenAI-compatible APIs",
    author="Your Name",
    py_modules=["main"],
    install_requires=[
        "rich>=10.0.0",
        "requests>=2.25.0",
    ],
    entry_points={
        "console_scripts": [
            "openaicli=main:main",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
