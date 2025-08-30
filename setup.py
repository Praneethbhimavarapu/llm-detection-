#!/usr/bin/env python3
"""
Setup script for LLM Memorization Detection System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "LLM Memorization Detection System"

# Read requirements
def read_requirements(filename):
    try:
        with open(filename, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

requirements = read_requirements("requirements.txt")
dev_requirements = read_requirements("requirements-dev.txt")

setup(
    name="llm-memorization-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive system for detecting memorization in LLM outputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-memorization-detection",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "web": ["streamlit>=1.25.0", "plotly>=5.0.0"],
        "api": ["flask>=2.0.0", "flask-cors>=3.0.10"],
        "all": ["streamlit>=1.25.0", "plotly>=5.0.0", "flask>=2.0.0", "flask-cors>=3.0.10"],
    },
    entry_points={
        "console_scripts": [
            "memorization-detect=cli.memorization_cli:main",
            "memorization-web=web.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt"],
        "config": ["*.json"],
        "data": ["sample_training_data/*", "test_files/*"],
    },
    keywords="llm memorization detection machine-learning nlp artificial-intelligence",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-memorization-detection/issues",
        "Documentation": "https://github.com/yourusername/llm-memorization-detection/docs",
        "Source": "https://github.com/yourusername/llm-memorization-detection",
    },
)
