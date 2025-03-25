import os

from setuptools import find_packages, setup

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="agentic-rag",
    version="0.1.0",
    description="A comprehensive Agentic Retrieval-Augmented Generation (RAG) system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-username/agentic-rag",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "pdf": ["PyPDF2>=3.0.0"],
        "eval": ["rouge-score>=0.1.2"],
        "rerank": ["sentence-transformers>=2.2.0"],
        "all": [
            "PyPDF2>=3.0.0",
            "rouge-score>=0.1.2",
            "sentence-transformers>=2.2.0",
            "sumy>=0.11.0",
            "langdetect>=1.0.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "agentic-rag=agentic_rag.cli:main",
        ],
    },
)
