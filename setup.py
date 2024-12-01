from setuptools import setup, find_packages

setup(
    name="intellifun",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "anthropic",
        "boto3",
        "pydantic",
        "rich"
    ],
    author="Eric Wong",
    author_email="eric.wong@proenergy.vip",
    description="A flexible library for building AI agents with multiple LLM backends",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ProEnergyVIP/intellifun",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 