import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
  long_description = fh.read()

setuptools.setup(
  name="tinysklearn",
  version="0.0.3",
  author="lyh",
  author_email="412929473@qq.com",
  description="A small machine learning package, which can help people learn ML more easier",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/WhatAboutMyStar/MachineLearningAlgorithm",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)