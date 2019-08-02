#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ktk",
    version="0.0.1",
    author="Felix Chenier",
    author_email="felix@felixchenier.com",
    description="KTK - Kinesiology Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://felixchenier.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
