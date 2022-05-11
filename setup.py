from setuptools import find_packages, setup

setup(
    name="on2logic",
    version="0.0.1",
    description="Small set of scripts for performing image analysis of IIIF images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wjm41/on2logic",
    author="William McCorkindale",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
)
