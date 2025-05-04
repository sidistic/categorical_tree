from setuptools import setup, find_packages

setup(
    name="categorical_tree",
    version="0.1.0",
    packages=find_packages(),
    description="A decision tree implementation with native support for categorical features",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)