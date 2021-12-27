import re
from os.path import abspath, dirname, join

from setuptools import find_packages, setup

PROJECT_PATH = dirname(abspath(__file__))


def readme():
    with open("README.md") as fl:
        return fl.read()


def _read_requirements(fl):
    with open(fl) as fh:
        return fh.read().splitlines()


def _version():
    version = None
    for line in open(join(PROJECT_PATH, "ramsey", "__init__.py")):
        if line.startswith("__version__"):
            version = re.match(r"__version__.*(\d+\.\d+\.\d+).*", line).group(1)
    if version is None:
        raise ValueError("couldn't parse version number from __init__.py")
    return version


setup(
    name="ramsey",
    version=_version(),
    description="A library for probabilistic models using Haiku and JAX",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dirmeier/ramsey",
    author="Simon Dirmeier",
    author_email="simon.dirmeier@protonmail.com",
    license="Apache 2.0",
    keywords="bayes jax probabilistic models gaussian process neural process",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "jaxlib",
        "jax",
        "chex",
        "optax",
        "dm-haiku",
        "numpyro",
    ],
    extras_require={"dev": ["pre-commit", "black", "isort", "pylint", "tox"]},
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
