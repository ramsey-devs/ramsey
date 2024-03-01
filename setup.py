from os.path import abspath, dirname

from setuptools import find_packages, setup

PROJECT_PATH = dirname(abspath(__file__))


def readme():
    with open("README.md") as fl:
        return fl.read()

setup(
    name="ramsey",
    version="0.2.2",
    description="Probabilistic deep learning using JAX",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ramsey-devs/ramsey",
    author="The Ramsey developers",
    license="Apache 2.0",
    keywords=[
        "Bayes",
        "jax",
        "probabilistic deep learning",
        "probabilistic models",
        "neural process",
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "chex",
        "flax>=0.7.2",
        "jax>=0.4.4",
        "jaxlib>=0.4.4",
        "numpyro",
        "optax",
        "pandas",
        "rmsyutls",
        "tqdm",
    ],
    extras_require={
        "dev": ["pre-commit", "black", "isort", "pylint", "tox", "pytest"],
        "examples": ["matplotlib"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
