from setuptools import setup, find_packages


def readme():
    with open('README.md') as fl:
        return fl.read()


def _read_requirements(fl):
    with open(fl) as fh:
        return fh.read().splitlines()

setup(
  name='pax',
  version='0.0.1',
  description='Probabilistic models using Jax',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/dirmeier/pax',
  author='Simon Dirmeier',
  author_email='simon.dirmeier@protonmail.com',
  license='Apache 2.0',
  keywords='bayes jax probabilistic models gaussian process neural process',
  packages=find_packages(),
  include_package_data=True,
  python_requires='>=3.8',
  install_requires=[

  ],
  extras_require={
      'dev': [
          'pre-commit',
          'black',
          'tox'
      ],
      'doc': _read_requirements('docs/requirements.txt'),
  },
  classifiers=[
    'Development Status :: 1 - Planning',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'
  ]
)
