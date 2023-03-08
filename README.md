# Ramsey

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/ramsey/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/ramsey/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/ramsey-devs/ramsey/branch/main/graph/badge.svg?token=dn1xNBSalZ)](https://codecov.io/gh/ramsey-devs/ramsey)
[![codacy](https://app.codacy.com/project/badge/Grade/ed13460537fd4ac099c8534b1d9a0202)](https://www.codacy.com/gh/ramsey-devs/ramsey/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ramsey-devs/ramsey&amp;utm_campaign=Badge_Grade)
[![documentation](https://readthedocs.org/projects/ramsey/badge/?version=latest)](https://ramsey.readthedocs.io/en/latest/?badge=latest)
[![version](https://img.shields.io/pypi/v/ramsey.svg?colorB=black&style=flat)](https://pypi.org/project/ramsey/)

> Probabilistic modelling using Haiku and JAX

## About

Ramsey is a library for probabilistic modelling using [Haiku](https://github.com/deepmind/dm-haiku) and [JAX](https://github.com/google/jax).
It builds upon the same module system that Haiku is using and is hence fully compatible with Haiku's and NumPyro's API.

## Example usage

Ramsey uses to Haiku's module system to construct probabilistic models
and define parameters. For instance, a simple neural process can be constructed like this:

```python
import haiku as hk
import jax.random as random

from ramsey import NP
from ramsey.data import sample_from_sine_function

def neural_process(**kwargs):
    dim = 128
    np = NP(
        decoder=hk.nets.MLP([dim] * 3 + [2]),
        latent_encoder=(
            hk.nets.MLP([dim] * 3), hk.nets.MLP([dim, dim * 2])
        )
    )
    return np(**kwargs)

key = random.PRNGKey(23)
(x, y), _ = sample_from_sine_function(key)

neural_process = hk.transform(neural_process)
params = neural_process.init(key, x_context=x, y_context=y, x_target=x)
```

## Installation

To install from PyPI, call:

```bash
pip install ramsey
```

To install the latest GitHub <RELEASE>, just call the following on the
command line:

```bash
pip install git+https://github.com/ramsey-devs/ramsey@<RELEASE>
```

See also the installation instructions for [Haiku](https://github.com/deepmind/dm-haiku) and [JAX](https://github.com/google/jax), if
you plan to use Ramsey on GPU/TPU.

## Contributing

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
["good first issue"](https://github.com/ramsey-devs/ramsey/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

In order to contribute:

1) Install Ramsey and dev dependencies via `pip install -e '.[dev]'`,
2) test your contribution/implementation by calling `tox` on the (Unix) command line before submitting a PR.

## Why Ramsey

Just as the names of other probabilistic languages are inspired by researchers in the field
(e.g., Stan, Edward, Turing), Ramsey takes its name from one of my favourite philosophers/mathematicians, [Frank Ramsey](https://plato.stanford.edu/entries/ramsey/).
