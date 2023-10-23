# Ramsey

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/ramsey/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/ramsey/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/ramsey-devs/ramsey/branch/main/graph/badge.svg?token=dn1xNBSalZ)](https://codecov.io/gh/ramsey-devs/ramsey)
[![codacy](https://app.codacy.com/project/badge/Grade/ed13460537fd4ac099c8534b1d9a0202)](https://www.codacy.com/gh/ramsey-devs/ramsey/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ramsey-devs/ramsey&amp;utm_campaign=Badge_Grade)
[![documentation](https://readthedocs.org/projects/ramsey/badge/?version=latest)](https://ramsey.readthedocs.io/en/latest/?badge=latest)
[![version](https://img.shields.io/pypi/v/ramsey.svg?colorB=black&style=flat)](https://pypi.org/project/ramsey/)

> Probabilistic deep learning using JAX

## About

Ramsey is a library for probabilistic modelling using [JAX](https://github.com/google/jax),
[Flax](https://github.com/google/flax) and [NumPyro](https://github.com/pyro-ppl/numpyro).
It offers high quality implementations of neural processes, Gaussian processes, Bayesian time series and state-space models, clustering processes,
and everything else Bayesian.

Ramsey makes use of

- Flax`s module system for models with trainable parameters (such as neural or Gaussian processes),
- NumPyro for models where parameters are endowed with prior distributions (such as Gaussian processes, Bayesian neural networks, ARMA models)

and is hence aimed at being fully compatible with both of them.

## Example usage

You can, for instance, construct a simple neural process like this:

```python
from jax import random as jr

from ramsey import NP
from ramsey.nn import MLP
from ramsey.data import sample_from_sine_function

def get_neural_process():
    dim = 128
    np = NP(
        decoder=MLP([dim] * 3 + [2]),
        latent_encoder=(
            MLP([dim] * 3), MLP([dim, dim * 2])
        )
    )
    return np

key = jr.PRNGKey(23)
data = sample_from_sine_function(key)

neural_process = get_neural_process()
params = neural_process.init(key, x_context=data.x, y_context=data.y, x_target=data.x)
```

The neural process takes a decoder and a set of two latent encoders as arguments. All of these are typically MLPs, but
Ramsey is flexible enough that you can change them, for instance, to CNNs or RNNs. Once the model is defined, you can initialize
its parameters just like in Flax.

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

See also the installation instructions for [JAX](https://github.com/google/jax), if
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
