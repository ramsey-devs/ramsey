# Ramsey

[![active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![ci](https://github.com/ramsey-devs/ramsey/actions/workflows/ci.yaml/badge.svg)](https://github.com/ramsey-devs/ramsey/actions/workflows/ci.yaml)
[![coverage](https://codecov.io/gh/ramsey-devs/ramsey/branch/main/graph/badge.svg?token=dn1xNBSalZ)](https://codecov.io/gh/ramsey-devs/ramsey)
[![documentation](https://readthedocs.org/projects/ramsey/badge/?version=latest)](https://ramsey.readthedocs.io/en/latest/?badge=latest)
[![version](https://img.shields.io/pypi/v/ramsey.svg?colorB=black&style=flat)](https://pypi.org/project/ramsey/)

> Probabilistic deep learning using JAX

## About

Ramsey is a library for probabilistic deep learning using [JAX](https://github.com/google/jax),
[Flax](https://github.com/google/flax) and [NumPyro](https://github.com/pyro-ppl/numpyro).  Its scope covers

- neural processes (vanilla, attentive, Markovian, convolutional, ...),
- neural Laplace and Fourier operator models,
- etc.

## Example usage

You can, for instance, construct a simple neural process like this:

```python
from flax import nnx

from ramsey import NP
from ramsey.nn import MLP  # just a flax.nnx module

def get_neural_process(in_features, out_features):
  dim = 128
  np = NP(
    latent_encoder=(
      MLP(in_features, [dim, dim], rngs=nnx.Rngs(0)),
      MLP(dim, [dim, dim * 2], rngs=nnx.Rngs(1))
    ),
    decoder=MLP(in_features, [dim, dim, out_features * 2], rngs=nnx.Rngs(2))
  )
  return np

neural_process = get_neural_process(1, 1)
```

The neural process above takes a decoder and a set of two latent encoders as arguments. All of these are typically `flax.nnx` MLPs, but
Ramsey is flexible enough that you can change them, for instance, to CNNs or RNNs.

Ramsey provides a unified interface where each method implements (at least) `__call__` and `loss`
functions to transform a set of inputs and compute a training loss, respectively:

```python
from jax import random as jr
from ramsey.data import sample_from_sine_function

data = sample_from_sine_function(jr.key(0))
x_context, y_context = data.x[:, :20, :],  data.y[:, :20, :]
x_target, y_target = data.x, data.y

# make a prediction
pred = neural_process(
  x_context=x_context,
  y_context=y_context,
  x_target=x_target,
)

# compute the loss
loss = neural_process.loss(
  x_context=x_context,
  y_context=y_context,
  x_target=x_target,
  y_target=y_target
)
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

See also the installation instructions for [JAX](https://github.com/google/jax), if you plan to use Ramsey on GPU/TPU.

## Contributing

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
["good first issue"](https://github.com/ramsey-devs/ramsey/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

In order to contribute:

1) Clone Ramsey and install  `uv` from [here](https://github.com/astral-sh/uv),
2) create a new branch locally `git checkout -b feature/my-new-feature` or `git checkout -b issue/fixes-bug`,
3) install all dependencies via `uv sync --all-extras`,
4) implement your contribution and ideally a test case,
5) test it by calling `make format`, `make lints` and `make tests` on the (Unix) command line,
6) submit a PR 🙂

## Why Ramsey

Just as the names of other probabilistic languages are inspired by researchers in the field
(e.g., Stan, Edward, Turing), Ramsey takes its name from one of my favourite philosophers/mathematicians, [Frank Ramsey](https://plato.stanford.edu/entries/ramsey/).
