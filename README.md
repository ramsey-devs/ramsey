# pax

[![Project Status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/pax/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/pax/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/dirmeier/pax/branch/main/graph/badge.svg)](https://codecov.io/gh/dirmeier/pax)
[![codacy](https://app.codacy.com/project/badge/Grade/98715c0867ff4136a9b3a05340a0e6d6)](https://www.codacy.com/gh/dirmeier/pax/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=dirmeier/pax&amp;utm_campaign=Badge_Grade)

> Nonparametric probabilistic models using `haiku`

## About

A library for nonparametric probabilistic models using `haiku`

## Installation

To install the latest GitHub release, just call the following on the command line:

```bash
pip install git+https://github.com/dirmeier/pax@v0.0.1
```

## Example usage

`pax` is fully compatible with `haiku`'s module system which it uses to build neural networks
and define parameters. For instance, a simple neural process can be constructed as shown below.

```python
import haiku as hk
import jax.random as random

from pax.data import sample_from_sinus_function
from pax.models import NP

def neural_process(**kwargs):
    dim = 128
    np = NP(
        decoder=hk.nets.MLP([dim] * 3 + [2]),
        latent_encoder=(hk.nets.MLP([dim] * 3), hk.nets.MLP([dim, dim * 2]))
    )
    return np(**kwargs)

neural_process = hk.transform(neural_process)
```

```python
key = random.PRNGKey(23)

(x, y), _ = sample_from_sinus_function(key)
params = neural_process.init(key, x_context=x, y_context=y, x_target=x)
```

## Author

Simon Dirmeier <a href="mailto:simon.dirmeier @ protonmail com">simon.dirmeier @ protonmail com</a>
