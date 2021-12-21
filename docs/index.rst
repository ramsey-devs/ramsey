:github_url: https://github.com/dirmeier/pax/tree/main/docs

Pax documentation
=================

Pax is a library for nonparametric probabilistic modelling using Haiku and Jax.
It builds upon the same module system that Haiku is using and hence fully compatible with
Haiku's and JAX's API.

.. code-block:: python

    import haiku as hk
    import jax.random as random
    from pax.data import sample_from_sinus_function
    from pax.models import NP

    def neural_process(**kwargs):
        dim = 128
        np = NP(
            decoder=hk.nets.MLP([dim] * 3 + [2]),
            latent_encoder=(
                hk.nets.MLP([dim] * 3), hk.nets.MLP([dim, dim * 2])
            )
        )
        return np(**kwargs)

    neural_process = hk.transform(neural_process)

    (x, y), _ = sample_from_sinus_function(random.PRNGKey(0))
    params = neural_process.init(random.PRNGKey(1), x_context=x, y_context=y, x_target=x)

Installation
------------

To install the latest GitHub release, just call the following on the command line:

```bash
pip install git+https://github.com/dirmeier/pax@v0.0.1
```

See also the installation instructions for Haiku and Jax.

..  toctree::
    :caption: Examples
    :maxdepth: 2

    notebooks/neural_process
    notebooks/neural_process

..  toctree::
    :caption: API
    :maxdepth: 2

    api

License
-------

Pax is licensed under the Apache 2.0 License
