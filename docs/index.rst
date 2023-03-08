:github_url: https://github.com/ramsey-devs/ramsey/

Ramsey: probabilistic modelling using Haiku
===========================================

Ramsey is a library for probabilistic modelling using `Haiku <https://github.com/deepmind/dm-haiku>`_ and `JAX <https://github.com/google/jax>`_.
It builds upon the same module system that Haiku is using and is hence fully compatible with its API. Ramsey implements **probabilistic** models, such as neural processes, Gaussian processes,
Bayesian neural networks, Bayesian timeseries models and state-space-models, and more.

Example
-------

Ramsey uses to Haiku's module system to construct probabilistic models
and define parameters. For instance, a simple neural process can be constructed like this:

.. code-block:: python

    import haiku as hk
    import jax.random as random
    from ramsey.data import sample_from_sinus_function
    from ramsey.models import NP

    def neural_process(**kwargs):
        dim = 128
        np = NP(
            decoder=hk.nets.MLP([dim] * 3 + [2]),
            latent_encoder=(
                hk.nets.MLP([dim] * 3), hk.nets.MLP([dim, dim * 2])
            )
        )
        return np(**kwargs)

    (x, y), _ = sample_from_sinus_function(random.PRNGKey(0))

    neural_process = hk.transform(neural_process)
    params = neural_process.init(
        random.PRNGKey(1), x_context=x, y_context=y, x_target=x
    )


Why Ramsey
----------

Just as the names of other probabilistic languages are inspired by researchers in the field
(e.g., Stan, Edward, Turing), Ramsey takes its name from one of my favourite philosophers/mathematicians,
`Frank Ramsey <https://plato.stanford.edu/entries/ramsey/>`_.

Installation
------------

To install from PyPI, call:

.. code-block:: bash

    pip install ramsey

To install the latest GitHub <RELEASE>, just call the following on the
command line:

.. code-block:: bash

    pip install git+https://github.com/ramsey-devs/ramsey@<RELEASE>

See also the installation instructions for `Haiku <https://github.com/deepmind/dm-haiku>`_ and `JAX <https://github.com/google/jax>`_, if
you plan to use Ramsey on GPU/TPU.

Contributing
------------

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
`"good first issue" <https://github.com/ramsey-devs/ramsey/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_.

In order to contribute:

1) Clone Ramsey and install it and its dev dependencies via :code:`pip install -e '.[dev]'`,
2) create a new branch locally :code:`git checkout -b feature/my-new-feature` or :code:`git checkout -b issue/fixes-bug`,
3) implement your contribution,
4) test it by calling :code:`tox` on the (Unix) command line,
5) submit a PR 🙂

License
-------

Ramsey is licensed under the Apache 2.0 License.


..  toctree::
    :maxdepth: 1
    :hidden:

    Home <self>

..  toctree::
    :caption: Tutorials
    :maxdepth: 1
    :hidden:

    notebooks/neural_process
    notebooks/forecasting

..  toctree::
    :caption: API
    :maxdepth: 1
    :hidden:

    ramsey
    ramsey.attention
    ramsey.contrib
    ramsey.family
    ramsey.kernels
    ramsey.train
