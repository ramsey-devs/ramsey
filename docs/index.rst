:github_url: https://github.com/ramsey-devs/ramsey/

👋 Welcome to Ramsey!
=====================

Ramsey is a library for probabilistic modelling using `JAX <https://github.com/google/jax>`_ ,
`Flax <https://github.com/google/flax>`_ and `NumPyro <https://github.com/pyro-ppl/numpyro>`_.
It offers high quality implementations of neural processes, Gaussian processes, Bayesian time series and state-space models, clustering processes,
and everything else Bayesian.

Ramsey makes use of

- Flax`s module system for models with trainable parameters (such as neural or Gaussian processes),
- NumPyro for models where parameters are endowed with prior distributions (such as Gaussian processes, Bayesian neural networks, etc.)

and is hence aimed at being fully compatible with both of them.

Example usage
-------------

You can, for instance, construct a simple neural process like this:

.. code-block:: python

    from jax import random as jr

    from ramsey import NP, MLP
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

The neural process takes a decoder and a set of two latent encoders as arguments. All of these are typically MLPs, but
Ramsey is flexible enough that you can change them, for instance, to CNNs or RNNs. Once the model is defined, you can initialize
its parameters just like in Flax.

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

See also the installation instructions for `JAX <https://github.com/google/jax>`_, if
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

    🏠 Home <self>
    📰 News <news>

..  toctree::
    :caption: 🎓 Tutorials
    :maxdepth: 1
    :hidden:

    notebooks/on_inference
    notebooks/neural_process

..  toctree::
    :caption: 🎓 Example code
    :maxdepth: 1
    :hidden:

    examples

..  toctree::
    :caption: 🧱 API
    :maxdepth: 1
    :hidden:

    ramsey
    ramsey.data
    ramsey.experimental
    ramsey.family
