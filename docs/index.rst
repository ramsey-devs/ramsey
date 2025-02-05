:github_url: https://github.com/ramsey-devs/ramsey/

👋 Welcome to Ramsey!
=====================

*Probabilistic deep learning using JAX*

Ramsey is a library for probabilistic modelling using `JAX <https://github.com/google/jax>`_ ,
`Flax <https://github.com/google/flax>`_ and `NumPyro <https://github.com/pyro-ppl/numpyro>`_.

Ramsey's scope covers

- neural processes (vanilla, attentive, Markovian, convolutional, ...),
- neural Laplace and Fourier operator models,
- flow matching and denoising diffusion models,
- etc.

Example
-------

You can, for instance, construct a simple neural process like this:

.. code-block:: python

    from flax import nnx

    from ramsey import NP
    from ramsey.nn import MLP

    def get_neural_process(in_features, out_features):
      dim = 128
      np = NP(
        decoder=MLP(in_features, [dim, dim, out_features * 2], rngs=nnx.Rngs(0)),\
        latent_encoder=(
          MLP(in_features, [dim, dim], rngs=nnx.Rngs(1)),
          MLP(dim, [dim, dim * 2], rngs=nnx.Rngs(2))
        )
      )
      return np

    neural_process = get_neural_process(1, 1)

    neural_process = get_neural_process()
    params = neural_process.init(key, x_context=data.x, y_context=data.y, x_target=data.x)

The neural process takes a decoder and a set of two latent encoders as arguments. All of these are typically `flax.nnx` MLPs, but
Ramsey is flexible enough that you can change them, for instance, to CNNs or RNNs. Once the model is defined, you can train
the model by accessing the ELBO given input-output pairs via

.. code-block:: python

    from jax import random as jr
    from ramsey.data import sample_from_sine_function

    key = jr.PRNGKey(0)
    data = sample_from_sine_function(key)

    x_context, y_context = data.x[:, :20, :],  data.y[:, :20, :]
    x_target, y_target = data.x, data.y
    loss = neural_process.loss(
      x_context=x_context,
      y_context=y_context,
      x_target=x_target,
      y_target=y_target
    )

Making predictions can be done like this:

.. code-block:: python

    pred = neural_process(x_context=x_context, y_context=y_context, x_target=x_target)


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

1) Clone Ramsey and install it and the package manager `uv` from `here <https://github.com/astral-sh/uv>`_.
2) create a new branch locally via :code:`git checkout -b feature/my-new-feature` or :code:`git checkout -b issue/fixes-bug`,
3) install all dependencies via `uv sync --all-extras`,
4) implement your contribution,
5) test it by calling ``make format``, ``make lints`` and ``make tests`` on the (Unix) command line,
6) submit a PR 🙂

License
-------

Ramsey is licensed under the Apache 2.0 License.

..  toctree::
    :maxdepth: 1
    :hidden:

    🏠 Home <self>
    📰 News <news>
    📚 References <references>

..  toctree::
    :caption: 🎓 Tutorials
    :maxdepth: 1
    :hidden:

    notebooks/neural_processes

..  toctree::
    :caption: 🎓 Example code
    :maxdepth: 1
    :hidden:

    examples

..  toctree::
    :caption: 🧱 API
    :maxdepth: 2
    :hidden:

    ramsey
    ramsey.data
    ramsey.experimental
    ramsey.family
    ramsey.nn
