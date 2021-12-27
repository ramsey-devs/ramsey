:github_url: https://github.com/dirmeier/ramsey/

Ramsey documentation
====================

Ramsey is a library for probabilistic modelling using Haiku and JAX.
It builds upon the same module system that Haiku is using
and is hence fully compatible with Haiku's and NumPyro's API. Ramsey implements
(or rather intends to implement) neural and Gaussian process models.

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

Installation
------------


To install from PyPI, call:

.. code-block:: bash

    pip install ramsey

To install the latest GitHub <RELEASE>, just call the following on the
command line:

.. code-block:: bash

    pip install git+https://github.com/dirmeier/ramsey@<RELEASE>


See also the installation instructions for Haiku and JAX.

Why Ramsey
----------

Just as the names of other probabilistic languages are inspired by researchers in the field
(e.g., Stan, Edward, Turing), Ramsey takes
its name from one of my favourite philosophers/mathematicians,
`Frank Ramsey <https://plato.stanford.edu/entries/ramsey/>`_.

License
-------

Ramsey is licensed under the Apache 2.0 License

..  toctree::
     Home <self>
    :hidden:

..  toctree::
    :caption: Tutorials
    :maxdepth: 1
    :hidden:

    notebooks/neural_process

..  toctree::
    :maxdepth: 1
    :caption: Examples
    :hidden:

    examples/attentive_neural_process

..  toctree::
    :caption: API
    :maxdepth: 1
    :hidden:

    api
