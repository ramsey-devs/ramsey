``ramsey.experimental``
=======================

WARNING: experimental code is not native Ramsey code and subject to change and might even get deleted in the future when
it doesn't fit properly into the Ramsey ecosystem. Experimental code is often trial code that has been implemented
to test the method and get experience with it. For instance, BNNs are one class which we are evaluating and considering to remove again
due to the fact that they seem to be very difficult to train (with many epochs and non-trivial optimizers).

.. currentmodule:: ramsey.experimental

.. automodule:: ramsey.experimental

Models
------

.. autosummary::
    BNN
    RANP
    GP
    SparseGP

Bayesian neural network
~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: BNN
    :members: __call__

Recurrent attentive neural process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: RANP
    :members: __call__

GP
~~

..  autoclass:: GP
    :members: __call__

SparseGP
~~~~~~~~

..  autoclass:: SparseGP
    :members: __call__

Modules
-------

.. autosummary::
    BayesianLinear

BayesianLinear
~~~~~~~~~~~~~~

..  autoclass:: BayesianLinear
    :members: __call__

Covariance functions
--------------------

.. autosummary::
    ExponentiatedQuadratic
    Linear
    Periodic
    exponentiated_quadratic
    linear
    periodic

ExponentiatedQuadratic
~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: ExponentiatedQuadratic
    :members: __call__

.. autofunction:: exponentiated_quadratic

Linear
~~~~~~

..  autoclass:: Linear
    :members: __call__

.. autofunction:: linear

Periodic
~~~~~~~~~

..  autoclass:: Periodic
    :members: __call__

.. autofunction:: periodic

Train functions
---------------

.. autosummary::
    train_gaussian_process
    train_sparse_gaussian_process

train_gaussian_process
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: train_gaussian_process


train_sparse_gaussian_process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: train_sparse_gaussian_process
