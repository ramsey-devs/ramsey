``ramsey.contrib``
==================

WARNING: contributed code is not native Ramsey code and subject to change.

.. currentmodule:: ramsey.contrib

.. automodule:: ramsey.contrib

Models
------

.. autosummary::
    GP
    SparseGP

GP
~~

..  autoclass:: GP
    :members: __call__

SparseGP
~~~~~~~~

..  autoclass:: SparseGP
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
