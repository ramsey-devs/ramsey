``ramsey.experimental``
=======================

.. currentmodule:: ramsey.experimental

Experimental modules such as Gaussian processes or Bayesian neural networks.

.. warning::

    Experimental code is not native Ramsey code and subject to change, and might even get deleted in the future.
    Better don't build critical code bases around the :code:`ramsey.experimental` submodule.


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

.. autoclass:: ExponentiatedQuadratic
   :members: __call__

.. autofunction:: exponentiated_quadratic

Linear
~~~~~~

.. autoclass:: Linear
   :members: __call__

.. autofunction:: linear

Periodic
~~~~~~~~~

.. autoclass:: Periodic
   :members: __call__

.. autofunction:: periodic
