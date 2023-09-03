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

Bayesian neural network
~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: BNN
    :members: __call__

Recurrent attentive neural process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: RANP
    :members: __call__

Modules
-------

.. autosummary::
    BayesianLinear

BayesianLinear
~~~~~~~~~~~~~~

..  autoclass:: BayesianLinear
    :members: __call__