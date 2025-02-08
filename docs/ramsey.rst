``ramsey``
==========

.. currentmodule:: ramsey

Module containing all implemented probabilistic models and training functions.

Models
------

.. autosummary::
    NP
    ANP
    DANP

Neural processes
~~~~~~~~~~~~~~~~

.. autoclass:: NP
   :members: __call__, loss

.. autoclass:: ANP
   :members: __call__, loss

.. autoclass:: DANP
   :members: __call__, loss

Train functions
---------------

.. autosummary::
    train_neural_process

.. autofunction:: train_neural_process
