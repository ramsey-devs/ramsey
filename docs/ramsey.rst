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
   :members: __call__

.. autoclass:: ANP
   :members: __call__

.. autoclass:: DANP
   :members: __call__

Train functions
---------------

.. autosummary::
    train_neural_process

.. autofunction:: train_neural_process
