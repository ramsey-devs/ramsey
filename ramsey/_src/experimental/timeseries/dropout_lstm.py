from typing import Optional

from jax import numpy as jnp, Array
from flax import linen as nn


class DropoutLSTM(nn.LSTMCell):
    """
    A LSTM with Dropout
    """

    def __init__(
        self, hidden_size: int, rate: float, name: Optional[str] = None
    ):
        super().__init__(hidden_size, name)
        self._rate = rate

    def __call__(self, inputs: Array, prev_state, is_training):
        out, state = super().__call__(inputs, prev_state)
        return nn.Dropout(self._rate, deterministic=not is_training)(out), state
