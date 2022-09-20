from typing import Optional

import haiku as hk
import jax.numpy as np
from haiku import LSTM, LSTMState


class DropoutLSTM(LSTM):
    """
    A LSTM with Dropout
    """

    def __init__(
        self, hidden_size: int, rate: float, name: Optional[str] = None
    ):
        super().__init__(hidden_size, name)
        self._rate = rate

    def __call__(self, inputs: np.ndarray, prev_state: LSTMState):
        out, state = super().__call__(inputs, prev_state)
        return hk.dropout(hk.next_rng_key(), self._rate, out), state
