import haiku as hk
import jax.numpy as np
from jax import numpy as jnp

from ramsey.family import Family, Gaussian

__all__ = ["Recurrent"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class Recurrent(hk.Module):
    def __init__(self, network: hk.DeepRNN, family: Family = Gaussian()):
        super().__init__()
        self._decoder = network
        self._family = family

    def __call__(
        self,
        x_target: np.ndarray,  # pylint: disable=invalid-name
        **kwargs,
    ):
        if "y_target" in kwargs:
            return self._loss(x_target, **kwargs)

        n_batch, _, _ = x_target.shape
        target, _ = hk.dynamic_unroll(
            self._decoder,
            x_target,
            self._decoder.initial_state(n_batch),
            time_major=False,
        )
        return self._family(target)

    def _loss(
        self,
        x_target: jnp.ndarray,
        y_target: jnp.ndarray,
        **kwargs
    ):
        n_batch, _, _ = x_target.shape
        target, _ = hk.dynamic_unroll(
            self._decoder,
            x_target,
            self._decoder.initial_state(n_batch),
            time_major=False,
        )
        mvn = self._family(target)

        lpp__ = np.sum(mvn.log_prob(y_target), axis=-1, keepdims=True)
        return mvn, -jnp.sum(lpp__, axis=1), lpp__, 0
