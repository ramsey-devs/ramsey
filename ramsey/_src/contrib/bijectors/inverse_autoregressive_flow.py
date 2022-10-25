import distrax
from jax import numpy as jnp

from ramsey._src.contrib.bijectors.util import unstack


class IAF(distrax.Bijector):
    """
    Doc
    """

    def __init__(self, net, event_ndims_in: int):
        super().__init__(event_ndims_in)
        self.net = net

    def forward_and_log_det(self, x):
        shift, log_scale = unstack(self.net(x), axis=-1)
        y = x * jnp.exp(log_scale) + shift
        logdet = self._forward_log_det(log_scale)
        return y, logdet

    def _forward_log_det(self, forward_log_scale):
        return jnp.sum(forward_log_scale, axis=-1)

    def inverse_and_log_det(self, y):
        x = jnp.zeros_like(y)
        for _ in jnp.arange(x.shape[-1]):
            shift, log_scale = unstack(self.net(x), axis=-1)
            x = (y - shift) * jnp.exp(-log_scale)
        logdet = -self._forward_log_det(log_scale)
        return x, logdet
