import distrax
from jax import numpy as jnp


class Permutation(distrax.Bijector):
    """
    Doc
    """

    def __init__(self, permutation, event_ndims_in: int):
        super().__init__(event_ndims_in)
        self.permutation = permutation

    def forward_and_log_det(self, x):
        return x[..., self.permutation], jnp.full(jnp.shape(x)[:-1], 0.0)

    def inverse_and_log_det(self, y):
        size = self.permutation.size
        permutation_inv = (
            jnp.zeros(size, dtype=jnp.result_type(int))
            .at[self.permutation]
            .set(jnp.arange(size))
        )
        return y[..., permutation_inv], jnp.full(jnp.shape(y)[:-1], 0.0)
