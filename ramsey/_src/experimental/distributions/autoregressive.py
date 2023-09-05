from jax import numpy as jnp
from jax import random as jr
from numpyro import distributions as dist
from numpyro.distributions import constraints

__all__ = ["Autoregressive"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class Autoregressive(dist.Distribution):
    """
    An autoregressive model.

    Attributes
    ----------
    parameters: jnp.Array
        an initializer object from Flax
    parameters: Optional[jnp.Array]
        an initializer object from Flax
    """

    arg_constraints = {
        "loc": constraints.real,
        "ar_coefficients": constraints.real_vector,
        "scale": constraints.positive,
    }
    support = constraints.real_vector
    reparametrized_params = ["loc", "scale", "ar_coefficients"]

    def __init__(self, loc, ar_coefficients, scale):
        super().__init__()
        self.loc = loc
        self.ar_coefficients = ar_coefficients
        self.scale = scale
        self.p = len(ar_coefficients)

    def sample(self, rng_key, length, initial_state, sample_shape=()):
        def body_fn(states, sample_key):
            states = jnp.atleast_1d(states)
            take = jnp.minimum(self.p, states.shape[0])
            loc = self.loc + jnp.einsum(
                "i,i->", states[-take:][::-1], self.ar_coefficients[:take]
            )
            yt = jnp.atleast_1d(dist.Normal(loc, self.scale).sample(sample_key))
            states = jnp.concatenate([states, yt], axis=-1)
            return states

        states = jnp.atleast_1d(initial_state)
        sample_keys = jr.split(rng_key, length)
        for sample_key in sample_keys:
            states = body_fn(states, sample_key)
        return states

    def log_prob(self, value):
        pass
