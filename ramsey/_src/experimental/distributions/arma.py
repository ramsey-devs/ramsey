from jax import numpy as jnp
from jax import random as jr
from numpyro import distributions as dist
from numpyro.distributions import constraints

__all__ = ["ARMA"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class ARMA(dist.Distribution):
    """
    An autoregressive moving-average model.

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
        "ma_coefficients": constraints.real_vector,
        "scale": constraints.positive,
    }
    support = constraints.real_vector
    reparametrized_params = [
        "loc",
        "scale",
        "ar_coefficients",
        "ma_coefficients",
    ]

    def __init__(self, loc, ar_coefficients, ma_coefficients, scale):
        super().__init__()
        self.loc = loc
        self.ar_coefficients = ar_coefficients
        self.ma_coefficients = ma_coefficients
        self.scale = scale
        self.p = len(ar_coefficients)

    def sample(self, rng_key, length, initial_state, sample_shape=()):
        def body_fn(states, errors, sample_key):
            states = jnp.atleast_1d(states)
            errors = jnp.atleast_1d(errors)
            take = jnp.minimum(self.p, states.shape[0])
            loc = self.loc
            loc += jnp.einsum(
                "i,i->", states[-take:][::-1], self.ar_coefficients[:take]
            )
            loc += jnp.einsum(
                "i,i->", errors[-take:][::-1], self.ma_coefficients[:take]
            )
            yt = jnp.atleast_1d(dist.Normal(loc, self.scale).sample(sample_key))
            states = jnp.concatenate([states, yt], axis=-1)
            errors = jnp.concatenate([errors, yt - loc], axis=-1)
            return states, errors

        states = jnp.atleast_1d(initial_state)
        errors = jnp.zeros_like(states)
        sample_keys = jr.split(rng_key, length)
        for sample_key in sample_keys:
            states, errors = body_fn(states, errors, sample_key)
        return states

    def log_prob(self, value):
        pass
