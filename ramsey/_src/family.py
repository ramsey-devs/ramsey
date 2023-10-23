import abc

import jax.nn
import numpyro.distributions as dist
from jax import numpy as jnp


# pylint: disable=too-few-public-methods
class Family(abc.ABC):
    """Distributional family."""

    def __init__(self, distribution):
        """Abstract constructor."""
        self._distribution = distribution

    @abc.abstractmethod
    def __call__(self, target: jnp.ndarray):
        """Compose a NumPyro distribution."""

    @abc.abstractmethod
    def get_canonical_parameters(self):
        """Getter for canonical parameters."""


class Gaussian(Family):
    """Family of Gaussian distributions."""

    def __init__(self, **kwargs):
        """Construct a Gaussian object."""
        super().__init__(dist.Normal)
        self._kwargs = kwargs

    def __call__(self, target: jnp.ndarray, **kwargs):
        """Compose a NumPyro distribution."""
        log_scale = kwargs.get("scale", None)
        if log_scale is not None:
            mean = target
            scale = jnp.exp(log_scale)
        else:
            mean, log_scale = jnp.split(target, 2, axis=-1)
            scale = 0.1 + 0.9 * jax.nn.softplus(log_scale)
        return dist.Normal(loc=mean, scale=scale)

    def get_canonical_parameters(self):
        """Getter for canonical parameters."""
        return "scale", dist.Normal.arg_constraints["scale"]


class NegativeBinomial(Family):
    """Family of negative binomial distributions."""

    def __init__(self):
        """Construct a NegativeBinomial object."""
        super().__init__(dist.NegativeBinomial2)

    def __call__(self, target: jnp.ndarray, **kwargs):
        """Compose a NumPyro distribution."""
        concentration = kwargs.get("concentration", None)
        if concentration is None:
            mean, concentration = jnp.split(target, 2, axis=-1)
        mean = jnp.exp(mean)
        concentration = jnp.exp(concentration)
        return dist.NegativeBinomial2(mean=mean, concentration=concentration)

    def get_canonical_parameters(self):
        """Getter for canonical parameters."""
        return (
            "concentration",
            dist.NegativeBinomial2.arg_constraints["concentration"],
        )
