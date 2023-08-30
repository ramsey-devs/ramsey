import warnings
from typing import Iterable, Optional

import haiku as hk
from jax import numpy as jnp

from ramsey._src.contrib.bayesian_neural_network.bayesian_linear import (
    BayesianLinear,
)
from ramsey._src.family import Family, Gaussian


class BNN(hk.Module):
    """
    Bayesian neural network

    Implements a Bayesian neural network. The BNN layers can a mix of Bayesian
    layers and conventional layers. The training objective is the ELBO and is
    calculated according to [1].

    References
    ----------
    [1] Blundell C., Cornebise J., Kavukcuoglu K., Wierstra D.
        "Weight Uncertainty in Neural Networks".
        ICML, 2015.
    """

    def __init__(
        self,
        layers: Iterable[hk.Module],
        family: Family = Gaussian(),
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Instantiates a Bayesian neural network

        Parameters
        ----------
        layers: Iterable[hk.Module]
            layers of the BNN
        family: Family
            exponential family of the response
        name: Optional[str]
            name of the layer
        kwargs: keyword arguments
            you can supply the initializers for the parameters of the likelihood
            as keyword arguments. For instance, if your likelihood belongs to
            a Gaussian family hk.initializers.Initializer objects with names
            scale_init, if it is a NegativeBinomial the initializer is called
            concentration_init
        """

        super().__init__(name=name)
        self._layers = layers
        self._family = family
        self._kwargs = kwargs
        warnings.warn(
            "The BNN class has not been tested properly. Use with caution!"
        )

    def __call__(self, x: jnp.ndarray, **kwargs):
        if "y" in kwargs:
            y = kwargs["y"]
            return self._loss(x, y)

        for layer in self._layers:
            if isinstance(layer, BayesianLinear):
                x = layer(x, is_training=False)
            else:
                x = layer(x)
        return self._as_family(x)

    def _loss(self, x, y):
        kl_div = 0.0
        for layer in self._layers:
            if isinstance(layer, BayesianLinear):
                x, kl_contribution = layer(x, is_training=True)
                kl_div += kl_contribution
            else:
                x = layer(x)

        likelihood_fn = self._as_family(x)
        likelihood = jnp.sum(likelihood_fn.log_prob(y))

        # we are building the ELBO by composing it as
        # E_q[p(y \mid z)] - KL[q(z \mid y) || p(z)] and hence
        # see Murphy (2023), chp. 21.17
        elbo = likelihood - kl_div
        return likelihood_fn, -elbo

    def _as_family(self, x):
        return self._family(x)
