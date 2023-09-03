from typing import Iterable, Optional

from flax import linen as nn
from jax import numpy as jnp, Array

from ramsey._src.experimental.bayesian_neural_network.bayesian_linear import (
    BayesianLinear,
)
from ramsey._src.family import Family, Gaussian


class BNN(nn.Module):
    """
    A Bayesian neural network.

    The BNN layers can a mix of Bayesian layers and conventional layers.
    The training objective is the ELBO and is calculated according to [1].

    Attributes
    ----------
    layers: Iterable[hk.Module]
        layers of the BNN
    family: Family
        exponential family of the response

    References
    ----------
    [1] Blundell C., Cornebise J., Kavukcuoglu K., Wierstra D.
        "Weight Uncertainty in Neural Networks".
        ICML, 2015.
    """

    layers: Iterable[nn.Module]
    family: Family = Gaussian()

    def setup(self):
        self._layers = self.layers
        self._family = self.family

    def __call__(self, x: Array, **kwargs):
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
        # see Murphy (2023), chp 21.2.2, eqn 21.17
        elbo = likelihood - kl_div
        return likelihood_fn, -elbo

    def _as_family(self, x):
        return self._family(x)
