from typing import Iterable, Optional

import haiku as hk
from distrax import Normal
from jax import numpy as jnp

from ramsey._src.contrib.bayesian_neural_network.BayesianLayer import (
    BayesianLayer,
)
from ramsey._src.contrib.bayesian_neural_network.BayesianLinear import (
    BayesianLinear,
)


class BayesianNeuralNetwork(hk.Module):
    """
    Bayesian Neural Network

    Implements a BNN. The BNN layers can a mix of Bayesian (BayesianLayer) and
    normal (e.g. hk.Linear) layers.

    Training objective is the ELBO and is calculated according [1].

    References
    ----------

    [1] Blundell C., Cornebise J., Kavukcuoglu K., Wierstra D.
        "Weight Uncertainty in Neural Networks".
        ICML, 2015.
    """

    def __init__(
        self,
        layers: Iterable[hk.Module],
        lklh_log_sigma_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        """
        Instantiates a Bayesian Neural Network

        Parameters
        ----------
        layers: Iterable[hk.Module]
            layers of the BNN
        lklh_log_sigma_init: Optional[hk.initializers.Initializer]
            an initializer object from Haiku or None
        name: Optional[str]
            name of the layer
        """

        super().__init__(name=name)
        self._layers = layers
        self._lklh_log_sigma_init = lklh_log_sigma_init

    def __call__(self, x: jnp.ndarray, **kwargs):

        if "y" in kwargs:
            y = kwargs["y"]
            return self._negative_elbo(x, y)

        return self._forward(x)

    def _forward(self, x):

        key = hk.next_rng_key()

        for layer in self._layers:
            if isinstance(layer, BayesianLayer):
                x = layer(x, key)
            else:
                x = layer(x)
        return x

    def _negative_elbo(self, x, y):

        dtype = x.dtype
        key = hk.next_rng_key()
        kl_div = 0

        for layer in self._layers:
            if isinstance(layer, BayesianLinear):
                x, kl_contribution = layer(x, key, is_training=True)
                kl_div += kl_contribution
            else:
                x = layer(x)

        mu = x
        sigma = self._get_sigma(dtype) * jnp.ones((mu.shape[0], 1))
        p_likelihood = Normal(loc=mu, scale=sigma)
        likelihood = jnp.sum(p_likelihood.log_prob(y))

        elbo = likelihood - kl_div

        return -elbo

    def _get_sigma(self, dtype):

        log_sigma_init = self._lklh_log_sigma_init

        if log_sigma_init is None:
            log_sigma_init = hk.initializers.RandomUniform(
                jnp.log(0.1), jnp.log(1.0)
            )

        log_sigma = hk.get_parameter(
            "lklh_log_sigma", [], dtype=dtype, init=log_sigma_init
        )
        sigma = jnp.exp(log_sigma)

        return sigma
