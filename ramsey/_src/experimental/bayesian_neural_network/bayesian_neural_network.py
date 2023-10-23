from typing import Iterable

from flax import linen as nn
from jax import Array
from jax import numpy as jnp

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

    @nn.compact
    def __call__(self, x: Array, **kwargs):
        """Transform the inputs through the Bayesian neural network.

        Parameters
        ----------
        inputs: jax.Array
            Input data of dimension (*batch_dims, spatial_dims..., feature_dims)
        **kwargs: kwargs
            Keyword arguments can include:
            - outputs: jax.Array. If an argument called outputs is provided,
              computes the loss (negative ELBO) together with a
              predictive posterior distribution

        Returns
        -------
        Union[numpyro.distribution, Tuple[numpyro.distribution, float]]
            if 'outputs' is provided as keyword argument, returns a tuple of
            the predictive distribution and the negative ELBO which can be used
            as loss for optimzation.
            If 'outputs' is not provided, returns the predictive distribution
            only.
        """
        if "y" in kwargs:
            y = kwargs["y"]
            return self._loss(x, y)

        outputs = x
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                outputs = layer(outputs, is_training=False)
            else:
                outputs = layer(outputs)
        return self._as_family(outputs)

    def _loss(self, inputs, outputs):
        kl_div = 0.0
        hidden = inputs
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                hidden, kl_contribution = layer(hidden, is_training=True)
                kl_div += kl_contribution
            else:
                hidden = layer(hidden)

        pred_fn = self._as_family(hidden)
        loglik = jnp.sum(pred_fn.log_prob(outputs))

        # we are building the ELBO by composing it as
        # E_q[p(y \mid z)] - KL[q(z \mid y) || p(z)] and hence
        # see Murphy (2023), chp 21.2.2, eqn 21.17
        elbo = loglik - kl_div
        return pred_fn, -elbo

    def _as_family(self, inputs):
        return self.family(inputs)
