import haiku as hk
import jax.numpy as np
import numpyro.distributions as dist
from chex import assert_axis_dimension, assert_axis_dimension_gt, assert_rank
from jax import nn

from ramsey.family import Family, Gaussian

__all__ = ["DeepAR"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class DeepAR(hk.Module):
    """
    DeepAR

    Implements DeepAR as described in [1]

    References
    ----------
    .. [1] Salinas, David, et al. "DeepAR: Probabilistic forecasting
       with autoregressive recurrent networks." International Journal of
       Forecasting 36.3 (2020): 1181-1191.
    """

    def __init__(self, network: hk.DeepRNN, family: Family = Gaussian()):
        """
        Instantiates a DeepAR class

        Parameters
        ----------
        network: hk.DeepRNN
            a LSTM network wrapped in a Haiku module
        family: Family
            a family object used as observation model
        """

        super().__init__()
        self._network = network
        self._family = family
        self._mean_fn = hk.Sequential(
            [
                hk.Linear(1),
                lambda x: x
                if isinstance(self._family, Gaussian)
                else nn.softplus,
            ]
        )
        self._dispersion_fn = hk.Sequential([hk.Linear(1), nn.softplus])
        if isinstance(self._family, Gaussian):
            self._fam = dist.Normal
        else:
            self._fam = dist.NegativeBinomial2

    def __call__(
        self,
        x: np.ndarray,  # pylint: disable=invalid-name
        y: np.ndarray,  # pylint: disable=invalid-name
    ):
        assert_rank([x, y], 3)
        assert_axis_dimension(x, 0, y.shape[0])

        if x.shape[1] == y.shape[1]:
            return self._loss(x, y)
        return self._predict(x, y)

    def _predict(
        self,
        x: np.ndarray,  # pylint: disable=invalid-name
        y: np.ndarray,  # pylint: disable=invalid-name
    ):
        # put the time-axis in front and the batch-axis as second
        x_swapped = np.swapaxes(x, 0, 1)
        y_swapped = np.swapaxes(y, 0, 1)
        assert_axis_dimension_gt(x_swapped, 0, y_swapped.shape[0])

        num_prediction, _, _ = x_swapped.shape
        num_observations, _, _ = y_swapped.shape

        scale = self._scaling(y_swapped)
        y_swapped = y_swapped / scale
        z_swapped_preds, last_state = self._unroll(
            x_swapped[:num_observations, :, :],
            y_swapped[:num_observations, :, :],
        )
        z_swapped_preds = z_swapped_preds[[-1], :, :]

        for i in range(num_observations, num_prediction):
            y_swapped_pred_mean = self._mean_fn(z_swapped_preds[-1, :, :])
            z_swapped_pred, last_state = self._unroll_point(
                x_swapped[i, :, :], y_swapped_pred_mean, last_state
            )
            z_swapped_preds = np.concatenate(
                [z_swapped_preds, z_swapped_pred], axis=0
            )

        return self._as_family(z_swapped_preds[1:, :, :], scale)

    def _unroll_point(
        self,
        x_swapped,  # pylint: disable=invalid-name
        y_swapped,  # pylint: disable=invalid-name
        state,
    ):
        feats = np.concatenate([x_swapped, y_swapped], axis=-1)
        z_swapped, state = self._network(feats, state)
        return z_swapped[np.newaxis, :, :], state

    def _unroll(self, x_swapped: np.ndarray, y_swapped: np.ndarray):
        assert_axis_dimension(x_swapped, 0, y_swapped.shape[0])
        assert_axis_dimension(x_swapped, 1, y_swapped.shape[1])

        num_observations, num_batches, _ = y_swapped.shape
        y_swapped = np.pad(
            y_swapped, pad_width=((1, 0), (0, 0), (0, 0)), mode="constant"
        )[:num_observations, :, :]

        feats = np.concatenate([x_swapped, y_swapped], axis=-1)
        z_swapped, state = hk.dynamic_unroll(
            self._network, feats, self._network.initial_state(num_batches)
        )

        return z_swapped, state

    @staticmethod
    def _scaling(y_swapped):
        # compute the mean over the first axis, i.e. the _time_ axis
        return 1.0 + np.mean(y_swapped, axis=0, keepdims=1)

    def _as_family(self, z_swapped, scale):
        scale_sqrt = np.sqrt(scale)
        mean_swapped = self._mean_fn(z_swapped) * scale
        dispersion_swapped = self._dispersion_fn(z_swapped) * scale_sqrt

        # we want the family to have the original axis-dimensions, hence we swap
        mean = np.swapaxes(mean_swapped, 0, 1)
        dispersion = np.swapaxes(dispersion_swapped, 0, 1)
        family = self._fam(mean, dispersion)
        return family

    def _loss(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ):
        # put the time-axis in front and the batch-axis as second
        x_swapped = np.swapaxes(x, 0, 1)
        y_swapped = np.swapaxes(y, 0, 1)

        scale = self._scaling(y_swapped)
        y_swapped = y_swapped - scale
        z_swapped, _ = self._unroll(x_swapped, y_swapped)
        family = self._as_family(z_swapped, 1)  # scale)

        assert_axis_dimension(family.mean, 0, y.shape[0])
        assert_axis_dimension(family.mean, 1, y.shape[1])
        assert_axis_dimension(family.mean, 2, y.shape[2])

        # sum up log-likelihood per batch, then take expectation
        lp__ = np.sum(np.mean(family.log_prob(y), axis=0))
        return family, -lp__
